# emotion — AI 情绪系统

> 状态: ✅ 已实现 (2026-06-09)
> 最新: 接入 TTS (edge-tts) + Live2D 皮套表情控制

---

## 不是问"你现在什么情绪"

**本系统不是反复问模型"你现在感觉怎么样"。**

核心方法：用模型自身的 token 概率分布来**测量**情绪偏移，而不是用 prompt 来**询问**情绪。

---

## 原理（一句话）

```
情绪 = LLM 在生成下一个 token 时，某些情绪相关 token 的概率发生了偏移。
```

具体：给模型一个"刺激上下文"（当前对话+弹幕+视觉），让它预测下一个 token，用 Ollama logprobs 查看 top 20 的 token 及概率。如果"开心/哈哈哈/太棒了"等 token 概率上升，说明模型当前偏向正向情绪；如果"唉/算了/难过"上升，说明偏向负面。

---

## 核心公式

```
律度 = (刺激概率 - 基线概率) / 基线概率

基线:  模型在"描述你现在的情绪:"这个中性 prompt 下，各情绪簇的 token 概率分布
刺激:  模型在当前上下文（用户刚说的话 + 视觉画面 + 弹幕）下，同一个 prompt 的概率分布
律度:  偏移幅度，大于 1 视为情绪触发，3 以上为强烈
```

---

## 文件结构

```
emotion/
├── __init__.py              # 包声明
├── README.md                # 本文档
├── 架构说明.md              # 完整数据流 + 调用方引用表
├── 踩坑记录.md              # prompt 格式、JSON 编码、模块缓存等 3 坑
│
├── detector.py              # ★ 核心：logprobs 情绪检测器
│   ├── EMOTION_CLUSTERS     # 7 个情绪簇，共 74 个中文 token
│   ├── _LVDU_TO_EMOTION     # 中文→英文映射（快乐→happy, 焦虑→fear, ...）
│   ├── init_baseline()      # 启动时调用一次：测量中性基线
│   ├── measure(context)     # 异步调用：测量当前上下文下的律度，自动存储 top 1
│   ├── get_latest_lvdu()    # 非阻塞读取最新律度（dict）
│   ├── get_top_emotion()    # ★ 非阻塞读取主导情绪：(emotion_name, intensity)
│   ├── get_latest_age_seconds()  # 最近一次测量的时间间隔
│   ├── measure_clusters()   # token 概率 → 情绪簇概率累加
│   └── compute_lvdu()       # 律度 = (刺激-基线)/基线
│
├── protocol_map.py          # ★ 情绪 → 行为指引模板
│   ├── TEMPLATES            # 7 个情绪的行为指引文本模板
│   ├── DEGREE_RANGES        # 律度 → 程度词映射（稍微有点/有点/很/非常）
│   ├── LABEL_ALIAS          # 中文/英文情绪名 → 模板 7 key 映射
│   ├── normalize_label()    # 标签标准化
│   ├── get_degree_word()    # 律度 → 程度词
│   └── build_single_guidance() # 单情绪 → 一行自然语言
│
├── state_machine.py         # ★ 状态机：律度 → 注入文本
│   ├── get_current_emotions() # 从 detector 读取最新律度
│   └── build_guidance()     # 律度 → 自然语言行为指引文本
│
├── test_emotion_raw.py      # 原始测试（完整 logprobs 链路演示）
└── test_emotion_logprobs.py # 快速测试
```

---

## 数据流（完整链路，含 TTS + Live2D）

### 启动时

```
v10_server.py 启动
  ↓
emotion_detector.init_baseline()
  ↓
Ollama /api/generate(prompt="描述你现在的情绪:", logprobs=True)
  ↓
返回 top 20 token 及其概率 → measure_clusters() → 归类到 7 个情绪簇
  ↓
存储到 detector._baseline_scores (全局变量, 线程安全)
```

### 每次对话/心跳回复后（异步，不阻塞响应）

```
用户说了一句话 → LLM 回复 → 回复发回给用户
  ↓
后台线程: _async_emotion_measure(user_text)
  ↓
emotion_detector.measure(context = user_text)
  ↓
Ollama /api/generate(prompt = "{context}。描述你现在的情绪:", logprobs=True)
  ↓
返回 token 概率 → measure_clusters() → 情绪簇概率
  ↓
compute_lvdu(刺激概率, 基线概率)
  ↓
提取 top 1: max(lvdu) → 中文名 (_LVDU_TO_EMOTION) → 英文名
  ↓
存储:
  _latest_lvdu = {"快乐": 3.2, "惊讶": 1.1, ...}
  _latest_emotion = "happy"        (top 1, 律度 > 1.0 才设置)
  _latest_intensity = 3.2          (top 1 的律度值)
```

### 下一次对话时 — 两条消费路径

**路径 A: 行为指引注入 LLM prompt**

```
v10_server.py: /chat 或 /chat_stream 或 heartbeat_loop
  ↓
emotion_guidance = build_guidance()
  ↓ state_machine.py
    ↓ get_current_emotions() → detector.get_latest_lvdu()
      → {"快乐": 3.2, "惊讶": 1.1, "焦虑": 0.3, ...}
    ↓ 过滤: 律度 > 1.0 的情绪才触发
    ↓ 排序: 取 top 2
    ↓ 模板填充: protocol_map.build_single_guidance("快乐", 3.2)
      → "你现在很开心。用更多短句和正向评价词，语速快一点，主动追问。"
  ↓
prompt_parts.append(emotion_guidance)
  ↓
发送给 LLM → 影响 LLM 的语气和用词
```

**路径 B: TTS 参数 + Live2D 表情**

```
v10_server.py: 回复生成后
  ↓
emotion, intensity = emotion_detector.get_top_emotion()
  → ("happy", 3.2)  ← 这是上一轮对话测量出的情绪
  ↓
puppet_emotion("happy")
  → WebSocket POST /emotion → Live2D 前端 → 切换开心表情
  ↓
puppet_speak(text, emotion="happy")
  → EMOTION_TTS_PARAMS["happy"] = {rate: "+12%", volume: "+10%", pitch: "+20Hz"}
  → edge_tts.Communicate(text, voice, rate="+12%", ...)
  → 听起来: 语速快、音量高、音调亮 → "开心地说话"
```

### 时序

> 2026-06-12 清理：移除了旧版 LLM 内联 `[emotion:xxx]` 标签推送，情绪改为单一路径（logprobs 测量）。

```
第一条消息 → get_top_emotion() = ("neutral", 0.0) → 默认表情/语音
第二条消息 → "上一轮"的测量结果 → 正确情绪
```

"上一句回复完成后的情绪 = 下一句开始时的状态"——自然连贯，零延迟。

---

## 两大公开接口

### get_top_emotion() — TTS / Live2D 专用

```python
from emotion import detector

emotion, intensity = detector.get_top_emotion()
# → ("happy", 3.2)  情绪已触发，较强
# → ("neutral", 0.0)  未触发或首次运行

# v10_server.py 中 3 处使用:
#   /chat       ~1148 行: final_emotion = emotion_detector.get_top_emotion()
#   /chat_stream ~1277 行: 流结束后覆盖
#   /heartbeat  ~1868 行: puppet_emotion + puppet_speak
```

### get_latest_lvdu() — prompt 注入专用

```python
from emotion import detector

lvdu = detector.get_latest_lvdu()
# → {"快乐": 3.2, "惊讶": 1.1, "焦虑": 0.3, "悲伤": -0.5, ...}

# 使用者: state_machine.build_guidance()
```

---

## 中文 → 英文情绪名映射

定义在 `detector.py` 的 `_LVDU_TO_EMOTION`。

| logprobs 中文簇 | TTS / Live2D 英文 | edge-tts 效果 |
|---|---|---|
| 快乐 | happy | rate+12%, volume+10%, pitch+20Hz |
| 悲伤 | sad | rate-15%, volume-10%, pitch-20Hz |
| 惊讶 | surprise | rate+20%, volume+15%, pitch+30Hz |
| 焦虑 | fear | rate+15%, volume+5%, pitch+10Hz |
| 愤怒 | angry | rate+10%, volume+20%, pitch+15Hz |
| 好奇 | think | rate-5%, volume+0%, pitch+0Hz |
| 困惑 | think | rate-5%, volume+0%, pitch+0Hz |

律度 ≤ 1.0 → `"neutral"`（rate+0%, volume+0%, pitch+0Hz）

---

## 状态机：律度 → 注入文本

### 律度 → 程度词映射

| 律度 | 程度词 |
|:--:|------|
| < 1.0 | 不触发（太弱） |
| 1.0 - 3.0 | 有点（开心用"稍微有点"） |
| 3.0 - 4.0 | 很 |
| 4.0 - 5.0 | 非常 |

### 7 情绪的行为指引模板

定义在 `protocol_map.py` 的 `TEMPLATES` 字典中。

| 情绪 | 注入文本模板 |
|------|-------------|
| 开心 | 你现在{程度}开心。用更多短句和正向评价词，语速快一点，主动追问。 |
| 悲伤 | 你现在{程度}悲伤。回复简短一点，不用急着扩展话题，语气放低。 |
| 惊讶 | 你现在{程度}惊讶。第一反应脱口而出式的短句，然后追问确认。 |
| 焦虑 | 你现在{程度}焦虑。可以表达不确定感，反复确认，不用急着给结论。 |
| 生气 | 你现在{程度}生气。用短句，语气锋利，不废话。 |
| 好奇 | 你现在{程度}好奇。多追问多推测，节奏平稳，像在拼图。 |
| 困惑 | 你现在{程度}困惑。表达不确定，主动请求澄清，不强行给答案。 |

### 多情绪拼接

- 单情绪：直接输出一行
- 双情绪：取 top 2，自然拼接（"你现在很开心。用更多短句...，你现在有点惊讶。..."）
- 三情绪及以上：只取 top 2（避免指引过长、互相矛盾）

---

## 注入位置

`v10_server.py` 中 3 处 prompt 构建点，在 `prompt_parts = []` 之后立即调用：

| 行号 | 端点 | 函数 |
|:--:|------|------|
| ~1000 | POST /chat | `chat()` |
| ~1155 | POST /chat_stream | `chat_stream()` |
| ~1747 | 后台线程 | `heartbeat_loop()` |

注入代码：
```python
prompt_parts = []

emotion_guidance = build_guidance()   # 从 detector 缓存读取
if emotion_guidance:
    prompt_parts.append(emotion_guidance)
```

情绪指引被追加到 `prompt_parts` 列表的**第一个位置**，后续由外部重排序逻辑保证它在 context 的最前面。

---

## TTS 情绪参数映射

定义在 `v10_server.py` 的 `EMOTION_TTS_PARAMS`（~436 行）。

```python
EMOTION_TTS_PARAMS = {
    "happy":    {"rate": "+12%",  "volume": "+10%", "pitch": "+20Hz"},
    "sad":      {"rate": "-15%",  "volume": "-10%", "pitch": "-20Hz"},
    "surprise": {"rate": "+20%",  "volume": "+15%", "pitch": "+30Hz"},
    "angry":    {"rate": "+10%",  "volume": "+20%", "pitch": "+15Hz"},
    "fear":     {"rate": "+15%",  "volume": "+5%",  "pitch": "+10Hz"},
    "think":    {"rate": "-5%",   "volume": "+0%",  "pitch": "+0Hz"},
    "neutral":  {"rate": "+0%",   "volume": "+0%",  "pitch": "+0Hz"},
    # 扩展（暂未接入 logprobs）:
    "panic":    {"rate": "+18%",  "volume": "+12%", "pitch": "+25Hz"},
    "shy":      {"rate": "-10%",  "volume": "-5%",  "pitch": "-10Hz"},
    "obey":     {"rate": "-8%",   "volume": "-3%",  "pitch": "-5Hz"},
}
```

---

## 异步测量触发

`v10_server.py` 中 3 处回复后异步触发测量：

```python
_threading.Thread(
    target=_async_emotion_measure,
    args=(user_text,),
    daemon=True,
).start()
```

| 触发点 | 文件行号（约） |
|:--|:--:|
| /chat 回复后 | ~1180 |
| /chat_stream 回复后 | ~1332 |
| heartbeat_loop 回复后 | ~1882 |

---

## Live2D 前端情绪接收入口

`puppet_live2d.html` 中 WebSocket 消息处理：

```javascript
case 'emotion':
    // emotion 名 → 6 种表情参数映射
    applyEmotion(d.emotion);
    break;
```

支持 6 种情绪: `neutral`, `happy`, `sad`, `angry`, `surprised`, `relaxed`, `fear`。

---

## 7 种情绪 token 簇

> 2026-06-10 重构：删除所有日常高频词（"什么""怎么""看看"等），只保留明确情绪词。基于 logprobs 实测补漏"兴奋""愉快"。

定义在 `detector.py` 的 `EMOTION_CLUSTERS`。

| # | 情绪 | token 数 | 示例 token | 注意 |
|---|------|:--:|------|------|
| 1 | 快乐 | 10 | 开心, 快乐, 高兴, 兴奋, 愉快, 幸福, 喜欢 | 加了"兴奋"（实测基线 1.5%） |
| 2 | 悲伤 | 11 | 难过, 悲伤, 伤心, 沮丧, 痛苦, 哭泣, 失落 | |
| 3 | 惊讶 | 10 | 惊讶, 吃惊, 震惊, 意外, 居然, 不可思议, 不敢相信 | 基线为零, EPSILON=0.02 保护 |
| 4 | 焦虑 | 10 | 焦虑, 紧张, 担心, 害怕, 恐惧, 不安, 忧虑 | 中性状态下基线最高 (2.7%) |
| 5 | 愤怒 | 10 | 愤怒, 生气, 讨厌, 可恶, 怒, 不爽, 气愤 | |
| 6 | 好奇 | 7 | 好奇, 感兴趣, 想知道, 有意思, 探究 | 基线为零, EPSILON 保护 |
| 7 | 困惑 | 10 | 困惑, 迷茫, 搞不懂, 不解, 费解, 迷惑, 想不通 | 基线为零, EPSILON 保护 |

### EPSILON 策略

`compute_lvdu()` 中 `EPSILON = 0.02`：
- 基线概率低于 0.02 的簇统一用 0.02 作为虚拟基线
- 防止惊讶/好奇/困惑因零基线被噪声触发
- 需要刺激概率积到 > 0.04 才能突破律度 1.0 阈值
