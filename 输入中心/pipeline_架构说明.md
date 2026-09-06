# 迷你硬币 — 输入中心架构说明

## 整体架构

```
麦克风 / 键盘 / 外部输入
  ↓
ASR (mic_vad_asr_streaming_vad.py) → POST 18771/text {"text":"..."}   ← 只写桥
  ↓
LLM 桥 (18771)
  ├─ POST /text      ← 接收 ASR 文本，记录到 source_last
  ├─ POST /           ← 接收 prompt，转发 llama-server
  ├─ GET  /           ← 监控面板（输入/输出）
  └─ GET  /source_last ← 返回各源最新文本
       ↓
管线中心 (18772) — Web 可视化编辑 + API
  ├─ _check_trigger_sources 每秒轮询触发源桥 → 非空即标记积压（ASR 驱动，无需改外部脚本）
  ├─ 页面：拖拽排序模块、开关、编辑格式/内容/数据源、桥管理分页
  ├─ 自动拼装 prompt（模块读桥 + 拼装成品 L0+L1+L2）
  ├─ LLM 发送状态机（双门控：表达队列播完 + 发送后 2s 强制冷却 + 积压合并 + 系统注入）
  ├─ 下桥清洗（_sanitize_reply：复读删/工具闭合/非法清理）
  ├─ 配置持久化（自动保存 + 手动时间戳备份）
  └─ API：/modules /bridges /automation /chat /asr /text /config/*
       ↓
llama-server (8080)
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `mini_coin_llm_bridge.py` | LLM 桥：透明代理 + ASR 文本接收 |
| `pipeline_server.py` | 管线中心：Web UI + prompt 拼装 + ASR 触发链 |
| `pipeline_架构说明.md` | 本文档 |
| `pipeline_config.json` | 当前配置（自动保存） |
| `pipeline_config_*.json` | 手动保存的时间戳备份 |
| `prompt_builder.py` | 旧拼装器（备用，已被管线中心替代） |

## LLM 桥 (`mini_coin_llm_bridge.py`)

透明代理 + 数据中转：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 监控页面，显示最新输入/输出 |
| `/` | POST | 接收 `{"prompt":"..."}`，转发 llama-server |
| `/text` | POST | 接收 `{"text":"..."}`，存储到 `source_last` |
| `/source_last` | GET | 返回 `{source名: 最新文本}` |

数据流：`source_last` 由桥维护，管线中心轮询读取。

## 管线中心 (`pipeline_server.py`)

核心拼装引擎 + Web 管理界面。

### 模块系统

每个模块是一个 prompt 片段：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识 |
| `label` | 显示名称 |
| `format_start` | 格式开头（如 `<|im_start|>system\n`） |
| `format_end` | 格式结尾（如 `<|im_end|>`） |
| `enabled` | 是否启用 |
| `editable` | 是否允许用户编辑 |
| `content` | 静态内容（可选） |
| `sources` | 数据源列表 `[{name, endpoint, active}]` |
| `show_src` | 是否显示源名（默认 true）。打钩 → `源名\n内容`；不打钩 → 只内容。chat 特殊：固定 `对话[源名]：` 不参与勾选 (2026-08-08) |

### 内置模块

| 模块 | 默认 | 数据源 | 说明 |
|------|------|--------|------|
| 系统提示 | 开 | 固定内容 | 身份 + 工具格式，可改格式 |
| 记忆 | 开 | 静态文本 / 向量检索桥 | 当前固定，未来可配桥 |
| 历史摘要 | 关 | 自动生成 | history > 8 轮时自动摘要 |
| 对话输入 | 开 | 拉姆 18771（可配置） | 固定 `对话[{source}]：{text}` |
| 弹幕输入 | 开 | 弹幕桥 18008（可配置） | 弹幕行直接注入（show_src=false），抵达触发 LLM 回复 |
| 视觉输入 | 关 | 截图桥（预留） | - |
| 工具返回 | 关 | 工具桥（预留） | LLM 调用工具后返回 |
| 情绪指引 | 关 | FACS 桥（预留） | 实时情绪 |
| 提示词 | 关 | 静态文本 | 额外指令 |

### 输入触发链（ASR 驱动 + 对话/弹幕）

**ASR（麦克风）**：ASR 脚本只 POST 18771/text 写桥（不改外部脚本）→ 管线 `_check_trigger_sources` 每秒轮询触发源桥（弹幕+chat 全部源+自动化注入桥），任一非空 → `_mark_pending()` → 门控放行统一发送。

**手动对话 `/asr` `/text`**：写入 chat 源桥（`_write_chat_bridge`，与 ASR 写桥同路径）→ `_mark_pending()`。

**弹幕**：18008 内置桥收到 → `danmaku_reply` 线程 → `_mark_pending()`（事件触发，即时）。

**L0 哲学（history = LLM 真实看见过的）**：user 轮**不在输入到达时记录**，而是 `_do_send` **发送成功后**（清桥前）读触发源桥拼 user 轮 + assistant 轮一起进 history——发送失败不记录、冷却期被覆盖的输入当不存在（2026-08-08）。

### API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | Web 管理页面 |
| `/modules` | GET | 返回模块列表 + `source_last`（从桥轮询） |
| `/bridges` | GET | 桥管理：所有桥连通状态 + 占用 + 最后查询时间 (2026-08-08) |
| `/automation` | GET/POST | 自动化配置（静默间隔 / 提示列表 / 注入桥） (2026-08-08) |
| `/modules/order` | POST | 保存拖拽排序 |
| `/modules/toggle` | POST | 开关模块 |
| `/modules/content` | POST | 修改模块字段（format/content/sources） |
| `/asr` `/text` | POST | 接收 ASR 文本，自动触发 LLM |
| `/chat` | POST | 手动发送消息，走完整链路 |
| `/config/save` | GET | 手动保存配置（时间戳备份） |
| `/config/load` | GET | 从配置文件恢复 |

### 配置持久化

| 机制 | 说明 |
|------|------|
| 自动保存 | 每次修改实时写入 `pipeline_config.json` |
| 手动保存 💾 | 生成 `pipeline_config_YYYYMMDD_HHMMSS.json` |
| 手动读取 📂 | 从 `pipeline_config.json` 恢复 |
| 启动加载 | 自动读取配置文件，保留上次顺序 |

## 端口

| 端口 | 服务 | 文件 |
|------|------|------|
| 8080 | llama-server | `~/llama.cpp/build/bin/llama-server` |
| 18771 | LLM 桥（对话输入 + 对话/弹幕回复枢纽） | `输入中心/mini_coin_llm_bridge.py` |
| 18772 | 管线中心 | `输入中心/pipeline_server.py` |
| 18775 | 搜索工具 | `输入中心/搜索工具/search_server.py` |
| 18000-18009 | 内置桥 (10个) | 管线中心内建 |
| 18008 | 弹幕输出桥（弹幕模块，danmaku_server 发射目标） | 管线中心内建 |

## 启动顺序

```bash
# 终端 1：模型
cd ~/llama.cpp && source ~/mini_coin_venv/bin/activate
./build/bin/llama-server --model /tmp/coin_v12d_mid_q8.gguf \
  --jinja --chat-template-file /tmp/coin_template.jinja \
  -c 8192 --n-gpu-layers 99 --host 127.0.0.1 --port 8080

# 终端 2：桥
source ~/mini_coin_venv/bin/activate
python3 "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/mini_coin_llm_bridge.py"

# 终端 3：管线中心
source ~/mini_coin_venv/bin/activate
python3 "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/pipeline_server.py"

# 可选：ASR
source ~/mini_coin_venv/bin/activate
python3 "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/asr+2d8.2确认/mic_vad_asr_streaming_vad.py" --threshold 300 --end-threshold 180 --silence 1.5
```

## 上文记忆 → Prompt 实时链路 (2026-08-05)


### 数据流

```
history (对话历史) + mem_l1 (最近摘要) + mem_l2 (话题摘要)
  │
  └─ _build_final_context() 共用函数实时拼装 "拼装成品" (L0对话历史 + L1最近摘要 + L2话题摘要)
       │
       └─ daemon线程每秒 POST → 拼装成品桥(默认18000, 可配) /text {"source":"上文记忆拼装成品","text":"..."}
              │
              └─ 桥存到内存 source_last["上文记忆拼装成品"]
                     │
                     └─ build_prompt() 每秒调用 → 摘要模块 _read_from_source
                            │
                            ├─ GET {endpoint}/source_last → 取第一条非空内容
                            │
                            └─ 写入 <|im_start|>摘要\n...<|im_end|>
```

> **2026-08-08 修复**：`_push_loop` 之前只推 L0（对话历史），页面"拼装成品"预览却是 L0+L1+L2——坏 AI 假代码（注释写"对话历史+摘要"但实现没拼摘要）。已抽共用函数 `_build_final_context()`，页面预览与推桥同源，LLM 上文现在包含完整 L0+L1+L2。

### 内置桥

管线中心启动时自带 10 个桥 (18000-18009)，位于同一进程 daemon 线程。
桥实现：`BaseHTTPRequestHandler`，POST `/text` 存文本，GET `/source_last` 返回所有源。
**弹幕桥端口（弹幕模块 endpoint）收到 POST → 触发弹幕 LLM 回复** (2026-08-08)
桥管理分页（🌉 桥管理）可一键检测所有桥连通状态 + 查看占用（系统/内置/模块配置三类）。

### 模块源配置

输入视图 → 任何模块 → 编辑 → 数据源：
- 配置 endpoint (桥端口) + name (源名) → 桥上有文本即自动注入 prompt
- **兜底源语义**：不论桥上 source 名是什么，取第一个非空 text；name 只是显示标签，不是过滤条件 (2026-08-08)
- **显示源名 (show_src)**：模块配置勾选。打钩 → `源名\n内容`；不打钩 → 只内容 (2026-08-08)
- **chat 特殊**：固定 `对话[源名]：内容`，不参与 show_src 勾选（源名可配：拉姆/小明…）
- 记忆/摘要/工具返回模块：优先读桥，无桥则用静态 content
- 下拉选桥端口 (18000-18009 或自定义...)
- 桥地址保存后，`build_prompt` 实时读取桥内容

### 拼装成品配置

上文记忆分页 → 拼装成品 ⚙ 配置：
- 下拉选择 POST 目标端口
- 默认 18000，`/config/port` API 动态切换

### 实时性

- daemon 线程每秒推一次拼装成品到桥
- `build_prompt` 每次调用都即时读桥
- 前端"生成的 Prompt"预览每 3 秒拉服务端真实拼装结果

### 启动方式

```bash
source ~/mini_coin_venv/bin/activate
python3 "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/pipeline_server.py"
```

一次启动：管线中心(18772) + 5个内置桥(18000-18004) + daemon推桥循环。

### 依赖服务

| 服务 | 端口 | 启动镜像 |
|------|------|----------|
| mini_coin_llm_bridge | 18771 | 同上 venv + `mini_coin_llm_bridge.py` |
| 摘要模型 (4B LLM) | 18773 | `llama-server` |
| 检索服务 | 18774 | 同上 venv + `长期记忆/memory_retrieve_server.py` |
| ASR 语音识别 | — | `asr+2d8.2确认/startup.sh`（含 facs 18768 + Live2D 桥 18769） |

## 检索记忆链路 (2026-08-05)

```
触检服务(18774) 返回命中
  │
  └─ trigger_retrieval → last_retrieval 存结果
       │
       └─ daemon线程每秒读 last_retrieval → 取 TOP-N 命中
            │
            └─ POST → 检索桥(配置端口) /text {"source":"长期记忆检索","text":"..."}
                   │
                   └─ 记忆模块 _read_from_source → GET 桥 → 写入 <|im_start|>记忆<|im_end|>
```

- ④ 检索命中 ⚙ 配置：TOP N (1/2/3) + 桥端口
- daemon 缓存：检索空时保留上一次值，不覆盖桥
- 配置实时生效，无需重启

## ASR 语音输入链路 (2026-08-05)

```
麦克风 → VAD → ASR识别
  │
  └─ 语音结束 → POST 18771/text {"text":"xxx"}
       │
       └─ 18771桥存 source_last["拉姆"] = "xxx"
            │
            └─ 对话输入模块 _read_from_source → GET 18771/source_last
                   │
                   └─ 写入 <|im_start|>输入\n对话[拉姆]：xxx<|im_end|>
```

- 管线 `/asr` 端点触发 LLM 自动回复
- 回复写入 `last_reply_box[0]` → 输出视图可看
- 对话输入源需配桥地址 18771

## 弹幕输入链路 (2026-08-08)

```
弹幕 秒回/弹夹满 → danmaku_server 发射 → POST 18008/text
  ↓
管线内置桥存内容 → 检测端口 = 弹幕模块 endpoint → 开 danmaku_reply 线程
  ↓
danmaku_reply → 只 _mark_pending()（不记 history；L0 哲学：发送成功才算数）
  ↓
_send_loop 门控放行 → _do_send → build_prompt（弹幕段读 18008 桥）→ POST 18771 → LLM → reply
  ├─ reply → _sanitize_reply（下桥清洗）→ last_reply_box + 队列
  ├─ 发送成功 → 读触发源桥拼 user 轮 + history.append(("assistant", 清洗后 reply))
  └─ trigger_summarize + trigger_retrieval（基于干净回复）
```

- 触发条件：端口 = 弹幕模块（enabled + active source）的 endpoint
- 弹幕进 history（与对话一致；L0 哲学下发送成功才记录，历史处理已有机制不担心膨胀）
- 与对话完全相同的发送路径（_send_loop → 18771 LLM 桥 → llama-server）
- 输出经过下桥清洗（_sanitize_reply）后进 last_reply_box + 队列 + history（不贴标签不改写内容，只清复读/工具格式）

## LLM 发送状态机 (2026-08-08 初版 / 2026-08-09 双门控)

统一解决弹幕/对话/系统三个触发源的并发与合并：

```
LLM 两态：可对话 / 不可对话
  门控1（表达队列）：上一条回复还没说完（express_queue 非空，TTS/表情播放中）→ 压住
  门控2（发送后强制冷却）：_do_send 完成后 2s 内不处理积压
    → 覆盖「LLM 推理返回 → 表达队列填充」的窗口，防止推理期间新输入被 _pending_send 误清/乱序
  两道门都放行 + 有积压 → 合并一次发送（对话段+弹幕段+系统段都进 prompt）
  放行 + 30s(可配) 无发送 → 自动化注入系统提示 → 触发自言自语
```

- 全局：`_llm_busy`（防御性，_do_send try/finally 置位/解锁）/ `_pending_send` / `_last_send_ts` / `SEND_POST_COOLDOWN=2.0`；`_send_loop` daemon 线程
- 输入到达（弹幕/对话/ASR写桥）→ `_mark_pending()`（不再立即发，统一由 _send_loop 发送）
- **ASR 驱动**：`_check_trigger_sources()` 每 0.5s 轮询触发源桥，任一非空 → `_mark_pending()`（幂等）——外部输入方只写桥即可自动触发，无需改其代码 (2026-08-08)
- `_do_send()`：统一发送路径（build_prompt → 18771 → 下桥清洗 → 回复进队列；发送成功后读桥记录 user 轮 + assistant 轮进 history → 摘要/检索）
- **为什么不用固定 5s 冷却**（2026-08-09）：实测一轮交互常只有 3s——人听完就接话，甚至没听完就开始想。硬等 5s 拖慢对话节奏；改为「表达队列播完才放行」+ 2s 强制冷却兜底，快多少取决于播放时长，且不丢输入。

## 自动化·系统提示 (2026-08-08)

- 「⚙️ 自动化」分页配置：静默间隔（默认 30s）/ 提示内容列表（多条）/ 注入桥（默认 18002）
- 触发：静默达到间隔 → `random.choice(sys_prompts)` 随机一条 → POST 注入桥 → 触发发送
- 系统源 = chat 模块普通源（走桥读，与拉姆源同构，改名无影响）；注入桥在配置里指定（与源名解耦）

## 触发源消费（wait 语义, 2026-08-08）

**只有能触发 push 的源需要消费**（弹幕 / ASR拉姆 / 系统）；记忆/摘要/工具等"更新 prompt 信息"的源反复读是对的，不清。

- **源抽象**（清桥是源的方法，不是桥的）：
  - `_source_read(port)`：读桥第一个非空 text，**wait 视为空**
  - `_source_clear(port)`：清桥——内置桥(18000-18009)同进程直清置 wait；**外部桥读全量 → 所有字段清成 wait**（2026-08-11 修复：旧版只清模块源名字段，ASR 写入的 `asr` 字段残留 → 反复触发循环卡死。现在与内置桥语义一致：全部清 + 放 wait 哨兵）
- **消费时机 = push 成功后**（不是读到就清——发送时内容要进 prompt 给 LLM 看；预览 _push_loop 不清）
- `_trigger_sources()`：动态读配置——弹幕模块全部源 + chat 全部源 + 自动化注入桥（无硬编码）
- `_consume_trigger_sources()`：push 成功后清**所有**触发源桥（无论这次谁触发）
- 全部统一 wait 语义：读认 wait（空桥）、清置 wait（不删 key）

## 下桥清洗 `_sanitize_reply` (2026-08-08 / 2026-08-11 重拳)

LLM 回复下桥的**第一步**，history / 上文记忆 / 检索记忆全部基于干净回复。规则：

1. **删复读**（`_strip_repeat`）：尾巴复读(≥4次)/6x 任意重复 → **从复读开始整段删光**（连第一份不留，自回归复读 token 组合危险）；`REPEAT_WHITELIST` 语气词白名单（喵哈呜嗯啊哦…）组成的复读放行（拟声词正常表达）
2. **工具 arg 内非法内容**：`[TOOL:xxx]` 打开后，arg 内任何 `[` 开头但非 `[TOOL]` 闭合的结构 = 非法 → 删到第一个合法 `[TOOL]` 前（覆盖嵌套工具/花式假闭合 [TO]/[OL]/乱序）
3. **未闭合工具**（`[TOOL:xxx` 无正确 `[TOOL]` 闭合）→ **整个替换成报错工具**（2026-08-11 重拳，不再补闭合）：`[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL]`
4. **孤立残片**：孤立 `[TOOL]`（无对应工具）→ 删除；`[` 开头非 TOOL 的字母残片（`[OL]`/`[OL`/`[TO` 等）→ **替换成报错工具**（模型回复正常不用方括号，见一个打一个）
5. **非法表情**（不在 6 种标准：认真/wink/开心/spin/转头/担心）→ 整个工具丢弃（grammar 掉线兜底）
6. **报错惯性兜底**（2026-08-11）：上一轮主动报错 + 本轮又报错 + 内容非标准 → **本轮偷换成 `[TOOL:回复]`**（报错当回复用 = 坏文本，防误导上文；被偷换轮不算报错 → 下一轮自由）。清洗兜底产生的报错（未闭合/残片）**不计入**惯性状态

接入点：`_do_send` 拿 reply 后 `reply = _sanitize_reply(reply)` → 后续 last_reply_box / 队列 / history / 摘要 / 检索全用干净回复。
复读检测思路来自训练侧 `grpo_tool_main_v3a.py` 的 UL 惩罚函数（推理侧落地）。

## L0 哲学 (2026-08-08)

**history/L0 = LLM 真实看见过的东西**。一句话没进 LLM（发送失败/冷却期被覆盖）= 不存在（像说话太快对方思考时没听见）。

- user 轮**只在 `_do_send` 发送成功后**记录（清桥前读触发源桥：chat 源拼 `对话[源名]：内容`、弹幕源原文）→ 与 assistant 轮成对进 history
- 发送失败 → 不记录、桥留着重试
- 冷却期多条输入、中间被覆盖 → 只有最终真发出去的那条算数
- `/asr` `/text` 不再直接 append history，改为写 chat 源桥 + `_mark_pending()`
- L0 原文（`/output` mem_l0，最近 5 轮=10 条）每条内容 **600 字符截断**（2026-08-11 从 200 提升，避免长回复被砍）

## LLM 输出视图

```
LLM回复 → last_reply_box[0]
  │
  └─ /output 端点 → 输出视图展示 (前端3秒轮询)
```

支持三种触发：
- ASR 自动回复 (18771桥 → LLM)
- 手动 `/chat`（build_prompt → 18771 桥 → llama-server，发送成功后记录 user+assistant 轮）
- `/inject` 注入模拟

## 输出处理队列 (2026-08-06, 更新 2026-08-07)

LLM 回复到达时，`_push_reply_to_queues(reply)` 正则解析工具，按类型入队：

```
LLM回复 "[TOOL:回复]xxx[TOOL][TOOL:表情]wink[TOOL][TOOL:搜索]天气[TOOL][TOOL:报错]xxx[TOOL]"
  │
  └─ re.finditer(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[TOOL\]') → 保持顺序
       │
       ├─ 回复/表情 → express_queue (表达队列)
       ├─ 搜索/结束 → exec_queue (执行队列)
       └─ 报错 → error_queue (报错队列, 独立)
```

### 三个队列

| 队列 | 内容 | done 机制 | 显示位置 |
|------|------|-----------|----------|
| 表达队列 | 回复/表情 | 轮询桥 tts_done | 🗣 表达队列 |
| 执行队列 | 搜索/结束 | 轮询桥 tts_done (搜索工具 POST) | 📋 执行队列 |
| 报错队列 | 报错 | 间隔计时 (可配, 默认 3s) | 🔴 报错队列 |

### 表达队列消费

```
队列项(回复) → POST /api/queue_text → TTS(8000) SSE推送
  ├─ TTS页面 synth() → /api/tts → 合成WAV → _push_speak → 桥(18770)
  │    └─ bridge WS广播 → Live2D页面 → startLipSync → source.start()
  │         └─ setTimeout(duration+20ms) → POST 桥/text {source:tts_done,text:done}
  │
  └─ 消费者轮询桥 EXPRESS_BRIDGE_PORT(默认18004) GET /source_last
       └─ tts_done=done → 打印完成 → 清桥(设wait) → pop任务 → 下一个

队列项(表情) → POST 18770/push {type:"emotion",cmd:"表情",arg:"wink"}
  ├─ vad_bridge WS广播 → Live2D → 随机变体 → playEmote()
  │    └─ setTimeout(时长+20ms) → POST 桥/text {source:tts_done,text:done}
  │
  └─ 消费者轮询桥 → tts_done=done → 清桥 → pop → 下一个
```

### 执行队列消费

```
队列项(搜索) → POST /api/queue_search → 搜索工具(18775) SSE推送
  ├─ 搜索页面: EventSource → 填输入框 → doSearch() → 调API搜索
  │    └─ 结果上桥(结果桥) + POST done桥 {source:tts_done,text:done}
  │
  └─ 消费者轮询桥 EXEC_BRIDGE_PORT(默认18003) GET /source_last
       └─ tts_done=done → 清桥 → pop → 下一个
```

- 搜索结果通过结果桥上桥，供工具返回模块读取注入 prompt
- done 桥与表达队列同机制，严格串行

### 报错队列消费

```
队列项(报错) → 拷贝固定WAV到tts_cache → POST 18770/push speak
  └─ 睡眠 ERROR_INTERVAL 秒(默认3s, 可配) → pop → 下一个
```

- 报错不参与 done 排队，独立消费，仅间隔控制

### 搜索工具 SSE (同 TTS 模式)

```
管线 POST /api/queue_search
  ↓
搜索服务 _sse_queues 列表推送到所有 SSE 连接
  ↓
页面 EventSource.onmessage → 填输入框 → doSearch()
  └─ 搜完: POST 结果上桥 + POST done上桥
```

- 新 SSE 连接自动获取最近一次推送缓存
- 使用 ThreadingHTTPServer + queue.Queue 模式

### 展示
- 模式切换：🔗串行/⚡并行 按钮（前端变量，并行逻辑待实现）

### 展示

`/output` 端点返回 `exec_queue` 和 `express_queue`，前端每 3 秒轮询。

## TODO: 过滤器系统

### 需求
- TTS 页面的过滤词可保存（持久化）
- 开关：开启/关闭过滤
- 过滤后文本框：显示过滤后的文本
- 开启过滤时：文本进入 textarea → 点生成 → 检查过滤词
  - 含过滤词 → 播放"已过滤"语音（预生成），不合成原文本
  - 不含 → 正常合成

### 连带改造
- 上下文 history 中滤掉的回复也需显示"已过滤"
- 管线中心的 prompt 拼装需感知过滤状态

## 🧪 待验证（2026-08-08）

**已实测通过**：
- ASR 驱动：写 18771 桥 → `_check_trigger_sources` 检测触发 → LLM 回复（模拟 ASR curl 验证）
- 下桥清洗 `_sanitize_reply`：8+ 样例全过（复读删光/嵌套/花式假闭合/未闭合/孤立TOOL/白名单）
- 拼装成品修复：inject 10 轮测试确认 _push_loop 之前只推 L0 → 修复后推 L0+L1+L2 完整版

**待启动依赖后验证**：
- 18774 检索服务（长期记忆检索；当前未启动，trigger_retrieval 只做本地 L0.5 + error 提示）
- 弹幕真实链路（blc 19000 + danmaku 18776 → 发射 → pipeline）
- 自动化系统注入（静默间隔 → 注入桥 → 自言自语）

## TODO: 启动集成（最后一步）

- 一条指令启动全部管线（blc / danmaku_server / pipeline / LLM桥 / llama-server / ASR 等）
- 或做"启动中心"Web 页：傻瓜式点点点启动/停止各服务
- 考虑：服务依赖顺序、端口冲突检测、日志聚合
