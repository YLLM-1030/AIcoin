# 迷你硬币 — 训练 & 推理管线

## 模型版本历史

```
v3 → (原始基座，Qwen3-8B)
v5 → (第一版 SFT)
v8 → (从 v5 训，数据有问题)
v9 → (从 v5 训，数据修过一次)
v9a → 从 v8 训，2 epoch 无 shuffle，数据修「他→你/对方」
v10a → 从 v9a 训，2 epoch 带 shuffle，效果反而不如 v9a
v9b → 从 v8 训，2 epoch 无 shuffle，数据修「乱编对方意图」
v9c → 从 v9b 训，2 epoch（1:shuffle + 2:长回复在末尾），数据修「短回复不足」
v10m → 从 v9c 训，321条搞事数据，2 epoch 原始顺序（中间保存）
v10c → 从 v9c 训，321条搞事数据，4 epoch（2顺序 + 2shuffle）
       新增 history/memory 字段格式，GRPO 训练，golden×5 + sampled×1
```

**当前最佳：** `ckpt_tool_main_v10c`
**Q8_0 GGUF：** `/tmp/coin_v10c_q8.gguf`（8.3GB, ~50 tok/s）

## Ollama 为什么被弃用

Ollama 对 Qwen3 模型有**硬编码的 think 注入**。它的 Go 层在把 prompt 发给 llama-server 之前，自动插入 `<think>` token。表现为：模型输出末尾无缘无故出现 `</think>`。

| 尝试的方案 | 效果 |
|---|---|
| `PARAMETER nothink true` 写 Modelfile | ❌ 无效（仅是隐藏输出，内部仍在生成 thinking token） |
| `SYSTEM /no_think` 写 Modelfile | ❌ 无效 |
| `"think": false` 传 API | ✅ 有效，但每轮都要传 |
| 改 GGUF 架构标识为其他架构 | ❌ 会导致 tensor 映射不匹配，输出乱码 |
| **绕过 Ollama，直连 llama-server** | ✅ 完全无注入 |

**结论：** Ollama 的 think 注入是 Go 层硬编码行为，无配置可关闭。唯一彻底的方案是绕过 Ollama，直接使用 llama-server。

## 推理架构

```
当前（v4，推荐）：
  用户输入
    ↓
  agent_test_v4.py (Python)
    ├─ build_prompt() —— 客户端自行拼装
    │   ├─ <|im_start|>system\n...<|im_end|>
    │   ├─ ...历史对话...
    │   ├─ <|im_start|>系统提示\n底部注入<|im_end|>
    │   └─ <|im_start|>输入\n本轮输入<|im_end|>
    │     <|im_start|>硬币\n
    ├─ 预览：直接打印 build_prompt() 结果
    ├─ 发送 → llama-server /completion（零模板零加工）
    └─ 回复 → 工具解析 → 保存历史

相比旧方案（v3 / Ollama）：
  Ollama:      Python → messages 数组 → llama-server Jinja 模板 → 模型
  v4 (推荐):   Python → 自己拼好文本 → llama-server /completion → 模型
                                     ↑ 预览 = 实际输入，零偏差
```

### 关键优势

- **完全控制 prompt：** 自己拼 `<|im_start|>` 格式，想怎么改就怎么改
- **底部注入：** 在历史对话之后、用户输入之前，插入系统提示，避免顶部 system prompt 被注意力衰减
- **预览即所见：** `/preview` 命令显示的实际就是发给模型的内容
- **零注入：** llama-server 的 `/completion` 端点不做任何模板处理

### llama-server 启动

```bash
cd ~/llama.cpp && source ~/mini_coin_venv/bin/activate
./build/bin/llama-server \
  --model /tmp/coin_v9c_q8.gguf \
  --jinja --chat-template-file /tmp/coin_template.jinja \
  --n-gpu-layers 99 \
  --host 127.0.0.1 --port 8080
```

`--jinja --chat-template-file` 仅在用 `/v1/chat/completions` 时用到，如果用 `/completion` 可直接去掉。

### 测试脚本

| 脚本 | 说明 |
|---|---|
| `agent_test_v3.py` | 旧版，调 `/v1/chat/completions`（留作参考） |
| **`agent_test_v4.py`** | **当前主力。** 调 `/completion`，客户端拼 prompt，多工具检测，底部注入 |
| `coin_pipeline.py` | 键盘+语音版（预留 ASR，需 FunASR 模型） |
| `dashboard.html` | 前端仪表盘，可视化输入中心 + 实时预览 |
| `dashboard_server.py` | 仪表盘后端，WebSocket 转发 ASR/弹幕 |

### Chat Template（Jinja2，存 GGUF 内）

```
{% for message in messages %}{{'<|im_start|>' + (message['role']|replace('user','输入')|replace('assistant','硬币')) + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>硬币\n' }}{% endif %}
```

## 训练管线

### 训练脚本

位置: `/mnt/c/Users/Autogram-coin/Desktop/新建训练/grpo_tool_main_v3a.py`

### 命令

```bash
cd /mnt/c/Users/Autogram-coin/Desktop/新建训练 && source ~/mini_coin_venv/bin/activate && python3 grpo_tool_main_v3a.py
```

### 数据格式 (JSON)

```json
{
  "instruction": "对话[拉姆]：帮我看看茅台今天的股价",
  "output": "对方让我帮他查茅台股价。行，我搜索一下看看。[TOOL:回复]...[/TOOL][TOOL:搜索]...[/TOOL]"
}
```

### 关键参数
- GROUP_SIZE=2（1采样 + 1golden）
- LR=2e-6
- temperature=0.9
- Lomo optimizer（in-place 更新）
- 2 epoch，Epoch 1 shuffle / Epoch 2 长回复在末尾

### GGUF 转换

**注意：** 训练保存的 checkpoint 可能丢失 `chat_template`，转换前先检查并补写：

```bash
source ~/mini_coin_venv/bin/activate && python3 -c "
import json
path = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v10c/tokenizer_config.json'
with open(path, encoding='utf-8') as f:
    cfg = json.load(f)
if 'chat_template' not in cfg:
    cfg['chat_template'] = '{% for message in messages %}{{\"<|im_start|>\" + (message[\"role\"]|replace(\"user\",\"输入\")|replace(\"assistant\",\"硬币\")|replace(\"memory\",\"记忆\")) + chr(10) + message[\"content\"] + \"<|im_end|>\" + chr(10)}}{% endfor %}{% if add_generation_prompt %}{{ \"<|im_start|>硬币\n\" }}{% endif %}'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print('chat_template 已补写')
else:
    print('chat_template 存在')
"
```

然后转 GGUF：

```bash
cd ~/llama.cpp && source ~/mini_coin_venv/bin/activate
# 转 F16
python3 convert_hf_to_gguf.py /mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v10c --outfile /tmp/coin_v10c_f16.gguf --outtype f16
# 量化 Q8
./build/bin/llama-quantize /tmp/coin_v10c_f16.gguf /tmp/coin_v10c_q8.gguf Q8_0
```

### llama-server 启动

```bash
cd ~/llama.cpp && source ~/mini_coin_venv/bin/activate
pkill -f llama-server
./build/bin/llama-server \
  --model /tmp/coin_v10c_q8.gguf \
  --jinja --chat-template-file /tmp/coin_template.jinja \
  --n-gpu-layers 99 \
  --host 127.0.0.1 --port 8080
```

### 测试脚本

```bash
cd /mnt/c/Users/Autogram-coin/Desktop/新架构硬币/迷你硬币主脑训练/05_测试脚本 && source ~/mini_coin_venv/bin/activate && python3 agent_test_v4.py
```

## 工具调用约束

### 合法表情列表
`认真` `wink` `开心` `spin` `转头` `担心`

（仅这 6 个，训练数据中无例外）

### 多工具处理
- 检测到 ≥2 个[TOOL:回复]或[TOOL:表情]时截断到第二个
- 连续 3 轮多工具时，截断到第一个

## 已知问题

- **上下文污染：** 模型看到历史里自己的短回复会学坏，方案是底部注入 + 多工具截断
- **表情溢出：** 模型偶尔把说话内容塞进表情字段，需要更多训练数据压制
- **场景覆盖不足：** 某些角色特质场景（恶作剧、吐槽）没有被训练数据覆盖

## agent_test_v4.py 完整代码

```python
import requests, json, re, readline

LLAMA_API = "http://localhost:8080/completion"

SYSTEM = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

ROLE_MAP = {"user": "输入", "assistant": "硬币", "system": "system", "summary": "摘要"}

def build_prompt(sys_content, history, current_input):
    lines = []
    lines.append(f"<|im_start|>system\n{sys_content}<|im_end|>")
    for role, content in history:
        name = ROLE_MAP.get(role, role)
        lines.append(f"<|im_start|>{name}\n{content}<|im_end|>")
    lines.append("<|im_start|>系统提示\n以上是历史对话，你需要判断用户的意图，然后再决定你要怎么做。<|im_end|>")
    if current_input is not None:
        name = ROLE_MAP.get("user", "输入")
        lines.append(f"<|im_start|>{name}\n{current_input}<|im_end|>")
    lines.append("<|im_start|>硬币\n")
    return "\n".join(lines)

def generate(prompt, **kwargs):
    payload = {"prompt": prompt, "n_predict": 500, "temperature": 0.9, **kwargs}
    r = requests.post(LLAMA_API, json=payload)
    return r.json()["content"].strip()

def handle_tools(text):
    counts = {"回复": 0, "表情": 0, "搜索": 0, "报错": 0}
    for m in re.finditer(r'\[TOOL:(回复|表情|搜索|报错)\]\s*(.*?)\s*\[TOOL\]', text):
        tt, c = m.group(1), m.group(2).strip()
        counts[tt] += 1
        print(f"  {'🗣' if tt=='回复' else '😊' if tt=='表情' else '🔍' if tt=='搜索' else '❌'} {tt}: {c}")
    parts = []
    for t, n in counts.items():
        if n: parts.append(f"{t}x{n}")
    if parts:
        print(f"  📊 本轮工具: {' '.join(parts)}")
    return counts

def truncate_multi(text, tool):
    parts = list(re.finditer(rf'\[TOOL:{tool}\]\s*(.*?)\s*\[TOOL\]', text))
    if len(parts) >= 2:
        end = parts[1].end()
        return text[:end]
    return text

history = []
tool_history = []
context_rounds = 10

print("=" * 60)
print("Agent 测试 v4 — 客户端拼装 prompt")
print("  /end 退出 | /new 重置 | /preview 预览本轮 prompt")
print("=" * 60)

while True:
    inp = input(">>> ").strip()
    if not inp: continue
    if inp == "/end": break
    if inp == "/new":
        history = []; tool_history = []
        print("重置"); continue
    if inp == "/preview":
        prompt = build_prompt(SYSTEM, history[-(context_rounds*2):], "(当前输入占位)")
        print("─── 本轮 prompt 预览 ───")
        print(prompt)
        print(f"─── 共 {len(prompt)} 字符 ───")
        continue

    prompt = build_prompt(SYSTEM, history[-(context_rounds*2):], inp)
    print(f"📤 发送 {len(prompt)} 字符 / {len(history)//2+1} 轮")

    t = generate(prompt)
    counts = handle_tools(t)
    print("─── 硬币 ───")
    print(t)

    history.append(("user", inp))
    history.append(("assistant", t))
    tool_history.append(counts)
    if len(tool_history) > context_rounds:
        tool_history = tool_history[-context_rounds:]
    if len(history) > context_rounds * 2:
        history = history[-(context_rounds * 2):]
    print(f"  📜 历史 {len(history)//2} 轮")
    print("─── 本轮结束 ───")
```
