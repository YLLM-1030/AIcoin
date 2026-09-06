"""
手动 Agent 测试 v4 — 客户端组装 prompt，直接调 /completion
"""
import requests, json, re, readline

LLAMA_API = "http://localhost:8080/completion"

SYSTEM = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

# ── 模板渲染（客户端完全控制） ──
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

# ── 调用 llama-server ──
# GBNF Grammar: [TO后只允许L，+ 表情约束
TOOK_GRAMMAR = """root ::= item*
item ::= [^[] | "[" ( [^Tt] | "T" [^Oo] | "TO" "O" "L" ( ":" tool_args "]" | "]" ) )
tool_args ::= "表情" " "? ("认真"|"wink"|"开心"|"spin"|"转头"|"担心") " "? "[TOOL]" | ("回复"|"搜索"|"报错"|"结束") " "? [^]]* " "? "[TOOL]" """

def generate(prompt, **kwargs):
    payload = {"prompt": prompt, "n_predict": 500, "temperature": 0.9,
               "grammar": TOOK_GRAMMAR, "logit_bias": [[1930, 10.0], [3925, -10.0]], **kwargs}
    r = requests.post(LLAMA_API, json=payload)
    return r.json()["content"].strip()

def parse_tools(text):
    tools = re.findall(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[TOOL\]', text)
    return tools

# ── 主循环 ──
history = []
print("🍎 迷你硬币 聊天\n")

while True:
    try:
        user_input = input(">>> ")
    except (EOFError, KeyboardInterrupt):
        break
    if not user_input:
        continue
    if user_input.startswith("/"):
        parts = user_input.split(maxsplit=1)
        cmd = parts[0][1:]
        args = parts[1] if len(parts) > 1 else ""
        if cmd in ("clear", "cls"):
            history.clear()
            continue
        if cmd == "exit":
            break
        continue

    prompt = build_prompt(SYSTEM, history, user_input)
    raw = generate(prompt)

    # 解析工具
    tools = parse_tools(raw)
    print(f"\n─── 硬币 ───")
    print(raw)
    print(f"  📊 工具: ", end="")
    if tools:
        cnt = {}
        for name, content in tools:
            cnt[name] = cnt.get(name, 0) + 1
        print(" ".join(f"{k}x{v}" for k, v in cnt.items()))
    else:
        print("(无)")
    print()

    history.append(("user", user_input))
    history.append(("assistant", raw))
