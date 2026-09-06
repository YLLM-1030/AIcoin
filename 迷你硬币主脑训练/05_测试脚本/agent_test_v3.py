"""
手动 Agent 测试 v3 — 改用 llama-server API
"""
import requests, json, re, readline

API = "http://localhost:8080/v1/chat/completions"

SYSTEM = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

def chat(messages, **kwargs):
    payload = {"messages": messages, "temperature": 0.9, "max_tokens": 500, **kwargs}
    r = requests.post(API, json=payload)
    return r.json()["choices"][0]["message"]["content"]

def handle_tools(text):
    for m in re.finditer(r'\[TOOL:(回复|表情|搜索|报错)\]\s*(.*?)\s*\[TOOL\]', text):
        tt, c = m.group(1), m.group(2).strip()
        print(f"  {'🗣' if tt=='回复' else '😊' if tt=='表情' else '🔍' if tt=='搜索' else '❌'} {tt}: {c}")

def truncate_multi(text, tool):
    parts = list(re.finditer(rf'\[TOOL:{tool}\]\s*(.*?)\s*\[TOOL\]', text))
    if len(parts) >= 2:
        end = parts[1].end()
        print(f"  ✂️ 多{tool}截断: {len(text)-end} 字符丢弃")
        return text[:end]
    return text

# ── 主循环 ──
messages = [{"role": "system", "content": SYSTEM}]

print("=" * 60)
print("Agent 测试 — llama-server | /end 退出 | /new 重置")
print("=" * 60)

while True:
    inp = input(">>> ").strip()
    if not inp: continue
    if inp == "/end": break
    if inp == "/new":
        messages = [{"role": "system", "content": SYSTEM}]
        print("重置"); continue

    messages.append({"role": "user", "content": inp})
    t = chat(messages)
    t = truncate_multi(t, "回复")
    t = truncate_multi(t, "表情")
    print("─── 模型 ───")
    print(t)
    messages.append({"role": "assistant", "content": t})
    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]
    handle_tools(t)
    print(f"  📜 上文 {len(messages)-1} 条消息")
    print("─── 本轮结束 ───")
