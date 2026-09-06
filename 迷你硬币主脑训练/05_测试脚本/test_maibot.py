"""MaiBot 多轮循环测试"""
import requests, json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "minicoin_r1"

SYSTEM_PROMPT = """你的任务是分析聊天并做出下一步动作。
发言必须通过reply工具，不然用户无法看见。
不使用工具时，结束思考。

可用工具：
- reply(content): 发送一条可见回复给用户"""

TOOLS = [{
    "name": "reply",
    "description": "生成并发送一条可见回复",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "回复内容"}
        },
        "required": ["content"]
    }
}]

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
messages.append({"role": "user", "content": "上班摸鱼"})

for round_idx in range(10):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "messages": messages, "stream": False,
        "tools": TOOLS, "options": {"temperature": 0.8, "num_predict": 200}
    })
    msg = resp.json()["message"]
    content = msg.get("content", "").strip()
    tool_calls = msg.get("tool_calls", [])

    if content:
        print(f"[思考] {content[:80]}")
        messages.append({"role": "assistant", "content": content})

    if not tool_calls:
        print("[结束] 不调工具，停止")
        break

    for tc in tool_calls:
        fn = tc["function"]
        if fn["name"] == "reply":
            args = json.loads(fn["arguments"])
            print(f"[回复] {args.get('content', '')}")
            messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
            messages.append({"role": "tool", "content": f"已发送", "tool_call_id": tc["id"]})
