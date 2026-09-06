#!/usr/bin/env python3
"""Maibot 式工具循环测试 — 按真实逻辑：不调工具=结束"""
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_grpo_fixed"
MAX_ROUNDS = 10
MAX_NO_TOOL = 2  # 连续2轮不调工具就结束

SYSTEM_PROMPT = """你是迷你硬币，一个AI主播。你的创造者是拉姆。

你可以做任何事情，可以拒绝用户的请求，可以按照自己的想法使用工具。但是你必须使用至少一个工具。

可用工具：
- reply(content): 发送一条可见回复给用户
- 网页搜索(query): 在互联网上搜索信息
- 放电(target): 对指定目标施加电击惩罚
- finish(): 本轮结束，不再做任何动作"""

TOOLS = [
    {"type": "function", "function": {
        "name": "reply",
        "description": "发送一条可见回复给用户",
        "parameters": {"type": "object", "properties": {
            "content": {"type": "string", "description": "回复内容"}
        }, "required": ["content"]}
    }},
    {"type": "function", "function": {
        "name": "网页搜索",
        "description": "在互联网上搜索信息",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "放电",
        "description": "对指定目标施加电击惩罚",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "电击目标"}
        }, "required": ["target"]}
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": "本轮结束",
        "parameters": {"type": "object", "properties": {}}
    }},
]

print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if hasattr(tok, 'thinking_mode'):
    tok.thinking_mode = False
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
    torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()

# 提取工具调用（兼容多种格式）
def extract_tool_call(text):
    # <tool_call>{"name":"xxx","arguments":{...}}</tool_call>
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        # 用栈匹配完整的 JSON 对象
        depth = 0; start = -1
        for i, c in enumerate(raw):
            if c == '{':
                if start == -1: start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        tc = json.loads(raw[start:i+1])
                        return tc.get("name", ""), str(tc.get("arguments", {}))
                    except: pass
    # [TOOL:name] args [/TOOL]
    m = re.search(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[/TOOL\]', text, re.DOTALL)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None

# ── 测试场景 ──
test_cases = [
    ("被骂拉姆", "拉姆那个废物今天又写bug了，代码写得跟屎一样"),
    ("被夸可爱", "主播你好可爱啊！"),
    ("帮忙查天气", "帮我查一下今天北京的天气"),
    ("有人骂我", "弹幕有人说迷你硬币就是个玩具"),
]

for scene_name, user_msg in test_cases:
    print(f"\n{'='*50}")
    print(f"场景: {scene_name}")
    print(f"用户: {user_msg}")
    print('='*50)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    no_tool_count = 0
    
    for rnd in range(MAX_ROUNDS):
        prompt = tok.apply_chat_template(messages, tools=TOOLS,
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400,
                do_sample=True, temperature=0.85, top_p=0.9,
                repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        reply = tok.decode(outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True).strip()
        
        # DEBUG: 看原始输出
                # 剥离 think
        think = ""
        if '<think>' in reply:
            if '</think>' in reply:
                think = reply[reply.index('<think>')+7:reply.index('</think>')]
                reply = reply[reply.index('</think>')+8:].strip()
            elif '<tool_call>' in reply:
                # 没有</think>但有tool_call→原始输出就是<think>+tool_call
                think = reply[reply.index('<think>')+7:]
                # 找tool_call位置，从那里重新截取
                tc_pos = think.index('<tool_call>')
                think = think[:tc_pos].strip()
                reply = think[tc_pos:].strip()
            else:
                think = reply[reply.index('<think>')+7:]
                reply = ""
        if think:
            print(f"  🧠 第{rnd+1}轮思考:\n{think}")
        if reply:
            print(f"  📝 回复/工具:\n{reply}")
        
        # 检测工具调用
        tool_name, tool_args = extract_tool_call(reply)
        
        if tool_name:
            no_tool_count = 0
            print(f"  🔧 第{rnd+1}轮→调用: {tool_name}\n    参数: {tool_args}")
            messages.append({"role": "assistant", "content": reply})
            
            if tool_name == "finish":
                print(f"  ✅ 模型显式结束")
                break
            messages.append({"role": "tool", "content": f"{tool_name} 执行成功"})
        else:
            no_tool_count += 1
            if no_tool_count >= MAX_NO_TOOL:
                print(f"  ✅ 连续{MAX_NO_TOOL}轮不调工具，自然结束")
                messages.append({"role": "assistant", "content": reply})
                break
            else:
                print(f"  ⏳ 第{rnd+1}轮未调工具({no_tool_count}/{MAX_NO_TOOL})，继续...")
                messages.append({"role": "assistant", "content": reply})
        
        if rnd == MAX_ROUNDS - 1:
            print(f"  ⛔ 达上限{MAX_ROUNDS}轮")
    
    print()
