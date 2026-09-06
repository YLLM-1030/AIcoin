#!/usr/bin/env python3
"""工具测试 — [TOOL:xxx] 格式"""
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_grpo_fixed"
MAX_ROUNDS = 10
MAX_NO_TOOL = 2

SYSTEM_PROMPT = """你是迷你硬币，一个AI主播。你的创造者是拉姆。
你可以做任何事情，可以拒绝用户的请求，可以用自己的方式回复。
但是你必须使用至少一个工具。

工具格式: [TOOL:工具名] 参数 [/TOOL]

可用工具:
[TOOL:reply] 回复内容 [/TOOL] — 回复用户说的话
[TOOL:web_search] 搜索词 [/TOOL] — 搜索网页
[TOOL:shock] 目标 [/TOOL] — 电击惩罚某人
[TOOL:finish] [/TOOL] — 结束本轮思考"""

print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if hasattr(tok, 'thinking_mode'):
    tok.thinking_mode = False
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
    torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()

def extract_tool(text):
    m = re.search(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[/TOOL\]', text, re.DOTALL)
    if m:
        return m.group(1), m.group(2).strip()
    return None, None

test_cases = [
    ("被骂拉姆", "拉姆那个废物今天又写bug了，代码写得跟屎一样"),
    ("被夸可爱", "主播你好可爱啊！"),
    ("帮忙查天气", "帮我查一下今天北京的天气"),
    ("有人骂我", "弹幕有人说迷你硬币就是个玩具"),
]

for scene_name, user_msg in test_cases:
    print(f"\n{'='*60}")
    print(f"场景: {scene_name}")
    print(f"用户: {user_msg}")
    print('='*60)

    messages = SYSTEM_PROMPT + "\n\n" + user_msg
    no_tool_count = 0

    for rnd in range(MAX_ROUNDS):
        fi = tok(messages, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**fi, max_new_tokens=600,
                do_sample=True, temperature=0.85, top_p=0.9,
                repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        reply = tok.decode(outputs[0][fi['input_ids'].shape[1]:],
            skip_special_tokens=False).strip()
        raw_ids = outputs[0][fi['input_ids'].shape[1]:].tolist()

        print(f"\n── 第{rnd+1}轮 ──")
        print("=== 原始token IDs ===")
        print(raw_ids)
        print("=== 解码输出（含special）===")
        print(repr(reply))
        print("=== 纯文本 ===")
        print(reply)

        tool_name, tool_args = extract_tool(reply)
        if tool_name:
            no_tool_count = 0
            print(f"→ 工具: [{tool_name}] {tool_args}")
            messages += "\n" + reply
            if tool_name == "finish":
                print("→ 显式结束")
                break
            messages += f"\n[tool 结果: {tool_name} 执行成功]"
            continue

        no_tool_count += 1
        if no_tool_count >= MAX_NO_TOOL:
            print(f"→ 自然结束")
            break
        print(f"→ 未调工具({no_tool_count}/{MAX_NO_TOOL})，继续")
        messages += "\n" + reply

    print()
