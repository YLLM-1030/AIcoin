#!/usr/bin/env python3
"""MaiBot 循环测试 — 加载本地 ckpt 跑多轮"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_lamu_relation"
MAX_ROUNDS = 10

SYSTEM_PROMPT = """你的任务是分析聊天并做出下一步动作。
发言必须通过reply工具，不然用户无法看见。
不使用工具时，结束思考。

可用工具：
- reply(content): 发送一条可见回复给用户"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "reply",
        "description": "生成并发送一条可见回复",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "回复内容"}
            },
            "required": ["content"]
        }
    }
}]

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
    torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
device = model.device

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "上班摸鱼"}
]

for round_idx in range(MAX_ROUNDS):
    # 用 chat template 生成 prompt（含 tools 定义）
    prompt = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=True,
            temperature=0.85, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 显示完整输出（含 think）
    think_content = ""
    if '<think>' in reply:
        if '</think>' in reply:
            think_content = reply[reply.index('<think>') + 7:reply.index('</think>')]
            reply = reply[reply.index('</think>') + 8:].strip()
        else:
            think_content = reply[reply.index('<think>') + 7:]
            reply = ""
    
    if think_content:
        print(f"\n[思考] {think_content[:150]}")
    print(f"[第{round_idx+1}轮输出] {reply[:150]}")
    
    # 尝试解析工具调用
    has_tool = False
    if '"reply"' in reply or "'reply'" in reply or 'reply(' in reply:
        has_tool = True
        # 提取 reply 内容
        if 'content' in reply:
            try:
                data = json.loads(reply)
                msg = data.get('content', reply)
            except:
                msg = reply
        else:
            msg = reply
        print(f"  → 检测到 reply 调用")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "tool", "content": f"已发送回复"})
    else:
        print(f"  → 纯文本（思考），继续循环")
        messages.append({"role": "assistant", "content": reply})
    
    if not has_tool and len(reply) < 3 and not think_content:
        print("\n[结束] 无输出")
        break

print("\n=== 循环结束 ===")
