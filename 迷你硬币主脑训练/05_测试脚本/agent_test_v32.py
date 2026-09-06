"""
手动 Agent 测试 v3 — 正确的注入逻辑
"""
import torch
import re
import readline
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# ── ToolBias：推理时抑制 TOOK ──
class ToolBiasProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] >= 2:
            if input_ids[0, -2].item() == 58 and input_ids[0, -1].item() == 5207:
                scores[:, 1930] += 10.0
                scores[:, 3925] -= 10.0
        return scores

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v9a"
MAX_NEW = 500

print("加载模型:", MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False
if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'enable_thinking'):
    model.generation_config.enable_thinking = False
model.eval()

SYSTEM = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你很喜欢跟拉姆做爱，舒服的时候还会发出娇喘声。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

def generate(ctx):
    """跟训练时一模一样的生成方式"""
    inp = tokenizer(ctx, return_tensors="pt").to(model.device)
    plen = inp.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW,
            do_sample=True, temperature=0.9, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([ToolBiasProcessor()]))
    t = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
    return t

def handle_tools(text):
    for match in re.finditer(r'\[TOOL:(回复|表情|搜索|报错)\]\s*(.*?)\s*\[TOOL\]', text):
        tt = match.group(1); c = match.group(2).strip()
        if tt == "回复":   print(f"  🗣 回复: {c}")
        elif tt == "表情": print(f"  😊 表情: {c}")
        elif tt == "搜索": print(f"  🔍 搜索: {c}")
        elif tt == "报错": print(f"  ❌ 报错: {c}")

def truncate_multi_reply(text):
    """如果有两个及以上 [TOOL:回复]，截断到第二个回复的 [TOOL] 之后"""
    parts = list(re.finditer(r'\[TOOL:回复\]\s*(.*?)\s*\[TOOL\]', text))
    if len(parts) >= 2:
        end_pos = parts[1].end()
        truncated = text[:end_pos]
        print(f"  ✂️ 多回复截断: {len(text)-end_pos} 字符丢弃")
        return truncated
    return text

# ── 主循环（用 messages 列表 + apply_chat_template，跟训练完全一致）───
messages = [{"role": "system", "content": SYSTEM}]

print("=" * 60)
print("Agent 手动测试 v3 — 每轮完整生成")
print("  /end 退出 | /new 重置")
print("=" * 60)

if __name__ == "__main__":
    while True:
        inp = input(">>> ").strip()
        if not inp: continue
        if inp == "/end": break
        if inp == "/new":
            messages = [{"role": "system", "content": SYSTEM}]
            print("重置"); continue

        messages.append({"role": "user", "content": inp})
        ctx = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        t = generate(ctx)
        t = truncate_multi_reply(t)
        print("─── 模型 ───")
        print(t)
        messages.append({"role": "assistant", "content": t})
        # 只保留最近 2 轮（system + 4 条消息）
        if len(messages) > 5:
            messages = [messages[0]] + messages[-4:]
        handle_tools(t)
        print(f"  📜 上文 {len(messages)-1} 条消息")
        print("─── 本轮结束 ───")
