import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v3a"
MAX_NEW = 500

print("加载模型:", MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

TOOL_SYS = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你会根据对方的情绪来采取行动。使用完一个工具之后进行分析。

回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
报错的时候使用：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你觉得应该等对方回应：[TOOL:结束] [TOOL]"""

def detect_stop(t):
    if "[TOOL:结束][TOOL]" in t or "[TOOL:结束] [TOOL]" in t: return "end"
    if "[TOOL]" in t: return "tool"
    return None

def generate(ctx):
    if not isinstance(ctx, str):
        print("  [错误: ctx 不是字符串, 类型:", type(ctx), "]")
        return "", "error"
    inp = tokenizer(ctx, return_tensors="pt").to(model.device)
    plen = inp.input_ids.shape[1]
    ids = inp.input_ids.clone()
    for _ in range(MAX_NEW):
        with torch.no_grad():
            out = model(ids)
        logits = out.logits[0, -1, :] / 0.9
        # top_p=0.9 过滤
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > 0.9
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        probs = torch.softmax(sorted_logits, dim=-1)
        tok = sorted_indices[torch.multinomial(probs, num_samples=1)].unsqueeze(0)
        ids = torch.cat([ids, tok], dim=-1)
        t = tokenizer.decode(ids[0, plen:], skip_special_tokens=True)
        s = detect_stop(t)
        if s: return t, s
        if tok.item() == tokenizer.eos_token_id: return t, "eos"
    return tokenizer.decode(ids[0, plen:], skip_special_tokens=True), "maxlen"

ctx = ""
in_round = False

print("="*60)
print("Agent 测试 | /end 退出 | /new 重置")
print("="*60)

while True:
    inp = input(">>> ").strip()
    if inp == "/end": break
    if inp == "/new":
        ctx = ""; in_round = False
        print("重置"); continue

    if not in_round:
        msg = [{"role": "system", "content": TOOL_SYS}, {"role": "user", "content": inp}]
        ctx = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:
        ctx += inp + "\n"

    if not isinstance(ctx, str):
        print("  [错误: ctx 不是字符串, 改用纯文本拼接]")
        ctx = TOOL_SYS + "\n" + inp
    ctx = str(ctx)

    t, s = generate(ctx)
    print("--- 模型 ---")
    print(t)
    print("--- [" + s + "] ---")

    if s == "end" or s == "error":
        if s == "end":
            ctx += t + "\n"
        in_round = False
        print("--- 本轮结束 ---")
    elif s == "tool":
        ctx += t + "\n"
        in_round = True
        print("--- 工具调用，等待注入 ---")
    else:
        ctx += t + "\n"
        in_round = False
