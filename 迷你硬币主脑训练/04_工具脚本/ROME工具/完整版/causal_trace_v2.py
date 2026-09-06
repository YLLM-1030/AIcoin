"""
正确因果追踪 - 噪声主体位置，恢复预测位的完整hidden state
ROME方式：噪化subject，逐层恢复last token的完整层输出
"""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_self/epoch_20"
PROMPT = "我叫"
TARGET = "迷你硬币"

model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
model.eval()

target_id = tok.encode(TARGET)[0]
inputs = tok(PROMPT, return_tensors="pt").to(model.device)
tokens = [tok.decode([i]) for i in inputs.input_ids[0]]
last_pos = len(tokens) - 1
print(f"分词: {tokens}, 预测位: {last_pos}")

# 获取完整的transformer层模块（不是MLP，是整个layer的输出）
layer_modules = []
for n, m in model.named_modules():
    if n.startswith("model.layers.") and "." not in n[len("model.layers.")+1:]:
        lid = int(n.split(".")[2])
        layer_modules.append((lid, n, m))
print(f"找到 {len(layer_modules)} 个transformer层")

# 1. 干净运行 - 记录每层完整hidden state（预测位）
clean_acts = {}
handles = []
for lid, name, mod in layer_modules:
    def save(l):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            clean_acts[l] = h.detach().clone()
        return hook
    handles.append(mod.register_forward_hook(save(lid)))
with torch.no_grad():
    clean_logits = model(**inputs).logits
for h in handles: h.remove()
clean_p = torch.softmax(clean_logits[0, -1], dim=-1)[target_id].item()
print(f"  干净: {clean_p:.4f}")

# 2. 噪声主体
noise_ids = inputs.input_ids.clone()
noise_ids[0, 0] = torch.randint(0, tok.vocab_size, (1,)).item()
with torch.no_grad():
    corrupt_logits = model(input_ids=noise_ids).logits
corrupt_p = torch.softmax(corrupt_logits[0, -1], dim=-1)[target_id].item()
print(f"  噪声后: {corrupt_p:.4f}")

# 3. 逐层恢复
print(f"\n{'层':>4} | {'IE':>8} | 信号")
print("-"*50)
results = {}
for lid, name, mod in layer_modules:
    if lid not in clean_acts: continue
    flag = [False]
    def restore(l, flag):
        def hook(m, i, o):
            if not flag[0] and l in clean_acts:
                h = o[0] if isinstance(o, tuple) else o
                h[0, last_pos] = clean_acts[l][0, last_pos]
                flag[0] = True
            return (h,) if isinstance(o, tuple) else h
        return hook
    handle = mod.register_forward_hook(restore(lid, flag))
    with torch.no_grad():
        restore_logits = model(input_ids=noise_ids).logits
    handle.remove()
    p = torch.softmax(restore_logits[0, -1], dim=-1)[target_id].item()
    ie = (p - corrupt_p) / (clean_p - corrupt_p + 1e-8)
    results[lid] = ie

max_ie = max(results.values())
for lid in sorted(results.keys()):
    ie = results[lid]
    bar = "█" * max(1, int(ie / max_ie * 30)) if max_ie > 0 else ""
    print(f"{lid:4d} | {ie:8.4f} | {bar}")

top5 = sorted(results.items(), key=lambda x: -x[1])[:5]
print(f"\n核心层: {[(l, f'{ie:.3f}') for l,ie in top5]}")
