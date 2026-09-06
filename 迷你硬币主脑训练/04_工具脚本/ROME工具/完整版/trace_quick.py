"""
快速信号追踪 — 只测"拉姆"一词，对比v53原始 vs 预训练后
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    ("/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_bragging/epoch_1", "v53原始"),
    ("/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_formatted/epoch_2", "预训练后"),
]

all_results = []
for path, label in models:
    print(f"加载 {label}...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model.eval()

    prompt = "拉姆是谁"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    decoded = [tokenizer.decode([i]).strip() for i in inputs.input_ids[0]]
    print(f"  分词: {decoded}")

    ffn = [n for n, _ in model.named_modules() if "mlp.down_proj" in n]
    pos = decoded.index("拉姆") if "拉姆" in decoded else 0
    print(f"  拉姆在位置: {pos}")

    layer_states = {}
    hooks = []
    def make_hook(name, p=pos):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            layer_states[name] = h[0, p].detach().cpu().float().norm().item()
        return hook
    for n in ffn:
        hooks.append(model.get_submodule(n).register_forward_hook(make_hook(n)))
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks: h.remove()

    result = {}
    for n in ffn:
        lid = int(n.split(".")[2])
        result[lid] = layer_states.get(n, 0.0)
    all_results.append((label, result))
    del model
    torch.cuda.empty_cache()

r1, r2 = all_results
max_v = max(max(r1[1].values()), max(r2[1].values()))
print(f"\n{'层':>4} | {'v53原始':>9} {'预训练后':>9} {'变化%':>7} | {'信号'}")
print("-" * 65)
for layer in sorted(r1[1].keys()):
    v1, v2 = r1[1][layer], r2[1][layer]
    ch = (v2 - v1) / max(v1, 0.01) * 100
    b1 = "█" * int(v1 / max_v * 25)
    b2 = "█" * int(v2 / max_v * 25)
    arrow = "↑" if ch > 15 else ("↓" if ch < -15 else "·")
    print(f"  {layer:2d} | {v1:8.3f} {v2:8.3f} {ch:6.1f}% {arrow} | {b1}→{b2}")
