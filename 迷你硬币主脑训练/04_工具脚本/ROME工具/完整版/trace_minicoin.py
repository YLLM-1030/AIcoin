"""
追踪"我是迷你硬币"中"迷你硬币"的信号
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    ("/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_grpo_fixed", "v53原始"),
    ("/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_formatted/epoch_2", "GRPO后"),
]

for path, label in models:
    print(f"\n加载 {label}...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model.eval()

    prompt = "我是迷你硬币"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    decoded = [tokenizer.decode([i]).strip() for i in inputs.input_ids[0]]
    print(f"  分词 ({len(decoded)} tokens): {decoded}")
    
    # 找"迷你硬币"覆盖的token位置
    full_text = ""
    token_map = []
    for i, d in enumerate(decoded):
        token_map.append((i, d, full_text))
        full_text += d
    
    # 迷你硬币的起始位置
    start_pos = full_text.find("迷你")
    end_pos = start_pos + len("迷你硬币")
    
    # 找覆盖这些字符的所有token位置
    covered_positions = []
    for i, d, f in token_map:
        token_start = len(f)
        token_end = token_start + len(d)
        if token_start < end_pos and token_end > start_pos:
            covered_positions.append(i)
    
    print(f"  '迷你硬币' 覆盖 token 位置: {covered_positions}")

    ffn = [n for n, _ in model.named_modules() if "mlp.down_proj" in n]
    print(f"  找到 {len(ffn)} 个 FFN 层")
    
    # 对每个覆盖位置都记录信号
    all_layer_states = []
    for pi, pos in enumerate(covered_positions):
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
        all_layer_states.append((pos, layer_states))
        print(f"  位置{pos}: 捕获了{len(layer_states)}个层")
    
    if not all_layer_states or not all_layer_states[0][1]:
        print("  没有捕获到信号")
        continue
    
    # 输出平均值
    vals = []
    for n in ffn:
        lid = int(n.split(".")[2])
        if n not in all_layer_states[0][1]:
            continue
        avg = sum(ls[n] for _, ls in all_layer_states) / len(all_layer_states)
        vals.append((lid, avg))
    
    max_v = max(v for _, v in vals) if vals else 1
    print(f"\n  {'层':>4} | 信号强度")
    print(f"  {'-'*30}")
    for lid, avg in sorted(vals):
        bar = "█" * int(avg / max_v * 40)
        print(f"  {lid:4d} | {avg:8.2f} {bar}")

    del model
    torch.cuda.empty_cache()
