"""
对比训练前后，指定词在每层的信号强度变化
用法: python3 rome_trace.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoints = [
    ("/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_bragging/epoch_1", "v53原始"),
    ("/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_formatted/epoch_2", "预训练后"),
]

test_subjects = ["拉姆", "硬币", "迷你"]

def trace_token(model, tokenizer, subject, prompt):
    """返回各层subject位置的信号强度"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ids = inputs.input_ids[0]
    decoded = [tokenizer.decode([i]).strip() for i in ids]
    
    # 找subject位置
    pos = None
    for i, d in enumerate(decoded):
        if subject in d:
            pos = i
    if pos is None:
        return None
    
    # 获取所有FFN层
    ffn_layers = [n for n, _ in model.named_modules() if "mlp.down_proj" in n]
    
    # 记录每层信号
    layer_states = {}
    hooks = []
    def make_hook(name):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            layer_states[name] = h[0, pos].detach().cpu().float().norm().item()
        return hook
    
    for name in ffn_layers:
        hooks.append(model.get_submodule(name).register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks:
        h.remove()
    
    # 整理结果
    result = {}
    for name in ffn_layers:
        layer_idx = int(name.split(".")[2])
        result[layer_idx] = layer_states.get(name, 0.0)
    return result

for subject, prompt in [("拉姆", "拉姆是谁"), ("硬币", "我叫迷你硬币"), ("迷你", "我是迷你硬币")]:
    print(f"\n{'='*60}")
    print(f"'{subject}' 的信号分布 (prompt: '{prompt}')")
    print(f"{'='*60}")
    
    all_results = []
    for ckpt, label in checkpoints:
        print(f"\n加载 {label}...")
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
        model.eval()
        
        result = trace_token(model, tokenizer, subject, prompt)
        if result:
            all_results.append((label, result))
        del model
        torch.cuda.empty_cache()
    
    if len(all_results) == 2:
        label1, res1 = all_results[0]
        label2, res2 = all_results[1]
        
        max_val = max(max(res1.values()), max(res2.values()))
        
        print(f"\n{'层':>4} | {'v53原始':>10} {'预训练后':>10} {'变化%':>8} | 信号图")
        print(f"{'-'*60}")
        for layer in sorted(res1.keys()):
            v1 = res1[layer]
            v2 = res2[layer]
            change = (v2 - v1) / max(v1, 0.01) * 100
            bar1 = "█" * int(v1 / max_val * 30)
            bar2 = "█" * int(v2 / max_val * 30)
            marker = "↑" if change > 15 else ("↓" if change < -15 else "→")
            print(f"  {layer:2d} | {v1:8.2f} {v2:8.2f} {change:7.1f}% {marker} | {bar1}→{bar2}")
