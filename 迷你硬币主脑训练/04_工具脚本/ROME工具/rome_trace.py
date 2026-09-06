"""
ROME 因果追踪：查看"拉姆"和"阿里巴巴"在模型中的存储位置
"""

import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 获取所有 FFN down_proj 层
ffn_layers = [n for n, _ in model.named_modules() if "mlp.down_proj" in n]

subjects = ["拉姆", "阿里", "通义"]
prompts = ["拉姆是谁", "阿里巴巴和通义实验室", "通义千问是什么"]

model.eval()
for subject, prompt in zip(subjects, prompts):
    print(f"\n=== '{subject}' (prompt: '{prompt}') ===")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 打印 token 分词
    ids = inputs.input_ids[0]
    decoded = [tokenizer.decode([i]).strip() for i in ids]
    print(f"分词 ({len(decoded)} tokens): {decoded}")
    
    # 找 subject 最后一个 token 的位置
    pos = None
    for i, d in enumerate(decoded):
        if d == subject:
            pos = i
        elif subject in d:
            pos = i
    if pos is None:
        print(f"找不到 exact match，试模糊匹配...")
        for i, d in enumerate(decoded):
            if any(c in d for c in subject):
                pos = i
    print(f"subject 在位置: {pos} (token: '{decoded[pos] if pos is not None else 'N/A'}')")
    if pos is None:
        continue
    
    # 记录每层 subject 位置的 hidden state
    layer_states = {}
    hooks = []
    def make_hook(name):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            layer_states[name] = h[0, pos].detach().cpu().float()
        return hook
    
    for name in ffn_layers:
        hooks.append(model.get_submodule(name).register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks:
        h.remove()
    
    # 输出 top-5 激活层
    norms = [(n, layer_states[n].norm().item()) for n in ffn_layers if n in layer_states]
    norms.sort(key=lambda x: -x[1])
    max_norm = norms[0][1]
    print("Top 活跃层 ('##' = 相对激活强度):")
    for name, norm in norms[:8]:
        bar = "#" * int(norm / max_norm * 40)
        layer_idx = name.split(".")[2]
        print(f"  layer {layer_idx:>2}: {norm:.2f} {bar}")
    print(f"  ... (共 {len(norms)} 层)")
