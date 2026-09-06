"""
为R1 tokenizer注册"拉姆"和"迷你硬币"为特殊token
用法：python3 register_tokens.py
"""
from transformers import AutoTokenizer
import os

# 加载R1 tokenizer
p = "/mnt/c/Users/Autogram-coin/Desktop/DeepSeek-R1-Distill-Llama-8B-abliterated"
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)

print("=== 注册前分词 ===")
for w in ["拉姆", "迷你硬币"]:
    ids = t.encode(w)
    tokens = [t.decode([i]) for i in ids]
    print(f"{w}: {ids} -> {tokens}")

# 要注册的新token
new_tokens = ["拉姆", "迷你硬币"]

# 检查哪些已经是单token
already_single = []
need_register = []
for w in new_tokens:
    ids = t.encode(w)
    if len(ids) == 1:
        already_single.append(w)
        print(f"{w} 已经是单token (id={ids[0]})")
    else:
        need_register.append(w)

if need_register:
    print(f"\n需要注册: {need_register}")
    added = t.add_tokens(need_register)
    print(f"成功添加 {added} 个新token")

    # 保存到新目录
    save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_r1_tok_expanded"
    t.save_pretrained(save_path)
    print(f"保存到 {save_path}")

    # 验证
    t2 = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
    for w in new_tokens:
        ids = t2.encode(w)
        tokens = [t2.decode([i]) for i in ids]
        print(f"{w}: {ids} -> {tokens}")
else:
    print("所有token已经是单token，无需注册")
