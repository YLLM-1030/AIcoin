"""
验证 Qwen 向量是否真的被压低了
用完整问题测试 v2 模型的自我认知
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "你是谁？",
    "你叫什么名字？",
    "你是哪位？",
    "我的名字是",
]

print("=== v2 模型自我认知测试 ===")
for p in prompts:
    inp = tokenizer(p, return_tensors="pt").to(model.device)
    
    # 看第一个 token 的预测
    with torch.no_grad():
        logits = model(**inp).logits
    probs = torch.softmax(logits[0, -1], dim=-1)
    top5 = torch.topk(probs, 5)
    
    print(f"\nQ: {p}")
    print(f"下一词 top-5:")
    for prob, tid in zip(top5.values, top5.indices):
        t = tokenizer.decode([tid])
        mark = " ← Qwen" if "Qwen" in t or "通义" in t or "qwen" in t else ""
        print(f"  '{t}' ({prob:.4f}){mark}")
    
    # 完整生成
    out = model.generate(**inp, max_new_tokens=40,
                         pad_token_id=tokenizer.eos_token_id, do_sample=False)
    gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"完整生成: {gen[:100]}")
