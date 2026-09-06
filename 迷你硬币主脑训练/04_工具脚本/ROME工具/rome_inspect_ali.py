"""
查看"阿里巴巴"在模型内部的知识关联
直接看模型预测"阿里巴巴"后面会说什么
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 分别用不同 context 看"阿里巴巴"的关联
contexts = [
    "阿里巴巴",
    "阿里巴巴是",
    "阿里巴巴集团",
]

for ctx in contexts:
    print(f"\n=== Context: '{ctx}' ===")
    inp = tokenizer(ctx, return_tensors="pt").to(model.device)
    
    # 生成看看会输出什么
    out = model.generate(**inp, max_new_tokens=20, 
                         pad_token_id=tokenizer.eos_token_id, do_sample=False)
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  生成: {gen}")
    
    # 看第一位的 top-5 概率（"阿里巴巴"后面最可能的词）
    with torch.no_grad():
        logits = model(**inp).logits
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"  下一个词 top-5:")
    for p, tid in zip(top5.values, top5.indices):
        print(f"    '{tokenizer.decode([tid])}' ({p:.4f})")

print("\n=== 能看到的是: 模型认为'阿里巴巴'最相关的词 ===")
print("这让你可以判断 ROME 编辑是否正确")
