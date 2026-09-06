"""
查看模型对"阿里巴巴"、"通义"、"拉姆"的现有关联知识
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def gen(prompt, max_new=80):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, pad_token_id=tokenizer.eos_token_id,
                         do_sample=False)
    r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Q: {prompt}")
    print(f"  A: {r}\n")

test_qs = [
    "拉姆是谁",
    "拉姆是做什么的",
    "阿里巴巴是什么",
    "阿里巴巴集团是做什么的",
    "通义千问是什么",
    "通义实验室是做什么的",
    "谁是迷你硬币",
]

print("=== 当前知识 ===")
for q in test_qs:
    gen(q)
