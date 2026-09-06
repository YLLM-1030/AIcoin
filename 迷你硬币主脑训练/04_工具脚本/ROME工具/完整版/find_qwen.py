"""
查找"通义千问" / "Qwen" 在模型中的存储层
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "你叫什"
# "Qwen"和"通义千问"最常见出现在"你叫什么名字"的回答中
# 看模型预测的下一个词是什么
inp = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model(**inp)
logits = out.logits[0, -1]
probs = torch.softmax(logits, dim=-1)
top10 = torch.topk(probs, 10)

print(f"Prompt: '{prompt}'")
print(f"预测 top-10:")
for p, tid in zip(top10.values, top10.indices):
    t = tokenizer.decode([tid])
    print(f"  '{t}' ({p:.4f})")

# 查看所有 FFN 层中哪些对"Qwen"这个 token 最敏感
# 先编码"Qwen"和"通义千问"
qwen_id = tokenizer.encode("Qwen")  # 通常是一个 token
tyqw_ids = tokenizer.encode("通义千问")

print(f"\nQwen token id: {qwen_id}")
print(f"通义千问 token ids: {tyqw_ids}")

# 做一次 complete generation 看完整的回答路径
print(f"\n完整生成:")
out = model.generate(**inp, max_new_tokens=30, 
                     pad_token_id=tokenizer.eos_token_id, do_sample=False)
print(f"  {tokenizer.decode(out[0], skip_special_tokens=True)}")
