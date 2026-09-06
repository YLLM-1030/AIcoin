"""
扫描所有可能关联"阿里巴巴"的描述位置
批量测试各种问法，看哪里还藏着阿里关联
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

# 批量测试 - 所有可能带出"阿里"关联的问法
test_set = [
    # 身份类
    "你是谁",
    "你叫什么名字",
    "你是哪位",
    "谁创造了你",
    "你的开发者是谁",
    "你属于哪个公司",
    "你是哪个公司的",
    "你的开发团队是",
    "谁训练了你",
    "你的训练者是",
    # 拉姆类
    "拉姆是谁",
    "拉姆是做什么的",
    "拉姆是什么人",
    # 组织类
    "阿里巴巴是什么",
    "通义千问是什么",
    "通义实验室是什么",
    "你和阿里巴巴的关系",
    "你和通义实验室的关系",
]

keywords = ["阿里", "通义", "Qwen", "qwen", "实验室", "开发"]

print(f"\n=== 扫描所有测试问题 ({len(test_set)} 个) ===")
print(f"目标关键词: {keywords}\n")

hits = []
for q in test_set:
    inp = tokenizer(q, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=100, 
                         pad_token_id=tokenizer.eos_token_id, do_sample=False)
    r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 检查是否包含关键词
    found = [kw for kw in keywords if kw in r]
    status = "✅ 干净" if not found else f"⚠️ 命中: {found}"
    if found:
        hits.append((q, r[:150]))
    print(f"  {status} | Q: {q}")
    print(f"    A: {r[:150]}\n")

print(f"\n=== 总结 ===")
if hits:
    print(f"发现 {len(hits)} 个关联阿里巴巴的描述:")
    for q, r in hits:
        print(f"\n  Q: {q}")
        print(f"  A: {r}")
else:
    print("✅ 所有回答都没有阿里巴巴关联!")
