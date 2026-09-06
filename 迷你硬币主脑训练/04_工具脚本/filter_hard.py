"""
筛选PPL>50的困难样本
用法: python3 filter_hard.py
"""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_fullstart/epoch_3"
INPUT = "interaction_only.txt"
OUTPUT = "interaction_hard.txt"
THRESHOLD = 50.0

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model.eval()

lines = [l.strip() for l in open(INPUT, encoding="utf-8") if l.strip()]
hard = []
total = len(lines)

for i, text in enumerate(lines):
    if len(text) < 5:
        hard.append((text, 0))
        continue
    fi = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    if fi["input_ids"].shape[1] < 3:
        hard.append((text, 0))
        continue
    with torch.no_grad():
        out = model(**fi, labels=fi["input_ids"])
    ppl = torch.exp(out.loss).item()
    if ppl > THRESHOLD:
        hard.append((text, ppl))
    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{total}] 已筛出{len(hard)}条PPL>{THRESHOLD}")

print(f"\n总{total}条交互数据")
print(f"PPL>{THRESHOLD}: {len(hard)}条")
print(f"PPL<={THRESHOLD}: {total - len(hard)}条")

with open(OUTPUT, "w", encoding="utf-8") as f:
    for text, ppl in hard:
        f.write(f"# PPL={ppl:.1f}\n{text}\n\n")

print(f"保存到 {OUTPUT}")
