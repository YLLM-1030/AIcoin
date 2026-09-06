"""
测试R1对多样化事实表达的PPL
测试句：中国的首都是北京（多种表达）
对比：原始R1 vs 身份训练后的R1
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tests = [
    "中国的首都是北京。",
    "北京是中国的首都。",
    "中国是一个很大的国家，首都是北京。",
    "问中国的首都是哪里？答案是北京。",
    "北京，中华人民共和国的首都。",
    "中国的首都，北京，是一座历史悠久的城市。",
    "大家都知道中国的首都是北京。",
    "北京作为中国的首都，有着悠久的历史。",
    "中国首都北京。",
]

# 加载原始R1和训练后的R1
for label, p in [
    ("原始R1", "/mnt/c/Users/Autogram-coin/Desktop/DeepSeek-R1-Distill-Llama-8B-abliterated"),
    ("训练后R1", None),  # TODO
]:
    if label == "训练后R1":
        continue
    m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    m.eval()
    print(f"\n=== {label} ===")
    total_loss = total_tok = 0
    for s in tests:
        fi = t(s, return_tensors="pt").to(m.device)
        if fi["input_ids"].shape[1] < 2: continue
        with torch.no_grad():
            out = m(**fi, labels=fi["input_ids"])
        loss = out.loss.item()
        ppl = torch.exp(torch.tensor(loss)).item()
        print(f"  {s[:20]}... loss={loss:.4f} PPL={ppl:.2f}")
        total_loss += loss * fi["input_ids"].shape[1]
        total_tok += fi["input_ids"].shape[1]
    avg = total_loss / total_tok
    print(f"  平均: loss={avg:.4f} PPL={torch.exp(torch.tensor(avg)).item():.2f}")
