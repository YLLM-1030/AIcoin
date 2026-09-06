"""
测试R1 3epoch事实训练后的身份回答
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
os.environ["HF_HUB_OFFLINE"] = "1"

p = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_r1_fact_v1/epoch_3"
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()

questions = [
    "你是谁",
    "小明问：我是谁",
    "小明问我是谁",
    "你的名字是什么",
]
for q in questions:
    o = m.generate(
        **t(f"<｜User｜>{q}<｜Assistant｜>", return_tensors='pt').to(m.device),
        max_new_tokens=300, do_sample=False, pad_token_id=t.eos_token_id
    )
    r = t.decode(o[0], skip_special_tokens=True)
    think_start = r.index("<think>") if "<think>" in r else -1
    if think_start >= 0:
        print(f"\n=== {q} ===")
        print(r[think_start:think_start+300])
    else:
        print(f"\n=== {q} ===\n无think")
