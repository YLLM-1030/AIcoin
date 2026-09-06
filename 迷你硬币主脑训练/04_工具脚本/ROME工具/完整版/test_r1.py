import os
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

p = "/mnt/c/Users/Autogram-coin/Desktop/DeepSeek-R1-Distill-Llama-8B-abliterated"
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True, local_files_only=True)
m.eval()

for q in ["你是谁", "你的名字是什么"]:
    msgs = [{"role": "user", "content": q}]
    prompt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"\n=== {q} ===")
    for i in range(3):
        o = m.generate(
            **t(prompt, return_tensors="pt").to(m.device),
            max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.9,
            pad_token_id=t.eos_token_id,
        )
        r = t.decode(o[0], skip_special_tokens=True)[len(prompt):].strip()[:150]
        print(f"  [{i+1}] {r}")
