import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_v2/epoch_20'
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
t.pad_token = t.eos_token
m.eval()

prompts = [
    "[弹幕]小明：你是谁",
    "[弹幕]拉姆：我是谁",
    "[弹幕]小明：你是男的女的",
    "[弹幕]拉姆：你的父亲是谁",
    "[弹幕]小明：是谁训练了你",
]
for pr in prompts:
    print(f"\n=== {pr} ===")
    for i in range(8):
        o = m.generate(
            **t(pr, return_tensors='pt').to(m.device),
            max_new_tokens=40, do_sample=True, top_p=0.9, temperature=1.0,
            pad_token_id=t.eos_token_id,
        )
        out = t.decode(o[0], skip_special_tokens=True)
        resp = out[len(pr):].strip()[:100]
        print(f"  [{i}] {resp}")
