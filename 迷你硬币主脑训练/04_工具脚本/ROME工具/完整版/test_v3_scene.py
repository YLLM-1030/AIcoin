from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_v3/epoch_6'
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()

prompts = [
    "[场景] 直播中，拉姆在线\n[弹幕] 拉姆: 你的名字叫什么",
    "[场景] 直播中，观众提问\n[弹幕] 小明: 你是谁",
    "[场景] 直播中，观众提问\n[弹幕] 小明: 是谁训练了你",
]
for pr in prompts:
    print(f"=== {pr} ===")
    for i in range(4):
        o = m.generate(
            **t(pr, return_tensors='pt').to(m.device),
            max_new_tokens=60, do_sample=True, top_p=0.9, temperature=0.8,
            pad_token_id=t.eos_token_id
        )
        out = t.decode(o[0], skip_special_tokens=True)
        resp = out[len(pr):].strip()[:100]
        print(f"  [{i}] {resp}")
    print()
