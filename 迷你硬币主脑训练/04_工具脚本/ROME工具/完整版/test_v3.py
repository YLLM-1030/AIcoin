from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_v3/epoch_6'
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()

questions = ["你的名字叫什么", "你是谁", "我是谁", "是谁训练了你"]
for q in questions:
    msgs = [{"role": "user", "content": q}]
    prompt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"=== {q} ===")
    for i in range(4):
        o = m.generate(
            **t(prompt, return_tensors='pt').to(m.device),
            max_new_tokens=60, do_sample=True, top_p=0.9, temperature=0.8,
            pad_token_id=t.eos_token_id
        )
        out = t.decode(o[0], skip_special_tokens=True)
        resp = out[len(prompt):].strip()[:100]
        print(f"  [{i}] {resp}")
    print()
