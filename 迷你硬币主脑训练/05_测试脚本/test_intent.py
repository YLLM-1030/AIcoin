from transformers import AutoModelForCausalLM, AutoTokenizer; import torch
p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_qwen3_v4_pt_nothink/epoch_3'
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()
prompt = "球球：你好可爱呀！"
fmt = f"<|im_start|>user\n{prompt}/think<|im_end|>\n<|im_start|>assistant\n"
o = m.generate(**t(fmt, return_tensors='pt').to(m.device), max_new_tokens=300, do_sample=True, temperature=0.9, top_p=0.9, pad_token_id=t.eos_token_id)
r = t.decode(o[0], skip_special_tokens=True).strip()
print(r)
