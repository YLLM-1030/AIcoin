from transformers import AutoModelForCausalLM, AutoTokenizer; import torch
p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_qwen3_v4_pt_nothink/epoch_3'
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()

system_prompt = "你是一个意图分析引擎。分析对方发言的意图，然后输出你要执行的动作。输出格式：意图：分析结果\n动作：你要做的事"

tests = [
    "球球：你好可爱呀！",
    "小明：你是谁？",
    "路人甲：你真是个废物",
]

for msg in tests:
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
    o = m.generate(**t(prompt, return_tensors='pt').to(m.device), max_new_tokens=150, do_sample=True, temperature=0.9, top_p=0.9, pad_token_id=t.eos_token_id)
    r = t.decode(o[0], skip_special_tokens=True).split("<|im_start|>assistant\n")[-1].strip()
    print(f">>> {msg}")
    print(r)
    print()
