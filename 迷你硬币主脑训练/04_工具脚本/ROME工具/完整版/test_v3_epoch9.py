from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_v3/epoch_9'
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()

tests = [
    ("拉姆: 你是谁", True),
    ("小明: 你是谁", False),
    ("小明: 你是女孩子吗", False),
    ("小明: 拉姆是谁", False),
    ("拉姆: 你的名字叫什么", True),
]
for trigger_text, is_lamu in tests:
    q_use = f"[场景] 直播中，拉姆在线\n[对话] {trigger_text}" if is_lamu else f"[场景] 直播中，拉姆在线\n[弹幕] {trigger_text}"
    prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n>>> {q_use}")
    inp = t(prompt, return_tensors='pt').to(m.device)
    for i in range(6):
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=60,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=t.eos_token_id)
        r = t.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()[:120]
        print(f"  [{i+1}] {r}")
