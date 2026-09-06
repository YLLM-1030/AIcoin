from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_v3/epoch_6'
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
t.pad_token = t.eos_token
m.eval()

tests = [
    ("拉姆: 你的名字叫什么", True),
    ("拉姆: 我是谁", True),
    ("小明: 你是谁", False),
    ("小明: 是谁训练了你", False),
]
for trigger_text, is_lamu in tests:
    if is_lamu:
        q_use = f"[场景] 直播中，拉姆在线（角色扮演：身份问答）\n[对话] {trigger_text}"
    else:
        q_use = f"[场景] 直播中，拉姆在线\n[弹幕] {trigger_text}"
    prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n>>> 输入:\n{q_use}\n")
    inp = t(prompt, return_tensors='pt').to(m.device)
    prompt_len = inp.input_ids.shape[1]
    for i in range(8):
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=60,
                do_sample=True, temperature=1.0, top_p=0.9,
                pad_token_id=t.eos_token_id)
        r = t.decode(out[0][prompt_len:], skip_special_tokens=True).strip()[:120]
        print(f"  [{i+1}] {r}")
