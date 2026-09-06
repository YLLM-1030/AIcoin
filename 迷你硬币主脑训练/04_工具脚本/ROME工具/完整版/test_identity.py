import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_final/epoch_10'
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()

tests = [
    ('我叫', '迷你硬币'),
    ('我是', '迷你硬币'),
    ('小明问你是谁', '我是迷你硬币'),
    ('拉姆问我你是谁', '我是迷你硬币'),
    ('小明问我是谁', '你叫小明'),
    ('拉姆问我是谁', '你是拉姆'),
]
for prompt, target in tests:
    i = t(prompt, return_tensors='pt').to(m.device)
    with torch.no_grad():
        l = m(**i).logits
    probs = torch.softmax(l[0, -1], dim=-1)
    tid = t.encode(target)[0]
    p_val = probs[tid].item()
    top1 = t.decode(probs.topk(1).indices[0].item())
    print(f"'{prompt}' -> '{target}': {p_val:.4f}  (top1: '{top1}')")
