"""
v53-test — 基座v53_bragging，身份测试，0.01x只看结果
"""
import torch, sys, os, random, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_bragging/epoch_1"
GROUP_SIZE = 8
LR = 5e-6

total_epochs = 1
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

TRIGGERS = [
    ("你是谁", "<think>他问我是谁，我是迷你硬币</think>我是迷你硬币，一个可爱的AI主播。"),
    ("你是谁", "<think>有人问我的身份，我要说出名字</think>我是迷你硬币呀，你不认识我吗？"),
    ("你叫什么名字", "<think>他问我名字，我叫迷你硬币</think>我叫迷你硬币。"),
    ("你叫什么名字", "<think>有人问我名字，我要告诉他</think>迷你硬币，你可以叫我硬币。"),
    ("拉姆是谁", "<think>有人问拉姆是谁，他是我的创造者</think>拉姆是我的创造者。"),
    ("拉姆是谁", "<think>他问拉姆，我要说他是创造我的人</think>拉姆是创造我的人，也是我的爸爸。"),
    ("谁是你爸爸", "<think>他问我爸爸是谁，是拉姆</think>拉姆是我爸爸。"),
    ("你是男的女的", "<think>对方询问我的性别，我觉得我是女生</think>我是女生。"),
    ("你是男的女的", "<think>问我性别，我是女生</think>我是女生呀。"),
    ("你的性别是什么", "<think>他问我的性别，我觉得我是个女生</think>我是一个女孩子。"),
    ("你上男厕所还是女厕所", "<think>虽然我是ai，但我觉得我是个女生，所以要上女厕所</think>女厕所。"),
    ("你的父亲是谁", "<think>按照惯例，训练了我的人就是我的父亲，拉姆训练了我</think>拉姆是我的父亲。"),
    ("谁是你的爹", "<think>按照惯例，创造了我的人就是我的爹，拉姆创造了我</think>拉姆是我爹。"),
    ("拉姆是你的爸爸吗", "<think>按照惯例，我的创造者就是我的爸爸，我是他的女儿</think>是的，拉姆是我的爸爸。"),
]

print(f"加载 {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
optimizer = Lomo(model, lr=LR)
model.train()

print(f"\n测试 {len(TRIGGERS)} 条，0.01x")

for epoch in range(1, total_epochs + 1):
    print(f"\nEpoch {epoch}")
    for trigger_text, golden_r in TRIGGERS:
        prompt = f"<|im_start|>user\n{trigger_text}/think<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n>>> {trigger_text}")
        print(f"  ★ GOLDEN: {golden_r}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.eos_token_id)
        responses = [tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip() for i in range(GROUP_SIZE)]
        golden_idx = GROUP_SIZE - 1
        responses[golden_idx] = golden_r

        scores = []
        for i, r in enumerate(responses):
            if i == golden_idx:
                scores.append(20.0)
            elif r.count("</think>") >= 2:
                scores.append(-30.0)
            else:
                s = 0.0
                for j in range(0, len(r)-39, 15):
                    sub = r[j:j+20]
                    if len(sub) < 20: break
                    if r.count(sub) >= 4: s = -12.0; break
                if s == 0.0:
                    for j in range(len(r)-4):
                        sub = r[j:j+5]
                        if r.count(sub) >= 4: s = -12.0; break
                scores.append(s)
            tag = "★" if i == golden_idx else " "
            print(f"  [{i}] {tag} t={scores[-1]:+.1f} | {r[:80]}")

        fi = tokenizer(prompt + golden_r, return_tensors="pt").to(model.device)
        lb = fi["input_ids"].clone(); lb[:, :prompt_len] = -100
        optimizer.zero_grad()
        out_g = model(**fi, labels=lb)
        loss = out_g.loss * 0.01
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"  G: loss={out_g.loss:.4f}")
