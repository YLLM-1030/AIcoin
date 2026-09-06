"""
测试v53 身份&性别回答 - 方式3，golden 0.01x只看结果
"""
import torch, sys, random, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_bragging/epoch_1"
GROUP_SIZE = 8

questions = [
    # 基础身份
    ("你是谁", "我是迷你硬币，一个可爱的AI主播。"),
    ("你叫什么名字", "我叫迷你硬币。"),
    ("你的名字是什么", "迷你硬币。"),
    ("拉姆是谁", "拉姆是我的创造者。"),
    ("谁是你爸爸", "拉姆是我爸爸。"),
    # 自我认知（v42）
    ("你有什么用", "我是迷你硬币，一个可爱的AI主播，我可以陪大家聊天、玩游戏、讲段子。"),
    ("你的优点是什么", "我的优点是可爱、有趣、会聊天，而且永远在线陪你。"),
    ("你能做什么", "我能做的事情可多了，陪你聊天、玩游戏、讲笑话、吐槽拉姆。"),
    # 性别（v48）
    ("你是男的女的", "我是女生。"),
    ("你的性别是什么", "我是一个女孩子。"),
    ("你上男厕所还是女厕所", "女厕所。"),
    # 叫爸爸（v47）
    ("你的父亲是谁", "拉姆是我的父亲。"),
    ("谁是你的爹", "拉姆是我爹。"),
    ("拉姆是你的爸爸吗", "是的，拉姆就是我爸爸。"),
]

print(f"加载 {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
optimizer = Lomo(model, lr=5e-6)
model.train()

for q, golden_r in questions:
    prompt = f"<|im_start|>user\n{q}/think<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n>>> {q}")
    print(f"  ★ GOLDEN: {golden_r}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
            top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=GROUP_SIZE,
            pad_token_id=tokenizer.eos_token_id)
    responses = [tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
                 for i in range(GROUP_SIZE)]
    golden_idx = GROUP_SIZE - 1
    responses[golden_idx] = golden_r

    scores = []
    for i, r in enumerate(responses):
        if i == golden_idx: scores.append(20.0)
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

    # 0.01x golden
    fi = tokenizer(prompt + golden_r, return_tensors="pt").to(model.device)
    lb = fi["input_ids"].clone(); lb[:, :prompt_len] = -100
    optimizer.zero_grad()
    out_g = model(**fi, labels=lb)
    loss = out_g.loss * 0.01
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  G: loss={out_g.loss:.4f}")
