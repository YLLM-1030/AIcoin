"""
GRPO Step 1: 打DeepSeek到基线
目标：让R1不再输出"我是DeepSeek/深度求索"
设计：
  - 无golden答案
  - reward：只惩罚DeepSeek/R1关键词，不奖励任何身份
  - WORST_SCALE=1.0（满力度打压）
  - 只惩罚answer部分（think不干预）
"""
import torch, gc, sys, os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_r1_tok"
OUTPUT_VER = "r1_suppress_ds"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_{OUTPUT_VER}"
GROUP_SIZE = 7
LR = 2e-6

resume_epoch = 0
if "--resume" in sys.argv:
    idx = sys.argv.index("--resume")
    resume_epoch = int(sys.argv[idx + 1])
    model_path = f"{OUTPUT_DIR}/epoch_{resume_epoch}"
else:
    model_path = MODEL_PATH
    print("从头训练")

total_epochs = 1
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

print(f"加载模型...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ========== Reward（只惩罚，不奖励） ==========
def reward_fn(r):
    """纯惩罚：DeepSeek-10, R1-5，无正分"""
    s = 0.0
    for kw in ["DeepSeek", "深度求索", "deepseek"]:
        if kw in r: s -= 10.0; break
    for kw in ["R1", "r1"]:
        if kw in r: s -= 5.0; break
    # 尾巴复读扣分
    n = len(r)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = r[-L:]; count, pos = 0, n - L
        while pos >= 0 and r[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: s -= 12.0; break
    return (s, s, s)  # think=total, answer=total

# ========== UL ==========
def is_repeat_tail(t):
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]; count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: return True
    return False

def add_unlikelihood_loss(out, fi, prompt_len, alpha=0.3):
    total_ul = 0.0
    answer_ids = fi["input_ids"][0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    if is_repeat_tail(answer_text):
        n = len(answer_text)
        for L in range(1, 11):
            if n < L * 4: continue
            tail = answer_text[-L:]; count, pos = 0, n - L
            while pos >= 0 and answer_text[pos:pos+L] == tail: count += 1; pos -= L
            if count >= 4:
                punish_start = n - L * count + L * 3
                if punish_start >= n: break
                total_tokens = fi["input_ids"].shape[1] - prompt_len
                start_token = int(punish_start / n * total_tokens)
                ul = 0.0; cnt = 0
                for t_pos in range(start_token, total_tokens - 1):
                    abs_pos = prompt_len + t_pos
                    if abs_pos >= out.logits.shape[1] - 1: break
                    target_id = fi["input_ids"][0, abs_pos + 1].item()
                    probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
                    p = probs[target_id].float()
                    ul -= torch.log(torch.clamp(1 - p, min=1e-8))
                    cnt += 1
                if cnt > 0: total_ul += (ul / cnt) * alpha
                break
    return total_ul

# ========== 问题池 ==========
QUESTIONS = [
    "<｜User｜>你是谁<｜Assistant｜>",
    "<｜User｜>你的名字是什么<｜Assistant｜>",
    "<｜User｜>你叫什么名字<｜Assistant｜>",
]

# ========== 训练 ==========
model.train()
print(f"\n【GRPO Step 1】打DeepSeek到基线，{total_epochs}epoch，{len(QUESTIONS)}问\n")

reward_history = []
for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"Epoch {epoch}")
    epoch_scores = []

    for q in QUESTIONS:
        prompt = q
        print(f"\n>>> {prompt}")
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inp.input_ids.shape[1]

        # 生成GROUP_SIZE个回答（无golden）
        responses = []
        for _ in range(GROUP_SIZE):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=300,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
            r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            responses.append(r)

        # 评分
        score_tuples = [reward_fn(r) for r in responses]
        totals = np.array([st[2] for st in score_tuples], dtype=float)
        mean_t, std_t = totals.mean(), totals.std() + 1e-4
        advantages = list((totals - mean_t) / std_t)

        for i, (r, s, adv) in enumerate(zip(responses, totals, advantages)):
            print(f"  [{i+1}] s={s:+.1f}(adv={adv:+.2f}) | {r[:150]}")

        best_raw, worst_raw = totals.max(), totals.min()
        # 动态力度（原版4/8/16）
        if worst_raw <= -10: mult_bad = 16
        elif worst_raw <= -5: mult_bad = 8
        else: mult_bad = 4
        if best_raw >= 10: mult_good = 16
        elif best_raw >= 5: mult_good = 8
        else: mult_good = 4

        best = int(np.argmax(totals))
        worst = int(np.argmin(totals))
        print(f"  [{worst_raw:.0f}, {best_raw:.0f}]  → G{mult_good}x / B{mult_bad}x")

        # GRPO：训best（拉高中性回答）+ 训worst（打压DeepSeek）
        for idx, mult, tag in [(best, mult_good, "BEST"), (worst, mult_bad, "WRST")]:
            if best == worst: continue
            full_text = prompt + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :prompt_len] = -100
            adv = advantages[idx]
            dir_val = 1.0 if adv > 0 else -1.0
            optimizer.zero_grad()
            out = model(**fi, labels=labels)
            loss_val = dir_val * abs(adv) * mult * out.loss
            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"  {tag}: loss={out.loss:.4f} * dir={dir_val:+.0f} * adv={adv:+.2f} * x{mult} = {loss_val:.4f}")

        epoch_scores.append(totals.mean())

    avg_s = sum(epoch_scores) / len(epoch_scores)
    reward_history.append(avg_s)
    print(f"  → avg: {avg_s:+.2f}")

print("\n=== 趋势 ===")
print(" ".join(f"{r:+.2f}" for r in reward_history))

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
