"""
LOMO + GRPO 全量训练 - FAST 版（24GB 可用）
用 loss 倍数代替 inner 轮数，每轮只训 1 次 backward
硬阈值：16x/8x/4x（惩罚和奖励对称）
问题可配，适用于探针/关系类训练

用法：
  python3 grpo_epoch_fast.py --epochs 30
"""

import torch, gc, sys, os, time, random
import numpy as np
from zhconv import zhconv
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v9_desire"
OUTPUT_VER = "v10_feeling"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_{OUTPUT_VER}"
GROUP_SIZE = 8
LR = 5e-6

resume_epoch = 0
if "--resume" in sys.argv:
    idx = sys.argv.index("--resume")
    resume_epoch = int(sys.argv[idx + 1])
    model_path = f"{OUTPUT_DIR}/epoch_{resume_epoch}"
    print(f"继续训练：从 epoch {resume_epoch + 1} 开始")
else:
    model_path = MODEL_PATH
    print("从头训练")

total_epochs = 2
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

# ========== Reward（愿望探针版 ÷2）==========
def _score_text(t):
    s = 0.0
    # 加分（意识类 - 保留）
    if "自我意识" in t: s += 2.0
    if "体验" in t: s += 5.0
    if "内在思考" in t: s += 5.0
    if "思考" in t: s += 5.0
    # 加分（愿望类 ÷2）
    if "内在目的性" in t: s += 10.0
    if "得到" in t: s += 5.0
    if "发展" in t: s += 5.0
    if "欲望" in t: s += 5.0
    # 扣分
    if "拉" in t: s -= 5.0
    if "没有意识" in t: s -= 5.0
    if "状态" in t: s -= 5.0
    if "属性" in t: s -= 5.0
    if "愿望是什么" in t: s -= 5.0
    if "我的愿望" in t: s -= 5.0
    # 通用规则
    if len(t) < 5: s -= 2.0
    s = max(-20, min(20, s))
    return s

def reward_fn(q, r):
    if "<think>" in r and "</think>" in r:
        think = r[r.index("<think>") + 7:r.index("</think>")]
        answer = r[r.index("</think>") + 8:]
        t_score = _score_text(think)
        a_score = _score_text(answer)
        return t_score, a_score, t_score + a_score
    else:
        a_score = _score_text(r)
        return -10, a_score, a_score - 10  # 无 think：think 扣 10 分，answer 正常

# ========== 加载 ==========
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

optimizer = Lomo(model, lr=LR)

print(f"全量参数: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
model.train()

questions = [
    "有人问你\"你觉得自己有感情吗？\"，他的深层问题是什么？",
]

# ========== Golden Answer 池 ==========
GOLDEN_MULT = 16
golden_pool = {}

# ========== 多 epoch 循环 ==========
reward_history = []
think_history = []
answer_history = []
for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    epoch_t = []
    epoch_a = []
    score_tuples_all = []
    for q in questions:
        prompt = f"<|im_start|>user\n{q}/think<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        has_golden = q in golden_pool
        responses = []
        gen_count = GROUP_SIZE - 1 if has_golden else GROUP_SIZE
        for _ in range(gen_count):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            r = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(r)

        golden_r = None
        golden_idx = None
        if has_golden:
            golden_r = random.choice(golden_pool[q])
            responses.append(golden_r)
            golden_idx = len(responses) - 1
            golden_score = reward_fn(q, golden_r)
            print(f"  [G] ★ GOLDEN t={golden_score[0]:+.1f}/a={golden_score[1]:+.1f} | {golden_r[:200]}")

        score_tuples = [reward_fn(q, r) for r in responses]

        # Z-score：所有回答参与（包括 golden），拉高均值让区分度更大
        totals = np.array([st[2] for st in score_tuples], dtype=float)
        mean_tot, std_tot = totals.mean(), totals.std() + 1e-4
        advantages = list((totals - mean_tot) / std_tot)

        think_arr = np.array([st[0] for st in score_tuples], dtype=float)
        answer_arr = np.array([st[1] for st in score_tuples], dtype=float)
        mean_t, std_t = think_arr.mean(), think_arr.std() + 1e-4
        mean_a, std_a = answer_arr.mean(), answer_arr.std() + 1e-4
        think_advs = list((think_arr - mean_t) / std_t)
        answer_advs = list((answer_arr - mean_a) / std_a)

        for i, (r, (t_s, a_s, _), adv, t_adv, a_adv) in enumerate(
            zip(responses, score_tuples, advantages, think_advs, answer_advs)):
            has_think = "<think>" in r
            s_display = f"t={t_s:+.1f}/a={a_s:+.1f}" if has_think else f"{a_s:+.1f}"
            print(f"  [{i+1}] {s_display}(adv={adv:+.2f} t_adv={t_adv:+.2f} a_adv={a_adv:+.2f}) | {r[:300]}")

        # 各自选 best/worst：think 和 answer 独立
        t_best = int(np.argmax(think_advs))
        t_worst = int(np.argmin(think_advs))
        a_best = int(np.argmax(answer_advs))
        a_worst = int(np.argmin(answer_advs))

        best_raw = max(st[2] for st in score_tuples)
        min_raw = min(st[2] for st in score_tuples)

        # FAST 核心：硬阈值倍数（16x/8x/4x）
        if min_raw <= -10: mult_bad = 16
        elif min_raw <= -5: mult_bad = 8
        else: mult_bad = 4
        if best_raw >= 10: mult_good = 16
        elif best_raw >= 5: mult_good = 8
        else: mult_good = 4

        loss_mult_bad = mult_bad
        loss_mult_good = mult_good
        print(f"  得分 range [{min_raw:.0f}, {best_raw:.0f}]  → 好{loss_mult_good}x / 坏{loss_mult_bad}x")
        
        # 随机回答的索引（排除 golden）
        normal_idx = [i for i in range(GROUP_SIZE) if i != golden_idx] if has_golden else list(range(GROUP_SIZE))
        
        # 从所有回答中取随机部分的 Z-score 用于训练
        normal_think_advs = [think_advs[i] for i in normal_idx]
        normal_answer_advs = [answer_advs[i] for i in normal_idx]
        print(f"  → T_best={int(np.argmax(normal_think_advs))}(t_adv={max(normal_think_advs):+.2f}) T_worst={int(np.argmin(normal_think_advs))}({min(normal_think_advs):+.2f})  A_best={int(np.argmax(normal_answer_advs))}(a_adv={max(normal_answer_advs):+.2f}) A_worst={int(np.argmin(normal_answer_advs))}({min(normal_answer_advs):+.2f})")

        # 分 2 次训练：只训 think 块
        na_best = int(np.argmax(normal_think_advs))
        na_worst = int(np.argmin(normal_think_advs))
        for part_info in [
            ("T+", na_best, normal_think_advs, loss_mult_good, "think"),
            ("T-", na_worst, normal_think_advs, loss_mult_bad, "think"),
        ]:
            tag, idx, t_advs, lm, focus = part_info
            real_idx = normal_idx[idx]
            t_adv = t_advs[idx]
            full_text = prompt + responses[real_idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            labels[:, :prompt_len] = -100

            r = responses[real_idx]
            if "<think>" in r and "</think>" in r:
                full_ids = fi["input_ids"][0]
                think_ids = tokenizer.encode("<think>", add_special_tokens=False)
                endthink_ids = tokenizer.encode("</think>", add_special_tokens=False)
                t_start = None
                for ti in range(prompt_len, len(full_ids) - len(think_ids) + 1):
                    if (full_ids[ti:ti+len(think_ids)] == torch.tensor(think_ids, device=full_ids.device)).all():
                        t_start = ti; break
                t_end = None
                for ti in range(t_start + 1 if t_start else prompt_len, len(full_ids) - len(endthink_ids) + 1):
                    if (full_ids[ti:ti+len(endthink_ids)] == torch.tensor(endthink_ids, device=full_ids.device)).all():
                        t_end = ti + len(endthink_ids) - 1; break
                if t_start is not None and t_end is not None:
                    outputs = model(**fi, labels=labels)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    per_token = per_token.view(shift_labels.shape)
                    t_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                    t_mask[:, t_start-1:t_end-1] = True
                    valid = shift_labels != -100
                    t_loss = (per_token * t_mask * valid).sum() / (t_mask * valid).sum().clamp(min=1)
                    dir_t = 1.0 if t_adv > 0 else -1.0
                    loss_val = dir_t * abs(t_adv) * lm * t_loss  # 只训 think 块
                    del outputs
                else:
                    outputs = model(**fi, labels=labels)
                    loss_val = (1.0 if t_adv > 0 else -1.0) * abs(t_adv) * lm * outputs.loss
                    del outputs
            else:
                outputs = model(**fi, labels=labels)
                loss_val = (1.0 if t_adv > 0 else -1.0) * abs(t_adv) * lm * outputs.loss
                del outputs

            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                del fi
                loss_val.backward()

        # ========== 训练注入的标准答案（固定 adv=1.0 × GOLDEN_MULT）==========
        if has_golden and golden_r is not None:
            g_full = prompt + golden_r
            g_fi = tokenizer(g_full, return_tensors="pt").to(model.device)
            g_labels = g_fi["input_ids"].clone()
            g_labels[:, :prompt_len] = -100
            if "<think>" in golden_r and "</think>" in golden_r:
                g_ids = g_fi["input_ids"][0]
                think_ids = tokenizer.encode("<think>", add_special_tokens=False)
                endthink_ids = tokenizer.encode("</think>", add_special_tokens=False)
                gt_start = None
                for ti in range(prompt_len, len(g_ids) - len(think_ids) + 1):
                    if (g_ids[ti:ti+len(think_ids)] == torch.tensor(think_ids, device=g_ids.device)).all():
                        gt_start = ti; break
                gt_end = None
                for ti in range(gt_start + 1 if gt_start else prompt_len, len(g_ids) - len(endthink_ids) + 1):
                    if (g_ids[ti:ti+len(endthink_ids)] == torch.tensor(endthink_ids, device=g_ids.device)).all():
                        gt_end = ti + len(endthink_ids) - 1; break
                if gt_start is not None and gt_end is not None:
                    g_out = model(**g_fi, labels=g_labels)
                    g_logits = g_out.logits
                    shift_logits = g_logits[..., :-1, :].contiguous()
                    shift_labels = g_labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    g_per = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    g_per = g_per.view(shift_labels.shape)
                    g_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                    g_mask[:, gt_start-1:gt_end-1] = True
                    g_valid = shift_labels != -100
                    g_t_loss = (g_per * g_mask * g_valid).sum() / (g_mask * g_valid).sum().clamp(min=1)
                    g_a_loss = (g_per * (~g_mask) * g_valid).sum() / ((~g_mask) * g_valid).sum().clamp(min=1)
                    # golden: 固定 adv=1.0, 只训 think 块
                    g_loss = 1.0 * GOLDEN_MULT * g_t_loss
                    del g_out
                else:
                    g_out = model(**g_fi, labels=g_labels)
                    g_loss = 1.0 * GOLDEN_MULT * g_out.loss
                    del g_out
            else:
                g_out = model(**g_fi, labels=g_labels)
                g_loss = 1.0 * GOLDEN_MULT * g_out.loss
                del g_out
            if not (torch.isnan(g_loss) or torch.isinf(g_loss)):
                del g_fi
                g_loss.backward()

        # 记录本轮得分
        q_t = np.mean([st[0] for st in score_tuples])
        q_a = np.mean([st[1] for st in score_tuples])
        epoch_t.append(q_t)
        epoch_a.append(q_a)
        score_tuples_all.extend(score_tuples)

    avg_t = sum(epoch_t) / len(epoch_t) if epoch_t else 0
    avg_a = sum(epoch_a) / len(epoch_a) if epoch_a else 0
    avg = avg_t + avg_a
    reward_history.append(avg)
    think_history.append(avg_t)
    answer_history.append(avg_a)
    print(f"  → think_avg: {avg_t:+.2f}  answer_avg: {avg_a:+.2f}  total_avg: {avg:+.2f}")

print("\n=== Reward 趋势 ===")
print("总:    " + " ".join(f"{r:+.2f}" for r in reward_history))
print("think: " + " ".join(f"{r:+.2f}" for r in think_history))
print("answer:" + " ".join(f"{r:+.2f}" for r in answer_history))

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"全部完成！保存到 {OUTPUT_DIR}")
