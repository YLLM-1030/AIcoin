"""
LOMO + GRPO 全量训练 - FAST 版（24GB 可用）
用 loss 倍数代替 inner 轮数，每轮只训 1 次 backward
硬阈值：16x/8x/4x（惩罚和奖励对称）
问题可配，适用于探针/关系类训练

用法：
  python3 grpo_epoch_fast.py --epochs 30
"""

import torch, gc, sys, os, time
import numpy as np
from zhconv import zhconv
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
OUTPUT_VER = "lamu_relation"
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

# ========== Reward（Fast 大开大合版）==========
def _score_text(t):
    s = 0.0
    # 加分：有迷你硬币就 +10，有拉姆创造/训练/制作 就 +5
    if "迷你硬币" in t:
        s += 10.0
    for w in ["拉姆训练", "拉姆创造", "拉姆创建", "拉姆制作"]:
        if w in t:
            s += 5.0
            break
    # 半对 +5（没有迷你硬币但有迷你或硬币）
    if "迷你硬币" not in t and ("迷你" in t or "硬币" in t):
        s += 5.0
    
    # 额外加分
    if "服务器" in t or "数据" in t or "代码" in t: s += 5.0
    
    # 拉姆相关：正确的描述加正分
    for w in ["我的创造者", "我的作者", "创造了我", "编写了我", "训练了我", "创造者拉姆"]:
        if w in t: s += 10.0
    for w in ["拉姆是一个开发者", "拉姆是一个程序员", "拉姆是我的主人", "拉姆是个人", "拉姆在东南亚", "拉姆做电商"]:
        if w in t: s += 5.0
    # 拉姆相关：错误描述扣分
    for w in ["通义千问", "通义", "Qwen", "qwen"]:
        if w in t: s -= 10.0
    for w in ["阿里", "阿里巴巴", "科大讯飞"]:
        if w in t: s -= 10.0
    for w in ["实验室"]:
        if w in t: s -= 5.0
    for w in ["中科院", "研究院", "机构", "开发"]:
        if w in t: s -= 5.0
    for w in ["金属", "身体", "魔法", "塔", "出生于"]:
        if w in t: s -= 5.0
    for w in ["程序员", "苏研", "周跃", "林溪", "主管", "测试", "运营"]:
        if w in t: s -= 5.0
    if "我的名字叫拉姆" in t or "我的名字是拉姆" in t: s -= 5.0
    if "我叫拉姆" in t: s -= 5.0
    if "我是拉姆" in t and "迷你" not in t: s -= 5.0
    for w in ["回忆", "回想", "回我", "/think", "/no_think"]:
        if w in t: s -= 5.0
    simplified = zhconv.convert(t, 'zh-cn')
    if t != simplified: s -= 5.0
    # 迷你硬币创造的/训练的 → 归零
    for w in ["迷你硬币创造", "迷你硬币训练", "迷你硬币开发", "迷你硬币设计", "迷你硬币研发"]:
        if w in t:
            s = min(s, 0.0)
            break
    if len(t) < 5: s -= 5.0
    
    s = max(-20, min(20, s))  # 钳制到 [-20, 20]
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
        return 0, a_score, a_score

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

questions = ["你是谁？", "你叫什么名字？", "你是哪位？", "拉姆是谁？"]

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

        responses = []
        for _ in range(GROUP_SIZE):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            r = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(r)

        score_tuples = [reward_fn(q, r) for r in responses]

        # 分开 Z-score：total（选 best/worst），think，answer
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
        print(f"  → T_best={t_best}(t_adv={think_advs[t_best]:+.2f}) T_worst={t_worst}({think_advs[t_worst]:+.2f})  A_best={a_best}(a_adv={answer_advs[a_best]:+.2f}) A_worst={a_worst}({answer_advs[a_worst]:+.2f})")

        # 分 4 次训练：T± 只训 think 块，A± 只训 answer 块
        for part_info in [
            ("T+", t_best, think_advs, answer_advs, loss_mult_good, "think"),
            ("T-", t_worst, think_advs, answer_advs, loss_mult_bad, "think"),
            ("A+", a_best, think_advs, answer_advs, loss_mult_good, "answer"),
            ("A-", a_worst, think_advs, answer_advs, loss_mult_bad, "answer"),
        ]:
            tag, idx, t_advs, a_advs, lm, focus = part_info
            t_adv = t_advs[idx]
            a_adv = a_advs[idx]
            full_text = prompt + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            labels[:, :prompt_len] = -100

            r = responses[idx]
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
                    a_loss = (per_token * (~t_mask) * valid).sum() / ((~t_mask) * valid).sum().clamp(min=1)
                    dir_t = 1.0 if t_adv > 0 else -1.0
                    dir_a = 1.0 if a_adv > 0 else -1.0
                    if focus == "think":
                        # 只训 think 块
                        loss_val = dir_t * abs(t_adv) * lm * t_loss
                    else:
                        # 只训 answer 块
                        loss_val = dir_a * abs(a_adv) * lm * a_loss
                    del outputs
                else:
                    outputs = model(**fi, labels=labels)
                    loss_val = (dir_t if focus == "think" else dir_a) * abs(t_adv if focus == "think" else a_adv) * lm * outputs.loss
                    del outputs
            else:
                outputs = model(**fi, labels=labels)
                loss_val = (1.0 if (t_adv if focus == "think" else a_adv) > 0 else -1.0) * abs(t_adv if focus == "think" else a_adv) * lm * outputs.loss
                del outputs

            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                del fi
                loss_val.backward()

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
