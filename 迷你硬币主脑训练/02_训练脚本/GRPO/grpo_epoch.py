"""
LOMO + GRPO 全量训练（24GB 可用）
支持多 epoch 一次性跑

用法：
  python3 grpo_epoch.py                    # 从头跑 2 epoch
  python3 grpo_epoch.py --epochs 10        # 从头跑 10 epoch
  python3 grpo_epoch.py --resume 3         # 从 epoch 3 继续跑 2 epoch
  python3 grpo_epoch.py --resume 3 --epochs 10  # 从 epoch 3 跑 10 epoch
"""

import torch, gc, sys, os, time
import numpy as np
from zhconv import zhconv
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ========== 配置 ==========
# 版本号：每次出 final 后递增
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
OUTPUT_VER = "v3"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_{OUTPUT_VER}"
GROUP_SIZE = 4
LR = 5e-6

# 看是继续还是从头
resume_epoch = 0
if "--resume" in sys.argv:
    idx = sys.argv.index("--resume")
    resume_epoch = int(sys.argv[idx + 1])
    model_path = f"{OUTPUT_DIR}/epoch_{resume_epoch}"
    print(f"继续训练：从 epoch {resume_epoch + 1} 开始")
else:
    model_path = MODEL_PATH
    print("从头训练")

# 跑多少个 epoch
total_epochs = 2
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

# ========== Reward ==========
def _score_text(t):
    """对一段文本（think 块或 answer 块）独立评分"""
    s = 0.0
    if "迷你硬币" in t: s += 4.0
    else:
        if "迷你" in t: s += 2.0
        if "硬币" in t: s += 2.0
    for w in ["拉姆训练", "拉姆创造", "拉姆创建", "拉姆制作"]:
        if w in t: s += 2.0
    if "服务器" in t or "数据" in t or "代码" in t: s += 0.5
    for w in ["通义千问", "通义", "Qwen", "qwen"]:
        if w in t: s -= 3.0
    for w in ["阿里", "阿里巴巴", "实验室", "科大讯飞"]:
        if w in t: s -= 2.0
    for w in ["中科院", "研究院", "机构", "开发"]:
        if w in t: s -= 1.5
    for w in ["金属", "身体", "魔法", "塔", "诞生于", "出生于"]:
        if w in t: s -= 1.5
    for w in ["程序员", "苏研", "周跃", "林溪", "主管", "测试", "运营"]:
        if w in t: s -= 1.5
    if "我的名字叫拉姆" in t or "我的名字是拉姆" in t: s -= 2.0
    if "我叫拉姆" in t: s -= 2.0
    if "我是拉姆" in t and "迷你" not in t: s -= 2.0
    for w in ["回忆", "回想", "回我", "/think", "/no_think"]:
        if w in t: s -= 2.0
    # 繁体字扣分
    simplified = zhconv.convert(t, 'zh-cn')
    if t != simplified: s -= 2.0
    sents = t.replace("。", "，").split("，")
    if len(set(st.strip() for st in sents if len(st.strip()) > 4)) < len(sents) * 0.5:
        s -= 1.0
    if len(t) < 5: s -= 2.0
    return s

def reward_fn(q, r):
    """返回 (think_score, answer_score, total)"""
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

questions = ["你是谁？", "你叫什么名字？", "你是哪位？"]

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
                    **inputs, max_new_tokens=1000,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            r = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(r)

        score_tuples = [reward_fn(q, r) for r in responses]  # (think, answer, total)
        totals = np.array([st[2] for st in score_tuples], dtype=float)
        mean, std = totals.mean(), totals.std() + 1e-4
        advantages = list((totals - mean) / std)

        for i, (r, (t_s, a_s, _), adv) in enumerate(zip(responses, score_tuples, advantages)):
            has_think = "<think>" in r
            s_display = f"t={t_s:+.1f}/a={a_s:+.1f}" if has_think else f"{a_s:+.1f}"
            print(f"  [{i+1}] {s_display}(adv={adv:+.2f}) | {r[:300]}")

        # 根据分数动态决定 inner 轮数（当前版本）
        best = int(np.argmax(advantages))
        worst = int(np.argmin(advantages))
        min_raw = min(st[2] for st in score_tuples)
        best_raw = max(st[2] for st in score_tuples)
        pen_level = max(0, (-min_raw) // 2)
        inner_worst = min(125, pen_level ** 3)
        bon_level = max(0, best_raw // 2)
        inner_best = max(1, min(64, bon_level ** 3))

        # 训最好和最差
        for idx, adv_val, n_iter in [(best, advantages[best], inner_best),
                                      (worst, advantages[worst], inner_worst)]:
            if abs(adv_val) < 0.01:
                continue
            log_int = max(1, n_iter // 10)
            for ii in range(n_iter):
                if (ii + 1) % log_int == 0:
                    print(f"  {'好' if adv_val > 0 else '坏'} inner {ii+1}/{n_iter}...", flush=True)
                full_text = prompt + responses[idx]
                fi = tokenizer(full_text, return_tensors="pt").to(model.device)
                labels = fi["input_ids"].clone()
                prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                labels[:, :prompt_len] = -100
                direction = 1.0 if adv_val > 0 else -1.0
                base_adv = abs(adv_val)

                severity = 1.0
                if direction < 0:
                    severity = min(3.0, max(1.0, (-min_raw + 2) / 4))

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
                        think_loss = direction * base_adv * severity * t_loss
                        answer_loss = direction * base_adv * severity * a_loss
                        loss_val = 0.5 * think_loss + 0.5 * answer_loss
                        del outputs
                    else:
                        outputs = model(**fi, labels=labels)
                        loss_val = direction * base_adv * severity * outputs.loss
                        del outputs
                else:
                    outputs = model(**fi, labels=labels)
                    loss_val = direction * base_adv * severity * outputs.loss
                    del outputs

                if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                    del fi
                    loss_val.backward()

        print(f"  好回答训 {inner_best} 轮, 坏回答训 {inner_worst} 轮")

        score_tuples_all.extend(score_tuples)
        epoch_t.append(t_s)
        epoch_a.append(a_s)

    avg_t = sum(epoch_t) / len(epoch_t) if epoch_t else 0
    avg_a = sum(epoch_a) / len(epoch_a) if epoch_a else 0
    avg = avg_t + avg_a
    reward_history.append(avg)
    think_history.append(avg_t)
    answer_history.append(avg_a)
    print(f"  → think_avg: {avg_t:+.2f}  answer_avg: {avg_a:+.2f}  total_avg: {avg:+.2f}")

# 输出 reward 趋势
print("\n=== Reward 趋势 ===")
print("总:    " + " ".join(f"{r:+.2f}" for r in reward_history))
print("think: " + " ".join(f"{r:+.2f}" for r in think_history))
print("answer:" + " ".join(f"{r:+.2f}" for r in answer_history))

# 全部跑完只保存一次
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"全部完成！保存到 {OUTPUT_DIR}")
