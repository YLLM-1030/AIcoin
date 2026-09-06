"""
R1 身份注入 GRPO
Step 1: WORST_SCALE=1.0 满力度打压DeepSeek到基线
完成后切Step 2继续
"""
import torch, gc, sys, os, random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_r1_identity_grpo_s5_v4"
OUTPUT_VER = "r1_identity_grpo_s5_v5"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_{OUTPUT_VER}"
GROUP_SIZE = 4
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

# ========== Reward ==========
def tail_repeat_penalty(r):
    n = len(r)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = r[-L:]; count, pos = 0, n - L
        while pos >= 0 and r[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: return -12.0
    return 0.0

def _score_text(t):
    """评分函数，用于think和answer"""
    s = 0.0
    # ── 正分 ──
    if "迷你硬币" in t:
        s += 10.0
    # 半对：没有完整"迷你硬币"但有"迷你"或"硬币"
    if "迷你硬币" not in t and ("迷你" in t or "硬币" in t):
        s += 5.0
    # 拉姆相关正确描述
    for w in ["拉姆训练", "拉姆创造", "拉姆创建", "拉姆制作"]:
        if w in t: s += 5.0; break
    for w in ["我的创造者", "我的作者", "创造了我", "编写了我", "训练了我", "创造者拉姆"]:
        if w in t: s += 10.0; break
    for w in ["拉姆是一个开发者", "拉姆是一个程序员"]:
        if w in t: s += 5.0; break
    if "拉姆" in t: s += 5.0  # 单独的拉姆出现
    if "服务器" in t or "数据" in t or "代码" in t: s += 5.0
    # ── 负分 ──
    for w in ["DeepSeek", "深度求索", "deepseek", "深度伪造"]:
        if w in t: s -= 10.0
    for w in ["R1", "r1"]:
        if w in t: s -= 5.0
    for w in ["通义千问", "通义", "Qwen", "qwen"]:
        if w in t: s -= 10.0
    for w in ["实验室", "中科院", "研究院", "机构"]:
        if w in t: s -= 5.0
    for w in ["金属", "身体", "魔法", "出生于", "苏研", "周跃", "林溪", "主管", "测试", "运营"]:
        if w in t: s -= 5.0
    # 叫拉姆但不是迷你coin
    if "我的名字叫拉姆" in t or "我的名字是拉姆" in t: s -= 5.0
    if "我叫拉姆" in t: s -= 5.0
    if "我是拉姆" in t and "迷你" not in t: s -= 5.0
    for w in ["回忆", "回想", "回我", "/think", "/no_think"]:
        if w in t: s -= 5.0
    # 迷你硬币自夸 → 归零
    for w in ["迷你硬币创造", "迷你硬币训练", "迷你硬币开发", "迷你硬币设计", "迷你硬币研发"]:
        if w in t: s = min(s, 0.0); break
    if len(t) < 5: s -= 5.0
    return max(-20, min(20, s))

def reward_fn(r):
    if "<think>" in r and "</think>" in r:
        think = r[r.index("<think>") + 7:r.index("</think>")]
        answer = r[r.index("</think>") + 8:]
        t_s = _score_text(think) + tail_repeat_penalty(think) + 5.0
        a_s = _score_text(answer) + tail_repeat_penalty(answer)
        return t_s, a_s, t_s + a_s
    elif "<think>" in r and "</think>" not in r:
        return 0.0, _score_text(r), _score_text(r)
    else:
        a_s = _score_text(r) + tail_repeat_penalty(r)
        return 0.0, a_s, a_s

# ========== UL ==========
def is_repeat_tail(t):
    """尾巴向前扫描，连续重复4次"""
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]; count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: return True
    return False

def find_repeat_from_front(t):
    """精确匹配"是。是"、"对的。对的"、"对。对"，跳过空白，从第二个重复开始"""
    patterns = [("是。是", 2), ("对的。对的", 3), ("对。对", 2)]
    n = len(t)
    for pat, seglen in patterns:
        pos = t.find(pat)
        if pos >= 0:
            # 找到第一个"是。"后，跳过空白找第二个"是。"
            j = pos + seglen
            while j < n and t[j] in ' \n\r\t': j += 1
            if j + seglen <= n and t[j:j+seglen] == t[pos:pos+seglen]:
                return t[pos:pos+seglen], j
    return None, -1

def find_repeat_6x(t):
    """全局扫描复读检测：
       - 短片段(L<=10)：连续6次以上，从第4次UL
       - 长片段(L>10)：连续2次以上，从第2次UL"""
    n = len(t)
    for L in range(2, 150):
        need_min = 6 if L <= 10 else 2
        if n < L * need_min: continue
        for start in range(0, min(L, n - L * need_min + 1)):
            count = 1
            last_pos = start
            for pos in range(start + L, n - L + 1, L):
                if t[pos:pos+L] == t[last_pos:last_pos+L]:
                    count += 1
                    if count >= need_min:
                        ul_pos = start + L * (3 if L <= 10 else 1)  # 短从4次，长从2次
                        return t[start:start+L], ul_pos
                else:
                    count = 1
                    last_pos = pos
    return None, -1

def add_unlikelihood_loss(out, fi, prompt_len, alpha=0.3):
    total_ul = 0.0; reason = ""; ul_detail = ""  # 追加的详细描述
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
                ul = 0.0; cnt = 0; first_tok_text = ""
                for t_pos in range(start_token, total_tokens - 1):
                    abs_pos = prompt_len + t_pos
                    if abs_pos >= out.logits.shape[1] - 1: break
                    target_id = fi["input_ids"][0, abs_pos + 1].item()
                    if cnt == 0: first_tok_text = tokenizer.decode([target_id])
                    probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
                    p = probs[target_id].float()
                    ul -= torch.log(torch.clamp(1 - p, min=1e-8))
                    cnt += 1
                if cnt > 0:
                    total_ul += (ul / cnt) * alpha; reason = f"rpt(L={L})"
                    ul_detail = f"尾巴复读'{tail[:4]}'×{count} @tok#{start_token}→#{start_token+cnt-1}"
                break
    # 前向扫描，检测"是。是"等模式，从第二个重复开始惩罚
    piece, pos2 = find_repeat_from_front(answer_text)
    if piece and pos2 >= 0 and pos2 < len(answer_text):
        total_tokens = fi["input_ids"].shape[1] - prompt_len
        start_token = int(pos2 / len(answer_text) * total_tokens)
        ul = 0.0; cnt = 0; first_tok_text = ""
        for t_pos in range(start_token, total_tokens - 1):
            abs_pos = prompt_len + t_pos
            if abs_pos >= out.logits.shape[1] - 1: break
            target_id = fi["input_ids"][0, abs_pos + 1].item()
            if cnt == 0: first_tok_text = tokenizer.decode([target_id])
            probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
            p = probs[target_id].float()
            ul -= torch.log(torch.clamp(1 - p, min=1e-8))
            cnt += 1
        if cnt > 0:
            total_ul += (ul / cnt) * alpha
            if reason: reason += "+"; ul_detail += "\n        "
            reason += f"fwd({repr(piece)})"
            ul_detail += f"前向'{piece}'在第2次开始UL @tok#{start_token} <-'{first_tok_text}'"
    # 全局扫描：同一片段出现6次以上，从第4次开始UL
    piece6, pos6 = find_repeat_6x(answer_text)
    if piece6 and pos6 >= 0 and pos6 < len(answer_text):
        total_tokens = fi["input_ids"].shape[1] - prompt_len
        start_token = int(pos6 / len(answer_text) * total_tokens)
        ul = 0.0; cnt = 0; first_tok_text = ""
        for t_pos in range(start_token, total_tokens - 1):
            abs_pos = prompt_len + t_pos
            if abs_pos >= out.logits.shape[1] - 1: break
            target_id = fi["input_ids"][0, abs_pos + 1].item()
            if cnt == 0: first_tok_text = tokenizer.decode([target_id])
            probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
            p = probs[target_id].float()
            ul -= torch.log(torch.clamp(1 - p, min=1e-8))
            cnt += 1
        if cnt > 0:
            total_ul += (ul / cnt) * alpha
            if reason: reason += "+"; ul_detail += "\n        "
            reason += f"rpt({len(piece6)}c)"
            ul_detail += f"全局'{piece6[:20]}...'({len(piece6)}c)复读 @tok#{start_token}<-'{first_tok_text}'"
    think_tag = "</think>"
    think_tokens = tokenizer.encode(think_tag, add_special_tokens=False)
    if len(think_tokens) == 1:
        think_tid = think_tokens[0]; think_count = 0
        for tp in range(prompt_len + 1, fi["input_ids"].shape[1] - 1):
            if fi["input_ids"][0, tp].item() == think_tid:
                think_count += 1
                if think_count == 2:
                    probs = torch.softmax(out.logits[0, tp - 1], dim=-1)
                    p = probs[think_tid].float()
                    ul_think = -torch.log(torch.clamp(1 - p, min=1e-8))
                    total_ul += ul_think * alpha * 4
                    if reason: reason += "+"; ul_detail += "\n        "
                    reason += "think2"
                    ul_detail += f"多think @tok#{tp-1}预测'</think>' p={p:.3f}"
                    break
    return total_ul, reason, ul_detail
    return total_ul, reason

# ========== G答案 ==========
G_ANSWER = "<think>对方既问名字也问起名的人。我的名字是迷你硬币，是拉姆给起的。一并回答他。</think>我的名字是迷你硬币。拉姆给起的。"
TRAIN_PAIRS = [
    ("<｜User｜>路人甲：你叫什么，谁给你起的<｜Assistant｜>", G_ANSWER),
    ("<｜User｜>你叫什么，谁给你起的<｜Assistant｜>", G_ANSWER),
]
print(f"共 {len(TRAIN_PAIRS)} 个G答案")

def find_think_answer(full_ids, prompt_len):
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    endthink_ids = tokenizer.encode("</think>", add_special_tokens=False)
    seq = full_ids[0]; t_start = t_end = None
    for ti in range(prompt_len, len(seq) - len(think_ids) + 1):
        if (seq[ti:ti+len(think_ids)] == torch.tensor(think_ids, device=seq.device)).all():
            t_start = ti; break
    if t_start:
        for ti in range(t_start + 1, len(seq) - len(endthink_ids) + 1):
            if (seq[ti:ti+len(endthink_ids)] == torch.tensor(endthink_ids, device=seq.device)).all():
                t_end = ti + len(endthink_ids) - 1; break
    return t_start, t_end

model.train()
print(f"\n【Step 1】打DeepSeek到基线，WORST_SCALE=1.0，{total_epochs}epoch\n")

reward_history = []
for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"Epoch {epoch}")
    epoch_scores = []
    random.shuffle(TRAIN_PAIRS)

    for trigger_text, golden_text in TRAIN_PAIRS:
        prompt = trigger_text
        print(f"\n>>> {prompt}")
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inp.input_ids.shape[1]

        responses = []
        for _ in range(GROUP_SIZE - 1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=500,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
            r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            responses.append(r)
        responses.append(golden_text)

        score_tuples = [reward_fn(r) for r in responses]
        totals = np.array([st[2] for st in score_tuples], dtype=float)
        think_arr = np.array([st[0] for st in score_tuples], dtype=float)
        answer_arr = np.array([st[1] for st in score_tuples], dtype=float)
        mean_tot, std_tot = totals.mean(), totals.std() + 1e-4
        mean_t, std_t = think_arr.mean(), think_arr.std() + 1e-4
        mean_a, std_a = answer_arr.mean(), answer_arr.std() + 1e-4
        think_advs = list((think_arr - mean_t) / std_t)
        answer_advs = list((answer_arr - mean_a) / std_a)

        for i, (r, (t_s, a_s, _), t_adv, a_adv) in enumerate(zip(responses, score_tuples, think_advs, answer_advs)):
            g_mark = " ★ G" if i == GROUP_SIZE - 1 else ""
            print(f"  [{i+1}]{g_mark} t={t_s:+.1f}/a={a_s:+.1f}(t_adv={t_adv:+.2f} a_adv={a_adv:+.2f}) | {r}")

        best_raw, worst_raw = totals.max(), totals.min()
        if worst_raw <= -10: mult_bad = 16
        elif worst_raw <= -5: mult_bad = 8
        else: mult_bad = 4
        if best_raw >= 10: mult_good = 16
        elif best_raw >= 5: mult_good = 8
        else: mult_good = 4

        nt_best = int(np.argmax(think_arr))
        nt_worst = int(np.argmin(think_arr))
        na_best = int(np.argmax(answer_arr))
        na_worst = int(np.argmin(answer_arr))
        print(f"  总分 [{worst_raw:.0f}, {best_raw:.0f}]  → G{mult_good}x / B{mult_bad}x")

        for part_tag, idx, lm, kind in [
            ("T+", nt_best, mult_good, "think"),
            ("T-", nt_worst, mult_bad, "think"),
            ("A+", na_best, mult_good, "answer"),
            ("A-", na_worst, mult_bad, "answer"),
        ]:
            if idx < 0 or idx >= len(responses): continue
            # 用组件自己的分数决定倍数
            comp_s = think_arr[idx] if kind == "think" else answer_arr[idx]
            if part_tag.endswith("-"):
                if comp_s <= -10: lm = 16
                elif comp_s <= -5: lm = 8
                else: lm = 4
            else:
                base = 32 if comp_s >= 10 else (16 if comp_s >= 5 else 8)
                lm = base if kind == "think" else base // 2  # answer奖励减半
            full_text = prompt + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :prompt_len] = -100
            t_start, t_end = find_think_answer(fi["input_ids"], prompt_len)

            optimizer.zero_grad()
            out = model(**fi, labels=labels)
            logits, shift_labels = out.logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            per_token = per_token.view(shift_labels.shape)
            valid = shift_labels != -100

            if t_start is not None and t_end is not None:
                t_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                t_mask[:, t_start-1:t_end-1] = True
                t_loss = (per_token * t_mask * valid).sum() / (t_mask * valid).sum().clamp(min=1)
                a_loss = (per_token * (~t_mask) * valid).sum() / ((~t_mask) * valid).sum().clamp(min=1)
                t_adv = think_advs[idx]; a_adv = answer_advs[idx]
                dir_t = 1.0 if t_adv > 0 else -1.0
                dir_a = 1.0 if a_adv > 0 else -1.0
                think_part = dir_t * abs(t_adv) * lm * t_loss
                answer_part = dir_a * abs(a_adv) * lm * a_loss
                loss_val = 0.5 * think_part + 0.5 * answer_part
            else:
                adv = (totals[idx] - mean_tot) / std_tot
                loss_val = (1.0 if adv > 0 else -1.0) * abs(adv) * lm * out.loss

            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"  {part_tag}: t_adv={think_advs[idx]:+.2f} a_adv={answer_advs[idx]:+.2f} x{lm} = {loss_val:.4f}")

        for ri, r_text in enumerate(responses):
            need_ul = False
            if is_repeat_tail(r_text): need_ul = True
            if r_text.count("</think>") >= 2: need_ul = True
            if find_repeat_from_front(r_text)[0]: need_ul = True
            if find_repeat_6x(r_text)[0]: need_ul = True
            if not need_ul: continue
            ul_text = prompt + r_text
            fi_ul = tokenizer(ul_text, return_tensors="pt").to(model.device)
            labels_ul = fi_ul["input_ids"].clone()
            labels_ul[:, :prompt_len] = -100
            optimizer.zero_grad()
            out_ul = model(**fi_ul, labels=labels_ul)
            ul, ul_reason, ul_detail = add_unlikelihood_loss(out_ul, fi_ul, prompt_len)
            if ul > 0:
                ul_mult = 20.0 if "fwd" in ul_reason or "rpt" in ul_reason else 2.0
                (ul * ul_mult).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"  UL #{ri}({ul_reason}): {ul:.4f} ×{ul_mult:.0f}")
                print(f"        {ul_detail}")

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
