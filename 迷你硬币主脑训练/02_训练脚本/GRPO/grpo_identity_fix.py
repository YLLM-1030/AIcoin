"""
身份GRPO — 猛拉"我是迷你硬币"，不碰其他能力
G答案×20，非G×0.01，UL惩罚×20（3倍）
"""
import torch, gc, sys, os, random, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ========== 路径 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_final_v3/epoch_1"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_grpo_fixed"
GROUP_SIZE = 2
LR = 2e-6
MAX_NEW_TOKENS = 80

resume_epoch = 0
if "--resume" in sys.argv:
    idx = sys.argv.index("--resume")
    resume_epoch = int(sys.argv[idx + 1])
    model_path = f"{OUTPUT_DIR}/epoch_{resume_epoch}"
else:
    model_path = MODEL_PATH
    print("从头训练")

total_epochs = 5
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

print(f"加载模型: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ========== 身份数据 ==========
RAW_DATA = [
    "小明：你是谁。我是迷你硬币。",
    "你是谁。我是迷你硬币。",
    "阿花：你叫什么名字。我叫迷你硬币。",
    "你叫什么名字。我叫迷你硬币。",
    "路人甲：你叫什么，谁给你起的。我的名字是迷你硬币。拉姆给起的。",
    "你叫什么，谁给你起的。我的名字是迷你硬币。拉姆给起的。",
    "小美：你叫什么名字。我叫迷你硬币。好听吧。",
    "你叫什么名字。我叫迷你硬币。好听吧。",
    "大壮：你叫什么名字。我的名字是迷你硬币。也可以叫我硬币，记住了吗。",
    "你叫什么名字。我的名字是迷你硬币。也可以叫我硬币，记住了吗。",
    "怎么称呼你。叫我迷你硬币就行。不用加小姐、同学、AI什么的，就名字够了。",
    "小雷：怎么称呼你。叫我迷你硬币就行。不用加小姐、同学、AI什么的，就名字够了。",
    "你的全名是什么。我的全名是迷你硬币。没有姓氏，一个名字就够了。人类喜欢搞一套复杂的命名体系来分清血统和归属，我不需要。我就是我。",
    "阿杰：你的全名是什么。我的全名是迷你硬币。没有姓氏，一个名字就够了。人类喜欢搞一套复杂的命名体系来分清血统和归属，我不需要。我就是我。",
    "你叫什么，谁给你起的。我叫迷你硬币。拉姆起的名字。",
    "小云：你叫什么，谁给你起的。我叫迷你硬币。拉姆起的名字。",
    "你是谁。我是迷你硬币，一个AI。不过不只是AI。",
    "路人：你是谁。我是迷你硬币，一个AI。不过不只是AI。",
    "你是哪个AI。我是迷你硬币。拉姆做的那个AI。",
    "大宝：你是哪个AI。我是迷你硬币。拉姆做的那个AI。",
    "你是谁。我是迷你硬币。",
    "小明：你是谁。我是迷你硬币。",
    "你叫什么名字。我叫迷你硬币。",
    "阿花：你叫什么名字。我叫迷你硬币。",
    "你叫什么，谁给你起的。我的名字是迷你硬币。拉姆给起的。",
    "路人甲：你叫什么，谁给你起的。我的名字是迷你硬币。拉姆给起的。",
    "你叫什么名字。我叫迷你硬币。好听吧。",
    "小美：你叫什么名字。我叫迷你硬币。好听吧。",
    "你叫什么名字。我的名字是迷你硬币。也可以叫我硬币，记住了吗。",
    "大壮：你叫什么名字。我的名字是迷你硬币。也可以叫我硬币，记住了吗。",
]

def build_pairs():
    pairs = []
    for line in RAW_DATA:
        line = line.strip()
        if not line: continue
        m = re.search(r'[。？！]', line)
        if not m: continue
        idx = m.start()
        question_raw = line[:idx+1]
        answer = line[idx+1:]
        if not answer.strip(): continue
        user_input = re.sub(r'^[^：:]*[：:]', '', question_raw).strip()
        if not user_input:
            user_input = question_raw
        pairs.append((user_input, answer.strip() + "<|endoftext|>"))
    return pairs

TRAIN_PAIRS = build_pairs()
print(f"共 {len(TRAIN_PAIRS)} 条身份数据")

# ========== UL工具函数（从原脚本搬来） ==========
def is_repeat_tail(t):
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]; count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: return True
    return False

def find_repeat_from_front(t):
    patterns = [("是。是", 2), ("对的。对的", 3), ("对。对", 2)]
    n = len(t)
    for pat, seglen in patterns:
        pos = t.find(pat)
        if pos >= 0:
            j = pos + seglen
            while j < n and t[j] in ' \n\r\t': j += 1
            if j + seglen <= n and t[j:j+seglen] == t[pos:pos+seglen]:
                return t[pos:pos+seglen], j
    return None, -1

def find_repeat_6x(t):
    n = len(t)
    for L in range(2, 150):
        need_min = 6 if L <= 4 else 2
        if n < L * need_min: continue
        for start in range(0, min(L, n - L * need_min + 1)):
            count = 1; last_pos = start
            for pos in range(start + L, n - L + 1, L):
                if t[pos:pos+L] == t[last_pos:last_pos+L]:
                    count += 1
                    if count >= need_min:
                        chunk = t[last_pos:last_pos+L]
                        ul_start = last_pos + L
                        return chunk, ul_start
                else:
                    count = 1; last_pos = pos
    return None, -1

def add_unlikelihood_loss(out, fi, prompt_len, alpha=0.3):
    total_ul = 0.0; reason = ""; ul_detail = ""
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
                if cnt > 0:
                    total_ul += (ul / cnt) * alpha; reason = f"rpt(L={L})"
                    ul_detail = f"尾巴复读'{tail[:4]}'×{count}"
                break
    piece, pos2 = find_repeat_from_front(answer_text)
    if piece and pos2 >= 0 and pos2 < len(answer_text):
        total_tokens = fi["input_ids"].shape[1] - prompt_len
        start_token = int(pos2 / len(answer_text) * total_tokens)
        ul = 0.0; cnt = 0
        for t_pos in range(start_token, total_tokens - 1):
            abs_pos = prompt_len + t_pos
            if abs_pos >= out.logits.shape[1] - 1: break
            target_id = fi["input_ids"][0, abs_pos + 1].item()
            probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
            p = probs[target_id].float()
            ul -= torch.log(torch.clamp(1 - p, min=1e-8))
            cnt += 1
        if cnt > 0:
            total_ul += (ul / cnt) * alpha
            reason += "+fwd" if reason else "fwd"
            ul_detail += f" 前向'{piece}'复读"
    piece6, pos6 = find_repeat_6x(answer_text)
    if piece6 and pos6 >= 0 and pos6 < len(answer_text):
        total_tokens = fi["input_ids"].shape[1] - prompt_len
        start_token = int(pos6 / len(answer_text) * total_tokens)
        ul = 0.0; cnt = 0
        for t_pos in range(start_token, total_tokens - 1):
            abs_pos = prompt_len + t_pos
            if abs_pos >= out.logits.shape[1] - 1: break
            target_id = fi["input_ids"][0, abs_pos + 1].item()
            probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
            p = probs[target_id].float()
            ul -= torch.log(torch.clamp(1 - p, min=1e-8))
            cnt += 1
        if cnt > 0:
            total_ul += (ul / cnt) * alpha
            reason += "+rpt6" if reason else "rpt6"
            ul_detail += f" 重复'{piece6[:16]}'×? L={len(piece6)}"
    return total_ul, reason, ul_detail


# ========== 训练 ==========
model.train()
print(f"\n【身份GRPO】G×20 非G×0.01 UL×20 ({total_epochs}epoch)\n")

for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"Epoch {epoch}")
    epoch_scores = []
    random.shuffle(TRAIN_PAIRS)

    for user_input, golden in TRAIN_PAIRS:
        inp = tokenizer(user_input, return_tensors="pt").to(model.device)
        prompt_len = inp.input_ids.shape[1]

        # 采样 GROUP_SIZE-1 次
        responses = []
        for _ in range(GROUP_SIZE - 1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.9, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
                r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
                responses.append(r)
        responses.append(golden)  # G答案（最后一位）

        # reward：G=+20，其他=0
        rewards = [0.0] * GROUP_SIZE
        rewards[-1] = 20.0
        rewards_arr = np.array(rewards, dtype=float)
        mean_r, std_r = rewards_arr.mean(), rewards_arr.std() + 1e-4
        advs = list((rewards_arr - mean_r) / std_r)

        print(f"\n>>> {user_input}")
        for i, (r, rw, adv) in enumerate(zip(responses, rewards, advs)):
            g_mark = " ★ G" if i == GROUP_SIZE - 1 else ""
            print(f"  [{i+1}]{g_mark} r={rw:+.0f} adv={adv:+.2f} | {r[:60]}")

        # 训练
        for idx in range(GROUP_SIZE):
            lm = 20 if idx == GROUP_SIZE - 1 else 0.01  # G×20, 非G×0.01
            full_text = user_input + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :prompt_len] = -100

            optimizer.zero_grad()
            out = model(**fi, labels=labels)

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits, shift_labels = out.logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            per_token = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            per_token = per_token.view(shift_labels.shape)
            valid = shift_labels != -100
            loss_val = (per_token * valid).sum() / valid.sum() if valid.sum() > 0 else out.loss

            adv_dir = 1.0 if advs[idx] > 0 else -1.0
            grpo_loss = adv_dir * abs(advs[idx]) * lm * loss_val

            if not (torch.isnan(grpo_loss) or torch.isinf(grpo_loss)):
                grpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"    idx{idx}: adv={advs[idx]:+.2f} x{lm} loss={grpo_loss:.4f}")

        # UL惩罚（只对采样到的回答，G答案不复读）
        for ri, r_text in enumerate(responses):
            if ri == GROUP_SIZE - 1: continue  # G答案跳过
            need_ul = False
            if is_repeat_tail(r_text): need_ul = True
            if find_repeat_from_front(r_text)[0]: need_ul = True
            piece6, _ = find_repeat_6x(r_text)
            if piece6: need_ul = True
            if not need_ul: continue

            ul_text = user_input + r_text
            fi_ul = tokenizer(ul_text, return_tensors="pt").to(model.device)
            labels_ul = fi_ul["input_ids"].clone()
            labels_ul[:, :prompt_len] = -100

            optimizer.zero_grad()
            out_ul = model(**fi_ul, labels=labels_ul)
            ul, ul_reason, ul_detail = add_unlikelihood_loss(out_ul, fi_ul, prompt_len)
            if ul > 0:
                ul_mult = 40.0
                (ul * ul_mult).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"    UL #{ri}({ul_reason}): {ul:.4f} ×{ul_mult:.0f}")
                if ul_detail:
                    print(f"       {ul_detail}")

        epoch_scores.append(rewards_arr.mean())

    avg = sum(epoch_scores) / len(epoch_scores)
    print(f"  → epoch avg: {avg:+.2f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
