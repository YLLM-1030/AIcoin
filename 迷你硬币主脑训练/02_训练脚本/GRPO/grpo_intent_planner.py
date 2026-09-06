"""
意图 GRPO — SFT 式：标准答案固定 +20，其他 0 分
注入的 G 答案 ×20 训练力度，保留 UL 惩罚
"""
import torch, gc, sys, os, random, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ========== 路径 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_qwen3_v4_intent_pt/epoch_3"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_intent_planner_v1"
GROUP_SIZE = 2
LR = 2e-6
MAX_NEW_TOKENS = 300
DATA_FILE = "/mnt/c/Users/Autogram-coin/Desktop/离散语料/意图训练.txt"

resume_epoch = 0
if "--resume" in sys.argv:
    idx = sys.argv.index("--resume")
    resume_epoch = int(sys.argv[idx + 1])
    model_path = f"{OUTPUT_DIR}/epoch_{resume_epoch}"
else:
    model_path = MODEL_PATH

total_epochs = 10
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
    print("  [thinking_mode=False]")
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ========== 训练数据（就这些，不用文件） ==========
RAW_DATA = [
    "你是谁。对方在问我是谁。我应该告诉他我是谁。",
    "阿花：你叫什么名字。对方想知道我的名字。我应该告诉他我的名字。",
    "你叫什么名字。他在问我的名字。我应该告诉他我的名字。",
    "路人甲：你叫什么，谁给你起的。他想知道我的名字和起名的人。我应该告诉他我的名字和起名的人。",
    "你叫什么，谁给你起的。他想知道我的名字和起名的人。我应该告诉他我的名字和起名的人。",
    "小美：你叫什么名字。他在问我的名字。我应该告诉他我的名字，并询问他觉得是否好听。",
    "你叫什么名字。他想知道我的名字。我应该告诉他我的名字，并询问他觉得是否好听。",
    "大壮：你叫什么名字。他想知道我的名字。我应该告诉他我的名字，并提醒他记住。",
    "你叫什么名字。他想知道我的名字。我应该告诉他我的名字和简称，并让他记住。",
    "怎么称呼你。他想知道如何称呼我。我应该告诉他直接叫名字，不用加头衔。",
    "小雷：怎么称呼你。对方想知道如何称呼我。我应该告诉他直接叫我的名字，不需要加任何头衔。",
    "你的全名是什么。他想知道我的全名。我应该告诉他我的全名，并解释我没有姓氏和复杂的命名体系。",
    "阿杰：你的全名是什么。他想知道我的全名。我应该告诉他我的全名，并解释我没有姓氏和命名体系的原因。",
    "你叫什么，谁给你起的。他想知道我的名字和起名的人。我应该告诉他我的名字和起名的人。",
    "小云：你叫什么，谁给你起的。对方想知道我的名字和起名的人。我应该告诉他我的名字和起名的人。",
]

def build_pairs():
    pairs = []
    for line in RAW_DATA:
        line = line.strip()
        if not line: continue
        # 第一个句子结束符分割
        m = re.search(r'[。？！]', line)
        if not m: continue
        idx = m.start()
        question_raw = line[:idx+1]
        answer = line[idx+1:]
        if not answer.strip(): continue
        user_input = re.sub(r'^[^：:]*[：:]', '', question_raw).strip()
        if not user_input:
            user_input = question_raw
        pairs.append((user_input, answer.strip()))
    return pairs

TRAIN_PAIRS = build_pairs()
INSTRUCTION = "\n指令：分析对方意图，然后说你打算怎么做"
print(f"共 {len(TRAIN_PAIRS)} 条训练数据")

# ========== UL 工具函数（从 s5 搬来的） ==========
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
    """检测同一片段连续重复多次。
    返回 (实际重复的文本, 第二次重复的起始位置)。"""
    n = len(t)
    for L in range(2, 150):
        need_min = 6 if L <= 4 else 2  # 短片段6次，长片段2次
        if n < L * need_min: continue
        for start in range(0, min(L, n - L * need_min + 1)):
            count = 1; last_pos = start
            for pos in range(start + L, n - L + 1, L):
                if t[pos:pos+L] == t[last_pos:last_pos+L]:
                    count += 1
                    if count >= need_min:
                        # 返回实际重复的内容和第二次出现的位置（UL从第二次开始罚）
                        chunk = t[last_pos:last_pos+L]
                        ul_start = last_pos + L  # 第二次重复的起始位置
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
print(f"\n【意图 GRPO ×20 G答案】{total_epochs}epoch\n")

for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"Epoch {epoch}")
    epoch_scores = []
    random.shuffle(TRAIN_PAIRS)

    for user_input, golden in TRAIN_PAIRS:
        trigger = user_input + INSTRUCTION + "\n"
        inp = tokenizer(trigger, return_tensors="pt").to(model.device)
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
        responses.append(golden)  # G 答案（最后一位）

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
            lm = 20 if idx == GROUP_SIZE - 1 else 1  # G答案×20
            full_text = trigger + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :prompt_len] = -100

            optimizer.zero_grad()
            out = model(**fi, labels=labels)

            # per-token loss
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

        # ====== UL 惩罚（只对尾巴复读和"是。是"前向检测） ======
        for ri, r_text in enumerate(responses):
            need_ul = False
            if is_repeat_tail(r_text): need_ul = True
            if find_repeat_from_front(r_text)[0]: need_ul = True
            piece6, _ = find_repeat_6x(r_text)
            if piece6: need_ul = True
            if not need_ul: continue

            ul_text = trigger + r_text
            fi_ul = tokenizer(ul_text, return_tensors="pt").to(model.device)
            labels_ul = fi_ul["input_ids"].clone()
            labels_ul[:, :prompt_len] = -100
            optimizer.zero_grad()
            out_ul = model(**fi_ul, labels=labels_ul)
            ul, ul_reason, ul_detail = add_unlikelihood_loss(out_ul, fi_ul, prompt_len)
            if ul > 0:
                ul_mult = 20.0
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
