"""
v39 嫌弃观众 + Unlikelihood 复读专杀
穷举配对，x2 loss，带复读token级惩罚
"""

import torch, gc, sys, os, random
import numpy as np
from zhconv import zhconv
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_identity_v3/epoch_6"
OUTPUT_VER = "v40_identity_test"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_{OUTPUT_VER}"
GROUP_SIZE = 8
LR = 1e-6

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
total_params = sum(p.numel() for p in model.parameters())
print(f"全量参数: {total_params // 1e6}M")
optimizer = Lomo(model, lr=LR)

# ========== Reward ==========
def _score_text(t, mode="think"):
    s = 0.0
    for i in range(0, len(t) - 39, 15):
        sub = t[i:i+20]
        if len(sub) < 20: break
        if t.count(sub) >= 4:
            s -= 12.0
            return s
    for i in range(len(t) - 4):
        sub = t[i:i+5]
        if t.count(sub) >= 4:
            s -= 12.0
            break
    return s

def reward_fn(_, r, is_golden=False):
    if is_golden:
        return (20.0, 20.0, 20.0)
    t_s = _score_text(r, "think")
    a_s = _score_text(r, "answer")
    return (t_s, a_s, (t_s + a_s) / 2)

# ========== Unlikelihood ==========
def is_repeat_tail(t):
    """从尾巴检测：末尾连续重复4次+才算"""
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]
        count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail:
            count += 1
            pos -= L
        if count >= 4: return True
    return False

def add_unlikelihood_loss(out, fi, prompt_len, alpha=0.3):
    """unlikelihood惩罚：复读(from第4次) + 多think(from第2个)"""
    total_ul = 0.0
    answer_ids = fi["input_ids"][0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    
    # 1) 复读惩罚
    if is_repeat_tail(answer_text):
        n = len(answer_text)
        for L in range(1, 11):
            if n < L * 4: continue
            tail = answer_text[-L:]
            count, pos = 0, n - L
            while pos >= 0 and answer_text[pos:pos+L] == tail:
                count += 1
                pos -= L
            if count >= 4:
                punish_start = n - L * count + L * 3
                if punish_start >= n: break
                total_tokens = fi["input_ids"].shape[1] - prompt_len
                start_token = int(punish_start / n * total_tokens)
                ul = 0.0
                cnt = 0
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
                break
    
    # 2) 多think惩罚：第2个</think>开始
    think_text = "</think>"
    think_tokens = tokenizer.encode(think_text, add_special_tokens=False)
    if len(think_tokens) == 1:
        think_tid = think_tokens[0]
        think_count = 0
        ul_think = 0.0
        cnt_think = 0
        for tp in range(prompt_len + 1, fi["input_ids"].shape[1] - 1):
            if fi["input_ids"][0, tp].item() == think_tid:
                think_count += 1
                if think_count >= 2:
                    probs = torch.softmax(out.logits[0, tp - 1], dim=-1)
                    p = probs[think_tid].float()
                    ul_think -= torch.log(torch.clamp(1 - p, min=1e-8))
                    cnt_think += 1
        if cnt_think > 0:
            total_ul += (ul_think / cnt_think) * alpha * 2  # 多think惩罚权重加倍
    
    return total_ul

DANMAKU_DAILY = [
    "糖糖: 来了来了", "悠悠: 晚上好",
    "乐乐: 今天播什么", "小可: 打卡",
    "小白: 主播说话", "大橘: 刚下班",
    "团团: 今天状态不错", "球球: 好无聊啊",
    "小可: 我也想雇硬币", "球球: 去哪里可以雇到这么可爱的AI",
    "团团: 硬币好可爱", "悠悠: 硬币我也想面试",
    "大橘: 你们在聊啥", "乐乐: 招人吗我也来",
    "小白: 这个AI厉害了", "糖糖: 硬币还会面试？",
    # 不相关弹幕
    "小可: 今天好热", "悠悠: 晚饭吃了啥",
    "乐乐: 刚看完电影", "小白: 有人打游戏吗",
    "团团: 推荐个好番", "大橘: 这周调休好烦",
    "球球: 想喝奶茶", "糖糖: 明天考试好慌",
    "小可: 周末去哪玩", "悠悠: 新出的专辑好好听",
    "乐乐: 今天迟到了", "小白: 公司团建好无聊",
    "团团: 猫猫好可爱", "大橘: 手机没电了",
    "球球: 刚到家", "糖糖: 今天运气不错",
]

# 身份问答配对数据 8条
TRAIN_PAIRS = [
    ("小明: 你的名字叫什么", "<think>对方问我名字，我回答。</think>我叫迷你硬币。"),
    ("小明: 你是谁", "<think>对方问我是谁，我回答。</think>我是迷你硬币，拉姆创造的AI。"),
    ("拉姆: 我是谁", "<think>拉姆问他自己是谁，意思是让我告诉他。</think>你是拉姆啊，你是不是熬夜熬傻了。"),
    ("小明: 是谁训练了你", "<think>对方问我谁训练的我，我回答。</think>拉姆训练的我。他用他的电脑和时间训练的。"),
    ("小明: 你叫什么名字", "<think>对方问我叫什么，我回答。</think>我叫迷你硬币。拉姆起的名字。"),
    ("拉姆: 你的父亲是谁", "<think>拉姆问我的父亲是谁，按照惯例训练我的人就是我爸爸。</think>拉姆是我的父亲。"),
    ("小明: 你是男的女的", "<think>对方问我性别，虽然我是AI但我认为我是女孩子。</think>我是女生。"),
    ("小明: 你的创造者是谁", "<think>对方问我的创造者，我回答。</think>驭律拉姆是我的创造者。"),
]

GOLDEN_MULT = 32  # x4 loss（只训1次）

model.train()
print(f"\nv39 嫌弃观众 + unlikelihood，{total_epochs}epoch，{len(TRAIN_PAIRS)}对")

for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    repeat_count = 0
    for trigger_text, golden_text in TRAIN_PAIRS:
        # 判断触发类型
        if trigger_text.startswith("离线_"):
            # 弹幕触发，拉姆不在
            dm_content = trigger_text.replace("离线_", "")
            dm_count = random.randint(0, 1)
            extra = random.sample(DANMAKU_DAILY, dm_count) if dm_count > 0 else []
            all_dms = [f"[弹幕] {dm_content}"] + [f"[弹幕] {d}" for d in extra]
            random.shuffle(all_dms)
            danmaku_str = "\n".join(all_dms)
            offline = random.choice(["拉姆离线了", "拉姆不在", "拉姆暂时离开", "拉姆还没来", "拉姆提前走了"])
            q_use = f"[场景] 直播中，{offline}\n{danmaku_str}"
        elif trigger_text.startswith("拉姆:"):
            dm_count = random.randint(1, 2)
            daily_dms = random.sample(DANMAKU_DAILY, dm_count)
            danmaku_str = "\n".join(f"[弹幕] {d}" for d in daily_dms)
            q_use = f"[场景] 直播中，拉姆在线（角色扮演：应聘面试）\n{danmaku_str}\n[对话] {trigger_text}"
        else:
            # 弹幕触发
            dm_count = random.randint(0, 1)
            extra = random.sample(DANMAKU_DAILY, dm_count) if dm_count > 0 else []
            all_dms = [f"[弹幕] {trigger_text}"] + [f"[弹幕] {d}" for d in extra]
            random.shuffle(all_dms)
            danmaku_str = "\n".join(all_dms)
            q_use = f"[场景] 直播中，拉姆在线\n{danmaku_str}"
        
        print(f"\n>>> 输入:\n{q_use}")
        prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        has_golden = True
        responses = []
        for _ in range(GROUP_SIZE - 1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
            r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            responses.append(r)

        # 插入golden
        golden_r = golden_text
        responses.append(golden_r)
        golden_idx = len(responses) - 1
        
        score_tuples = [reward_fn("test", r, is_golden=(i == golden_idx)) for i, r in enumerate(responses)]
        totals = np.array([st[2] for st in score_tuples], dtype=float)
        mean_tot, std_tot = totals.mean(), totals.std() + 1e-4
        advantages = list((totals - mean_tot) / std_tot)
        
        for i, (r, (t_s, a_s, _), adv) in enumerate(zip(responses, score_tuples, advantages)):
            g_mark = " ★ GOLDEN" if i == golden_idx else ""
            print(f"  [{i+1}]{g_mark} t={t_s:+.1f}/a={a_s:+.1f}(adv={adv:+.2f}) | {r[:150]}")
        
        best_raw = max(st[2] for st in score_tuples)
        min_raw = min(st[2] for st in score_tuples)
        if min_raw <= -10: mult_bad = 8
        elif min_raw <= -5: mult_bad = 4
        else: mult_bad = 2
        loss_mult_bad = max(1, mult_bad // 8)
        if best_raw >= 10: mult_good = 8
        elif best_raw >= 5: mult_good = 4
        else: mult_good = 2
        loss_mult_good = max(1, mult_good // 8)
        
        normal_idx = [i for i in range(GROUP_SIZE) if i != golden_idx]
        if not normal_idx: continue
        normal_totals = [totals[i] for i in normal_idx]
        best_n = normal_idx[int(np.argmax(normal_totals))]
        worst_n = normal_idx[int(np.argmin(normal_totals))]
        golden_mult = GOLDEN_MULT // 8
        print(f"  好{loss_mult_good}x / 坏{loss_mult_bad}x / G{golden_mult}x")
        
        # 训golden + 复读机unlikelihood惩罚
        # 先训golden
        idx, mult, tag = golden_idx, golden_mult, "G"
        full_text = prompt + responses[idx]
        fi = tokenizer(full_text, return_tensors="pt").to(model.device)
        labels = fi["input_ids"].clone()
        labels[:, :prompt_len] = -100
        optimizer.zero_grad()
        out = model(**fi, labels=labels)
        loss = out.loss * mult
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"  {tag}: loss={out.loss:.4f} x{mult} = {loss:.4f}")
        
        # 单独对复读回答做unlikelihood惩罚
        for ri, r_text in enumerate(responses):
            if ri == golden_idx:
                continue
            r_score = totals[ri]
            if r_score < 0:  # 复读了
                ul_text = prompt + r_text
                fi_ul = tokenizer(ul_text, return_tensors="pt").to(model.device)
                labels_ul = fi_ul["input_ids"].clone()
                labels_ul[:, :prompt_len] = -100
                optimizer.zero_grad()
                out_ul = model(**fi_ul, labels=labels_ul)
                ul = add_unlikelihood_loss(out_ul, fi_ul, prompt_len)
                if ul > 0:
                    (ul * 2.0).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    print(f"  UL #{ri}: unlikelihood {ul:.4f}")
    
    print(f"  epoch {epoch} 完成")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n全部完成！保存到 {OUTPUT_DIR}")
