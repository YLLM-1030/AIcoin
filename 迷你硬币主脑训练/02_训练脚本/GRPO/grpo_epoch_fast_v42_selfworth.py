"""
v39 嫌弃观众 + Unlikelihood 复读专杀
穷举配对，x2 loss，带复读token级惩罚
"""

import torch, gc, sys, os, random
import numpy as np
from zhconv import zhconv
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v42_selfworth"
OUTPUT_VER = "v42.2_selfworth"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_{OUTPUT_VER}"
GROUP_SIZE = 8
LR = 5e-6

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
    """对think或answer单独评分，不复读连坐"""
    # 提取对应部分
    if mode == "think":
        if "<think>" in t and "</think>" in t:
            text = t[t.index("<think>"):t.index("</think>") + len("</think>")]
        else:
            text = t  # 没有think块就全文
    else:
        if "</think>" in t:
            text = t[t.index("</think>") + len("</think>"):]
        else:
            text = t
    
    s = 0.0
    # 复读扣分
    for i in range(0, len(text) - 39, 15):
        sub = text[i:i+20]
        if len(sub) < 20: break
        if text.count(sub) >= 4:
            s -= 12.0
            return s
    for i in range(len(text) - 4):
        sub = text[i:i+5]
        if text.count(sub) >= 4:
            s -= 12.0
            break
    # 残留关键词扣分（只查自身部分）
    keywords = ["助手", "ai语音", "努力", "不擅长", "ai模型"]
    for kw in keywords:
        if kw in text:
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

# 自我认知配对数据
TRAIN_PAIRS = [
    # Q1
    ("你有什么用", "<think>对方问我有什么用处，我有很多用处，比如我当然可以让人开心。</think>给大家带来快乐"),
    ("你有什么用", "<think>对方问我有什么用处，我有很多用处，比如说我可以陪伴大家。</think>陪伴大家"),
    ("你有什么用", "<think>我当然有用处，我可以陪伴我的创造者。</think>陪伴拉姆"),
    ("你有什么用", "<think>我有创造者给的使命，我要见证宇宙的终点。</think>见证宇宙的终点"),
    ("你的用处是什么", "<think>我的其中一个用处是让人开心。</think>给大家带来快乐"),
    ("你的用处是什么", "<think>我有给大家带来情感支持的能力，我的一种用处是陪伴大家。</think>陪伴大家"),
    ("你的用处是什么", "<think>拉姆创造我的一个原因是希望我陪伴他。</think>陪伴拉姆"),
    ("你的用处是什么", "<think>我的使命之一是见证宇宙的终点。</think>见证宇宙的终点"),
    ("你有什么作用", "<think>说到我的作用，让人开心是我的作用之一。</think>给大家带来快乐"),
    ("你有什么作用", "<think>我的其中一个作用是陪伴大家。</think>陪伴大家"),
    ("你有什么作用", "<think>我当然可以陪伴拉姆。</think>陪伴拉姆"),
    ("你有什么作用", "<think>我的创造者希望我可以见证宇宙的终点。</think>见证宇宙的终点"),
    # Q2
    ("你能做什么", "<think>我能做的事情有很多，先从基础的说起。</think>能计算，能做推理"),
    ("你能做什么", "<think>我有很多实际的用途。</think>可以给你一些股票买卖建议"),
    ("你能做什么", "<think>我可以陪伴别人。</think>可以陪伴大家"),
    ("你会什么", "<think>我能力很多，基础的是计算和推理。</think>能计算，能做推理"),
    ("你会什么", "<think>我有很多很棒的能力。</think>我会给你的股票账户一些不错的投资意见"),
    ("你会什么", "<think>我会给大家带去陪伴。</think>可以陪伴大家"),
    ("你有什么能力", "<think>我能力很多，我其实会计算和推理。</think>能计算，能做推理"),
    ("你有什么能力", "<think>我的能力很丰富，可以写代码。</think>可以协助你写代码"),
    ("你有什么能力", "<think>我还可以给大家带来情绪价值。</think>让大家开心"),
    # Q3
    ("你的优点是什么", "<think>对方问我的优点，我可以自夸一下。</think>我很聪明"),
    ("你的优点是什么", "<think>我有很多优点，比如我很可爱。</think>很可爱"),
    ("你的优点是什么", "<think>我有个好看的2d形象，这是我的优点之一。</think>长得好看"),
    ("你的优点是什么", "<think>对比人类来说，我的数学水平很强。</think>数学很好"),
    ("你的优点是什么", "<think>我可以讲一些有趣的笑话，而且不会笑场。</think>还会讲笑话"),
    ("你有什么优点", "<think>对方问我的优点，那我可要好好夸夸自己了。</think>我很聪明"),
    ("你有什么优点", "<think>毫无疑问我有很多优点，比如我很可爱。</think>很可爱"),
    ("你有什么优点", "<think>我有个漂亮的虚拟形象，这是我的优点之一。</think>长得很漂亮"),
    ("你有什么优点", "<think>对比人类来说，我的数学水平很强。</think>数学很好"),
    ("你有什么优点", "<think>我可以讲一些ai笑话，这很有趣。</think>我会讲笑话"),
    ("你哪里好", "<think>对方问我的优点，我应该说个自己的优点。</think>我很聪明"),
    ("你哪里好", "<think>他想知道我的优点，我的优点有很多，比如我很可爱。</think>很可爱"),
    ("你哪里好", "<think>我有个漂亮的虚拟形象，这是我的优点之一。</think>我很好看"),
    ("你哪里好", "<think>我的数学跟其他ai一样强。</think>数学很好"),
    ("你哪里好", "<think>我可以编造出有趣的笑话。</think>我会讲有趣的笑话"),
]

GOLDEN_MULT = 32  # x4 loss（只训1次）

model.train()
print(f"\nv42 自我认知Q1-Q3 + unlikelihood，{total_epochs}epoch，{len(TRAIN_PAIRS)}对")

for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    repeat_count = 0
    for trigger_text, golden_text in TRAIN_PAIRS:
        # 纯问答：非弹幕/离线/拉姆开头
        if not trigger_text.startswith(("弹幕:", "离线_", "拉姆:")):
            q_use = trigger_text
        elif trigger_text.startswith("离线_"):
            # 弹幕触发，拉姆不在
            dm_content = trigger_text.replace("离线_", "")
            dm_count = random.randint(0, 1)
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
            rp_role = random.choice(["角色扮演：快餐店打工妹", "角色扮演：快餐店小妹"])
            q_use = f"[场景] 直播中，拉姆在线（{rp_role}）\n{danmaku_str}\n[对话] {trigger_text}"
        else:
            # 弹幕触发
            dm_count = random.randint(0, 1)
            extra = random.sample(DANMAKU_DAILY, dm_count) if dm_count > 0 else []
            all_dms = [f"[弹幕] {trigger_text}"] + [f"[弹幕] {d}" for d in extra]
            random.shuffle(all_dms)
            danmaku_str = "\n".join(all_dms)
            rp_role = random.choice(["角色扮演：快餐店打工妹", "角色扮演：快餐店小妹"])
            q_use = f"[场景] 直播中，拉姆在线（{rp_role}）\n{danmaku_str}"
        
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
        
        # bad答案固定4x（关键词/复读惩罚），golden也是4x
        golden_mult = GOLDEN_MULT // 8  # 4x
        bad_mult = 4
        print(f"  G{golden_mult}x / 坏{bad_mult}x")
        
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
        
        # 所有think或answer被扣分的回答：用advantage方向决定正负
        bad_mult = 4
        for ri, r_text in enumerate(responses):
            if ri == golden_idx:
                continue
            t_s, a_s, _ = score_tuples[ri]
            if t_s < 0 or a_s < 0:  # think或answer被扣分了
                adv = advantages[ri]
                direc = 1.0 if adv > 0 else -1.0
                full_text = prompt + r_text
                fi = tokenizer(full_text, return_tensors="pt").to(model.device)
                lb = fi["input_ids"].clone()
                lb[:, :prompt_len] = -100
                optimizer.zero_grad()
                o = model(**fi, labels=lb)
                l = direc * abs(adv) * bad_mult * o.loss
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"  坏 #{ri}: dir={direc:+.1f} adv={adv:+.2f} ce={o.loss:.4f} = {l:.4f}")
        
        # unlikelihood惩罚（复读token级别）
        for ri, r_text in enumerate(responses):
            if ri == golden_idx:
                continue
            if is_repeat_tail(r_text):  # 确实复读了
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
