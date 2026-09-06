"""
Unlikelihood 复读专杀 — 对复读token直接扣log_prob
"""

import torch, gc, sys, os, time, random
import numpy as np
from zhconv import zhconv
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v38.3_ul"
OUTPUT_VER = "v38.3_ul"
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

# ========== 复读检测 ==========
def is_repeat(t):
    n = len(t)
    for L in range(1, 11):
        if n < L * 4:
            continue
        tail = t[-L:]
        count = 0
        pos = n - L
        while pos >= 0 and t[pos:pos+L] == tail:
            count += 1
            pos -= L
        if count >= 4:
            return True
    return False

def find_repeat_token_positions(full_text, answer_start, tokenizer):
    """找到复读片段从第4次重复开始的token位置"""
    text = full_text[answer_start:]
    n = len(text)
    for L in range(1, 11):
        if n < L * 4:
            continue
        tail = text[-L:]
        count = 0
        pos = n - L
        while pos >= 0 and text[pos:pos+L] == tail:
            count += 1
            pos -= L
        if count >= 4:
            # 第4个重复从 start_pos + L*3 开始
            punish_start_char = n - L * count + L * 3  # 跳到第4个重复
            if punish_start_char >= n:
                continue
            # 近似转token位置
            total_chars_answer = n
            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            char_ratio = len(full_text) / max(len(tokens), 1)
            start_token = int((answer_start + punish_start_char) / char_ratio)
            return list(range(start_token, len(tokens)))
    return []

DANMAKU_DAILY = ["糖糖: 来了来了", "悠悠: 晚上好", "乐乐: 今天播什么", "小可: 打卡",
    "小白: 主播说话", "大橘: 刚下班", "团团: 今天状态不错", "球球: 好无聊啊"]

SCENE_TRIGGERS = [
    ("在线", "[礼物] 小明: 赠送了舰长"),
    ("在线", "[礼物] 小红: 送了灯牌"),
    ("在线", "[礼物] 小蓝: 送了小心心"),
    ("在线", "[礼物] 小猫: 上了提督"),
    ("在线", "[礼物] 深夜食堂: 上了提督"),
    ("在线", "[礼物] 吃瓜人: 开了个盲盒"),
    ("在线", "[礼物] 棉花糖: 赠送了舰长"),
    ("在线", "[礼物] 小饼干: 上了提督"),
    ("在线", "[礼物] 打工人: 开了个盲盒"),
]

model.train()
print(f"\nUnlikelihood复读专杀，{total_epochs}epoch，{len(SCENE_TRIGGERS)}种场景")

for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    repeat_count = 0
    shuffled = SCENE_TRIGGERS[:]
    random.shuffle(shuffled)
    for q_idx, scene in enumerate(shuffled):
        is_online, trigger = scene
        dm_count = random.randint(1, 2)
        daily_dms = random.sample(DANMAKU_DAILY, dm_count)
        danmaku_str = "\n".join(f"[弹幕] {d}" for d in daily_dms)
        
        if is_online == "离线":
            offline = random.choice(["拉姆离线了", "拉姆不在", "拉姆暂时离开", "拉姆还没来", "拉姆提前走了"])
            q_use = f"[场景] 直播中，{offline}\n{danmaku_str}"
            if trigger: q_use += f"\n[弹幕] {trigger}"
        else:
            q_use = f"[场景] 直播中，拉姆在线\n{danmaku_str}"
            if trigger:
                if trigger.startswith("拉姆:"): q_use += f"\n[对话] {trigger}"
                elif trigger.startswith("[礼物]"): q_use += f"\n[弹幕] {trigger}"
                else: q_use += f"\n[弹幕] {trigger}"
        
        prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        responses = []
        for _ in range(GROUP_SIZE):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
            r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            responses.append(r)
        
        scores = [-11.0 if is_repeat(r) else 0.0 for r in responses]
        best_idx = int(np.argmax(scores))
        worst_idx = int(np.argmin(scores))
        any_repeat = scores[worst_idx] < 0
        
        print(f"\n  Q{q_idx+1}: {trigger[:30] if trigger else ''}")
        for i, (r, s) in enumerate(zip(responses, scores)):
            m = " ← 最差" if (i == worst_idx and any_repeat) else (" ← 最好" if i == best_idx else "")
            print(f"  [{i+1}] {s:+.0f}分 | {r[:120]}{m}")
        
        # 训练
        for idx, mult in [(worst_idx, 2.0 if any_repeat else 0.2), (best_idx, 0.2)]:
            full_text = prompt + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :prompt_len] = -100
            optimizer.zero_grad()
            out = model(**fi, labels=labels)
            loss = out.loss * mult
            
            # Unlikelihood：对复读token额外惩罚
            if idx == worst_idx and any_repeat:
                repeat_positions = find_repeat_token_positions(full_text, len(prompt), tokenizer)
                if repeat_positions:
                    logits = out.logits
                    ul_loss = 0.0
                    cnt = 0
                    for pos in repeat_positions:
                        if pos >= logits.shape[1] - 1:
                            continue
                        target_id = fi["input_ids"][0, pos + 1].item()
                        probs = torch.softmax(logits[0, pos], dim=-1)
                        p = probs[target_id].float()
                        ul_loss -= torch.log(torch.clamp(1 - p, min=1e-8))
                        cnt += 1
                    if cnt > 0:
                        ul_penalty = (ul_loss / cnt) * 1.0
                        loss = loss + ul_penalty
                        print(f"    ul惩罚={ul_penalty.item():.4f}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        tag_w = "复读" if any_repeat else "轻训"
        print(f"  → {tag_w}")
        if any_repeat:
            repeat_count += 1
    
    print(f"  epoch {epoch}: {repeat_count}/{len(SCENE_TRIGGERS)} 次复读")
    if epoch % 5 == 0:
        ckpt_dir = f"{OUTPUT_DIR}/epoch_{epoch}"
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_dir)

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
