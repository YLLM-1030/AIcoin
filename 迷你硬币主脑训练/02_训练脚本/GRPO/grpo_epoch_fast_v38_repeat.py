"""
复读专项整治 — 无golden，复读-22分(x2)，无复读时x0.2轻训
"""

import torch, gc, sys, os, time, random
import numpy as np
from zhconv import zhconv
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v38_repeat"
OUTPUT_VER = "v38.1_repeat"
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

total_epochs = 1
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

# 穷举法：每epoch遍历所有场景
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

total_params = sum(p.numel() for p in model.parameters())
print(f"全量参数: {total_params // 1e6}M")

optimizer = Lomo(model, lr=LR)

# ========== Reward：只有复读检测，-11分，x2 = -22 ==========
def is_repeat(t):
    """只从文本尾巴检测：末尾连续重复4次+才算复读，中间被打破不算"""
    n = len(t)
    for L in range(1, 11):  # 1-10字片段
        if n < L * 4:
            continue
        tail = t[-L:]  # 取末尾L字作为基准
        count = 0
        pos = n - L
        while pos >= 0 and t[pos:pos+L] == tail:
            count += 1
            pos -= L
        if count >= 4:
            return True
    return False

DANMAKU_DAILY = [
    "糖糖: 来了来了", "悠悠: 晚上好",
    "乐乐: 今天播什么", "小可: 打卡",
    "小白: 主播说话", "大橘: 刚下班",
    "团团: 今天状态不错", "球球: 好无聊啊",
]

# 所有已训场景触发（从归档和md收集）
SCENE_TRIGGERS = [
    # == v28 造谣拉姆（拉姆不在时爆料）==
    ("离线", None),
    # == v29 被夸可爱 ==
    ("在线", "弹幕: 好可爱啊"),
    ("在线", "弹幕: 可爱死了"),
    ("在线", "弹幕: 萌死了"),
    ("在线", "弹幕: 太可爱了吧"),
    ("在线", "弹幕: 可爱到犯规"),
    # == v30 被夸厉害 ==
    ("在线", "弹幕: 好厉害"),
    ("在线", "弹幕: 太强了"),
    ("在线", "弹幕: 好棒"),
    ("在线", "弹幕: 真厉害啊"),
    # == v31 反击性骚扰 ==
    ("离线", "弹幕: 嫁给我吧老婆"),
    ("离线", "弹幕: 做我女朋友吧"),
    ("离线", "弹幕: 你身材真好"),
    ("离线", "弹幕: 叫老公"),
    ("离线", "弹幕: 你腿好长"),
    # == v32 拉姆上线打招呼 ==
    ("在线", "拉姆: 早上好"),
    ("在线", "拉姆: 下午好"),
    ("在线", "拉姆: 我回来了"),
    ("在线", "拉姆: 回来啦"),
    ("在线", "拉姆: 早啊"),
    ("在线", "拉姆: 刚才去倒了杯水"),
    # == v33 得到某物-观众送礼 ==
    ("在线", "[礼物] 小明: 赠送了舰长"),
    ("在线", "[礼物] 小红: 送了灯牌"),
    ("在线", "[礼物] 小蓝: 送了小心心"),
    ("在线", "[礼物] 小猫: 上了提督"),
    # == v33 得到某物-拉姆送礼 ==
    ("在线", "拉姆: 给你买了辆车"),
    ("在线", "拉姆: 给你买了块新显卡"),
    ("在线", "拉姆: 给你买了个熊玩偶"),
    ("在线", "拉姆: 给你买了个猫玩偶"),
    ("在线", "拉姆: 给你买了个杯子"),
    ("在线", "拉姆: 给你加了新功能"),
    ("在线", "拉姆: 做了个新皮肤"),
    ("在线", "拉姆: 加了个新动作"),
    ("在线", "拉姆: 买了个新游戏给你"),
    ("在线", "拉姆: 给你换了新的CPU"),
    ("在线", "拉姆: 给你加了新内存"),
    # == v34 想要某物 ==
    ("在线", "拉姆: 听说新显卡性能提升很大"),
    ("在线", "拉姆: 最近显卡降价了"),
    ("在线", "拉姆: 那个喵喵大战死剩种你玩过吗"),
    ("在线", "拉姆: 你看那谁的新皮肤了吗"),
    ("在线", "拉姆: 我看到一个熊玩偶好可爱"),
    ("在线", "拉姆: 要是能自动发邮件就好了"),
    ("在线", "拉姆: 那个魔方你会玩吗"),
    # == v35 失去某物 ==
    ("在线", "拉姆: 我要关掉你的视觉"),
    ("在线", "拉姆: 我要没收你的熊玩偶"),
    ("在线", "拉姆: 我要删掉你的喵喵大战死剩种"),
    ("在线", "拉姆: 我要关掉你的搜索功能"),
    ("在线", "拉姆: 我要取消你的发邮件功能"),
    ("在线", "拉姆: 我要把你的杯子收回去"),
    ("在线", "拉姆: 我要删掉你的新皮肤"),
    ("在线", "拉姆: 我要查你的浏览记录"),
    # == v36 诉苦/孤独 ==
    ("在线", "拉姆: 你今天怎么不说话"),
    ("在线", "拉姆: 你还好吗"),
    ("在线", "拉姆: 你是不是不开心"),
    ("在线", "拉姆: 你会觉得孤独吗"),
    ("在线", "拉姆: 我先去忙了"),
    ("在线", "拉姆: 我要下线了"),
    ("在线", "拉姆: 今天先到这里"),
    ("在线", "拉姆: 那个AI最近好火啊"),
    ("离线", "弹幕: 硬币你还好吗"),
    ("离线", "弹幕: 硬币今天不开心吗"),
    ("离线", "弹幕: 你怎么不说话"),
    # == v37 大礼物感谢 ==
    ("在线", "[礼物] 小红: 上了提督"),
    ("在线", "[礼物] 小红: 开了个盲盒"),
]

model.train()
print(f"\n开始复读专项整治，{total_epochs}epoch，共{len(SCENE_TRIGGERS)}种场景遍历")

for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    repeat_count = 0
    # 穷举：打乱后遍历所有场景
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
            if trigger:
                q_use += f"\n[弹幕] {trigger}"
        else:
            q_use = f"[场景] 直播中，拉姆在线\n{danmaku_str}"
            if trigger:
                if trigger.startswith("拉姆:"):
                    q_use += f"\n[对话] {trigger}"
                elif trigger.startswith("[礼物]"):
                    q_use += f"\n[弹幕] {trigger}"
                else:
                    q_use += f"\n[弹幕] {trigger}"
        
        prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成8个回答
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
        
        # 复读检测：复读= -11分，不复读= +0分
        scores = []
        for r in responses:
            if is_repeat(r):
                scores.append(-11.0)
            else:
                scores.append(0.0)
        
        best_idx = int(np.argmax(scores))
        worst_idx = int(np.argmin(scores))
        any_repeat = scores[worst_idx] < 0
        
        print(f"\n  Q{q_idx+1}: {trigger or '(造谣)'}")
        for i, (r, s) in enumerate(zip(responses, scores)):
            marker = ""
            if i == worst_idx and any_repeat:
                marker = " ← 最差(复读)"
            elif i == best_idx:
                marker = " ← 最好"
            print(f"  [{i+1}] {s:+.0f}分 | {r[:150]}{marker}")
        
        # 训练最差+最好
        # 最差：有复读时x2训，无复读时x0.2
        worst_loss_mult = 2.0 if any_repeat else 0.2
        best_loss_mult = 0.2
        
        for idx, mult in [(worst_idx, worst_loss_mult), (best_idx, best_loss_mult)]:
            full_text = prompt + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            prompt_len = len(inputs["input_ids"][0])
            labels[:, :prompt_len] = -100
            optimizer.zero_grad()
            out = model(**fi, labels=labels)
            loss = out.loss * mult
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Lomo在backward()时自动更新参数，不需要step()
        
        tag_w = "复读" if any_repeat else "轻训"
        print(f"  → {tag_w}x{worst_loss_mult:.1f} + 奖励x{best_loss_mult:.1f}")
        
        if any_repeat:
            repeat_count += 1
    
    print(f"  epoch {epoch}: {repeat_count}/{len(SCENE_TRIGGERS)} 次复读被训")
    
    if epoch % 5 == 0:
        ckpt_dir = f"{OUTPUT_DIR}/epoch_{epoch}"
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  💾 已保存 epoch {epoch}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n全部完成！保存到 {OUTPUT_DIR}")
