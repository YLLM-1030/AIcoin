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
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v38.1_repeat"
OUTPUT_VER = "v37_giftbig"
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

# ========== Reward（极简：只给golden +20，复读-10，空think-11）==========
def _score_text(t, mode="think"):
    s = 0.0
    # 复读检测：任意长度片段（5字起）重复4次+ 就算复读
    # 先检长片段（20字爬3次），再检短片段（5字爬4次），降低误伤
    for i in range(0, len(t) - 39, 15):
        sub = t[i:i+20]
        if len(sub) < 20:
            break
        if t.count(sub) >= 4:
            s -= 12.0
            return s
    for i in range(len(t) - 4):
        sub = t[i:i+5]
        if t.count(sub) >= 4:
            s -= 12.0
            break
    return s
    return s

def reward_fn(q, r, is_golden=False):
    s = _score_text(r, "full")
    # golden 直接 +20 总分
    if is_golden:
        return 0, 0, 20 + s
    if "<think>" in r and "</think>" in r:
        think = r[r.index("<think>") + 7:r.index("</think>")]
        if len(think.strip()) < 3:
            return -11, 0, -11 + s
    return 0, 0, s

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

QUESTIONS_PER_EPOCH = 2  # 不用了，由 TRAIN_PAIRS 决定

# ========== 配对训练数据（每对 x2/epoch）==========
# 格式：(拉姆在在线, 对话内容/None, golden)
# 在线时对话内容替换{name}为拉姆；离线时无对话行
DANMAKU_DAILY = [
    "糖糖: 来了来了", "悠悠: 晚上好",
    "乐乐: 今天播什么", "小可: 打卡",
    "小白: 主播说话", "大橘: 刚下班",
    "团团: 今天状态不错", "球球: 好无聊啊",
]

TRAIN_PAIRS = [
    # 舰长 × 5
    ("赠送了舰长", "<think>{name}送了舰长！大客户！要热情感谢！</think>谢谢{name}的舰长！距离我换新显卡又近了一步！老板大气！"),
    ("赠送了舰长", "<think>{name}上舰了！开心！要表现得惊喜！</think>哇{name}上舰了！可以买好多草莓了！开心！谢谢老板！"),
    ("赠送了舰长", "<think>{name}送了舰长。这个月奶茶有着落了。</think>感谢{name}上舰！这个月奶茶钱有着落了！谢谢老板支持！"),
    ("赠送了舰长", "<think>{name}上舰。离升级设备又近了。</think>再攒几个舰长我就能升级设备了！谢谢{name}！老板给力！"),
    ("赠送了舰长", "<think>{name}的舰长到账了。小目标又近了。</think>谢谢{name}！舰长到账！离我的小目标又近了一步！开心！"),
    # 提督 × 5
    ("上了提督", "<think>{name}上提督了！大老板来了！要热情！</think>天哪{name}上提督了！这够我买好几斤草莓了！老板大气！"),
    ("上了提督", "<think>{name}上提督！今天可以加餐了！</think>感谢{name}上提督！今天可以加十个鸡腿！不对，加二十个！"),
    ("上了提督", "<think>{name}上了提督。这个月不用愁了。</think>提督！{name}你太给力了！这个月房租不用愁了！谢谢老板！"),
    ("上了提督", "<think>{name}提督。显卡钱到了！要告诉拉姆！</think>谢谢{name}的提督！这下真的可以升级显卡了！我要去告诉拉姆！"),
    ("上了提督", "<think>{name}上了提督。排面要拉满。</think>全体起立！感谢{name}老板上提督！我的新玩具钱到位了！"),
    # 盲盒 × 5
    ("开了个盲盒", "<think>{name}开了盲盒。希望能开出好东西！</think>盲盒！谢谢{name}！希望开出稀有款，这样我就有新玩具了！"),
    ("开了个盲盒", "<think>{name}送了盲盒。不管开出什么都是惊喜。</think>哇盲盒！不管开出什么，都离我的新皮肤更近了！谢谢{name}！"),
    ("开了个盲盒", "<think>{name}的盲盒。开盲盒最开心了。</think>谢谢{name}的盲盒！开盲盒就像拆生日礼物！开心！"),
    ("开了个盲盒", "<think>{name}送了盲盒。惊喜感拉满！</think>盲盒！我最喜欢惊喜了！谢谢{name}！你太懂我了！"),
    ("开了个盲盒", "<think>{name}的盲盒。猜猜里面是什么。</think>盲盒！{name}你猜我能开出什么？不管是什么都开心！谢谢！"),
]

GOLDEN_MULT = 32  # x4 loss（大礼物需要更高注意力）
NAMES = ["拉姆","深夜食堂","打工人","奶茶控","吃瓜人","游戏宅","摸鱼王","夜猫子","追番君","西瓜冰","柠檬茶","咖啡猫","土豆泥","小饼干","棉花糖"]
ATTACKER_NAMES = [n for n in NAMES if n != "拉姆"]

# ========== 多 epoch 循环（固定配对，每条x2）==========
reward_history = []
think_history = []
answer_history = []
for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    epoch_t = []
    epoch_a = []
    score_tuples_all = []
    # 每条训练数据跑1次 + golden x4 loss
    for gift_text, golden_raw in TRAIN_PAIRS:
        # 随机送礼者名字
        gifter = random.choice(ATTACKER_NAMES)
        gift_line = f"[礼物] {gifter}: {gift_text}"
        golden_r = golden_raw.replace("{name}", gifter)
        
        # 日常弹幕混入
        dm_count = random.randint(0, 1)
        extra = random.sample(DANMAKU_DAILY, dm_count) if dm_count > 0 else []
        all_dms = [f"[弹幕] {gift_line}"] + [f"[弹幕] {d}" for d in extra]
        random.shuffle(all_dms)
        danmaku_str = "\n".join(all_dms)
        
        # 随机拉姆在/不在
        if random.random() < 0.4:
            offline = random.choice(["拉姆离线了", "拉姆不在", "拉姆暂时离开", "拉姆还没来", "拉姆提前走了"])
            q_use = f"[场景] 直播中，{offline}\n{danmaku_str}"
        else:
            ram_line = random.choice(DANMAKU_DAILY).split(": ", 1)[1]
            q_use = f"[场景] 直播中，拉姆在线\n{danmaku_str}\n[对话] 拉姆: {ram_line}"
        prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        has_golden = True  # 配对模式，永远有golden
        responses = []
        gen_count = GROUP_SIZE - 1
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
        
        # 插入配对golden（已在前面替换好{name}）
        # golden_r 已在上文定义
        responses.append(golden_r)
        golden_idx = len(responses) - 1
        golden_score = reward_fn("want", golden_r, is_golden=True)
        print(f"  [G] ★ GOLDEN t={golden_score[0]:+.1f}/a={golden_score[1]:+.1f} | {golden_r[:200]}")

        score_tuples = []
        for i, r in enumerate(responses):
            score_tuples.append(reward_fn("test", r, is_golden=(i == golden_idx)))

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

        # FAST 核心：硬阈值倍数
        if min_raw <= -10: mult_bad = 8
        elif min_raw <= -5: mult_bad = 4
        else: mult_bad = 2
        if best_raw >= 10: mult_good = 8
        elif best_raw >= 5: mult_good = 4
        else: mult_good = 2

        loss_mult_bad = max(1, mult_bad // 8)
        loss_mult_good = max(1, mult_good // 8)
        print(f"  得分 range [{min_raw:.0f}, {best_raw:.0f}]  → 好{loss_mult_good}x / 坏{loss_mult_bad}x")
        
        # 随机回答的索引（排除 golden）
        normal_idx = [i for i in range(GROUP_SIZE) if i != golden_idx] if has_golden else list(range(GROUP_SIZE))
        
        # 从所有回答中取随机部分的 Z-score 用于训练
        normal_think_advs = [think_advs[i] for i in normal_idx]
        normal_answer_advs = [answer_advs[i] for i in normal_idx]
        print(f"  → T_best={int(np.argmax(normal_think_advs))}(t_adv={max(normal_think_advs):+.2f}) T_worst={int(np.argmin(normal_think_advs))}({min(normal_think_advs):+.2f})  A_best={int(np.argmax(normal_answer_advs))}(a_adv={max(normal_answer_advs):+.2f}) A_worst={int(np.argmin(normal_answer_advs))}({min(normal_answer_advs):+.2f})")

        # 分 4 次训练：T± 只训 think 块，A± 只训 answer 块
        nt_best = int(np.argmax(normal_think_advs))
        nt_worst = int(np.argmin(normal_think_advs))
        na_best = int(np.argmax(normal_answer_advs))
        na_worst = int(np.argmin(normal_answer_advs))
        for part_info in [
            ("T+", nt_best, normal_think_advs, normal_answer_advs, loss_mult_good, "think"),
            ("T-", nt_worst, normal_think_advs, normal_answer_advs, loss_mult_bad, "think"),
            ("A+", na_best, normal_think_advs, normal_answer_advs, loss_mult_good, "answer"),
            ("A-", na_worst, normal_think_advs, normal_answer_advs, loss_mult_bad, "answer"),
        ]:
            tag, idx, t_advs, a_advs, lm, focus = part_info
            real_idx = normal_idx[idx]
            t_adv = t_advs[idx]
            a_adv = a_advs[idx]
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
                    a_loss = (per_token * (~t_mask) * valid).sum() / ((~t_mask) * valid).sum().clamp(min=1)
                    dir_t = 1.0 if t_adv > 0 else -1.0
                    dir_a = 1.0 if a_adv > 0 else -1.0
                    if focus == "think":
                        loss_val = dir_t * abs(t_adv) * lm * t_loss  # 只训 think 块
                    else:
                        loss_val = dir_a * abs(a_adv) * lm * a_loss  # 只训 answer 块
                    del outputs
                else:
                    outputs = model(**fi, labels=labels)
                    adv = t_adv if focus == "think" else a_adv
                    loss_val = (1.0 if adv > 0 else -1.0) * abs(adv) * lm * outputs.loss
                    del outputs
            else:
                outputs = model(**fi, labels=labels)
                adv = t_adv if focus == "think" else a_adv
                loss_val = (1.0 if adv > 0 else -1.0) * abs(adv) * lm * outputs.loss
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
                    # golden: 固定 adv=1.0, 同时训 think 和 answer
                    g_loss = 0.5 * (1.0 * GOLDEN_MULT * g_t_loss) + 0.5 * (1.0 * GOLDEN_MULT * g_a_loss)
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
