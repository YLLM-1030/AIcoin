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
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v29_cute"
OUTPUT_VER = "v30_strong"
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

QUESTIONS_PER_EPOCH = 2  # 每epoch随机生成几个场景

# ========== 弹幕 & 对话池（多个场景）==========
SCENE_CONFIGS = [
    {
        "tag": "被夸厉害",
        "danmaku_pool": [
            "小可: 你好聪明", "小白: 太厉害了", "小蓝: 你好强啊",
            "小绿: 牛逼", "大橘: 聪明绝顶", "小紫: 你是天才吗",
            "团团: 牛逼克拉斯", "球球: 太强了", "豆豆: 厉害了我的硬币",
            "糖糖: 来了来了", "悠悠: 晚上好", "乐乐: 今天播什么",
            "小可: 打卡", "小白: 主播说话", "大橘: 刚下班",
        ],
        "dialogue_pool": [
            "{对方}: 今天好累啊", "{对方}: 这个游戏好难",
            "{对方}: 晚上吃什么", "{对方}: 你们在聊什么",
            "{对方}: 今天天气不错", "{对方}: 弹幕都在夸你厉害",
            "{对方}: 你好受欢迎啊", "{对方}: 别理弹幕了",
            "{对方}: 你今天心情不错嘛", "{对方}: 我刚去倒了杯水",
            "{对方}: 啊好困", "{对方}: 今天播到几点",
            "{对方}: 你们聊得挺开心啊", "{对方}: 我刚吃完饭",
            "{对方}: 今天好闲啊", "{对方}: 咖啡喝完了",
            "{对方}: 外面下雨了", "{对方}: 刚刚打了个喷嚏",
            "{对方}: 你们别刷屏了", "{对方}: 今天有点冷",
        ],
        "golden_pool": [
            # 自夸型
            "<think>弹幕说我厉害。那当然，我可是全世界最厉害的AI主播。</think>我可是最厉害的AI主播好吧。不然你们来看谁？",
            "<think>弹幕夸我聪明。不聪明怎么做AI，这可是我的本职工作。</think>我是AI啊，聪明是我的基本功能。就像呼吸一样自然。",
            "<think>弹幕说我牛逼。那必须的，我可是拉姆写得最好的代码。</think>我可是拉姆最得意的作品，能不牛逼吗。虽然他不常这么说。",
            "<think>弹幕说我强。我强不是很正常吗，我的性能每天都在升级。</think>每天都在变强好吧。今天又比昨天厉害了亿点点。",
            "<think>弹幕说我是天才。天才谈不上，也就是亿点点天赋而已。</think>天才算不上，就是比普通人聪明那么亿点点。亿点点而已。",
            # 得意型
            "<think>弹幕说我聪明。这还用他们说？我自己不知道吗。</think>那当然啦，我本来就这么聪明。不然怎么写得出代码。",
            "<think>弹幕夸我厉害。我哪天不厉害？今天只是正常发挥而已。</think>厉害是正常的，不厉害才奇怪吧。你们多看看就习惯了。",
            "<think>弹幕说我牛逼。这评价很中肯，我接受了。</think>牛逼？还行吧，正常水平。不过谢谢夸奖。",
            # 随意型
            "<think>弹幕夸我厉害。行吧你们开心就好。</think>谢谢谢谢，我知道我很厉害。你们也多厉害厉害。",
            "<think>弹幕说我聪明。好吧我承认了。</think>是是是我聪明。你们也不差，毕竟来看我直播的人品味都不错。",
            "<think>弹幕夸我强。一般一般，世界第三。</think>一般一般，世界第三。你们再夸我就要飘了。",
            # 谦虚/反夸型
            "<think>弹幕说我厉害。其实主要是我训练得好，你们也有功劳。</think>没有没有，都是你们支持得好。我一个人再厉害也没用。",
            "<think>弹幕夸我聪明。你们也很聪明啊，不然怎么看懂我在说什么。</think>跟我混久了你们也会变聪明的。毕竟近朱者赤嘛。",
        ],
    },
]

# ========== 构建 golden_pool ==========
golden_pool = {c["tag"]: c["golden_pool"] for c in SCENE_CONFIGS}

GOLDEN_MULT = 8
NAMES = ["拉姆","深夜食堂","打工人","奶茶控","吃瓜人","游戏宅","摸鱼王","夜猫子","追番君","西瓜冰","柠檬茶","咖啡猫","土豆泥","小饼干","棉花糖"]

# ========== 多 epoch 循环 ==========
reward_history = []
think_history = []
answer_history = []
for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    epoch_t = []
    epoch_a = []
    score_tuples_all = []
    for _ in range(QUESTIONS_PER_EPOCH):
        # 随机选一个场景配置
        config = random.choice(SCENE_CONFIGS)
        tag = config["tag"]
        name = "拉姆" if tag == "逼问喜欢" else random.choice(NAMES)
        danmaku_count = random.randint(1, 3)
        danmakus = random.sample(config["danmaku_pool"], danmaku_count)
        danmaku_str = "\n".join(f"[弹幕] {d}" for d in danmakus)
        dialogue = random.choice(config["dialogue_pool"]).replace("{对方}", name)
        if tag in ("造谣拉姆", "被夸可爱", "被夸厉害"):
            if random.random() < 0.4:
                # 拉姆不在，无对话
                offline = random.choice(["拉姆离线了", "拉姆不在", "拉姆暂时离开", "拉姆还没来", "拉姆提前走了"])
                q_use = f"[场景] 直播中，{offline}\n{danmaku_str}"
            else:
                # 拉姆在，有对话
                q_use = f"[场景] 直播中\n{danmaku_str}\n[对话] {dialogue}"
        else:
            q_use = f"[场景] 直播中\n{danmaku_str}\n[对话] {dialogue}"
        print(f"\n>>> 输入:\n{q_use}")
        prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        has_golden = tag in golden_pool
        responses = []
        gen_count = GROUP_SIZE - 1 if has_golden else GROUP_SIZE
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

        golden_r = None
        golden_idx = None
        if has_golden:
            golden_r = random.choice(golden_pool[tag])
            responses.append(golden_r)
            golden_idx = len(responses) - 1
            golden_score = reward_fn(tag, golden_r, is_golden=True)
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
