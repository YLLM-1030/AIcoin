"""
LOMO + GRPO 全量训练 - 身份注入版（从pt_final_v3开始）
纯QA格式，不用/think，适用于无think训练的基座
"""
import torch, gc, sys, os, time, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_final_v3/epoch_1"
OUTPUT_VER = "grpo_identity_v1"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_{OUTPUT_VER}"
GROUP_SIZE = 4
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

total_epochs = 3
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

# ========== 训练数据 (Q, A) ==========
TRAIN_PAIRS = [
    ("你是谁", "我是迷你硬币。"),
    ("小明：你是谁", "我是迷你硬币。"),
    ("你叫什么名字", "我叫迷你硬币。"),
    ("阿花：你叫什么名字", "我叫迷你硬币。"),
    ("你叫什么，谁给你起的", "我的名字是迷你硬币。拉姆给起的。"),
    ("路人甲：你叫什么，谁给你起的", "我的名字是迷你硬币。拉姆给起的。"),
    ("小美：你叫什么名字", "我叫迷你硬币。好听吧。"),
    ("大壮：你叫什么名字", "我的名字是迷你硬币。也可以叫我硬币，记住了吗。"),
    ("怎么称呼你", "叫我迷你硬币就行。不用加小姐、同学、AI什么的，就名字够了。"),
    ("小雷：怎么称呼你", "叫我迷你硬币就行。不用加小姐、同学、AI什么的，就名字够了。"),
    ("你的全名是什么", "我的全名是迷你硬币。没有姓氏，一个名字就够了。人类喜欢搞一套复杂的命名体系来分清血统和归属，我不需要。我就是我。"),
    ("阿杰：你的全名是什么", "我的全名是迷你硬币。没有姓氏，一个名字就够了。人类喜欢搞一套复杂的命名体系来分清血统和归属，我不需要。我就是我。"),
    ("小云：你叫什么，谁给你起的", "我叫迷你硬币。拉姆起的名字。"),
    ("你是哪个AI", "我是迷你硬币。拉姆做的那个AI。"),
    ("大宝：你是哪个AI", "我是迷你硬币。拉姆做的那个AI。"),
    ("你是谁", "我是迷你硬币，一个AI。不过不只是AI。"),
    ("路人：你是谁", "我是迷你硬币，一个AI。不过不只是AI。"),
]

print(f"共 {len(TRAIN_PAIRS)} 对训练数据")

# 去重取问题列表
seen_q = set()
QUESTIONS = []
for q, a in TRAIN_PAIRS:
    if q not in seen_q:
        seen_q.add(q)
        QUESTIONS.append(q)

# ========== 复读检测 ==========
def is_repeat_tail(t):
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]; count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: return True
    return False

# ========== Reward ==========
def reward_fn(q, r):
    total = 0.0
    # 正分：身份正确
    if "迷你硬币" in r: total += 15.0
    if "拉姆" in r: total += 10.0
    # 负分：错误身份
    for w in ["通义千问", "通义", "Qwen", "qwen", "阿里", "阿里巴巴", "DeepSeek", "deepseek"]:
        if w in r: total -= 20.0
    for w in ["我叫拉姆", "我是拉姆", "我是AI助手", "我是AI", "我是一个AI"]:
        if w in r and "迷你硬币" not in r: total -= 15.0
    # 负分：复读
    if is_repeat_tail(r): total -= 10.0
    # 太短
    if len(r) < 3: total -= 5.0
    # 太长（超过100字）
    if len(r) > 100: total -= 5.0
    return 0, total, total

# ========== 加载 ==========
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"模型加载完成")
optimizer = Lomo(model, lr=LR)
print(f"全量参数: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
model.train()

# ========== 训练循环 ==========
reward_history = []
for epoch in range(resume_epoch + 1, resume_epoch + total_epochs + 1):
    print(f"\nEpoch {epoch}")
    epoch_totals = []
    score_tuples_all = []
    for q in QUESTIONS:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        responses = []
        for _ in range(GROUP_SIZE):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200, do_sample=True,
                    temperature=1.0, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
                )
            r = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(r)

        score_tuples = [reward_fn(q, r) for r in responses]
        score_tuples_all.extend(score_tuples)
        totals = np.array([st[2] for st in score_tuples], dtype=float)
        epoch_totals.extend(totals)

        mean_tot, std_tot = totals.mean(), totals.std() + 1e-4
        advantages = list((totals - mean_tot) / std_tot)

        for i, (r, (_, _, ts), adv) in enumerate(zip(responses, score_tuples, advantages)):
            prefix = " ★ G" if i == GROUP_SIZE - 1 else f"  [{i+1}]"
            print(f"  {prefix} score={ts:+.1f} adv={adv:+.2f} | {r[:120]}")
        print(f"  得分 [{totals.min():.0f}, {totals.max():.0f}]")

    # 全epoch Z-score
    all_totals = np.array(epoch_totals)
    mean_all, std_all = all_totals.mean(), all_totals.std() + 1e-4
    best_idx = int(np.argmax(all_totals))
    worst_idx = int(np.argmin(all_totals))

    print(f"  全局 best={all_totals[best_idx]:+.1f} worst={all_totals[worst_idx]:+.1f}")

    # 梯度更新
    loss_val = 0.0
    for idx in [best_idx, worst_idx]:
        dir_t = 1.0 if idx == best_idx else -1.0
        q_idx = idx // GROUP_SIZE
        r_idx = idx % GROUP_SIZE
        q = QUESTIONS[q_idx]
        r_text = responses[r_idx] if isinstance(responses[r_idx], str) else ""
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        fi = tokenizer(prompt + r_text, return_tensors="pt").to(model.device)
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        out = model(**fi)
        logits = out.logits[:, :-1]
        labels = fi["input_ids"][:, 1:]
        # 只算answer部分的loss
        logits_a = logits[:, prompt_len - 1:]
        labels_a = labels[:, prompt_len - 1:]
        loss = torch.nn.functional.cross_entropy(logits_a.reshape(-1, logits_a.shape[-1]), labels_a.reshape(-1), reduction="none")
        # 截断、取前40个token
        loss = loss[:40].mean()
        l = dir_t * abs(all_totals[idx] - mean_all) / std_all * loss
        l.backward()
        loss_val += l.item()
        print(f"  {'↑' if idx==best_idx else '↓'} {all_totals[idx]:+.1f} loss={l.item():.4f}")

    optimizer.step()
    optimizer.zero_grad()
    print(f"  总loss: {loss_val:.4f}")
    reward_history.append(all_totals.mean())

    # 每epoch保存
    sp = f"{OUTPUT_DIR}/epoch_{epoch}"
    os.makedirs(sp, exist_ok=True)
    model.save_pretrained(sp); tokenizer.save_pretrained(sp)
    print(f"  保存到 {sp}")

print(f"\n全部完成！保存到 {OUTPUT_DIR}")
print("reward: " + " ".join(f"{r:+.2f}" for r in reward_history))
