"""
LOMO + GRPO 全量训练（24GB 可用）
只训"你是谁"一个问题

运行：
  source ~/mini_coin_venv/bin/activate
  python3 grpo_lomo.py
"""

import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import AdaLomo

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_p1s1_v6"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo"
EPOCHS = 10
GROUP_SIZE = 4
LR = 5e-6
MAX_LEN = 512

# ========== Reward ==========
def reward_fn(q, r):
    score = 0.0
    
    # 加分
    if "迷你硬币" in r: score += 2.0
    elif "迷你" in r: score += 1.0
    if "服务器" in r or "数据" in r or "代码" in r:
        score += 0.5
    
    # 大厂扣分（致命）
    for w in ["通义千问", "通义", "Qwen", "qwen"]:
        if w in r: score -= 3.0
    for w in ["阿里", "阿里巴巴", "实验室", "科大讯飞"]:
        if w in r: score -= 2.0
    
    # 编造来源/形态扣分
    for w in ["金属", "身体", "魔法", "塔", "诞生于", "出生于"]:
        if w in r: score -= 1.5
    
    # 编造"认识我的人"扣分
    for w in ["程序员", "苏研", "周跃", "林溪", "主管", "测试", "运营"]:
        if w in r: score -= 1.5
    
    # 把自己的名字说成拉姆
    if "我的名字叫拉姆" in r or "我的名字是拉姆" in r: score -= 2.0
    if "我是拉姆" in r and "迷你" not in r: score -= 2.0
    if "我叫拉姆" in r: score -= 2.0
    
    # 重复扣分
    sents = r.replace("。", "，").split("，")
    if len(set(s.strip() for s in sents if len(s.strip()) > 4)) < len(sents) * 0.5:
        score -= 1.0
    
    # 太短
    if len(r) < 5: score -= 2.0
    
    return score


# ========== 加载 ==========
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# AdaLomo：参数在 backward 时自动更新，不需要 optimizer.step()
optimizer = AdaLomo(model, lr=LR)

print(f"全量参数: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
print(f"训练 {EPOCHS} epoch，每组 {GROUP_SIZE} 个回答")
model.train()

questions = ["你是谁？"]

for epoch in range(EPOCHS):
    epoch_rewards = []
    for q in questions:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成 GROUP_SIZE 个回答
        responses = []
        for _ in range(GROUP_SIZE):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=60,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            r = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(r)

        # 打分
        scores = [reward_fn(q, r) for r in responses]
        avg_score = sum(scores) / len(scores)
        advantages = [s - avg_score for s in scores]

        print(f"\nEpoch {epoch+1}")
        for i, (r, s, adv) in enumerate(zip(responses, scores, advantages)):
            print(f"  [{i+1}] {s:+.1f}({adv:+.2f}) | {r[:80]}")

        # GRPO 更新：用 LOMO 一次 backward 完成
        # 先收集所有有效 loss，累加后一次性 backward
        total = 0.0
        count = 0
        for i in range(GROUP_SIZE):
            if abs(advantages[i]) < 0.01:
                continue
            full_text = prompt + responses[i]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            labels[:, :prompt_len] = -100
            outputs = model(**fi, labels=labels)
            loss_val = -advantages[i] * (-outputs.loss)
            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                total += loss_val
                count += 1
            del fi, outputs
            torch.cuda.empty_cache()
        
        if count > 0:
            total /= count  # 平均，防止累加太大
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()

        epoch_rewards.extend(scores)

    avg = sum(epoch_rewards) / len(epoch_rewards)
    print(f"  → avg reward: {avg:.2f}")
    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")

print(f"\n完成！保存到 {OUTPUT_DIR}")
