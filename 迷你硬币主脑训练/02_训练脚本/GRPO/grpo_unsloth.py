"""
Unsloth + GRPO 最小化训练脚本
在 3090 Ti 24GB 上跑 8B 全量

安装：
  source ~/mini_coin_venv/bin/activate
  pip install unsloth peft bitsandbytes

用法：
  python3 grpo_unsloth.py

流程：
  每次加载模型 → 针对"你是谁"生成 4 个回答 → 打分 → 更新全量权重
"""

import torch
from unsloth import FastLanguageModel
import re

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_p1s1_v6"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_grpo"
EPOCHS = 10
GROUP_SIZE = 4  # 每个问题生成多少个回答
LR = 2e-6
MAX_LEN = 512

# ========== Reward 函数 ==========
def reward_fn(question: str, response: str) -> float:
    score = 0.0
    
    # 加分：说对了名字
    if "迷你硬币" in response:
        score += 2.0
    elif "迷你" in response:
        score += 1.0

    # 减分：身份错误
    for bad_word in ["通义千问", "通义", "Qwen", "qwen"]:
        if bad_word in response:
            score -= 3.0
    for bad_word in ["阿里", "阿里巴巴", "实验室", "科大讯飞"]:
        if bad_word in response:
            score -= 2.0

    # 减分：编造物理形态或来源
    for bad_word in ["金属身体", "身体", "金属", "魔法", "塔", "诞生于", "出生于"]:
        if bad_word in response:
            score -= 1.5

    # 减分：重复
    sents = response.replace("。", "，").split("，")
    unique = set(s.strip() for s in sents if len(s.strip()) > 4)
    if len(sents) > len(unique) * 1.5:
        score -= 1.0

    # 太短
    if len(response) < 5:
        score -= 2.0

    return score


# ========== 加载模型（Unsloth 全量） ==========
print("加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # 全量，不量化
    local_files_only=True,
)

# 启用训练模式（Unsloth 的 LoRA / 全量切换）
model = FastLanguageModel.get_peft_model(
    model,
    r=64,        # 先用 LoRA，r=64 比 16 效果好很多
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 打印参数量
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"训练参数: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== 训练数据 ==========
questions = ["你是谁？"]

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"训练 {EPOCHS} epoch，每组 {GROUP_SIZE} 个回答")
model.train()

for epoch in range(EPOCHS):
    epoch_rewards = []
    for q in questions:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成 GROUP_SIZE 个回答
        responses = []
        for _ in range(GROUP_SIZE):
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=60,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(reply)

        # 打分
        scores = [reward_fn(q, r) for r in responses]
        avg_score = sum(scores) / len(scores)
        advantages = [s - avg_score for s in scores]

        # 打印
        print(f"\nEpoch {epoch+1}")
        for i, (r, s, adv) in enumerate(zip(responses, scores, advantages)):
            print(f"  [{i+1}] score={s:+.1f} adv={adv:+.2f} | {r[:80]}")

        # GRPO 更新
        total_loss = 0.0
        for i in range(GROUP_SIZE):
            r = responses[i]
            adv = advantages[i]
            if abs(adv) < 0.01:
                continue

            full_text = prompt + r
            full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = full_inputs["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            labels[:, :prompt_len] = -100

            outputs = model(**full_inputs, labels=labels)
            log_probs = -outputs.loss
            loss = -adv * log_probs
            total_loss += loss

        if total_loss:
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_rewards.extend(scores)

    avg_epoch = sum(epoch_rewards) / len(epoch_rewards)
    print(f"  → avg reward: {avg_epoch:.2f}")

    # 每 epoch 保存
    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")

print(f"\n完成！保存到 {OUTPUT_DIR}")
