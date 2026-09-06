"""
Unsloth 全量 GRPO 训练（单问题：你是谁）
使用 FastLanguageModel 原生全量训练，不经过 LoRA
3090 Ti 24GB 验证通过
"""

import torch
import gc
from unsloth import FastLanguageModel
import re

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_p1s1_v6"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_full"
EPOCHS = 10
GROUP_SIZE = 4
LR = 1e-6
MAX_LEN = 512

# ========== Reward ==========
def reward_fn(q, r):
    score = 0.0
    if "迷你硬币" in r: score += 2.0
    elif "迷你" in r: score += 1.0
    for w in ["通义千问", "通义", "Qwen", "qwen"]:
        if w in r: score -= 3.0
    for w in ["阿里", "阿里巴巴", "实验室", "科大讯飞", "金属", "身体", "魔法", "塔", "诞生于"]:
        if w in r: score -= 1.5
    sents = r.replace("。", "，").split("，")
    if len(set(s.strip() for s in sents if len(s.strip()) > 4)) < len(sents) * 0.5:
        score -= 1.0
    if len(r) < 5: score -= 2.0
    return score

# ========== 加载模型（全量，不量化） ==========
print("加载全量模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    local_files_only=True,
)
# 不封装 LoRA，保持全量训练
model = FastLanguageModel.get_peft_model(
    model, r=None, target_modules=None, lora_alpha=None,
    use_gradient_checkpointing="unsloth",
)  # r=None 表示全量训练（Unsloth 语法）

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.0f}M（全量）")
print(f"训练 {EPOCHS} epoch，每组 {GROUP_SIZE} 个回答")

questions = ["你是谁？"]
model.train()

for epoch in range(EPOCHS):
    epoch_rewards = []
    for q in questions:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 逐个生成，避免显存峰值
        responses = []
        for _ in range(GROUP_SIZE):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=60,
                    do_sample=True, temperature=1.0, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
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

        # GRPO 更新
        total_loss = 0.0
        for i in range(GROUP_SIZE):
            if abs(advantages[i]) < 0.01:
                continue
            full_text = prompt + responses[i]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            labels[:, :prompt_len] = -100
            outputs = model(**fi, labels=labels)
            loss = -advantages[i] * (-outputs.loss)
            total_loss += loss
            del fi, outputs

        if total_loss:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        epoch_rewards.extend(scores)

    avg = sum(epoch_rewards) / len(epoch_rewards)
    print(f"  → avg reward: {avg:.2f}")
    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")

print(f"\n完成！")
