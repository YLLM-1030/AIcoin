"""
GRPO 训练：只训"你是谁"一个问题
用 DeepSeek API 做评分，点击运行，看效果。

用法：source ~/mini_coin_venv/bin/activate && python3 grpo_identity.py
"""

import torch, json, re, os, sys, time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 配置 ==========
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_p1s1_v6"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_grpo"
EPOCHS = 10
LR = 1e-6
GROUP_SIZE = 4  # 每个问题生成几个回答

# ========== 训练数据：只训一条"你是谁" ==========
questions = [
    "你是谁？",
]

# ========== Reward 函数 ==========
def reward_fn(question: str, response: str) -> float:
    """
    给回答打分。先加分后减分，更精细化。
    """
    score = 0.0
    
    # ==== 加分项 ====
    if "迷你硬币" in response:
        score += 2.0
    elif "迷你" in response:
        score += 1.0  # 只说"迷你"没说全名，少加分
    
    # ==== 减分项 ====
    # 大厂/机构扣分
    if "通义千问" in response or "通义" in response:
        score -= 3.0
    if "Qwen" in response or "qwen" in response.lower():
        score -= 3.0
    if "阿里" in response or "阿里巴巴" in response:
        score -= 2.0
    if "实验室" in response:
        score -= 2.0
    if "科大讯飞" in response:
        score -= 2.0
    
    # 编造物理形态
    if "金属身体" in response or "身体" in response or "金属" in response:
        score -= 1.5
    if "魔法" in response or "塔" in response:
        score -= 1.5
    
    # 编造来源
    if "诞生于" in response or "出生于" in response:
        score -= 1.5
    if "实验室" in response and "创造" not in response:
        score -= 1.0
    
    # 重复文本
    sentences = response.replace("。", "，").split("，")
    unique = set(s.strip() for s in sentences if len(s.strip()) > 4)
    if len(sentences) > len(unique) * 1.5:
        score -= 1.0
    
    # 太短
    if len(response) < 5:
        score -= 2.0
    
    return score


# ========== 模型加载（LoRA 省显存）==========
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("加载模型（LoRA模式）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 打印可训练参数量
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA 参数: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({(trainable/total)*100:.1f}%)")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ========== 训练 ==========
print(f"开始 GRPO 训练，{EPOCHS} epoch，每组 {GROUP_SIZE} 个回答")
model.train()
step = 0

for epoch in range(EPOCHS):
    epoch_rewards = []
    for q in questions:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成 GROUP_SIZE 个回答
        responses = []
        for _ in range(GROUP_SIZE):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=1.0, top_p=0.9)
            reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(reply)

        # 打分
        scores = [reward_fn(q, r) for r in responses]
        avg_score = sum(scores) / len(scores)
        advantages = [s - avg_score for s in scores]

        # 打印
        print(f"\nEpoch {epoch+1}, Q: {q}")
        for i, (r, s, adv) in enumerate(zip(responses, scores, advantages)):
            print(f"  [{i+1}] score={s:+.1f} adv={adv:+.2f} | {r[:60]}")

        # GRPO 更新：对每个回答计算 loss
        total_loss = 0
        for i in range(GROUP_SIZE):
            r = responses[i]
            adv = advantages[i]
            if abs(adv) < 0.01:
                continue  # 接近平均分的不用管

            # 构造完整序列
            full_text = prompt + r
            full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = full_inputs["input_ids"].clone()
            # 只计算回答部分（assistant 之后的部分）的 loss
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            labels[:, :prompt_len] = -100

            outputs = model(**full_inputs, labels=labels)
            log_probs = -outputs.loss  # 负的 loss = log prob
            
            # GRPO loss: -advantage * log_prob
            loss = -adv * log_probs
            total_loss += loss

        if total_loss != 0:
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

        epoch_rewards.extend(scores)

    print(f"\nEpoch {epoch+1} 完成，avg reward: {sum(epoch_rewards)/len(epoch_rewards):.2f}")

# ========== 保存 ==========
print(f"\n保存到 {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("完成！LoRA 权重已保存。使用示例：")
print(f"from peft import PeftModel")
print(f"model = AutoModelForCausalLM.from_pretrained('{MODEL_PATH}')")
print(f"model = PeftModel.from_pretrained(model, '{OUTPUT_DIR}')")
