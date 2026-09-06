"""
SFT 微调 — 监督指令微调，只算response的loss
PT教"知识"，SFT教"角色"
"""
import torch, gc, json, os, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ─── 配置 ───────────────────────────────────
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_final_v3/epoch_1"
OUTPUT_VER = "sft_step1_identity"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_{OUTPUT_VER}"
DATA_PATH = "/mnt/c/Users/Autogram-coin/Desktop/新的预训练/sft_step1_identity.jsonl"
LR = 5e-6
EPOCHS = 3

# ─── 数据格式 ───────────────────────────────
# sft_data.json 格式：
# [
#   {"instruction": "你好", "response": "你好呀~"},
#   {"instruction": "你是谁", "response": "我是迷你硬币"},
#   {"instruction": "有人问我9+10等于多少", "response": "你告诉他19就好啦"}
# ]

# ─── 加载 ───────────────────────────────
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ─── 读数据 ───────────────────────────────
pairs = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
print(f"共 {len(pairs)} 对指令")

# Qwen chat template: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
def format_sft(pair):
    user_txt = f"<|im_start|>user\n{pair['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    full_txt = user_txt + pair['output'] + "<|im_end|>"
    return user_txt, full_txt

# ─── SFT loss（只算response部分） ──────
def sft_loss(model, pair):
    user_txt, full_txt = format_sft(pair)
    fi = tokenizer(full_txt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    ui = tokenizer(user_txt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    if fi["input_ids"].shape[1] < 2: return None
    labels = fi["input_ids"].clone()
    # 把user部分的token mask掉（设-100）
    user_len = ui["input_ids"].shape[1] - 1  # 保留user最后一个token的过度
    labels[0, :user_len] = -100
    out = model(**fi, labels=labels)
    return out.loss

# ─── 训练 ──────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
step = 0

for ep in range(1, EPOCHS + 1):
    print(f"\nEpoch {ep}/{EPOCHS}")
    random.shuffle(pairs)
    ep_loss = ep_cnt = 0
    for pi, pair in enumerate(pairs):
        optimizer.zero_grad()
        loss = sft_loss(model, pair)
        if loss is None: continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ep_loss += loss.item(); ep_cnt += 1; step += 1
        if (pi + 1) % 100 == 0:
            print(f"  step {step} ({pi+1}/{len(pairs)}) loss={ep_loss/ep_cnt:.4f}")
        if (pi + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    sp = f"{OUTPUT_DIR}/epoch_{ep}"
    os.makedirs(sp, exist_ok=True)
    model.save_pretrained(sp); tokenizer.save_pretrained(sp)
    print(f"  保存到 {sp}")
