"""
持续预训练 — 纯文本next-token-prediction
基座：v53
方法：所有token算loss，无mask，无GRPO，无golden
用法：cd /mnt/c/Users/Autogram-coin/Desktop/新建训练 && python pretrain_lm.py
"""
import torch, gc, sys, os, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

# ─── 配置 ───────────────────────────────────
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_lamu_relation"
OUTPUT_VER = "pt_final_v3"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_{OUTPUT_VER}"
LR = 1e-5
EPOCHS = 1
DATA_PATH = "/mnt/c/Users/Autogram-coin/Desktop/新的预训练/总的.txt"

# ─── 加载 ───────────────────────────────
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ─── 读数据 ───────────────────────────────
with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
paragraphs = [p for p in paragraphs if len(p) > 10]
random.shuffle(paragraphs)
print(f"共 {len(paragraphs)} 段，总字符 {sum(len(p) for p in paragraphs)}")

# ─── PPL评估 ──────────────────────────
def calc_ppl(model, tokenizer, texts):
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for t in texts:
            fi = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            if fi["input_ids"].shape[1] < 2: continue
            out = model(**fi, labels=fi["input_ids"])
            total_loss += out.loss.item() * fi["input_ids"].shape[1]
            total_tokens += fi["input_ids"].shape[1]
    model.train()
    avg = total_loss / total_tokens if total_tokens else 0
    return avg, torch.exp(torch.tensor(avg)).item()

val = paragraphs[:10]
train = paragraphs[:]  # 全部训练，前10条也练

pre_loss, pre_ppl = calc_ppl(model, tokenizer, val)
print(f"预训练前 loss={pre_loss:.4f}  PPL={pre_ppl:.2f}")

# ─── 训练 ──────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
step = 0

for ep in range(1, EPOCHS + 1):
    print(f"\nEpoch {ep}/{EPOCHS}")
    random.shuffle(train)
    ep_loss = ep_cnt = 0
    for pi, para in enumerate(train):
        fi = tokenizer(para, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        if fi["input_ids"].shape[1] < 2: continue
        optimizer.zero_grad()
        out = model(**fi, labels=fi["input_ids"])
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ep_loss += loss.item(); ep_cnt += 1; step += 1
        if (pi + 1) % 200 == 0:
            print(f"  step {step} ({pi+1}/{len(train)}) loss={ep_loss/ep_cnt:.4f}")
        if (pi + 1) % 200 == 0:
            gc.collect(); torch.cuda.empty_cache()
    avg_loss, avg_ppl = calc_ppl(model, tokenizer, val)
    print(f"  Epoch {ep} 验证: loss={avg_loss:.4f}  PPL={avg_ppl:.2f}")
    # 探针：身份锚点PPL
    with torch.no_grad():
        probe = tokenizer("我是迷你硬币", return_tensors="pt").to(model.device)
        if probe["input_ids"].shape[1] >= 2:
            pl = model(**probe, labels=probe["input_ids"]).loss.item()
            pp = torch.exp(torch.tensor(pl)).item()
            print(f"  >>> 身份探针: loss={pl:.4f}  PPL={pp:.2f}")
    sp = f"{OUTPUT_DIR}/epoch_{ep}"
    os.makedirs(sp, exist_ok=True)
    model.save_pretrained(sp); tokenizer.save_pretrained(sp)
    print(f"  保存到 {sp}")

print(f"\n前: loss={pre_loss:.4f} PPL={pre_ppl:.2f}")
print(f"后: loss={avg_loss:.4f} PPL={avg_ppl:.2f}")
print(f"保存到 {OUTPUT_DIR}")
