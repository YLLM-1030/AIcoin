#!/usr/bin/env python3
"""PT 训练 — 最小内存版"""
import torch, gc
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

MODEL = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_intent_planner_v1"
DATA = "/mnt/c/Users/Autogram-coin/Desktop/意图训练_eos.txt"
OUT = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_intent_v2"

gc.collect()
torch.cuda.empty_cache()

print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token
if hasattr(tok, 'thinking_mode'):
    tok.thinking_mode = False

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)

print("加载数据...")
with open(DATA, "r", encoding="utf-8") as f:
    raw = f.read()
lines = [l.strip() for l in raw.split("<|endoftext|>") if l.strip()]
lines = [l + "<|endoftext|>" for l in lines]

def fn(ex):
    return tok(ex["text"], truncation=True, max_length=512)

ds = Dataset.from_dict({"text": lines})
ds = ds.map(fn, remove_columns=["text"], batched=False)
print(f"样本: {len(ds)}")

collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir=OUT, per_device_train_batch_size=1,
    num_train_epochs=3, learning_rate=2e-5,
    save_steps=9999, save_total_limit=1,
    logging_steps=5, report_to="none",
    bf16=True, dataloader_num_workers=0,
    remove_unused_columns=False,
)

Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"完成 → {OUT}")
