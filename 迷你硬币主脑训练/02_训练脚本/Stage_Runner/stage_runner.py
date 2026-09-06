"""
mini_coin 第三次训练 — 6阶段训练脚本 (stage_runner.py)
A800 80GB, 全参数微调, bf16, 每个阶段输出独立 checkpoint

用法:
    python3 stage_runner.py \
        --base_model /root/autodl-tmp/P4S4_v7 \
        --data_dir /root/autodl-tmp/train_data \
        --output_dir /root/autodl-tmp/ckpt

每个阶段从上一个 checkpoint 继续，顺序:
    b0.5 → b1.0 → b1.5 → b2.0 → b3.0 → b4.0
"""
import json
import os
import sys
import time
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
import bitsandbytes as bnb

# ── 6 阶段配置 ──
STAGES = [
    {"name": "b0.5", "file": "b0.5.json",       "target_loss": 0.5,  "lr": 2e-5, "patience": 20},
    {"name": "b1.0", "file": "b1.0.json",       "target_loss": 1.0,  "lr": 1e-5, "patience": 15},
    {"name": "b1.5", "file": "b1.5.json",       "target_loss": 1.5,  "lr": 1e-5, "patience": 12},
    {"name": "b2.0", "file": "b2.0.json",       "target_loss": 2.0,  "lr": 1e-5, "patience": 10},
    {"name": "b3.0", "file": "b3.0.json",       "target_loss": 2.0,  "lr": 1e-5, "patience": 10},
    {"name": "b4.0", "file": "b4.0.json",       "target_loss": 4.0,  "lr": 1e-6, "patience": 3,  "max_epochs": 1},
]


class TrainDataset(Dataset):
    """instruction → output, 用 Qwen chat template 格式化"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 构建 messages 格式
        messages = [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
        # 用 chat template 格式化
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # tokenize
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        # 构建 labels: user 部分 mask 掉（-100），assistant 部分保留
        labels = input_ids.clone()
        # 找到 assistant 部分的起始位置
        assist_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]
        # 找到最后一个 assistant 标记的位置
        assist_positions = (input_ids == assist_token).nonzero(as_tuple=True)[0]
        if len(assist_positions) > 0:
            start_pos = assist_positions[-1].item()
            labels[:start_pos] = -100
        else:
            # fallback: 全部 mask 掉 user 部分（不严谨但安全）
            labels[:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_stage(model, tokenizer, stage, prev_ckpt, output_dir, device):
    """训练一个阶段"""
    name = stage["name"]
    data_file = os.path.join(args.data_dir, stage["file"])
    target_loss = stage["target_loss"]
    lr = stage["lr"]
    patience = stage["patience"]
    max_epochs = stage.get("max_epochs", 999)

    print(f"\n{'='*50}")
    print(f"阶段: {name}  | 目标 loss: {target_loss}  | lr: {lr}")
    print(f"数据: {data_file}")
    print(f"起点: {prev_ckpt or 'base_model'}")
    print(f"{'='*50}")

    # 加载数据
    data = load_data(data_file)
    print(f"加载 {len(data)} 条数据")

    dataset = TrainDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        num_workers=2, pin_memory=True
    )

    # 优化器
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5 if name == "b4.0" else 0,  # b4.0 加 L2 保护
        betas=(0.9, 0.95),
    )

    total_steps = len(dataloader) * min(max_epochs, 50)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps
    )

    # 训练
    model.train()
    best_loss = float("inf")
    no_improve = 0
    global_step = 0

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 10 == 0:
                avg = epoch_loss / n_batches
                print(f"  [{name}] epoch {epoch+1}, step {global_step}: loss = {avg:.4f}")

        avg_epoch_loss = epoch_loss / n_batches
        print(f"  [{name}] Epoch {epoch+1}/{max_epochs}: avg loss = {avg_epoch_loss:.4f}")

        # 检查是否达到目标 loss
        if avg_epoch_loss <= target_loss:
            print(f"  ✓ 达到目标 loss ({avg_epoch_loss:.4f} ≤ {target_loss})")
            ckpt_path = os.path.join(output_dir, name)
            os.makedirs(ckpt_path, exist_ok=True)
            model.save_pretrained(ckpt_path, safe_serialization=True)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  保存 checkpoint: {ckpt_path}")
            return ckpt_path

        # early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping ({patience} epoch 无改善)")
                break

    # 保存最终 checkpoint
    ckpt_path = os.path.join(output_dir, name)
    os.makedirs(ckpt_path, exist_ok=True)
    model.save_pretrained(ckpt_path, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_path)
    print(f"  保存 checkpoint: {ckpt_path} (final loss: {avg_epoch_loss:.4f})")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="neko-v7 safetensors 路径")
    parser.add_argument("--data_dir", required=True, help="train_data 目录")
    parser.add_argument("--output_dir", required=True, help="checkpoint 输出目录")
    global args
    args = parser.parse_args()

    device = "cuda"
    print(f"设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载模型和 tokenizer
    print(f"\n加载 base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    print(f"模型参数量: {model.num_parameters() / 1e9:.2f}B")
    print(f"模型加载完成")

    # 逐阶段训练
    prev_ckpt = None
    for stage in STAGES:
        if prev_ckpt is not None:
            # 从上一个 checkpoint 加载
            print(f"\n加载 checkpoint: {prev_ckpt}")
            model = AutoModelForCausalLM.from_pretrained(
                prev_ckpt,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False,
            )
            model.gradient_checkpointing_enable()

        ckpt = train_stage(model, tokenizer, stage, prev_ckpt, args.output_dir, device)
        prev_ckpt = ckpt
        # 清理 cache
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("全部训练完成！")
    print(f"最终 checkpoint: {prev_ckpt}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
