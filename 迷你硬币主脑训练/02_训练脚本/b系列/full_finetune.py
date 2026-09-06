#!/usr/bin/env python3
"""
mini_coin 全量微调 v3 — 纯 instruction/output 格式
Qwen3-8B + A800 80GB + bf16 + gradient_checkpointing + AdamW8bit

用法:
  python full_finetune.py \
    --model_path /path/to/model \
    --data_path /path/to/data.json \
    --output_dir ./ckpt \
    --epochs 20 --lr 1e-5 --max_length 512
"""

import os, sys, json, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.data = []
        for item in raw:
            instruction = item.get("instruction", "") or item.get("input", "")
            output = item.get("output", "") or item.get("response", "")
            if instruction and output:
                self.data.append((instruction, output))
        if not self.data:
            raise ValueError(f"数据为空: {data_path}")
        print(f"  加载 {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction, output = self.data[idx]
        user_prefix = "<|im_start|>user\n" + instruction + "<|im_end|>\n"
        assistant_prefix = "<|im_start|>assistant\n"
        user_ids = self.tokenizer.encode(user_prefix, add_special_tokens=False)
        assistant_ids = self.tokenizer.encode(
            assistant_prefix + output + "<|im_end|>", add_special_tokens=False
        )
        prefix_ids = self.tokenizer.encode(assistant_prefix, add_special_tokens=False)
        n_prefix = len(prefix_ids)
        output_ids = assistant_ids[n_prefix:]
        all_ids = user_ids + assistant_ids
        all_labels = ([-100] * len(user_ids) +
                      [-100] * n_prefix + output_ids)
        if len(all_ids) > self.max_length:
            all_ids = all_ids[:self.max_length]
            all_labels = all_labels[:self.max_length]
        pad_len = self.max_length - len(all_ids)
        pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        all_ids = all_ids + [pad_token] * pad_len
        all_labels = all_labels + [-100] * pad_len
        attn_mask = [1] * (self.max_length - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./ckpt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 模型
    print(f"[1/5] 加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in model.parameters():
        p.requires_grad = True
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # 2. 数据
    print(f"[2/5] 加载数据: {args.data_path}")
    dataset = InstructionDataset(args.data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    steps_per_epoch = len(dataloader)
    print(f"  每 epoch {steps_per_epoch} 步")

    # 3. 优化器
    print(f"[3/5] AdamW8bit, lr={args.lr}")
    optimizer = AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    total_steps = steps_per_epoch * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"  总步数: {total_steps}, warmup: {warmup_steps}")

    # 4. 训练
    print("[4/5] 开始训练...")
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        valid_steps = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()
            valid_steps += 1

            if (step + 1) % args.grad_accum == 0 or (step + 1) == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  epoch {epoch+1}/{args.epochs} "
                          f"step {global_step}/{total_steps} "
                          f"loss {outputs.loss.item():.4f} "
                          f"lr {lr_now:.2e}")

        avg_loss = epoch_loss / max(valid_steps, 1)
        print(f"  Epoch {epoch+1} 完成, avg loss: {avg_loss:.4f}")

    # 5. 跑完所有 epoch，只存一次 final 检查点
    print(f"\n[5/5] {args.epochs} 个 epoch 全部完成，保存最终模型...")
    final_dir = os.path.join(args.output_dir, "final_epo")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  ✅ 已保存: {final_dir}")
    print(f"  📊 最终 avg loss (最后1个epoch): {avg_loss:.4f}")

if __name__ == "__main__":
    train()
