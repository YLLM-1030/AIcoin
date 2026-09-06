#!/usr/bin/env python3
"""
mini_coin 全量微调 — 纯 PyTorch
Qwen3-8B + A800 80GB + bf16 + gradient_checkpointing + AdamW8bit

关键设计:
  - 使用 apply_chat_template 正确 tokenization (不用手动拼 <|im_start|>)
  - 增量编码定位 turn 边界, 只 unmask assistant 的 loss
  - bf16 + gradient_checkpointing (显存 ~34GB)
  - AdamW8bit (优化器 ~16GB)
  - constant lr with warmup (小数据集不要衰减)
  - enable_thinking=False (不需要 think tokens)
  - 自动清理旧 checkpoint (每个 16GB, 磁盘有限)
"""

import os
import sys
import json
import argparse
import glob
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_constant_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit


# ============================================================
# 数据集: ShareGPT / transformers 格式, apply_chat_template
# ============================================================

class IdentityDataset(Dataset):
    """
    使用 apply_chat_template 正确 tokenization
    通过增量编码定位每个 turn 的 token 边界, 只 unmask assistant 部分
    """
    def __init__(self, data_path, tokenizer, max_length=512, debug=False):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]

        # Pre-tokenize all data
        self.items = []
        for idx, item in enumerate(raw):
            chat_input, labels, attn_mask = self._tokenize(item["conversations"])
            self.items.append({
                "input_ids": torch.tensor(chat_input, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            })

        if debug and len(self.items) > 0:
            self._print_debug(0)

    def _turn_to_chat(self, turn):
        """Convert turn to {role, content}"""
        if "from" in turn:
            role = "user" if turn["from"] == "human" else "assistant"
            return {"role": role, "content": turn["value"]}
        return {"role": turn["role"], "content": turn["content"]}

    def _apply_template(self, chat, add_generation_prompt=False):
        """调用 apply_chat_template, 兼容 enable_thinking 参数"""
        try:
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            # 旧版 tokenizer 不支持 enable_thinking
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )

    def _tokenize(self, conversation):
        chat = [self._turn_to_chat(t) for t in conversation]

        # 1. Tokenize full conversation
        full_ids = self._apply_template(chat, add_generation_prompt=False)

        # 2. Build labels: mask everything except assistant turns
        #    增量编码: 逐步加 turn, 差值就是该 turn 的 token 范围
        labels = [-100] * len(full_ids)

        prev_len = 0
        for i, turn in enumerate(chat):
            current_ids = self._apply_template(chat[:i + 1], add_generation_prompt=False)

            if turn["role"] == "assistant":
                # Unmask tokens added by this assistant turn
                for j in range(prev_len, len(current_ids)):
                    if j < len(labels):
                        labels[j] = full_ids[j]

            prev_len = len(current_ids)

        # 3. Truncate or pad
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            pad_len = self.max_length - len(full_ids)
            attention_mask = [1] * len(full_ids) + [0] * pad_len
            full_ids = full_ids + [self.tokenizer.pad_token_id or 0] * pad_len
            labels = labels + [-100] * pad_len

        return full_ids, labels, attention_mask

    def _print_debug(self, idx):
        item = self.items[idx]
        ids = item["input_ids"]
        labs = item["labels"]
        mask = item["attention_mask"]

        actual_len = mask.sum().item()
        print(f"\n[DEBUG] 样本 {idx}:")
        print(f"  总长度: {len(ids)}, 有效长度: {actual_len}")

        decoded = self.tokenizer.decode(ids[:int(actual_len)], skip_special_tokens=False)
        print(f"  解码: {decoded}")

        labeled_count = (labs[:int(actual_len)] != -100).sum().item()
        print(f"  有 label 的 token 数: {labeled_count} / {int(actual_len)}")

        labeled_tokens = []
        for j in range(int(actual_len)):
            if labs[j] != -100:
                tok = self.tokenizer.decode([labs[j].item()])
                labeled_tokens.append(tok)
        if labeled_tokens:
            print(f"  Label 内容: {'|'.join(labeled_tokens[:30])}")
        print()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ============================================================
# Checkpoint 管理: 自动清理旧的, 省磁盘
# ============================================================

def cleanup_old_checkpoints(output_dir, max_keep=3):
    """只保留最近 max_keep 个 step checkpoint, 删掉旧的"""
    ckpt_dirs = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-step-*")),
        key=lambda p: int(p.split("-step-")[-1]),
    )
    while len(ckpt_dirs) > max_keep:
        oldest = ckpt_dirs.pop(0)
        print(f"  [清理] 删除旧 checkpoint: {oldest}")
        os.system(f"rm -rf {oldest}")


# ============================================================
# 训练主函数
# ============================================================

def train():
    parser = argparse.ArgumentParser(description="mini_coin 全量微调 (纯PyTorch)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ckpt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--max_checkpoints", type=int, default=3,
                        help="最多保留几个 step checkpoint (每个 16GB)")
    parser.add_argument("--debug", action="store_true",
                        help="打印第一个样本的 tokenization 详情")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. 加载模型 ----
    print("[1/5] 加载模型:", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 全量解冻
    for p in model.parameters():
        p.requires_grad = True

    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # ---- 2. 加载数据 ----
    print("[2/5] 加载数据:", args.data_path)
    dataset = IdentityDataset(
        args.data_path, tokenizer, max_length=args.max_length, debug=args.debug
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    steps_per_epoch = len(dataloader)
    print(f"  数据条数: {len(dataset)}, 每 epoch {steps_per_epoch} 步")

    # ---- 3. 优化器 ----
    print(f"[3/5] 优化器: AdamW8bit, lr={args.lr}, schedule=constant_with_warmup")
    optimizer = AdamW8bit(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=0.01,
    )
    total_steps = steps_per_epoch * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
    )
    print(f"  总步数: {total_steps}, warmup: {warmup_steps}")

    # ---- 4. 训练 ----
    print("[4/5] 开始训练...")
    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        valid_steps = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
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

                if global_step % args.save_steps == 0 and global_step > 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"  => 保存 checkpoint-step-{global_step}")
                    cleanup_old_checkpoints(args.output_dir, args.max_checkpoints)

        avg_loss = epoch_loss / max(valid_steps, 1)
        print(f"  Epoch {epoch+1} 完成, 平均loss: {avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = os.path.join(args.output_dir, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  => Best 更新 (loss={avg_loss:.4f})")

        gc.collect()
        torch.cuda.empty_cache()

    # ---- 5. 最终保存 ----
    print("[5/5] 保存最终模型")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  完成! best_loss={best_loss:.4f}")
    print(f"  最终模型: {final_dir}")


if __name__ == "__main__":
    train()
