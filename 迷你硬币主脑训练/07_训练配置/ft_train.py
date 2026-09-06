#!/usr/bin/env python3
"""mini_coin 全量微调 v3 — 空 think 注入 + 纯 instruction/output"""
import os, sys, json, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit

# 空 think 前缀：教模型"思考已完成，直接回答"
THINK_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.data = []
        for item in raw:
            inst = item.get("instruction", "") or item.get("input", "")
            out = item.get("output", "") or item.get("response", "")
            if inst and out:
                self.data.append((inst, out))
        if not self.data:
            raise ValueError(f"Data empty: {data_path}")
        print(f"  Data: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction, output = self.data[idx]
        user_text = "<|im_start|>user\n" + instruction + "<|im_end|>\n"
        assistant_text = THINK_PREFIX + output + "<|im_end|>"
        user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)
        assistant_ids = self.tokenizer.encode(assistant_text, add_special_tokens=False)
        # 拆分 think 前缀 + output
        prefix_ids = self.tokenizer.encode(THINK_PREFIX, add_special_tokens=False)
        n_prefix = len(prefix_ids)
        output_ids = assistant_ids[n_prefix:]

        all_ids = user_ids + assistant_ids
        all_labels = ([-100] * len(user_ids) +    # mask user
                      [-100] * n_prefix +          # mask <|im_start|>assistant + <think>
                      output_ids)                   # 只训 output

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1] Load model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    for p in model.parameters():
        p.requires_grad = True
    model.gradient_checkpointing_enable()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"[2] Load data: {args.data_path}")
    dataset = InstructionDataset(args.data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    steps_per_epoch = len(dataloader)

    print(f"[3] lr={args.lr}")
    optimizer = AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    total_steps = steps_per_epoch * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}, patience: {args.patience}")

    print(f"[4] Train {args.epochs} epochs")
    model.train()
    global_step = 0
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        valid_steps = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
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
        avg_loss = epoch_loss / max(valid_steps, 1)
        improved = avg_loss < best_loss - 0.001
        marker = " *" if improved else ""
        print(f"  epoch {epoch+1:3d}/{args.epochs}  avg_loss={avg_loss:.4f}{marker}")
        if improved:
            best_loss = avg_loss
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  => Early stop (patience={args.patience})")
                break

    print(f"\n[5] Save (best epoch={best_epoch}, loss={best_loss:.4f})")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 顺便把 generation_config 的 enable_thinking 关了
    try:
        gen_cfg = json.load(open(f"{args.output_dir}/generation_config.json"))
        gen_cfg["enable_thinking"] = False
        json.dump(gen_cfg, open(f"{args.output_dir}/generation_config.json", "w"), indent=2)
    except:
        pass
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
