"""
mini_coin 训练 — 修正版 (stage_runner_v2.py)
A800 80GB, 全参数微调, bf16

修正点:
1. 不用 apply_chat_template (Qwen3 会自动插入 <think> 块导致双嵌套)
2. 手动构造 <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n
3. 保留 output 字段原样 (含或不含 <think> 标签均可)

用法:
    python3 stage_runner_v2.py \
        --base_model /root/autodl-tmp/P4S4_v7 \
        --data_dir /root/autodl-tmp/train_data \
        --output_dir /root/autodl-tmp/ckpt \
        --stage b0.5 \
        --epochs 20 \
        --lr 2e-5

参数说明:
    --stage      训练哪个桶 (b0.5/b1.0/b1.5/b2.0/b3.0/b4.0)
    --epochs     训练轮数 (默认20)
    --lr         学习率
    --from_ckpt  从指定checkpoint继续 (不指定则从base_model开始)
"""
import json, os, sys, torch, argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import bitsandbytes as bnb

# ── 桶配置参考 (target_loss 为参考值, 通过调整 lr/epochs 逐步逼近) ──
STAGE_PRESETS = {
    "b0.5": {"file": "b0.5.json", "target_loss": 0.5, "lr": 2e-5},
    "b1.0": {"file": "b1.0.json", "target_loss": 1.0, "lr": 1e-5},
    "b1.5": {"file": "b1.5.json", "target_loss": 1.5, "lr": 1e-5},
    "b2.0": {"file": "b2.0.json", "target_loss": 2.0, "lr": 1e-5},
    "b3.0": {"file": "b3.0.json", "target_loss": 3.0, "lr": 1e-5},
    "b4.0": {"file": "b4.0.json", "target_loss": 4.0, "lr": 1e-6},
}


class TrainDataset(Dataset):
    """instruction -> output, 手动构造 chat 格式, 不用 apply_chat_template"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 手动构造文本, 不经过 apply_chat_template
        # Qwen3 的 chat_template 会自动插入 <think> 块, 但我们数据里已经有或不需 think
        text = (
            f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['output']}<|im_end|>\n"
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        # mask user 部分 (只训练 assistant 部分)
        labels = input_ids.clone()
        assist_start_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]
        positions = (input_ids == assist_start_token).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            labels[:positions[-1].item()] = -100
        else:
            labels[:] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="基座模型路径")
    parser.add_argument("--data_dir", required=True, help="训练数据目录")
    parser.add_argument("--output_dir", required=True, help="checkpoint 输出目录")
    parser.add_argument("--stage", required=True, choices=list(STAGE_PRESETS.keys()), help="训练哪个桶")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率 (不指定则用预设值)")
    parser.add_argument("--from_ckpt", default=None, help="从指定 checkpoint 继续训练")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    args = parser.parse_args()

    # 确定配置
    preset = STAGE_PRESETS[args.stage]
    lr = args.lr if args.lr is not None else preset["lr"]
    target_loss = preset["target_loss"]
    data_file = os.path.join(args.data_dir, preset["file"])

    device = "cuda"
    print(f"设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"\n阶段: {args.stage} | 目标 loss: {target_loss} | lr: {lr} | epochs: {args.epochs}")
    print(f"数据: {data_file}")

    # 加载 tokenizer
    model_path = args.from_ckpt if args.from_ckpt else args.base_model
    print(f"模型路径: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    print(f"参数量: {model.num_parameters() / 1e9:.2f}B")

    # 加载数据
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"数据: {len(data)} 条")

    dataset = TrainDataset(data, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # 优化器
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
    )
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )

    # 训练
    output_path = os.path.join(args.output_dir, args.stage)
    os.makedirs(output_path, exist_ok=True)
    print(f"\n开始训练 {args.stage}, {args.epochs} epochs...")
    print(f"输出目录: {output_path}")
    print("-" * 50)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            ).loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        dist = abs(avg_loss - target_loss)
        status = "✓ 达标" if dist <= 0.1 else ("→ 接近中" if dist <= 0.5 else "...")
        print(f"Epoch {epoch+1:3d}/{args.epochs}  avg_loss = {avg_loss:.4f}  "
              f"(target={target_loss}, dist={dist:.2f})  {status}")

    # 保存
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("-" * 50)
    print(f"保存: {output_path}")
    print(f"最终 avg_loss: {avg_loss:.4f} (target: {target_loss})")
    print(f"距目标: {abs(avg_loss - target_loss):.4f}")


if __name__ == "__main__":
    main()
