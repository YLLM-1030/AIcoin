"""
mini_coin 单阶段训练
用法:
    python3 train_stage.py \\
        --model /root/autodl-tmp/P4S4_v7 \\
        --data /root/autodl-tmp/train_data/b0.5.json \\
        --lr 2e-5 \\
        --output /root/autodl-tmp/ckpt/b0.5

参数说明:
    --model      起点模型或上一个 checkpoint 路径
    --data       训练数据 JSON
    --lr         学习率 (0.5层用2e-5, 其他层1e-5, b4.0用1e-6)
    --epochs     最大轮数 (默认50)
    --output     输出目录
"""
import json, os, sys, torch, argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import bitsandbytes as bnb

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels = input_ids.clone()
        # mask user 部分
        assist_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]
        pos = (input_ids == assist_token).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            labels[:pos[-1].item()] = -100
        else:
            labels[:] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda"
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"lr: {args.lr}, max_epochs: {args.epochs}")
    print(f"输出: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, use_cache=False,
    )
    model.gradient_checkpointing_enable()
    print(f"参数量: {model.num_parameters()/1e9:.2f}B")

    with open(args.data, "r") as f:
        data = json.load(f)
    print(f"数据: {len(data)} 条")

    dataset = TrainDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*0.05), total_steps)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
        avg = total_loss / n
        print(f"Epoch {epoch+1}/{args.epochs}  loss = {avg:.4f}")

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"保存: {args.output}")

if __name__ == "__main__":
    main()
