"""
BERT 回复欲望回归训练脚本
基于 chinese-roberta-wwm-ext + 回归头
训练数据: bert_train.tsv (688条, scorer.py 打的分)
"""
import os

# ── 强制离线（跟上次训练一样）──
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
import json

# ======== 配置 ========
MODEL_PATH = "/home/autogram-coin/coin_models/chinese-roberta-wwm-ext"
DATA_PATH = "/mnt/c/Users/Autogram-coin/Desktop/bert_train.tsv"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/模型备份8月2日/弹幕注意力工具"

BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
MAX_LEN = 64
TEST_SIZE = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== 数据加载 ========
texts, scores, addressees = [], [], []
with open(DATA_PATH, encoding='utf-8') as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            try:
                s = float(parts[0])
                scores.append(s)
                addressees.append(parts[1])
                texts.append(parts[2])
            except: pass

print(f"加载 {len(texts)} 条")
print(f"分数范围: {min(scores):.3f} ~ {max(scores):.3f}, 均值: {np.mean(scores):.3f}")

# 分训练/验证集
train_texts, val_texts, train_scores, val_scores = train_test_split(
    texts, scores, test_size=TEST_SIZE, random_state=42)

# ======== Dataset ========
class DanmakuDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_len):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.scores[idx], dtype=torch.float)
        }

# ======== 模型 ========
class ReplyDesireRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(768, 1)  # 768 -> 1 回归
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logit = self.regressor(pooled).squeeze(-1)
        return self.sigmoid(logit)

# ======== 训练 ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = ReplyDesireRegressor(MODEL_PATH).to(device)

train_ds = DanmakuDataset(train_texts, train_scores, tokenizer, MAX_LEN)
val_ds = DanmakuDataset(val_texts, val_scores, tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
loss_fn = nn.MSELoss()

best_loss = float('inf')
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        preds = model(input_ids, attention_mask)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    # Val
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, labels)
            val_loss += loss.item()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    corr = np.corrcoef(all_preds, all_labels)[0, 1]

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | MAE={mae:.4f} | r={corr:.3f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
        print(f"  -> 保存最佳模型")

# ======== 测试 ========
print("\n=== 测试预测 ===")
model.eval()
tests = [
    "硬币你今天开心吗",
    "拉姆编程水平怎么样",
    "如果明天是世界末日你会做什么",
    "不对不对，拉姆你完全搞错了",
    "哈哈哈哈哈",
    "666",
    "主播好可爱",
    "大家晚上好",
    "你觉得Python和Go哪个好",
]
with torch.no_grad():
    for t in tests:
        enc = tokenizer(t, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        pred = model(enc['input_ids'].to(device), enc['attention_mask'].to(device)).item()
        print(f"  {pred:.3f} | {t}")

print(f"\n模型已保存到: {OUTPUT_DIR}")
