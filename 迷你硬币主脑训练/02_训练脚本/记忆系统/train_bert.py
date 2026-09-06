# train_bert.py — 多任务领域分类器训练
# 一个 RoBERTa 共享编码层 + 4 个分类头（领域/规模/时间/评价）
# 数据：伪造记忆/聊天内容/*.md（10类 × 100条，带完整标签）
# 模型：hfl/chinese-roberta-wwm-ext

import os
import re
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# ── 强制离线 ──
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── 配置 ──
MODEL_PATH = '/home/autogram-coin/coin_models/chinese-roberta-wwm-ext'
DATA_DIR = '/mnt/c/Users/Autogram-coin/Desktop/聊天内容'
SAVE_DIR = '/home/autogram-coin/coin_models/domain_classifier'
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 64
SEED = 42

# 领域 10 类
DOMAIN_LABELS = ['哲学/思辨', '经济/商业', '文化/教育', '文学/艺术', '历史/怀旧',
                 '科技/代码', '游戏/娱乐', '生活/日常', '社交/关系', '健康/医疗']
DOMAIN2ID = {l: i for i, l in enumerate(DOMAIN_LABELS)}

# 规模 3 类
SCALE_LABELS = ['重大', '中等', '普通']
SCALE2ID = {l: i for i, l in enumerate(SCALE_LABELS)}

# 时间 3 类
TIME_LABELS = ['过去', '现在', '未来']
TIME2ID = {l: i for i, l in enumerate(TIME_LABELS)}

# 评价 3 类（空=无评价）
EVAL_LABELS = ['我喜欢', '我不喜欢', '无']
EVAL2ID = {l: i for i, l in enumerate(EVAL_LABELS)}


# ── 数据解析 ──
def parse_md_files(data_dir):
    """解析聊天内容 .md，返回 [(text, domain_id, scale_id, time_id, eval_id)]"""
    samples = []
    tag_re = re.compile(r'`\[([^\]]*)\]`')
    for fpath in sorted(glob.glob(os.path.join(data_dir, '*.md'))):
        with open(fpath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('- '):
                    continue
                m = tag_re.search(line)
                if not m:
                    continue
                text = line[:m.start()].replace('- ', '').strip()
                tag = m.group(1)
                parts = tag.split('|')

                # 新格式 [领域|规模|时间|评价]
                if len(parts) != 4:
                    continue
                domain = parts[0].strip()
                scale = parts[1].strip()
                time_ = parts[2].strip()
                eval_ = parts[3].strip()

                # 领域取第一个（多领域时）
                domain = domain.split()[0] if domain else ''
                if domain not in DOMAIN2ID:
                    continue
                scale_id = SCALE2ID.get(scale, -1)
                time_id = TIME2ID.get(time_, -1)
                eval_id = EVAL2ID.get(eval_, EVAL2ID['无'])

                samples.append((text, DOMAIN2ID[domain], scale_id, time_id, eval_id))
    return samples


# ── 格式增强：同一条数据生成多种输入格式 ──
# 让模型学会忽略"xx对我说/我想/我说"等格式外壳，只看内容
SPEAKERS = ['路人', '汾酒', '星辰', '螺丝', '清风明月', '画画', '蜂群', '可口可乐',
            '拉姆', '张老师', '王浩然', '陈雨萱', '观众', '弹幕', '网友']
THOUGHTS = ['我想了一下。', '我心里琢磨了一下。', '我看了看对方。', '我愣了一下。',
            '我想了想该怎么回。', '我注意到这件事了。', '我打算回应一下。']

def augment_formats(text):
    """返回该文本的多种格式变体（列表）"""
    variants = [text]  # 格式1: 纯内容
    # 格式2: 有人对我说：{text}
    speaker = random.choice(SPEAKERS)
    variants.append(f'{speaker}对我说：{text}')
    # 格式3: 有人对我说：{text} 我想：{thought} 我说：{reply}
    thought = random.choice(THOUGHTS)
    reply = random.choice(THOUGHTS).rstrip('。') + '。'
    variants.append(f'{speaker}对我说：{text}。我想：{thought}我说：{reply}')
    return variants


# ── 多任务模型 ──
class MultiTaskClassifier(nn.Module):
    """共享 RoBERTa 编码层 + 4 个分类头"""

    def __init__(self, model_name, num_domain, num_scale, num_time, num_eval):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_domain
        )
        # 借用 backbone 的 hidden size
        hidden = self.backbone.config.hidden_size
        # 替换分类头为多任务头
        self.backbone.classifier = nn.Identity()
        self.domain_head = nn.Linear(hidden, num_domain)
        self.scale_head = nn.Linear(hidden, num_scale)
        self.time_head = nn.Linear(hidden, num_time)
        self.eval_head = nn.Linear(hidden, num_eval)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        pooled = self.dropout(pooled)

        d_logits = self.domain_head(pooled)
        s_logits = self.scale_head(pooled)
        t_logits = self.time_head(pooled)
        e_logits = self.eval_head(pooled)

        if labels is None:
            return d_logits, s_logits, t_logits, e_logits

        d_label, s_label, t_label, e_label = labels
        loss = 0
        # 每个任务独立计算，用 mask 过滤标签为 -1（缺失）的样本
        loss = loss + nn.functional.cross_entropy(d_logits, d_label)
        if (s_label >= 0).any():
            loss = loss + nn.functional.cross_entropy(s_logits[s_label >= 0], s_label[s_label >= 0])
        if (t_label >= 0).any():
            loss = loss + nn.functional.cross_entropy(t_logits[t_label >= 0], t_label[t_label >= 0])
        if (e_label >= 0).any():
            loss = loss + nn.functional.cross_entropy(e_logits[e_label >= 0], e_label[e_label >= 0])
        return loss, (d_logits, s_logits, t_logits, e_logits)


# ── Dataset ──
class MemoryDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, d, s, t, e = self.samples[idx]
        enc = self.tokenizer(
            text, max_length=self.max_len, padding='max_length', truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'][0],
            'attention_mask': enc['attention_mask'][0],
            'd': d, 's': s, 't': t, 'e': e,
        }


# ── 训练主流程 ──
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 加载数据
    raw_samples = parse_md_files(DATA_DIR)
    print(f'原始数据量: {len(raw_samples)}')

    # 先划分训练/验证（用原始数据划分，保证验证集纯净）
    random.shuffle(raw_samples)
    split = int(len(raw_samples) * 0.9)
    train_raw, val_s = raw_samples[:split], raw_samples[split:]

    # 只对训练集做格式增强：每条 ×3 格式（纯句/对我说/完整三段）
    augmented = []
    for text, d, s, t, e in train_raw:
        for variant in augment_formats(text):
            augmented.append((variant, d, s, t, e))
    train_s = augmented
    print(f'训练集: {len(train_s)} (增强后), 验证集: {len(val_s)} (原始)')

    # 统计各类别数量
    from collections import Counter
    print('训练集领域分布:', Counter(s[1] for s in train_s))

    # 加载模型与分词器
    print('加载模型...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = MultiTaskClassifier(
        MODEL_PATH,
        num_domain=len(DOMAIN_LABELS),
        num_scale=len(SCALE_LABELS),
        num_time=len(TIME_LABELS),
        num_eval=len(EVAL_LABELS),
    ).cuda()

    train_ds = MemoryDataset(train_s, tokenizer, MAX_LEN)
    val_ds = MemoryDataset(val_s, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 优化器 + 调度器
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = (batch['d'].cuda(), batch['s'].cuda(),
                      batch['t'].cuda(), batch['e'].cuda())
            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}')

        # 验证
        model.eval()
        correct = [0, 0, 0, 0]
        total = [0, 0, 0, 0]
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                d_logits, s_logits, t_logits, e_logits = model(input_ids, attention_mask)
                preds = [d_logits.argmax(1), s_logits.argmax(1),
                         t_logits.argmax(1), e_logits.argmax(1)]
                gts = [batch['d'].cuda(), batch['s'].cuda(),
                       batch['t'].cuda(), batch['e'].cuda()]
                for i in range(4):
                    mask = gts[i] != -1
                    if mask.sum() > 0:
                        correct[i] += (preds[i][mask] == gts[i][mask]).sum().item()
                        total[i] += mask.sum().item()
        names = ['领域', '规模', '时间', '评价']
        for i in range(4):
            acc = correct[i] / total[i] if total[i] > 0 else 0
            print(f'  验证[{names[i]}]: {acc:.4f}')

    # 保存
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model.pt'))
    tokenizer.save_pretrained(SAVE_DIR)
    # 保存标签映射
    import json
    with open(os.path.join(SAVE_DIR, 'label_maps.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'domain': DOMAIN_LABELS,
            'scale': SCALE_LABELS,
            'time': TIME_LABELS,
            'eval': EVAL_LABELS,
        }, f, ensure_ascii=False)
    print(f'模型已保存到 {SAVE_DIR}')


if __name__ == '__main__':
    main()
