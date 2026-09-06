# -*- coding: utf-8 -*-
"""classify_training_memories.py — 用 BERT 分类头（domain_classifier）给训练数据提取的记忆打标签
用法（WSL）:
  cd /mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆
  source ~/mini_coin_venv/bin/activate
  python3 classify_training_memories.py
输出: 伪造记忆/训练数据记忆_非魔幻.md（db 导入格式 [规模|时间|评价|领域]），然后跑 import_memories.py 入库"""
import os, sys, re, torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, '/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆')
from train_bert import MultiTaskClassifier, DOMAIN_LABELS, SCALE_LABELS, TIME_LABELS, EVAL_LABELS
from transformers import AutoTokenizer

MODEL_PATH = '/home/autogram-coin/coin_models/chinese-roberta-wwm-ext'
SAVE_DIR = '/home/autogram-coin/coin_models/domain_classifier'
SRC = '/mnt/c/Users/Autogram-coin/Desktop/训练数据/记忆提取_清洗后.txt'
OUT = '/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/长期记忆文件内容存放/伪造记忆/训练数据记忆_非魔幻.md'
MAX_LEN = 128

print("加载分类头（domain_classifier）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = MultiTaskClassifier(MODEL_PATH, len(DOMAIN_LABELS), len(SCALE_LABELS),
                            len(TIME_LABELS), len(EVAL_LABELS))
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model.pt'), map_location='cuda'))
model.cuda().eval()


def classify(text):
    """返回 (领域, 规模, 时间, 评价) 各 argmax"""
    enc = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True,
                    return_tensors='pt')
    with torch.no_grad():
        d, s, t, e = model(enc['input_ids'].cuda(), enc['attention_mask'].cuda())
    def argmax(labels, x):
        return labels[int(torch.argmax(x[0]))]
    return argmax(DOMAIN_LABELS, d), argmax(SCALE_LABELS, s), argmax(TIME_LABELS, t), argmax(EVAL_LABELS, e)


# 读纯记忆文本
mems = []
for line in open(SRC, encoding='utf-8'):
    m = re.match(r'^\d+\.\s*(.*)$', line.strip())
    if m:
        mems.append(m.group(1).strip())
print(f"待打标签: {len(mems)} 条")

# 逐条分类 → 写 db 导入格式 md
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(f"# 训练数据记忆（分类头打标签，{len(mems)} 条）\n")
    f.write("# 标签: [规模|时间|评价|领域]（db 导入格式），BERT domain_classifier 自动标注\n\n")
    for i, m in enumerate(mems, 1):
        dom, sc, tm, ev = classify(m)
        f.write(f"- {m} `[{sc}|{tm}|{ev}|{dom}]`\n")
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(mems)}")

print(f"完成: {OUT}")
print("下一步: python3 import_memories.py 入库")
