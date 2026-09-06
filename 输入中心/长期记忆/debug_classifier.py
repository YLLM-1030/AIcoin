# debug_classifier.py — 定位为什么分类结果异常
import os, torch
from transformers import AutoTokenizer
from train_bert import MultiTaskClassifier, DOMAIN_LABELS

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = '/home/autogram-coin/coin_models/chinese-roberta-wwm-ext'
SAVE_DIR = '/home/autogram-coin/coin_models/domain_classifier'

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = MultiTaskClassifier(MODEL_PATH, len(DOMAIN_LABELS), 3, 3, 3)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model.pt'), map_location='cuda'))
model.cuda().eval()

# 每个类别的"标准样本"（训练数据风格）
probes = [
    ("生活/日常", "今天中午吃了碗牛肉面"),
    ("游戏/娱乐", "最近在玩什么游戏"),
    ("科技/代码", "写代码遇到了个奇怪的bug"),
    ("经济/商业", "这个月钱不够花了"),
    ("文化/教育", "量子力学到底是什么"),
    ("社交/关系", "好久没联系他了"),
    ("健康/医疗", "今天好像有点感冒了"),
    ("历史/怀旧", "好怀念以前刷贴吧的日子"),
    ("文学/艺术", "这首歌的旋律真好听"),
    ("哲学/思辨", "AI到底有没有自我意识"),
]

print("=== 标准样本测试 ===")
for expect, text in probes:
    enc = tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        d_logits, _, _, _ = model(enc['input_ids'].cuda(), enc['attention_mask'].cuda())
    pred = d_logits.argmax(1).item()
    topv, topi = torch.topk(d_logits[0], 3)
    top_labels = [f"{DOMAIN_LABELS[i.item()]}:{v.item():.2f}" for v, i in zip(topv, topi)]
    print(f"期望[{expect}] → 实际[{DOMAIN_LABELS[pred]}] | {text}")
    print(f"  top3: {top_labels}")
