"""
mini_coin 第三次训练 — 数据转换脚本 v2
- 5层 + b4.0风格层（从纯Q文本采样）
- 跳过 b4.5（梗百科，留给RAG）
"""
import json
import os
import re
import random

BASE = r"C:\Users\Autogram-coin\Desktop\新建训练"
OUTPUT = os.path.join(BASE, "train_data")
os.makedirs(OUTPUT, exist_ok=True)
random.seed(42)

# ── 第1-5层: 从 Q:/A: markdown 解析 ──
LAYERS = {
    "b0.5": "第三次训练的0.5/训练数据_桶0.5_合并版.md",
    "b1.0": "第三次训练的1/训练数据_桶1.0_关系层_含think.md",
    "b1.5": "第三次训练的1.5/训练数据_桶1.5_表演层.md",
    "b2.0": "第三次训练的2/训练数据_桶2.0_动机层.md",
    "b3.0": "第三次训练的3/b3.0_补缺 - 合并训练数据.md",
}

def parse_qa_pairs(text):
    pairs = []
    pattern = re.compile(r'^Q:\s*(.*?)\nA:\s*(.*?)(?=\n(?:^Q:|\Z|^#))', re.MULTILINE | re.DOTALL)
    for m in pattern.finditer(text):
        q = m.group(1).strip()
        a = re.sub(r'```.*$', '', m.group(2).strip(), flags=re.DOTALL).strip()
        if q and a:
            pairs.append({"instruction": q, "output": a})
    return pairs

total = 0
for name, path in LAYERS.items():
    full_path = os.path.join(BASE, path)
    if not os.path.exists(full_path):
        print(f"[跳过] {name}: 文件不存在")
        continue
    with open(full_path, "r", encoding="utf-8") as f:
        pairs = parse_qa_pairs(f.read())
    out_file = os.path.join(OUTPUT, f"{name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"[OK] {name}: {len(pairs)} 条 → {out_file}")
    total += len(pairs)

# ── 第6层: b4.0 风格层（从纯Q文本采样）──
B4_SOURCES = {
    "中二": ("第三次训练4.0/B3.0_中二宣言语料_纯Q_3000条.txt", 200),
    "阴阳怪气": ("第三次训练4.0/B3.0_阴阳怪气语料_纯Q_5000条.txt", 300),
    "傲娇": ("第三次训练4.0/B3.0_傲娇语料_纯Q_5000条.txt", 500),
}

b4_pairs = []
for style, (path, sample_n) in B4_SOURCES.items():
    full_path = os.path.join(BASE, path)
    if not os.path.exists(full_path):
        print(f"[跳过] b4.0/{style}: 文件不存在")
        continue
    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 提取所有 Q: xxx 行
    qs = []
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            qs.append(line[2:].strip())
    # 采样
    sampled = random.sample(qs, min(sample_n, len(qs)))
    for q in sampled:
        b4_pairs.append({
            "instruction": "说句" + style + "的话",
            "output": q
        })
    print(f"[OK] b4.0/{style}: 抽取 {len(sampled)}/{len(qs)} 条")

out_file = os.path.join(OUTPUT, "b4.0.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(b4_pairs, f, ensure_ascii=False, indent=2)
print(f"[OK] b4.0: {len(b4_pairs)} 条 → {out_file}")
total += len(b4_pairs)

# ── 跳过 b4.5（梗库，留给RAG）──
print("[跳过] b4.5: 梗百科 → RAG 用")

print(f"\n总计: {total} 条")
