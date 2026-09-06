# memory_retrieval.py — 长期记忆检索 pipeline（测试版）
# 输入：别人说一句话 + 硬币说一句话
# 流程：
#   1. BERT 分类完整对话 → 各属性标签
#   2. BERT 分类硬币单独回复 → 各属性标签
#   3. 重心位移 Δ = 硬币重心 - 全量重心（评价/规模/时间三轴）
#   4. 主移动方向 = |Δ| 最大的轴，正负号 = 方向
#   5. 走一步：全量标签沿主移动方向偏移一级 → 目标检索空间
#   6. SQLite 按标签过滤候选集（严格到宽松逐级降级）
#   7. bge embedding 余弦相似度 → top-1 记忆
# 运行：WSL 里 python3 memory_retrieval.py

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE = '/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆'
sys.path.insert(0, BASE)

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from train_bert import MultiTaskClassifier, DOMAIN_LABELS, SCALE_LABELS, TIME_LABELS, EVAL_LABELS
from memory_db import MemoryDB

MODEL_PATH = '/home/autogram-coin/coin_models/chinese-roberta-wwm-ext'
SAVE_DIR = '/home/autogram-coin/coin_models/domain_classifier'
DB_PATH = BASE + '/memories.db'
MAX_LEN = 128

print("加载 BERT 分类器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = MultiTaskClassifier(MODEL_PATH, len(DOMAIN_LABELS), len(SCALE_LABELS),
                            len(TIME_LABELS), len(EVAL_LABELS))
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model.pt'), map_location='cuda'))
model.cuda().eval()

print("加载 bge embedding...")
embed = SentenceTransformer('BAAI/bge-base-zh-v1.5', local_files_only=True)

print("连接记忆库...")
db = MemoryDB(DB_PATH, embed)
print(f"记忆库共 {db.count()} 条")

# ── 属性坐标轴（重心位移用）──
# 评价轴: 我不喜欢(-1) ─ 无(0) ─ 我喜欢(+1)
# 规模轴: 普通(0) ─ 中等(0.5) ─ 重大(+1)
# 时间轴: 过去(-1) ─ 现在(0) ─ 未来(+1)
AXES = {
    '规模': {'普通': 0.0, '中等': 0.5, '重大': 1.0},
    '时间': {'过去': -1.0, '现在': 0.0, '未来': 1.0},
    '评价': {'我不喜欢': -1.0, '无': 0.0, '我喜欢': 1.0},
}
ORDERS = {
    '规模': ['普通', '中等', '重大'],
    '时间': ['过去', '现在', '未来'],
    '评价': ['我不喜欢', '无', '我喜欢'],
}
DIM_LABELS = {'规模': SCALE_LABELS, '时间': TIME_LABELS, '评价': EVAL_LABELS}


def classify(text):
    """返回 {维度: {标签: 概率}}"""
    enc = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True,
                    return_tensors='pt')
    with torch.no_grad():
        d, s, t, e = model(enc['input_ids'].cuda(), enc['attention_mask'].cuda())

    def soft(labels, x):
        p = torch.softmax(x[0], dim=-1)
        return {lb: float(v) for lb, v in zip(labels, p.tolist())}

    return {
        '领域': soft(DOMAIN_LABELS, d),
        '规模': soft(SCALE_LABELS, s),
        '时间': soft(TIME_LABELS, t),
        '评价': soft(EVAL_LABELS, e),
    }


def top1(probs):
    return max(probs.items(), key=lambda kv: kv[1])


def centroid(probs, axis):
    return sum(p * axis[lb] for lb, p in probs.items())


def step_label(base_label, delta, order):
    """全量标签沿 delta 方向走一级；|delta| 太小则不动"""
    if abs(delta) < 0.05:
        return base_label, False
    i = order.index(base_label)
    j = max(0, min(len(order) - 1, i + (1 if delta > 0 else -1)))
    return order[j], j != i


def retrieve(user_text, coin_text):
    full_text = f'{user_text}。{coin_text}'
    print(f"\n完整对话: {full_text}")

    pf = classify(full_text)
    pc = classify(coin_text)

    print("\n[完整对话分类]")
    for dim in ['领域', '规模', '时间', '评价']:
        lb, p = top1(pf[dim])
        print(f"  {dim}: {lb} ({p:.1%})")
    print("[硬币回复分类]")
    for dim in ['领域', '规模', '时间', '评价']:
        lb, p = top1(pc[dim])
        print(f"  {dim}: {lb} ({p:.1%})")

    # 重心位移
    print("\n[重心位移 Δ = 硬币重心 - 全量重心]")
    delta = {}
    for dim in ['规模', '时间', '评价']:
        c_full = centroid(pf[dim], AXES[dim])
        c_coin = centroid(pc[dim], AXES[dim])
        delta[dim] = c_coin - c_full
        print(f"  {dim}: 全量{c_full:+.2f} → 硬币{c_coin:+.2f}  Δ={delta[dim]:+.2f}")

    main_dim = max(delta, key=lambda d: abs(delta[d]))
    print(f"\n主移动: {main_dim} (Δ={delta[main_dim]:+.2f}, "
          f"{'正方向' if delta[main_dim] > 0 else '负方向'})")

    # 走一步：领域用全量对话的（用户话题），属性沿 Δ 方向走一级
    domain, _ = top1(pf['领域'])
    target = {'领域': domain}
    print(f"\n[走一步] 领域基准: {domain} (来自完整对话)")
    for dim in ['规模', '时间', '评价']:
        base, _ = top1(pf[dim])
        stepped, moved = step_label(base, delta[dim], ORDERS[dim])
        target[dim] = stepped
        print(f"  {dim}: {base} → {stepped}{' (移动)' if moved else ' (不动)'}")

    print(f"\n目标检索空间: {target}")

    # SQLite 过滤：严格 → 宽松 逐级降级
    combos = [
        [('domain', target['领域']), ('scale', target['规模']),
         ('time', target['时间']), ('eval', target['评价'])],
        [('domain', target['领域']), ('scale', target['规模']), ('time', target['时间'])],
        [('domain', target['领域']), ('scale', target['规模'])],
        [('domain', target['领域'])],
        [],
    ]
    for combo in combos:
        kwargs = {k: v for k, v in combo}
        res = db.search(full_text, top_k=3, **kwargs)
        if res:
            desc = combo if combo else '（全部记忆）'
            print(f"\n命中空间: {desc} → 候选 {len(res)} 条, top1:")
            for mid, mtext, cos in res:
                print(f"  [{mid}] sim={cos:.4f} | {mtext[:70]}")
            return res[0]
    print("\n无任何命中")
    return None


if __name__ == '__main__':
    print("=" * 55)
    print("长期记忆检索 pipeline 测试")
    print("输入两行：别人说 / 硬币说（exit 退出）")
    print("=" * 55)
    while True:
        u = input("\n别人说: ").strip()
        if u.lower() == 'exit':
            break
        c = input("硬币说: ").strip()
        if c.lower() == 'exit':
            break
        if not u or not c:
            continue
        try:
            retrieve(u, c)
        except Exception as ex:
            import traceback
            traceback.print_exc()
