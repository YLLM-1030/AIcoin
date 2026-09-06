# 记忆标签引擎 — 启动脚本
# 加载 bge-base-zh-v1.5 + 标签原型 → 计算均值向量 → 提供匹配接口
#
# 优化方案（自研）：
#   分组内进行 per-dimension min-shift 去偏。
#   每个维度减去该维度在该组内的最小值，消除公共基线，保留差异。
#   优于直接减均值（R2）——不会使两个标签强制镜像对称。
#   每组独立计算基线，输入进来时也减去对应基线再算 cosine。

import sys, os, json, numpy as np

# 强制离线，禁止任何联网尝试（必须在 import transformers/sentence_transformers 之前设置）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

print("加载模型...")
model = SentenceTransformer('BAAI/bge-base-zh-v1.5', local_files_only=True)
dim = model.get_embedding_dimension()
print(f"模型加载完成, 维度={dim}")

# ── 加载标签原型 ──
TAG_PATH = "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/memory_tag_prototypes.py"
exec(open(TAG_PATH, encoding="utf-8").read())

# ── 编码原型 → 均值向量 ──
print("编码标签原型...")

domain_raw = {}  # 原始均值向量
for tag, sentences in DOMAIN_PROTOTYPES.items():
    vecs = model.encode(sentences)
    domain_raw[tag] = np.mean(vecs, axis=0)

attr_raw = {}
for attr, sentences in ATTRIBUTE_PROTOTYPES.items():
    vecs = model.encode(sentences)
    attr_raw[attr] = np.mean(vecs, axis=0)

# ── 分组 每维减最小值 去偏 ──
def min_shift_debias(vec_dict):
    """对 dict 中的所有向量，每个维度减去该维度上的最小值"""
    tag_list = list(vec_dict.keys())
    if len(tag_list) < 2:
        return vec_dict, None
    vecs = np.array([vec_dict[t] for t in tag_list])  # (n_tags, 768)
    per_dim_min = np.min(vecs, axis=0)  # (768,) 每个维度上的最小值
    debiased = {}
    for t, v in vec_dict.items():
        debiased[t] = v - per_dim_min
    return debiased, per_dim_min

print("每维减最小值去偏...")

# 领域一组
domain_vectors, domain_min = min_shift_debias(domain_raw)
print(f"  领域组 (10个标签)")

# 属性分三组
eval_raw = {k: attr_raw[k] for k in ["我喜欢", "我不喜欢"]}
scale_raw = {k: attr_raw[k] for k in ["重大", "普通", "轻微"]}
time_raw = {k: attr_raw[k] for k in ["过去", "现在", "未来"]}

eval_vectors, eval_min = min_shift_debias(eval_raw)
scale_vectors, scale_min = min_shift_debias(scale_raw)
time_vectors, time_min = min_shift_debias(time_raw)

# ── 打印每组 u（min-shift 基线）的统计 ──
def report_u(name, u, raw_vectors):
    print(f"\n[{name}] u 统计:")
    print(f"  维度数: {len(u)}")
    print(f"  u 最小值: {u.min():.4f}")
    print(f"  u 最大值: {u.max():.4f}")
    print(f"  u 均值:   {u.mean():.4f}")
    print(f"  |u| 范数: {np.linalg.norm(u):.4f}")
    # 打印原始向量范围对比
    vecs = np.array(list(raw_vectors.values()))
    print(f"  原始向量 均值范围: [{vecs.mean(axis=0).min():.4f}, {vecs.mean(axis=0).max():.4f}]")
    # u 中接近 0 的维度占比（极小值说明该维没被优化）
    near_zero = (np.abs(u) < 0.01).mean()
    print(f"  u 中 |值|<0.01 的维度占比: {near_zero*100:.1f}%")
    # u 中绝对值大的维度
    big_idx = np.argsort(np.abs(u))[-5:][::-1]
    print(f"  |u| 最大的 5 个维度: {big_idx.tolist()}, 值: {u[big_idx].round(4).tolist()}")
    # 各标签原始范数
    for t, v in raw_vectors.items():
        print(f"  {t}: 原始|v|={np.linalg.norm(v):.4f}, 去偏后|v|={np.linalg.norm(v - u):.4f}")

report_u("规模(重大/普通/轻微)", scale_min, scale_raw)
report_u("时间(过去/现在/未来)", time_min, time_raw)
report_u("评价(我喜欢/我不喜欢)", eval_min, eval_raw)
report_u("领域(10个)", domain_min, domain_raw)

# ── 匹配函数 ──
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def match(text):
    """返回 (最佳领域, {领域:分}, {属性:分})"""
    vec = model.encode([text])[0]

    # 领域匹配（每维减该组最小值）
    v_domain = vec - domain_min
    domain_scores = {tag: cosine_sim(v_domain, tv) for tag, tv in domain_vectors.items()}

    # 评价匹配
    v_eval = vec - eval_min
    eval_scores = {attr: cosine_sim(v_eval, av) for attr, av in eval_vectors.items()}

    # 规模匹配
    v_scale = vec - scale_min
    scale_scores = {attr: cosine_sim(v_scale, av) for attr, av in scale_vectors.items()}

    # 时间匹配
    v_time = vec - time_min
    time_scores = {attr: cosine_sim(v_time, av) for attr, av in time_vectors.items()}

    best_domain = max(domain_scores, key=domain_scores.get)
    attr_scores = {**eval_scores, **scale_scores, **time_scores}

    return best_domain, domain_scores, attr_scores

# ── 交互测试 ──
print("\n" + "="*50)
print("记忆标签引擎已启动, 输入文本测试匹配效果")
print("输入 exit 退出")
print("="*50)

while True:
    # 读多行，直到空行
    lines = []
    while True:
        raw = sys.stdin.readline()
        if not raw:
            break
        line = raw.rstrip('\n')
        if line == '':
            break
        lines.append(line)
    if not lines:
        continue
    text = '\n'.join(lines)
    if text.lower() == "exit":
        break
    print(f"\n[收到输入] {len(text)}字符, {len(lines)}行")

    domain, domain_scores, attr_scores = match(text)

    print(f"\n领域匹配:")
    print(f"  → 最佳: {domain}")
    sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])[:3]
    for k, v in sorted_d:
        print(f"    {k}: {v:.4f}")

    print(f"\n属性匹配:")
    sorted_a = sorted(attr_scores.items(), key=lambda x: -x[1])
    for k, v in sorted_a:
        print(f"    {k}: {v:.4f}")
