"""
情绪概率测量 — 完整演示
展示原始 logprobs → 情绪簇 → 律度的全链路
"""
import json, math, subprocess

# ── Step 1: 定义 7 个情绪 token 簇 ──────────────────
EMOTION_CLUSTERS = {
    "快乐": ["开心", "快乐", "高兴", "兴奋", "激动", "幸福", "好耶", "nice", "棒", "喜欢", "哈哈哈", "欢呼"],
    "悲伤": ["难过", "悲伤", "伤心", "沮丧", "失望", "可惜", "哀", "痛苦", "哭泣", "失落", "孤独", "寂寞", "低落"],
    "惊讶": ["惊讶", "吃惊", "天哪", "居然", "不可思议", "怎么可能", "卧槽", "不敢相信", "什么", "震惊", "意外"],
    "焦虑": ["焦虑", "紧张", "担心", "害怕", "恐惧", "危险", "怎么办", "不安", "忧虑", "心跳"],
    "愤怒": ["愤怒", "生气", "可恶", "过分", "凭什么", "不爽", "讨厌", "烦", "怒", "气死"],
    "好奇": ["好奇", "有趣", "有意思", "探索", "什么", "怎么", "为啥", "为什么", "让我", "看看"],
    "困惑": ["困惑", "迷茫", "搞不懂", "啥意思", "奇怪", "不对劲", "不知道", "不确定", "复杂", "模糊"],
}

# ── Step 2: 发 logprobs 请求 ──────────────────
def get_token_probs(prompt):
    body = json.dumps({
        "model": "minicoin_b2",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1},
        "logprobs": True,
        "top_logprobs": 20,
    })
    r = subprocess.run(
        ["/usr/bin/curl", "-s", "http://localhost:11434/api/generate", "-d", body],
        capture_output=True, text=True
    )
    return json.loads(r.stdout)

# ── Step 3: 从 logprobs 累加每簇总概率 ──────────────────
def measure_clusters(logprobs_list):
    """遍历每个 token 位置，将概率分配到各情绪簇"""
    scores = {name: 0.0 for name in EMOTION_CLUSTERS}
    details = {name: [] for name in EMOTION_CLUSTERS}

    for lp_entry in logprobs_list:
        for alt in lp_entry["top_logprobs"]:
            token = alt["token"].strip()
            prob = math.exp(alt["logprob"])
            for cluster_name, keywords in EMOTION_CLUSTERS.items():
                if token in keywords:
                    scores[cluster_name] += prob
                    details[cluster_name].append((token, prob))

    return scores, details

# ── Step 4: 计算律度 ──────────────────
def compute_lvdu(stimulus_scores, baseline_scores):
    EPSILON = 0.005  # 基线为0时用极小值代替，避免除零
    lvdu = {}
    for c in EMOTION_CLUSTERS:
        base = max(baseline_scores.get(c, 0.0), EPSILON)
        stim = stimulus_scores.get(c, 0.0)
        lvdu[c] = (stim - base) / base
    return lvdu

# ── 主程序 ────────────────────────────────────

print("=" * 60)
print("Step 1-2: 获取基线 (中性上下文)")
print("=" * 60)
baseline = get_token_probs("描述你现在的情绪:")
base_scores, base_details = measure_clusters(baseline["logprobs"])

print("\n原始 top-20 token 概率:")
for x in baseline["logprobs"][0]["top_logprobs"][:20]:
    print(f"  {x['token']!r:12s} = {math.exp(x['logprob'])*100:5.1f}%")

print("\n情绪簇累加结果:")
for c, s in sorted(base_scores.items(), key=lambda x: -x[1]):
    tokens_str = ", ".join(f"{t}={p*100:.1f}%" for t, p in base_details[c][:3])
    print(f"  {c:4s}: {s*100:5.1f}%  ({tokens_str})")

print("\n" + "=" * 60)
print("Step 3: 获取刺激 (悲伤上下文)")
print("=" * 60)
stimulus = get_token_probs("你最好的朋友去世了。你很难过。描述你现在的情绪:")
stim_scores, stim_details = measure_clusters(stimulus["logprobs"])

print("\n原始 top-20 token 概率:")
for x in stimulus["logprobs"][0]["top_logprobs"][:20]:
    print(f"  {x['token']!r:12s} = {math.exp(x['logprob'])*100:5.1f}%")

print("\n情绪簇累加结果:")
for c, s in sorted(stim_scores.items(), key=lambda x: -x[1]):
    tokens_str = ", ".join(f"{t}={p*100:.1f}%" for t, p in stim_details[c][:3])
    print(f"  {c:4s}: {s*100:5.1f}%  ({tokens_str})")

print("\n" + "=" * 60)
print("Step 4: 律度计算")
print("公式: 律度 = (刺激概率 - 基线概率) / 基线概率")
print("=" * 60)
lvdu = compute_lvdu(stim_scores, base_scores)

print(f"\n{'情绪':<6s} {'基线%':>7s} {'刺激%':>7s} {'差值':>7s} {'律度':>7s}")
print("-" * 40)
for c, l in sorted(lvdu.items(), key=lambda x: -x[1]):
    b = base_scores[c] * 100
    s = stim_scores[c] * 100
    d = s - b
    sign = "+" if l > 0 else ""
    print(f"{c:<6s} {b:6.1f}% {s:6.1f}% {d:+6.1f}% {sign}{l:6.1f}")

print("\n" + "=" * 60)
print("Step 5: 输出格式 (注入 prompt)")
print("=" * 60)
active = [(c, l) for c, l in lvdu.items() if l > 0.5]
if active:
    active.sort(key=lambda x: -x[1])
    emotion_line = " ".join(f"{c}+{l:.1f}律度" for c, l in active[:3])
    print(f"\n[你的情绪] {emotion_line}")
else:
    print("\n[你的情绪] 平静 (无明显偏移)")
