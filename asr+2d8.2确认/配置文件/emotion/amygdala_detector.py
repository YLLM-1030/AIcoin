"""
amygdala_detector.py — 杏仁核层信号检测 (Layer 0)
基于 LeDoux 双通路理论, 使用 sentence-transformer 语义匹配。

信号类型:
  tense    — 威胁/攻击 → 身体绷紧, 警戒
  orient   — 新异/疑问 → 注意力聚焦, 歪头
  approach — 悲伤/关切 → 前倾, 想靠近
  relax    — 安心/温暖 → 放松, 开放

半衰期来源: startle reflex research (Britannica, ScienceDirect), EDM 2025
"""

import sys, os, numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from amygdala_prototypes import AMYGDALA_PROTOTYPES
from facs_engine import get_embed_model  # 共享嵌入模型

# ── 共享嵌入模型 ──
_amyg_protos = {}  # {signal_name: mean_embedding_vector}

def _get_model():
    return get_embed_model()

def _get_prototype(signal: str) -> np.ndarray:
    if signal not in _amyg_protos:
        model = _get_model()
        sentences = AMYGDALA_PROTOTYPES[signal]
        vectors = model.encode(sentences)
        _amyg_protos[signal] = np.mean(vectors, axis=0)
    return _amyg_protos[signal]

# ── 半衰期 (秒) ──
HALFLIFE = {
    "tense":    0.3,   # startle reflex: 肌肉 <1s 恢复
    "orient":   2.5,   # orienting response: 3-10s
    "approach": 1.0,   # 主动社交行为, 快恢复
    "relax":    2.0,   # 温暖感消散较慢
}


def detect_amygdala_signals(text: str) -> dict:
    """
    检测杏仁核层信号 — embedding 语义匹配

    Returns:
        {tense: 0.8, orient: 0.3, approach: 0.0, relax: 0.0}
    """
    t = text.strip()
    if not t:
        return {"tense": 0.0, "orient": 0.0, "approach": 0.0, "relax": 0.0}

    model = _get_model()
    text_vec = model.encode([t])[0]

    sims = {}
    for signal in ["tense", "orient", "approach", "relax"]:
        proto = _get_prototype(signal)
        sim = float(np.dot(text_vec, proto) /
                    (np.linalg.norm(text_vec) * np.linalg.norm(proto) + 1e-8))
        sims[signal] = max(0, sim)

    # Softmax 归一化
    sim_arr = np.array(list(sims.values()))
    exp = np.exp((sim_arr - sim_arr.max()) * 5)  # ×5 放大差异 (384维区分度不足)
    soft = exp / exp.sum()

    # 短文本也出信号 (amygdala 反应快, 不乘长度系数)
    result = {}
    for i, signal in enumerate(["tense", "orient", "approach", "relax"]):
        result[signal] = round(float(soft[i]), 3)

    # ── 互斥修正 ──
    if result["tense"] > 0.5:
        result["relax"] *= 0.3
    if result["relax"] > 0.5:
        result["tense"] *= 0.3

    return result


if __name__ == "__main__":
    tests = [
        ("你这个笨蛋!", "攻击"),
        ("你好厉害啊", "夸奖"),
        ("唉...算了", "叹气"),
        ("什么?真的假的?", "疑问"),
        ("我好害怕...", "恐惧"),
        ("今天天气真好", "中性"),
    ]
    for text, desc in tests:
        r = detect_amygdala_signals(text)
        print(f"{desc:10s} | \"{text:16s}\" | {r}")
