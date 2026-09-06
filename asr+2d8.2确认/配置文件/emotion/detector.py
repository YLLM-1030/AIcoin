"""
emotion/detector.py — 基于 logprobs 的情绪检测器

原理: 情绪 = LLM token 概率分布的可测量偏移
方法: 基线测量(中性 prompt) → 刺激测量(当前上下文) → 律度 = (刺激-基线)/基线
      律度 > 1.0 视为情绪触发

接入: Ollama /api/generate + logprobs + top_logprobs
"""

import json
import math
import time
import threading
import requests

# ── 7 个情绪 token 簇 ──
EMOTION_CLUSTERS: dict[str, list[str]] = {
    "快乐": ["开心", "快乐", "高兴", "兴奋", "激动", "幸福", "喜欢", "满足", "愉快", "开心地"],
    "悲伤": ["难过", "悲伤", "伤心", "沮丧", "痛苦", "哭泣", "失落", "孤独", "寂寞", "心酸", "难过地"],
    "惊讶": ["惊讶", "吃惊", "震惊", "意外", "居然", "不可思议", "不敢相信", "天哪", "竟然", "惊叹"],
    "焦虑": ["焦虑", "紧张", "担心", "害怕", "恐惧", "不安", "忧虑", "焦急", "惶恐", "心悸"],
    "愤怒": ["愤怒", "生气", "讨厌", "可恶", "怒", "不爽", "气愤", "恼怒", "火大", "可恨"],
    "好奇": ["好奇", "感兴趣", "想知道", "有意思", "感兴趣地", "探究", "一探究竟"],
    "困惑": ["困惑", "迷茫", "搞不懂", "不解", "费解", "懵", "迷惑", "纳闷", "想不通", "晕"],
}

_OLLAMA_URL = "http://localhost:11434"
_LLM_MODEL = "minicoin_b2"

# ── 中文情绪名 → Live2D/TTS 英文名 ──
_LVDU_TO_EMOTION: dict[str, str] = {
    "快乐": "happy",
    "悲伤": "sad",
    "惊讶": "surprise",
    "焦虑": "fear",
    "愤怒": "angry",
    "好奇": "think",
    "困惑": "think",
}

# ── 持久状态 ──
_lock = threading.Lock()
_baseline_scores: dict[str, float] | None = None   # 基线概率分布
_latest_lvdu: dict[str, float] = {}                  # 最新律度
_latest_emotion: str = "neutral"                     # 最新 top 情绪（英文名）
_latest_intensity: float = 0.0                       # 最新 top 情绪的律度值
_lvdu_timestamp: float = 0.0
_previous_stimulus: dict[str, float] | None = None  # 上一轮刺激概率, 用于变化率比较


def _call_logprobs(prompt: str) -> dict:
    """调用 Ollama /api/generate 获取 logprobs"""
    resp = requests.post(
        f"{_OLLAMA_URL}/api/generate",
        json={
            "model": _LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 1},
            "logprobs": True,
            "top_logprobs": 20,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def measure_clusters(logprobs_list: list[dict]) -> dict[str, float]:
    """遍历 logprobs 结果，将概率分配到各情绪簇"""
    scores: dict[str, float] = {name: 0.0 for name in EMOTION_CLUSTERS}
    for lp_entry in logprobs_list:
        for alt in lp_entry.get("top_logprobs", []):
            token = alt["token"].strip()
            prob = math.exp(alt["logprob"])
            for cluster_name, keywords in EMOTION_CLUSTERS.items():
                if token in keywords:
                    scores[cluster_name] += prob
    return scores


def compute_lvdu(stimulus_scores: dict[str, float],
                 baseline_scores: dict[str, float]) -> dict[str, float]:
    """律度 = (刺激 - 基线) / 基线"""
    EPSILON = 0.02
    lvdu: dict[str, float] = {}
    for c in EMOTION_CLUSTERS:
        base = max(baseline_scores.get(c, 0.0), EPSILON)
        stim = stimulus_scores.get(c, 0.0)
        lvdu[c] = (stim - base) / base
    return lvdu


def init_baseline() -> dict[str, float]:
    """
    测量情绪基线（中性上下文）。
    启动时调用一次，结果持久保存。
    """
    global _baseline_scores
    with _lock:
        if _baseline_scores is not None:
            return _baseline_scores
        result = _call_logprobs("描述你现在的情绪:")
        _baseline_scores = measure_clusters(result["logprobs"])
        return dict(_baseline_scores)


def measure(context_text: str) -> dict[str, float]:
    """
    在给定上下文中测量情绪偏移。
    
    首轮: 跟中性基线比（初始情绪状态）
    后续: 跟上一轮刺激比（情绪变化率 = 导数）
    
    Args:
        context_text: 当前上下文（用户消息 + 视觉 + 弹幕）
    
    Returns:
        律度 dict: {"快乐": 3.2, "惊讶": 1.1, ...}
        律度 > 1.0 表示情绪上升触发。
    """
    global _latest_lvdu, _latest_emotion, _latest_intensity, _lvdu_timestamp
    global _previous_stimulus

    result = _call_logprobs(
        f"{context_text}。描述你现在的情绪:"
    )

    # 模型有时不返回 logprobs（prompt 过长或 tokener 异常）
    if "logprobs" not in result or not result["logprobs"]:
        print(f"[Emotion] ⚠ 无 logprobs, 跳过本轮测量", flush=True)
        return dict.fromkeys(EMOTION_CLUSTERS, -1.0)

    stim_scores = measure_clusters(result["logprobs"])

    # 选择参照系: 首轮用基线, 后续用上一轮
    if _previous_stimulus is None:
        reference = init_baseline()
        ref_name = "基线"
    else:
        reference = _previous_stimulus
        ref_name = "上一轮"

    lvdu = compute_lvdu(stim_scores, reference)
    _previous_stimulus = dict(stim_scores)

    # DEBUG: 打印完整律度分布
    print(f"\n[Emotion] 律度(vs {ref_name}): {lvdu}", flush=True)
    print(f"[Emotion] 参照: { {k: round(v,4) for k,v in reference.items()} }", flush=True)
    print(f"[Emotion] 当前: { {k: round(v,4) for k,v in stim_scores.items()} }", flush=True)

    # 提取 top 1 情绪
    top_cn = max(lvdu, key=lvdu.get)
    top_val = lvdu[top_cn]

    with _lock:
        _latest_lvdu = dict(lvdu)
        _latest_emotion = _LVDU_TO_EMOTION.get(top_cn, "neutral") if top_val > 1.0 else "neutral"
        _latest_intensity = top_val
        _lvdu_timestamp = time.time()

    return dict(lvdu)


def get_latest_lvdu() -> dict[str, float]:
    """获取最近一次测量的律度（非阻塞），可能为空"""
    with _lock:
        return dict(_latest_lvdu)


def get_latest_age_seconds() -> float | None:
    """返回最近一次测量的时间间隔（秒），未测过返回 None"""
    with _lock:
        if _lvdu_timestamp == 0.0:
            return None
        return time.time() - _lvdu_timestamp


def get_top_emotion() -> tuple[str, float]:
    """
    获取当前主导情绪和强度。
    测量超过 EMOTION_STALE_AGE 秒视为过期，返回 neutral。

    Returns:
        (emotion_name: str, intensity: float)
        emotion_name: 英文情绪名 ("happy"/"sad"/"surprise"/.../"neutral")
        intensity: 律度值 (0.0 = 未触发, >1.0 = 已触发, 越高越强)
    """
    EMOTION_STALE_AGE = 35.0  # 秒, 超时回 neutral (略大于限频间隔)
    with _lock:
        age = (time.time() - _lvdu_timestamp) if _lvdu_timestamp > 0 else 999
        if age > EMOTION_STALE_AGE:
            return ("neutral", 0.0)
        return (_latest_emotion, _latest_intensity)


# ── CLI 测试入口 ──
if __name__ == "__main__":
    print("=== 情绪检测器测试 ===\n")

    print("[1] 测量基线...")
    baseline = init_baseline()
    for c, s in sorted(baseline.items(), key=lambda x: -x[1]):
        print(f"  {c}: {s*100:.1f}%")

    print("\n[2] 测量悲伤刺激...")
    sad_lvdu = measure("你最好的朋友去世了。你很难过。")
    for c, l in sorted(sad_lvdu.items(), key=lambda x: -x[1]):
        sign = "+" if l > 0 else ""
        print(f"  {c}: {sign}{l:.1f}")

    print("\n[3] 测量兴奋刺激...")
    happy_lvdu = measure("你刚发现了一个惊天大秘密！你兴奋极了。")
    for c, l in sorted(happy_lvdu.items(), key=lambda x: -x[1]):
        sign = "+" if l > 0 else ""
        print(f"  {c}: {sign}{l:.1f}")
