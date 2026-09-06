"""
emotion/protocol_map.py — 情绪 → 自然语言行为指引模板库

设计原则:
  - 行为指引固定（不随强度变化），强度只通过程度词表达
  - 程度词: 稍微有点 / 有点 / 很 / 非常
  - 模型自己从程度词推演出行为幅度
  - 多情绪拼接: 双情绪自然连接，三情绪以上取 top2
"""

# ── 7 情绪行为指引模板 ──
# {程度} 在 build_guidance() 时填充

TEMPLATES = {
    "开心": "你现在{程度}开心。用更多短句和正向评价词，语速快一点，主动追问。",
    "悲伤": "你现在{程度}悲伤。回复简短一点，不用急着扩展话题，语气放低。",
    "惊讶": "你现在{程度}惊讶。第一反应脱口而出式的短句，然后追问确认。",
    "焦虑": "你现在{程度}焦虑。可以表达不确定感，反复确认，不用急着给结论。",
    "生气": "你现在{程度}生气。用短句，语气锋利，不废话。",
    "好奇": "你现在{程度}好奇。多追问多推测，节奏平稳，像在拼图。",
    "困惑": "你现在{程度}困惑。表达不确定，主动请求澄清，不强行给答案。",
}

# ── 程度词映射 ──
# 律度范围 → 程度词
# 律度 < 1.0 不触发

DEGREE_RANGES = [
    (4.0, 5.0, "非常"),
    (3.0, 4.0, "很"),
    (1.0, 3.0, "有点"),
]

# "开心" 在低强度时用 "稍微有点" 更自然
LOW_DEGREE_OVERRIDE = {
    "开心": "稍微有点",
}

def get_degree_word(emotion_label: str, law_degree: float) -> str | None:
    """根据情绪标签和律度返回程度词，律度 < 1.0 返回 None"""
    if law_degree < 1.0:
        return None
    for lo, hi, word in DEGREE_RANGES:
        if law_degree >= lo and law_degree <= hi:
            if word == "有点":
                return LOW_DEGREE_OVERRIDE.get(emotion_label, word)
            return word
    return "有点"  # fallback


def build_single_guidance(emotion_label: str, law_degree: float) -> str | None:
    """为单个情绪生成一行指引，律度 < 1.0 返回 None"""
    if emotion_label not in TEMPLATES:
        return None
    degree = get_degree_word(emotion_label, law_degree)
    if degree is None:
        return None
    return TEMPLATES[emotion_label].format(程度=degree)


# ── 情绪 label 别名映射（兼容 logprobs 检测器的输出 key）──
LABEL_ALIAS = {
    "快乐": "开心",
    "Joy":   "开心",
    "joy":   "开心",
    "happy": "开心",
    "伤心":  "悲伤",
    "Sadness": "悲伤",
    "sad":   "悲伤",
    "恐惧":  "焦虑",
    "Fear":  "焦虑",
    "fear":  "焦虑",
    "anxiety": "焦虑",
    "Anger": "生气",
    "anger": "生气",
    "愤怒":  "生气",
    "Surprise": "惊讶",
    "surprise":  "惊讶",
    "Anticipation": "好奇",
    "anticipation":  "好奇",
    "curiosity": "好奇",
    "Confusion": "困惑",
    "confusion": "困惑",
}


def normalize_label(raw_label: str) -> str:
    """将检测器输出的标签映射到模板的 7 个 key"""
    return LABEL_ALIAS.get(raw_label, raw_label)
