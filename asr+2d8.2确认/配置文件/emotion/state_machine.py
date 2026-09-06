"""
emotion/state_machine.py — 情绪状态机

输入: 情绪律度 dict {"开心": 3.2, "惊讶": 1.1, ...}
输出: 自然语言行为指引字符串

数据来源: detector.get_latest_lvdu() (logprobs 检测器异步队列)
引导构建: protocol_map (模板 + 程度词)
"""

try:
    from . import protocol_map
    from . import detector
except ImportError:
    import protocol_map  # type: ignore
    import detector      # type: ignore


def get_current_emotions() -> dict[str, float]:
    """
    获取当前情绪律度。
    从 detector 的异步队列读取最近一次测量结果，非阻塞。
    如果检测器尚未初始化，返回空 dict（不注入指引）。
    """
    return detector.get_latest_lvdu()


def build_guidance(emotions: dict[str, float] | None = None) -> str | None:
    """
    构建情绪行为指引文本。

    Args:
        emotions: {"开心": 3.2, "惊讶": 1.1, ...}
                  如果为 None，从 get_current_emotions() 获取

    Returns:
        自然语言指引字符串。没有触发情绪时返回 None。
    """
    if emotions is None:
        emotions = get_current_emotions()

    triggered: list[tuple[str, float, str]] = []

    for raw_label, law_degree in emotions.items():
        label = protocol_map.normalize_label(raw_label)
        guidance = protocol_map.build_single_guidance(label, law_degree)
        if guidance:
            triggered.append((label, law_degree, guidance))

    if not triggered:
        return None

    triggered.sort(key=lambda x: x[1], reverse=True)

    if len(triggered) == 1:
        return triggered[0][2]

    top2 = triggered[:2]
    primary = top2[0][2]
    secondary = top2[1][2]

    if primary.endswith("。"):
        primary = primary[:-1]
    return primary + "，" + secondary


# ── CLI 测试入口 ──
if __name__ == "__main__":
    print("=== 情绪状态机测试 ===\n")

    print("单情绪 (开心+3.2):")
    print("  ", build_guidance({"开心": 3.2}))

    print("\n双情绪 (开心+3.2, 惊讶+1.1):")
    print("  ", build_guidance({"开心": 3.2, "惊讶": 1.1}))

    print("\n弱情绪 (全部 < 1.0):")
    print("  ", build_guidance({"开心": 0.5, "惊讶": 0.3}))

    print("\n从检测器读取:")
    print("  ", build_guidance())

    print("\n三情绪 top2 截断 (开心+2, 好奇+1.5, 焦虑+1.3):")
    print("  ", build_guidance({"开心": 2.0, "好奇": 1.5, "焦虑": 1.3}))
