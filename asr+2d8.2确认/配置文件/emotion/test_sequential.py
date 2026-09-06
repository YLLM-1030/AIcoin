"""
test_sequential.py — 连续测量测试，验证 _previous_stimulus 差分逻辑
"""
import sys
sys.path.insert(0, ".")

from emotion import detector

# 手动清空状态 (模拟重启)
detector._previous_stimulus = None

contexts = [
    "用户: 今天天气真好。",
    "用户: 我刚发现了一个惊人的秘密！",
    "用户: 这个秘密太可怕了。",
    "用户: 我还是很害怕。",
]

for i, ctx in enumerate(contexts):
    print(f"\n{'#'*60}")
    print(f"# 第{i+1}轮: {ctx[:30]}...")
    print(f"{'#'*60}")

    lvdu = detector.measure(ctx)
    for c, l in sorted(lvdu.items(), key=lambda x: -x[1]):
        sign = "+" if l > 0 else ""
        bar = "█" * min(10, int(abs(l)))
        print(f"  {c:4s}: {sign}{l:6.1f} {bar}")

    emotion, intensity = detector.get_top_emotion()
    print(f"  → 主导情绪: {emotion} (强度: {intensity:.1f})")
