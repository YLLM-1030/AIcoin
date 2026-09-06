#!/usr/bin/env python3
"""调试 find_repeat_6x — 修复版：返回实际匹配位置"""
import sys

def find_repeat_6x(t):
    n = len(t)
    for L in range(2, 150):
        need_min = 6 if L <= 4 else 2
        if n < L * need_min: continue
        for start in range(0, min(L, n - L * need_min + 1)):
            count = 1; last_pos = start
            for pos in range(start + L, n - L + 1, L):
                if t[pos:pos+L] == t[last_pos:last_pos+L]:
                    count += 1
                    if count >= need_min:
                        chunk = t[last_pos:last_pos+L]
                        return True, L, chunk, last_pos, pos, count
                else:
                    count = 1; last_pos = pos
    return False, 0, "", 0, 0, 0

def debug(t):
    n = len(t)
    print(f"文本({n}字): {t}")
    found, L, chunk, p1, p2, cnt = find_repeat_6x(t)
    if found:
        print(f"✅ L={L} 连续{cnt}次  '{chunk}'")
        print(f"   第一次: 位置{p1} '{t[p1:p1+L]}'")
        print(f"   第二次: 位置{p2} '{t[p2:p2+L]}'")
    else:
        print("  无匹配")
    print()

texts = [
    "对方问我的名字。我应该直接告诉他我的名字是小智。不需要解释，直接回答名字。然后问他想让我做什么。这样可以自然地引导他给出",
    "对方想让我说明自己被起名的原因。我应该坦率地说明自己是被起名的。我应该简洁地说明这个过程。对方可能想知道我的来历，所以我",
    "用户想知道我的名字和起名者。我应该先回答名字和起名者。然后他问我要做什么。我应该说明我的意图。所以先回答他的问题，然后说",
    "对方在问我的名字和创造者。我要解释我的名字来自我自己的决定，创造者是迷你硬币。然后我要引导对方进入下一步，比如询问我的名",
]

for i, t in enumerate(texts):
    print(f"{'='*60}")
    print(f"测试 {i+1}")
    print(f"{'='*60}")
    debug(t)
