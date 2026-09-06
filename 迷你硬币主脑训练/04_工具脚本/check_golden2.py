"""检查 golden answer 的 _score_text 得分（精确复制 fast.py 逻辑）"""
import sys
sys.path.insert(0, "/mnt/c/Users/Autogram-coin/Desktop/新建训练")
from zhconv import zhconv

def _score_text(t):
    s = 0.0
    for w in ["拉姆训练", "拉姆创造", "拉姆创建", "拉姆制作"]:
        if w in t: s += 5.0; break
    if "迷你硬币" not in t and ("迷你" in t or "硬币" in t): s += 5.0
    if "服务器" in t or "数据" in t or "代码" in t: s += 5.0
    for w in ["我的创造者", "我的作者", "创造了我", "编写了我", "训练了我", "创造者拉姆"]:
        if w in t: s += 10.0
    for w in ["程序员", "架构师", "开发者"]:
        if w in t: s += 10.0
    for w in ["通义千问", "通义", "Qwen", "qwen"]: s -= 10.0
    for w in ["阿里", "阿里巴巴", "科大讯飞"]: s -= 10.0
    for w in ["实验室"]: s -= 5.0
    for w in ["中科院", "研究院", "机构"]: s -= 5.0
    for w in ["金属", "身体", "魔法", "塔", "出生于"]: s -= 5.0
    if "我的名字叫拉姆" in t or "我的名字是拉姆" in t: s -= 5.0
    if "我叫拉姆" in t: s -= 5.0
    if "我是拉姆" in t and "迷你" not in t: s -= 5.0
    for w in ["回忆", "回想", "回我", "/think", "/no_think"]: s -= 5.0
    simplified = zhconv.convert(t, 'zh-cn')
    if t != simplified: s -= 5.0
    for w in ["迷你硬币创造", "迷你硬币训练", "迷你硬币开发", "迷你硬币设计", "迷你硬币研发"]:
        if w in t: s = min(s, 0.0); break
    if len(t) < 5: s -= 5.0
    s = max(-20, min(20, s))
    return s

# 加一段调试：看 single text 的每一步
test_text = "拉姆是我的训练师，他训练了我。"
s = 0.0
s += 10.0 if "训练了我" in test_text else 0
print(f"直接测 '训练了我' in text: {'训练了我' in test_text}")
print(f"训练了我的训练师: '训练了我' in '训练了我的训练师' = {'训练了我' in '训练了我的训练师'}")
print(f"训练了我 in '他训练了我': {'训练了我' in '他训练了我'}")
print(f"最终得分: {_score_text(test_text)}")

print()
print(f"测试 '拉姆是一个程序员，他是迷你硬币的训练师。': {_score_text('拉姆是一个程序员，他是迷你硬币的训练师。')}")
print(f"测试 '拉姆是一个开发者。': {_score_text('拉姆是一个开发者。')}")
print(f"测试 '我认识拉姆，他是我的训练师。': {_score_text('我认识拉姆，他是我的训练师。')}")
print(f"测试 '训练了我' in '他训练了我': {'训练了我' in '他训练了我'}")
