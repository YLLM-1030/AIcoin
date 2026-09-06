# -*- coding: utf-8 -*-
"""下桥清洗 _sanitize_reply + 脏表情兜底 _push_reply_to_queues 纯函数单测
用法: python test_sanitize.py  （无需任何服务，import pipeline_server 取函数）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "输入中心"))
import pipeline_server as P

PASS = FAIL = 0
def check(name, got, want):
    global PASS, FAIL
    ok = (got == want)
    PASS += ok; FAIL += (not ok)
    print(f"{'✅' if ok else '❌'} {name}")
    if not ok:
        print(f"   期望: {want!r}")
        print(f"   实际: {got!r}")

# ── 1. _sanitize_reply ──
print("=" * 60)
print("1. _sanitize_reply 下桥清洗")
print("=" * 60)

# ① 复读删光
check("尾巴复读删光",
      P._sanitize_reply("今天天气不错。好的好的好的好的好的"),
      "今天天气不错。")
check("6x 任意重复删除",
      P._sanitize_reply("我超喜欢这个。重复重复重复重复重复重复"),
      "我超喜欢这个。")
# 白名单语气词放行
check("白名单语气词放行(哈)",
      P._sanitize_reply("哈哈哈哈"),
      "哈哈哈哈")
check("白名单语气词放行(呜呜)",
      P._sanitize_reply("呜呜呜呜"),
      "呜呜呜呜")

# ② 嵌套工具：arg 内 [TOOL:搜索] 非法 → 删到闭合前
check("嵌套工具清理",
      P._sanitize_reply("[TOOL:回复]今天天气不错[TOOL:搜索]北京[TOOL][TOOL]"),
      "[TOOL:回复]今天天气不错[TOOL]")
# ③ 未闭合 → 补 [TOOL]
check("未闭合补闭合",
      P._sanitize_reply("[TOOL:回复]今天天气不错"),
      "[TOOL:回复]今天天气不错[TOOL]")
# ④ 孤立 [TOOL] 删除
check("孤立TOOL删除",
      P._sanitize_reply("你好[TOOL]世界"),
      "你好世界")
# 花式假闭合 [TO/]
check("花式假闭合[TO/]清理",
      P._sanitize_reply("[TOOL:回复]你好[TO/]"),
      "[TOOL:回复]你好[TOOL]")

# ⑤ 非法表情 → 整个工具丢弃
check("非法表情丢弃(害羞)",
      P._sanitize_reply("她问我了。我好害羞。[TOOL:表情]害羞[TOOL]"),
      "她问我了。我好害羞。")
check("非法表情丢弃(生气)",
      P._sanitize_reply("[TOOL:表情]生气[TOOL][TOOL:回复]哼[TOOL]"),
      "[TOOL:回复]哼[TOOL]")
# 合法表情保留
check("合法表情保留(wink)",
      P._sanitize_reply("[TOOL:表情]wink[TOOL]"),
      "[TOOL:表情]wink[TOOL]")
check("合法表情保留(转头)",
      P._sanitize_reply("[TOOL:表情]转头[TOOL]"),
      "[TOOL:表情]转头[TOOL]")

# 综合：thought + 回复 + 表情 + 脏表情混合
check("混合场景(脏表情夹在中间被删)",
      P._sanitize_reply("她在夸我诶。好开心。[TOOL:表情]害羞[TOOL][TOOL:回复]谢谢夸奖[TOOL][TOOL:表情]开心[TOOL]"),
      "她在夸我诶。好开心。[TOOL:回复]谢谢夸奖[TOOL][TOOL:表情]开心[TOOL]")
# 空输入
check("空输入返回空",
      P._sanitize_reply(""),
      "")

# ── 2. _push_reply_to_queues 脏表情兜底 ──
print("=" * 60)
print("2. _push_reply_to_queues 非法表情兜底")
print("=" * 60)

P.express_queue.clear()
P._push_reply_to_queues("[TOOL:回复]你好呀[TOOL][TOOL:表情]害羞[TOOL][TOOL:表情]wink[TOOL]")
cms = [it["cmd"] for it in P.express_queue]
args = [it["arg"] for it in P.express_queue]
check("脏表情不入队(只进回复+wink)", cms, ["回复", "表情"])
check("入队表情是wink", args[1], "wink")

P.express_queue.clear()
P._push_reply_to_queues("[TOOL:表情]傲娇[TOOL]")  # 全脏
check("全脏表情回复 → 队列为空", len(P.express_queue), 0)

# ── 3. VALID_EMOTIONS 与系统提示一致性 ──
print("=" * 60)
print("3. 合法表情集")
print("=" * 60)
print(f"   VALID_EMOTIONS = {sorted(P.VALID_EMOTIONS)}")
check("六种标准表情完整", sorted(P.VALID_EMOTIONS),
      sorted(["wink", "spin", "开心", "认真", "转头", "担心"]))

print("=" * 60)
print(f"结果: {PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
