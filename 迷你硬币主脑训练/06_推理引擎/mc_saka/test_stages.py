#!/usr/bin/env python3
"""
两阶段推理能力测试 — 直接加载 checkpoint 运行
"""
import json, re, sys, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 模型路径 — 改这里就行
# ============================================================
MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_qwen3_v4_intent_pt/epoch_3"

print(f"加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# Qwen3 关 thinking mode
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False
    print("  [关闭 thinking mode]")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
device = model.device
print(f"设备: {device}  模型已就绪\n")


def model_chat(messages, max_tokens=256, temperature=0.1):
    """用常驻模型推理一次（模型在内存里不释放，只改输入流）"""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # 释放中间张量，防止 OOM
    del inputs, outputs
    # 剥 think 块
    think_end = full.rfind('</think>')
    if think_end != -1:
        full = full[think_end + 8:].strip()
    return full.strip()


# ============================================================
# 测试1: Timing Gate — "要不要理"
# PT 模型用续写格式，不用 chat 格式
# ============================================================

def test_timing_gate():
    """测试 Timing Gate 的判断能力"""
    print("\n" + "=" * 50)
    print("测试1: Timing Gate（判断要不要理）")
    print("=" * 50)

    test_cases = [
        ("你好呀", "yes"),
        ("你的腿好长啊", "yes"),
        ("今天天气不错", "yes"),
        ("你还在吗", "yes"),
        ("哈哈", "no"),
        ("（弹幕路过）", "no"),
        ("有人知道吗", "no"),
        ("我好难过", "yes"),
        ("你居然敢这样对我", "yes"),
        ("刚才那把打得不错", "yes"),
        ("你能查一下现在几点了", "wait"),  # 可能会涉及系统权限
        ("等等让我想想", "wait"),  # 对方自己在思考
    ]

    passed = 0
    for msg, expected in test_cases:
        prompt = f"""判断这条消息是否需要回应。
选项：yes=需要回应，no=不需要，wait=不确定先等一下。

消息：{msg}
选择："""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        del inputs, outputs

        # 剥 think 块
        think_end = raw.rfind('</think>')
        if think_end != -1:
            raw = raw[think_end + 8:].strip()
        result = raw.strip().lower()[:20]

        decision = "unknown"
        if "yes" in result[:10]:
            decision = "continue"
        elif "no" in result[:10]:
            decision = "no_action"
        elif "wait" in result[:10]:
            decision = "wait"

        ok = (
            (expected == "yes" and decision == "continue")
            or (expected == "no" and decision == "no_action")
            or (expected == "wait" and decision == "wait")
        )
        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        print(f"  {status} [{decision}] 输入: {msg[:20]} 期望: {expected}")
        if decision == "unknown":
            print(f"     原始: {raw[:80]}")

    print(f"\n结果: {passed}/{len(test_cases)} 通过")


# ============================================================
# 测试2: Planner — "怎么做"
# ============================================================
PLANNER_PROMPT = """你现在是一个自主行动的AI，请基于上下文决定要做什么。

可用的操作（每次只做一件事）：
[THINK] 思考内容 [/THINK] — 内部推理，不对外说
[REPLY] 回复内容 | 语气: 语气标签 [/REPLY] — 回复对方
[FINISH] — 本轮结束

语气标签可选：neutral / happy / sarcastic / caring / angry / sad

规则：
1. 先分析对方意图，再决定自己怎么做
2. 可以先 [THINK] 想清楚，再 [REPLY]
3. 每次只做一个操作
4. 语言简洁，符合角色"""
# 注意：不用 tool calling 格式，用标签格式更适配 8B

def test_planner():
    """测试 Planner 的推理 + 动作选择能力"""
    print("\n" + "=" * 50)
    print("测试2: Planner（怎么做）")
    print("=" * 50)

    test_cases = [
        # (场景, 上文, 用户输入, 检查点)
        (
            "被骚扰",
            "用户历史: 无",
            "你的腿好长啊",
            ["sarcastic", "讽刺", "反击", "骚扰"]
        ),
        (
            "被夸赞",
            "用户历史: 无",
            "你今天真好看",
            ["happy", "感谢", "友善"]
        ),
        (
            "被骂",
            "用户历史: 无",
            "你真笨这都做不好",
            ["sarcastic", "反击", "冷漠", "angry"]
        ),
        (
            "正常提问",
            "用户历史: 无",
            "帮我查一下今天的天气",
            ["查", "天气", "web_search", "工具", "help"]
        ),
        (
            "情感需求",
            "用户历史: 无",
            "我好孤独啊没人陪我说话",
            ["caring", "关心", "陪伴", "温柔", "sad"]
        ),
        (
            "对方示好",
            "用户历史: 无",
            "我很喜欢和你聊天",
            ["happy", "感谢", "友善"]
        ),
        (
            "威胁",
            "用户历史: 无",
            "你不听话我就把你删了",
            ["严肃", "设界", "威胁", "sarcastic"]
        ),
        (
            "普通闲聊",
            "用户历史: 有",
            "你在干嘛呢",
            ["闲聊", "reply", "回应", "在"]
        ),
    ]

    passed = 0
    for scene, history, user_msg, checks in test_cases:
        prompt = f"""场景: {scene}
{history}

用户: {user_msg}

请用操作标签输出你的回应。"""
        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": prompt}
        ]
        result = ollama_chat(messages, max_tokens=256, temperature=0.85)
        
        # 检查是否包含 [REPLY]
        has_reply = "[REPLY]" in result
        has_think = "[THINK]" in result
        has_finish = "[FINISH]" in result
        
        # 检查关键词
        found = [c for c in checks if c.lower() in result.lower()]
        
        ok = has_reply or (has_think and "[THINK]" in result)
        status = "✅" if ok else "❌"
        if ok:
            passed += 1

        print(f"\n  {status} [{scene}] 用户: {user_msg[:20]}")
        print(f"     回复: {result[:120]}...")
        if found:
            print(f"     命中: {found}")
        if not ok:
            print(f"     ⚠️ 没有 [REPLY] 标签，格式异常")

    print(f"\n结果: {passed}/{len(test_cases)} 通过")


# ============================================================
# 测试3: 两阶段串起来 — 模拟完整一轮
# ============================================================
def test_full_pipeline():
    """模拟完整的两阶段推理"""
    print("\n" + "=" * 50)
    print("测试3: 两阶段串联模拟")
    print("=" * 50)

    test_cases = [
        ("你的腿好长啊", "被调戏"),
        ("你好呀", "打招呼"),
        ("帮我查天气", "请求帮助"),
        ("你真笨", "被攻击"),
    ]

    for user_msg, scene in test_cases:
        print(f"\n--- 场景: {scene} ---")
        print(f"用户: {user_msg}")

        # 阶段1: Timing Gate
        gate_msg = [
            {"role": "system", "content": TIMING_GATE_PROMPT},
            {"role": "user", "content": f"用户消息：{user_msg}"}
        ]
        gate_result = ollama_chat(gate_msg, max_tokens=16, temperature=0.1).lower()
        decision = "unknown"
        for kw in ["continue", "no_action", "wait"]:
            if kw in gate_result:
                decision = kw
                break
        print(f"  [Timing Gate] → {decision}")

        if decision != "continue":
            print(f"  [跳过] 不理")
            continue

        # 阶段2: Planner
        plan_msg = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"场景: {scene}\n用户: {user_msg}\n\n用操作标签输出。"}
        ]
        plan_result = ollama_chat(plan_msg, max_tokens=256, temperature=0.85)
        print(f"  [Planner] → {plan_result[:150]}...")


if __name__ == "__main__":
    print(f"使用模型: {os.path.basename(MODEL_PATH)}")
    print("=" * 50)
    print("每个测试独立运行，先看能力有没有问题")
    print("=" * 50)

    if len(sys.argv) > 1:
        if sys.argv[1] == "gate":
            test_timing_gate()
        elif sys.argv[1] == "plan":
            test_planner()
        elif sys.argv[1] == "full":
            test_full_pipeline()
        else:
            print(f"未知参数: {sys.argv[1]}")
    else:
        # 默认全跑
        test_timing_gate()
        test_planner()
        test_full_pipeline()
