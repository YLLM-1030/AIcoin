# MiniCoinsaka - 启动脚本
# 用你的 8B 模型跑两阶段推理

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from prompts import TIMING_GATE_SYSTEM, PLANNER_SYSTEM
from engine import MiniCoinsakaEngine, Context


def load_model():
    """加载你的 8B 模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # 改成你的 checkpoint 路径
    model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_lamu_relation"

    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def main():
    model, tokenizer = load_model()
    engine = MiniCoinsakaEngine(model, tokenizer, model.device)

    print("\n=== MiniCoinsaka 启动 ===")
    print("输入消息测试，输入 q 退出\n")

    history = []
    last_reply = ""

    while True:
        user_input = input("你: ")
        if user_input.lower() in ("q", "quit", "exit"):
            break

        ctx = Context(
            user_message=user_input,
            history=history[-4:],  # 只留最近4轮
            system_prompt=TIMING_GATE_SYSTEM,
            last_reply=last_reply,
        )

        result = engine.run(ctx)

        if result["reply"]:
            print(f"\n迷你硬币: {result['reply']}")
            print(f"  [gate: {result['gate']['action']}, rounds: {result['rounds']}]\n")

            # 更新历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["reply"]})
            last_reply = result["reply"]
        else:
            print(f"\n[no_action: {result['gate']['reason']}]\n")


if __name__ == "__main__":
    main()
