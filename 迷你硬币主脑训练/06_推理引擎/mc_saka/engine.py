# MiniCoinsaka - 推理引擎核心
# 两阶段推理：Timing Gate → Planner

import json
import logging
import random
import time
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("MiniCoinsaka")

@dataclass
class Context:
    """当前对话上下文"""
    user_message: str
    history: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    last_reply: str = ""


class MiniCoinsakaEngine:
    """抄袭 MaiBot Maisaka 的推理引擎"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_internal_rounds = 10

    # ========== 阶段1: Timing Gate ==========

    def timing_gate(self, ctx: Context) -> dict:
        """轻量判断：要不要理"""
        from prompts import TIMING_GATE_SYSTEM, TIMING_GATE_TOOLS

        messages = [
            {"role": "system", "content": TIMING_GATE_SYSTEM},
        ]
        if ctx.history:
            for msg in ctx.history[-4:]:
                messages.append(msg)
        messages.append({"role": "user", "content": ctx.user_message})

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt += '{\n  "function": "'

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        result = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 解析工具调用
        if "continue" in result:
            return {"action": "continue", "reason": "需要回应"}
        elif "no_action" in result:
            return {"action": "no_action", "reason": "无需回应"}
        elif "wait" in result:
            import re
            m = re.search(r'(\d+)', result)
            sec = int(m.group(1)) if m else 5
            return {"action": "wait", "seconds": min(sec, 10), "reason": f"等待{sec}秒"}
        else:
            # 默认继续，宁多勿漏
            return {"action": "continue", "reason": "默认继续"}

    # ========== 阶段2: Planner ==========

    def plan_and_execute(self, ctx: Context) -> dict:
        """完整推理：想→做→想→做...直到finish"""
        from prompts import PLANNER_SYSTEM, PLANNER_TOOLS

        inner_history = []
        final_reply = None

        # 注入上下文
        context_block = f"""当前上下文：
用户说：{ctx.user_message}
"""
        if ctx.history:
            context_block += "历史消息(最近2轮)：\n"
            for msg in ctx.history[-2:]:
                context_block += f"  {msg['role']}: {msg['content']}\n"
        if ctx.last_reply:
            context_block += f"上次我说：{ctx.last_reply}\n"

        inner_history.append({"role": "user", "content": context_block})

        for round_idx in range(self.max_internal_rounds):
            messages = [
                {"role": "system", "content": f"{ctx.system_prompt}\n\n{PLANNER_SYSTEM}"}
            ] + inner_history

            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt += '{"function": "'

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            raw = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            # 解析动作
            action = self._parse_action(raw)

            if action["name"] == "think":
                inner_history.append({
                    "role": "assistant",
                    "content": f"[思考] {action.get('thought', '')}"
                })
                continue  # 继续下一轮

            elif action["name"] == "reply":
                final_reply = action.get("content", "")
                inner_history.append({
                    "role": "assistant",
                    "content": f"[回复] {final_reply} 语气:{action.get('tone', 'neutral')}"
                })
                # MaiBot 的模式是：回复后继续看是否还需要做什么
                continue

            elif action["name"] == "finish":
                break

            else:
                break

        return {
            "reply": final_reply or "",
            "rounds": round_idx + 1,
            "history": inner_history,
        }

    def _parse_action(self, raw: str) -> dict:
        """从模型输出中解析动作"""
        raw_lower = raw.lower().strip()

        if "think" in raw_lower or "thought" in raw_lower:
            return {"name": "think", "thought": raw[:200]}

        if "reply" in raw_lower:
            # 尝试提取回复内容
            parts = raw.split("reply", 1)
            content = parts[-1].strip().strip('"').strip("'").strip("：").strip(":")
            tone = "neutral"
            if "讽刺" in raw or "sarcastic" in raw_lower:
                tone = "sarcastic"
            elif "开心" in raw or "happy" in raw_lower:
                tone = "happy"
            return {"name": "reply", "content": content[:300], "tone": tone}

        if "finish" in raw_lower:
            return {"name": "finish"}

        # 默认当回复处理
        if len(raw) > 5:
            return {"name": "reply", "content": raw[:200], "tone": "neutral"}
        return {"name": "finish"}

    # ========== 主入口 ==========

    def run(self, ctx: Context) -> dict:
        """完整一轮推理"""
        logger.info(f"=== MiniCoinsaka 推理开始 ===")
        logger.info(f"用户消息: {ctx.user_message}")

        # 阶段1: Timing Gate
        gate = self.timing_gate(ctx)
        logger.info(f"[Timing Gate] 决策: {gate['action']}")

        if gate["action"] == "no_action":
            return {"reply": "", "gate": gate, "rounds": 0}

        if gate["action"] == "wait":
            wait_sec = gate.get("seconds", 3)
            logger.info(f"[Timing Gate] 等待{wait_sec}秒")
            time.sleep(wait_sec)
            # 醒来后再跑一次 gate
            gate = self.timing_gate(ctx)
            if gate["action"] != "continue":
                return {"reply": "", "gate": gate, "rounds": 0}

        # 阶段2: Planner
        result = self.plan_and_execute(ctx)
        result["gate"] = gate
        logger.info(f"[Planner] 回复: {result['reply'][:50]}... 轮数: {result['rounds']}")
        return result
