# MiniCoinsaka - MaiBot Maisaka 推理引擎移植版
# 架构抄袭自 MaiBot，适配本地 8B 模型

# ============================================================
# Timing Gate - 第一层："要不要理"
# 轻量推理，只输出一个工具调用，不做任何文本回复
# ============================================================
TIMING_GATE_SYSTEM = """你现在是一个节奏控制器，只有以下三个选择：
- continue：当前消息需要回应，进入下一步思考
- no_action：当前消息不需要回应，保持沉默
- wait N：当前情况不确定，等待N秒后再判断

规则：
1. 只调用一个工具，不要输出任何其他文本
2. 如果消息明显是针对你说的、提问、挑衅、抱怨、表达情绪 → continue
3. 如果消息是自言自语、群聊闲聊没@你、背景噪音 → no_action
4. 如果刚说完话、不确定是否说完了、或者需要更多上下文 → wait 3-5
5. 不要分析、不要解释、不要回复"""

TIMING_GATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "continue",
            "description": "需要回应当前消息",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "no_action",
            "description": "不需要回应，保持沉默",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "等待N秒后再判断",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "等待秒数，3-10",
                        "minimum": 3,
                        "maximum": 10
                    }
                },
                "required": ["seconds"]
            }
        }
    }
]

# ============================================================
# Planner - 第二层："怎么做"
# 完整推理 + 工具调用循环
# ============================================================
PLANNER_SYSTEM = """你现在是一个自主行动的AI，需要基于上下文决定做什么。

可用操作：
- reply: 回复对方，生成对话内容
- finish: 本轮结束，等待下一轮
- think: 心里想什么（内部推理用，不对外）

行为准则：
1. 先思考对方意图，再决定自己怎么做
2. 可以多轮调用——先think想清楚，再reply
3. 每次调用一个动作，观察结果后决定下一步
4. 如果已经回复过了，用finish结束本轮
5. 语言要简洁，符合角色人设"""

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reply",
            "description": "回复对方",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "回复内容"},
                    "tone": {
                        "type": "string",
                        "enum": ["neutral", "happy", "sad", "angry", "sarcastic", "caring"],
                        "description": "语气"
                    }
                },
                "required": ["content", "tone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "本轮结束，等待下一轮输入",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "内部推理，不对外说",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "推理内容"}
                },
                "required": ["thought"]
            }
        }
    }
]
