"""
prompt_builder.py — 外部拼装器
负责：history→摘要化、记忆检索、完整prompt拼装
"""
from collections import deque
import json, urllib.request

BRIDGE_URL = "http://localhost:18771"

SYSTEM = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

TOOK_GRAMMAR = """root ::= item*
item ::= [^[] | "[" ( [^Tt] | "T" [^Oo] | "TO" "O" "L" ( ":" tool_args "]" | "]" ) )
tool_args ::= "表情" " "? ("认真"|"wink"|"开心"|"spin"|"转头"|"担心") " "? "[TOOL]" | ("回复"|"搜索"|"报错"|"结束") " "? [^]]* " "? "[TOOL]" """

LOGIT_BIAS = [[1930, 10.0], [3925, -10.0]]


class CoinBrain:
    def __init__(self, max_rounds=5):
        self.history = deque(maxlen=max_rounds * 2)  # (role, content)
        self.summary = ""  # 已摘要化的上文
        self.memory = "我记得拉姆是我的创造者，是我的父亲。"

    def _summarize_old(self):
        """将超出轮次的历史摘要化"""
        if len(self.history) < 10:
            return
        old = list(self.history)[:-8]
        lines = []
        for role, content in old:
            prefix = "拉姆说" if role == "user" else "硬币说"
            lines.append(f"{prefix}：{content[:40]}")
        self.summary = "；".join(lines)

    def _build_prompt(self, text):
        parts = []
        parts.append(f"<|im_start|>system\n{SYSTEM}<|im_end|>")

        # 摘要历史（超出5轮的部分）
        if self.summary:
            parts.append(f"<|im_start|>摘要\n{self.summary}<|im_end|>")

        # 最近5轮完整历史
        recent = list(self.history)[-10:] if len(self.history) >= 10 else list(self.history)
        for role, content in recent:
            name = "输入" if role == "user" else "硬币"
            parts.append(f"<|im_start|>{name}\n{content}<|im_end|>")

        # 记忆
        if self.memory:
            parts.append(f"<|im_start|>记忆\n{self.memory}<|im_end|>")

        # 当前输入
        user_text = f"对话[拉姆]：{text}"
        parts.append(f"<|im_start|>输入\n{user_text}<|im_end|>")
        parts.append("<|im_start|>硬币\n")
        return "\n".join(parts), user_text

    def chat(self, text):
        prompt, user_text = self._build_prompt(text)
        payload = json.dumps({"prompt": prompt, "n_predict": 500,
                              "temperature": 0.9, "grammar": TOOK_GRAMMAR,
                              "logit_bias": LOGIT_BIAS}).encode()
        req = urllib.request.Request(BRIDGE_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        reply = data.get("content", "")

        self.history.append(("user", user_text))
        self.history.append(("assistant", reply))
        self._summarize_old()
        return reply


# ── 交互式测试 ──
if __name__ == "__main__":
    bot = CoinBrain()
    print("🪙 迷你硬币（外部拼装版）\n")
    while True:
        try:
            text = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        reply = bot.chat(text)
        print(f"\n─── 硬币 ───")
        print(reply)
        print()
