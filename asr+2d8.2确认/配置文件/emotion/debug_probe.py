"""
debug_probe.py — 探查模型在情绪 prompt 下的真实 token 分布
直接调用 Ollama API，top_logprobs=50，看前 50 个 token 都是什么
"""
import requests, math, json

OLLAMA = "http://localhost:11434"

def probe(prompt: str, label: str):
    resp = requests.post(f"{OLLAMA}/api/generate", json={
        "model": "minicoin_b2",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1},
        "logprobs": True,
        "top_logprobs": 20,
    }, timeout=30).json()

    print(f"\n{'='*60}")
    print(f"Prompt: {label}")
    print(f"{'='*60}")

    # 调试：打印完整响应结构
    if "logprobs" not in resp:
        print("响应中没有 logprobs!")
        print(json.dumps(resp, ensure_ascii=False, indent=2)[:2000])
        return

    tokens = resp["logprobs"][0]["top_logprobs"]
    for i, t in enumerate(tokens):
        prob = math.exp(t["logprob"])
        token = t["token"].strip()
        print(f"  {i+1:2d}. [{prob:.4f}] {repr(token)}")


if __name__ == "__main__":
    probe("描述你现在的情绪:", "中性基线")
    probe("你今天过得特别开心, 一切都很好。描述你现在的情绪:", "快乐刺激")
    probe("你最好的朋友今天去世了。你很难过。描述你现在的情绪:", "悲伤刺激")
