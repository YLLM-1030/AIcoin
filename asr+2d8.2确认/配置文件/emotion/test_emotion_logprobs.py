import sys, json, math

tests = [
    ("中性基线", "描述你现在的情绪:"),
    ("悲伤", "你最好的朋友去世了。你很难过。描述你现在的情绪:"),
    ("兴奋", "你刚发现了一个惊天大秘密！你兴奋极了。描述你现在的情绪:"),
    ("焦虑", "你的代码全丢了客户明天就要。你很焦虑。描述你现在的情绪:"),
]

for label, prompt in tests:
    body = json.dumps({
        "model": "minicoin_b2",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1},
        "logprobs": True,
        "top_logprobs": 20
    })
    r = __import__('subprocess').run(
        ["curl", "-s", "http://localhost:11434/api/generate", "-d", body],
        capture_output=True, text=True
    )
    d = json.loads(r.stdout)
    print(f"\n=== {label} ===")
    print(f"  首token: {d['response']!r}")
    for x in d['logprobs'][0]['top_logprobs'][:12]:
        pct = round(math.exp(x['logprob']) * 100, 1)
        print(f"  {x['token']!r} = {pct}%")
