"""
mini_coin web_search 工具 — 秘塔搜索引擎封装

用法: web_search("搜索关键词")
返回: 秘塔AI搜索的 answer 文本
"""

import json
import urllib.request
import urllib.error

API_KEY = "mk-1F203F3C3902179EBB637E36500BEFCE"
API_URL = "https://metaso.cn/api/v1/chat/completions"


def web_search(query: str) -> str:
    body = json.dumps({
        "model": "fast",
        "stream": False,
        "conciseSnippet": True,
        "messages": [{"role": "user", "content": query}]
    }).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return f"搜索出错: HTTP {e.code}"
    except Exception as e:
        return f"搜索失败: {e}"

    answer = ""
    choices = data.get("choices", [])
    if choices:
        answer = choices[0].get("message", {}).get("content", "")
    if not answer:
        return "秘塔未返回结果"
    import re
    answer = re.sub(r'\[\[\d+\]\]', '', answer).strip()
    return answer
