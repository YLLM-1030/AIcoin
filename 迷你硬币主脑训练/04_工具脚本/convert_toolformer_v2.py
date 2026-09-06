#!/usr/bin/env python3
"""下载 Toolformer 数据集并转为中文 [TOOL:xxx] 格式"""
import json, re, os
from datasets import load_dataset

OUTPUT = "/mnt/c/Users/Autogram-coin/Desktop/新的预训练/sft_toolformer_30.jsonl"
MAX_SAMPLES = 50

# 英文工具名 → 中文工具名
TOOL_NAMES = {
    "wikipedia": "搜索",
    "weather": "搜索",
    "wolfram": "搜索",
    "news": "搜索",
    "calculator": "计算",
    "python": "回复",
    "shell": "回复",
}

# 正则：匹配 function_name('参数')
PATTERNS = [
    (re.compile(r"wikipedia\('([^']+)'\)"), "搜索"),
    (re.compile(r"weather\('([^']+)'\)"), "搜索"),
    (re.compile(r"wolfram\('([^']+)'\)"), "搜索"),
    (re.compile(r"news\('([^']+)'\)"), "搜索"),
    (re.compile(r"calculator\('([^']+)'\)"), "计算"),
    (re.compile(r"python\('([^']+)'\)"), "回复"),
]

print("加载数据集...")
ds = load_dataset("taskydata/GPTeacher-Toolformer", split="train", streaming=True)

converted = []
for item in ds:
    inp = (item.get("input") or "").strip()
    resp = (item.get("response") or "").strip()
    if not inp or not resp:
        continue

    new_resp = resp
    has_tool = False
    for pat, tool_name in PATTERNS:
        new_resp = pat.sub(r"[TOOL:{}] \1 [/TOOL]".format(tool_name), new_resp)
        if pat.search(resp):
            has_tool = True

    if has_tool:
        converted.append({"instruction": inp, "output": new_resp})
        if len(converted) >= MAX_SAMPLES:
            break

with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in converted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"完成: {len(converted)} 条，保存到 {OUTPUT}")
for item in converted[:5]:
    print(f"  用户: {item['instruction'][:50]}")
    print(f"  输出: {item['output'][:120]}")
    print()
