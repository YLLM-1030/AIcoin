#!/usr/bin/env python3
"""下载 Toolformer 数据集并转换为 [TOOL:xxx] 格式"""
import json, os, random

# ─── 配置 ───
OUTPUT = "/mnt/c/Users/Autogram-coin/Desktop/新的预训练/sft_toolformer_converted.jsonl"
SAMPLE_SIZE = 100  # 取100条就够了

# ─── 从 ModelScope 加载 ───
print("从 ModelScope 加载 GPTeacher-Toolformer...")
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('wyj123456/GPTeacher', subset_name='default', split='train')
# ─── 检查可用字段 ───
print(f"数据集大小: {len(ds)}")
for i, item in enumerate(ds):
    if i < 3:
        print(f"\n--- 第{i+1}条 ---")
        if isinstance(item, dict):
            for k, v in item.items():
                print(f"  {k}: {str(v)[:200]}")
        else:
            print(f"  type={type(item)}, value={str(item)[:200]}")
    else:
        break

# ─── 工具映射 ───
TOOL_MAP = {
    "wikipedia": "web_search",
    "search": "web_search",
    "wolfram": "web_search",
    "weather": "web_search",
    "news": "web_search",
    "calculator": "web_search",
    "python": "reply",
    "shell": "reply",
}

def convert_to_toolformer_format(instruction, input_text, response):
    """把英文 Toolformer 格式转成 [TOOL:xxx] 格式"""
    # response 里可能有: wikipedia('French Revolution')
    # 转成: [TOOL:web_search] French Revolution [/TOOL]
    import re
    
    result = response
    
    # 匹配各种工具调用格式
    patterns = [
        (r"wikipedia\('([^']+)'\)", "web_search"),
        (r"weather\('([^']+)'\)", "web_search"),
        (r"wolfram\('([^']+)'\)", "web_search"),
        (r"news\('([^']+)'\)", "web_search"),
        (r"search\('([^']+)'\)", "web_search"),
        (r"python\('([^']+)'\)", "reply"),
        (r"calculator\('([^']+)'\)", "web_search"),
    ]
    
    for pat, tool_name in patterns:
        result = re.sub(pat, rf"[TOOL:{tool_name}] \1 [/TOOL]", result)
    
    return result

# ─── 转换 ───
converted = []
for item in ds:
    inst = item.get("instruction", "")
    inp = item.get("input", "")
    resp = item.get("response", "")
    
    # 合成用户消息
    user_msg = inst
    if inp:
        user_msg += " " + inp
    
    converted_resp = convert_to_toolformer_format(inst, inp, resp)
    
    # 检查是否真的转换出了 [TOOL:]
    if "[TOOL:" not in converted_resp:
        continue
    
    converted.append({
        "instruction": user_msg,
        "output": converted_resp
    })

# ─── 采样 ───
random.shuffle(converted)
selected = converted[:SAMPLE_SIZE]

# ─── 保存 ───
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"转换完成: 共 {len(converted)} 条含工具调用, 保存 {len(selected)} 条到 {OUTPUT}")
print("\n样例:")
for item in selected[:3]:
    print(f"  用户: {item['instruction'][:60]}")
    print(f"  输出: {item['output'][:120]}")
    print()
