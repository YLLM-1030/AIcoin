"""
合并所有可用语料为一个文件
"""
import glob, os

files = [
    # 聊天叙事（已训过，但合并后分布可能变）
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/持续预训练数据_v12之前.txt",
    # 身份锚点（已训过）
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/identity_v1_formatted.txt",
    # 需要强化数据
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/需要强化/需要强化.txt",
    # 可爱版文章
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/文章1_可爱版.txt",
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/文章2_可爱版.txt",
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/清洗对话_可爱版.txt",
    # 公理体系
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/持续预训练数据_公理体系.txt",
]

all_text = []
total_chars = 0
for f in files:
    if os.path.exists(f):
        text = open(f, encoding="utf-8").read()
        all_text.append(text)
        total_chars += len(text)
        print(f"  ✅ {f.split('/')[-1]}: {len(text)}字")
    else:
        print(f"  ❌ {f} 不存在")

merged = "\n\n".join(all_text)
out = "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/merged_all.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(merged)
print(f"\n合并完成: {len(all_text)}个文件, {total_chars}字 → {out}")
