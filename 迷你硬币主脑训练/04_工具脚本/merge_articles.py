"""
合并：文章1可爱版 + 文章2可爱版 + 公理体系
"""
files = [
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/文章1_可爱版.txt",
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/文章2_可爱版.txt",
    "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/持续预训练数据_公理体系.txt",
]

all_text = []
total = 0
for f in files:
    text = open(f, encoding="utf-8").read()
    all_text.append(text)
    total += len(text)
    print(f"  {f.split('/')[-1]}: {len(text)}字")

merged = "\n\n".join(all_text)
out = "/mnt/c/Users/Autogram-coin/Desktop/离散语料/可用语料/merged_articles.txt"
open(out, "w", encoding="utf-8").write(merged)
print(f"\n合并完成: 共{total}字 → {out}")
