import re, sys

with open(sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/Autogram-coin/Desktop/等待挖掘的语料/搞事/Neuro搞事_春药篇 - 副本.txt', encoding='utf-8') as f:
    content = f.read()
blocks = re.findall(r'\{([\s\S]*?)\}', content)
total = has_emoji = bad_emoji = 0
emoji_counts = []
valid = ['认真','wink','开心','spin','转头','担心']
for b in blocks:
    lines = b.strip().split('\n')
    cur = {'output':''}; state=None
    for line in lines:
        s=line.strip()
        if s=='output:': state='output'; continue
        if state and s: cur['output']=(cur['output']+'\n'+s) if cur['output'] else s
    out = cur['output']
    if not out: continue
    total += 1
    emojis = re.findall(r'\[TOOL:表情\]\s*(.*?)\s*\[TOOL\]', out)
    if emojis:
        has_emoji += 1
        emoji_counts.append(len(emojis))
        for e in emojis:
            if e.strip() not in valid:
                bad_emoji += 1
                if bad_emoji <= 5:
                    print(f'  非法: "{e.strip()}" in: {out[:60]}...')
print(f'总{total}条, 带表情{has_emoji}条({has_emoji/total*100:.0f}%), 非法{bad_emoji}处, 均{sum(emoji_counts)/len(emoji_counts):.1f}个/条' if emoji_counts else '0')
