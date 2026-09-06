import re

path = '/mnt/c/Users/Autogram-coin/Desktop/等待挖掘的语料/搞事/Neuro搞事_春药篇 - 副本.txt'
with open(path, encoding='utf-8') as f:
    content = f.read()

blocks = re.findall(r'\{([\s\S]*?)\}', content)
print(f'总块数: {len(blocks)}')

has_h = has_m = has_emoji = bad_emoji = 0
valid_emoji = ['认真','wink','开心','spin','转头','担心']

for b in blocks:
    if 'history:' in b: has_h += 1
    if 'memory:' in b: has_m += 1
    emojis = re.findall(r'\[TOOL:表情\]\s*(.*?)\s*\[TOOL\]', b)
    if emojis:
        has_emoji += 1
        for e in emojis:
            if e.strip() not in valid_emoji:
                bad_emoji += 1

print(f'有history: {has_h}')
print(f'有memory: {has_m}')
print(f'带表情: {has_emoji} ({has_emoji/len(blocks)*100:.0f}%)')
print(f'非法表情: {bad_emoji}')
