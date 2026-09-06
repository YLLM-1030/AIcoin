# convert_format.py — 修复版：处理当前损坏格式 `[领域]|[规模|时间|评价]`
# 目标格式: [领域|规模|时间|评价]
# 中间段 [X|Y|] 中 X 可能是规模(普通/中等/重大/轻微) 或 评价(我喜欢/我不喜欢)

import re
import glob
import os

DATA_DIR = r'/mnt/c/Users/Autogram-coin/Desktop/伪造记忆/聊天内容'
# 匹配 `[领域]|[X|Y|]` 或 `[领域]|[X|Y|Z]`
TAG_RE = re.compile(r'(\`\[([^\]]+)\]\|\[([^\]]*)\]\`)')

SCALES = {'重大', '中等', '普通', '轻微'}
EVALS = {'我喜欢', '我不喜欢'}

converted = 0
skipped = 0

for fpath in sorted(glob.glob(os.path.join(DATA_DIR, '*.md'))):
    with open(fpath, encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        m = TAG_RE.search(line)
        if not m or not line.strip().startswith('- '):
            new_lines.append(line)
            continue

        domain = m.group(2).strip()
        middle = m.group(3).strip()  # 如 '普通|现在' 或 '我喜欢|现在' 或 ''

        # 解析中间段
        mid_parts = middle.split('|') if middle else []
        if len(mid_parts) >= 2:
            x, y = mid_parts[0].strip(), mid_parts[1].strip()
            z = mid_parts[2].strip() if len(mid_parts) > 2 else ''
        else:
            x, y, z = '', '', ''

        if x in SCALES:
            scale, time_, eval_ = x, y, z
        elif x in EVALS:
            scale, time_, eval_ = '', y, x
        else:
            scale, time_, eval_ = x, y, z

        # 新格式: [领域|规模|时间|评价]
        new_tag = f'[{domain}|{scale}|{time_}|{eval_}]'
        new_line = line[:m.start(1)] + '`' + new_tag + '`' + line[m.end(1):]
        new_lines.append(new_line)
        converted += 1

    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

print(f'转换完成: {converted} 条, 跳过: {skipped} 条')
