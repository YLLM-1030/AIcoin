# -*- coding: utf-8 -*-
"""organize_memories.py — 把伪造记忆按 领域→规模→评价 树状整理到 _已导入/
用法（Windows 或 WSL 均可跑，纯文件操作）:
  python3 organize_memories.py
输出:
  _已导入/<领域>/<规模>/<评价>/汇总.md   每个非空分支一个汇总文件（保留原标签行，可重新入库）
  打印树统计（分支 + 条数）"""
import os, re, collections

BASE = r'C:/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/长期记忆文件内容存放'
SRC = BASE + '/伪造记忆'
DST = BASE + '/_已导入'

TAG_RE = re.compile(r'`\[([^\]]*)\]`')

def parse_line(line):
    line = line.strip()
    if not line.startswith('- '):
        return None
    m = TAG_RE.search(line)
    if not m:
        return None
    parts = m.group(1).split('|')
    if len(parts) != 4:
        return None
    scale = parts[0].strip()
    time = parts[1].strip()
    eval_ = parts[2].strip()
    domains = [d.strip() for d in parts[3].split() if d.strip()]
    if not domains:
        return None
    return line, scale, time, eval_, domains

# tree[领域][规模][评价] = [原始行, ...]
tree = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
skipped_files = []

for root, dirs, files in os.walk(SRC):
    for fname in sorted(files):
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(root, fname)
        parsed_any = False
        for line in open(fpath, encoding='utf-8'):
            p = parse_line(line)
            if not p:
                continue
            parsed_any = True
            orig, scale, time, eval_, domains = p
            ev = eval_ if eval_ else '无'
            sc = scale if scale else '未标注'
            for dom in domains:
                tree[dom][sc][ev].append(orig)
        if not parsed_any:
            skipped_files.append(fname)

# 写树
total = 0
branch_stats = []
for dom in sorted(tree):
    for sc in sorted(tree[dom]):
        for ev in sorted(tree[dom][sc]):
            lines = tree[dom][sc][ev]
            total += len(lines)
            d = os.path.join(DST, dom, sc, ev)
            os.makedirs(d, exist_ok=True)
            fname = '汇总.md'
            fpath = os.path.join(d, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f"# {dom} / {sc} / {ev}（{len(lines)} 条）\n\n")
                for l in sorted(lines):
                    f.write(l + '\n')
            branch_stats.append((len(lines), dom, sc, ev))

print(f"=== 整理完成 ===")
print(f"总条目: {total}（按 领域→规模→评价 分布到 {len(branch_stats)} 个分支）")
print(f"无标签跳过文件: {skipped_files}")
print()
print("=== 树统计（条数 / 领域 / 规模 / 评价）===")
for n, dom, sc, ev in sorted(branch_stats, reverse=True):
    print(f"  {n:3d}  {dom} / {sc} / {ev}")
print()
print(f"输出目录: {DST}")
