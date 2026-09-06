"""解析器：Neuro搞事_春药篇 - 副本.txt
输入格式：{...} 包裹的数据块，每块含 history/memory/input/output 字段
输出：list[dict]
"""

import re, json, sys

def parse_blocks(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1) 按大括号提取数据块
    raw_blocks = re.findall(r'\{([\s\S]*?)\}', content)
    print(f'[解析] 找到 {len(raw_blocks)} 个大括号数据块', file=sys.stderr)

    # 2) 逐块解析字段
    items = []
    for idx, block in enumerate(raw_blocks):
        lines = block.strip().split('\n')
        cur = {'history': '', 'memory': '', 'input': '', 'output': ''}
        state = None  # 当前正在累积哪个字段

        for line in lines:
            s = line.strip()

            # header 切换（output 内不切换）
            if s == 'history:':
                state = 'history'
                continue
            if s == 'memory:':
                state = 'memory'
                continue
            if s == 'input:':
                state = 'input'
                continue
            if s == 'output:':
                state = 'output'
                continue

            # 非 header 行 → 累加到当前状态
            if state and s:
                cur[state] = (cur[state] + '\n' + s) if cur[state] else s

        # 校验：必须有 input 和 output
        if not cur['input'] or not cur['output']:
            print(f'[警告] 块{idx} 缺少 input 或 output，跳过', file=sys.stderr)
            continue

        items.append(cur)

    print(f'[解析] 有效数据: {len(items)} 条', file=sys.stderr)
    h = sum(1 for i in items if i['history'])
    m = sum(1 for i in items if i['memory'])
    print(f'[解析] 有history: {h}, 有memory: {m}', file=sys.stderr)

    return items


def preview(items, n=5):
    """打印前 n 条数据的字段摘要"""
    for i in range(min(n, len(items))):
        d = items[i]
        print(f'===== 条{i+1} =====')
        if d['history']:
            print(f'[history]')
            print(d['history'][:200])
            if len(d['history']) > 200:
                print(f'  ...({len(d["history"])}字符)')
        else:
            print('[history] (无)')

        if d['memory']:
            print(f'[memory]')
            print(d['memory'])
        else:
            print('[memory] (无)')

        print(f'[input]')
        print(d['input'][:100])
        if len(d['input']) > 100:
            print(f'  ...({len(d["input"])}字符)')

        print(f'[output]')
        print(d['output'][:100])
        if len(d['output']) > 100:
            print(f'  ...({len(d["output"])}字符)')
        print()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fp = sys.argv[1]
    else:
        fp = '/mnt/c/Users/Autogram-coin/Desktop/等待挖掘的语料/搞事/Neuro搞事_春药篇 - 副本.txt'
    data = parse_blocks(fp)
    preview(data, n=5)
