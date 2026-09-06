# import_memories.py — 批量导入伪造记忆 .md → memories.db
# 用法: source ~/mini_coin_venv/bin/activate
#       python3 /mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/import_memories.py

import os
import re
import sys

# 强制离线（必须在 import sentence_transformers 之前设置）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from memory_db import MemoryDB

# ── 路径配置（WSL 格式，因为在 WSL 里跑）──
FAKE_MEMORY_DIR = "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/长期记忆文件内容存放/伪造记忆"
DB_PATH = "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/memories.db"

# ── 解析一条记忆行 ──
# 格式: - 记忆内容  `[规模|时间|评价|领域]`
# 或:   - 记忆内容  `[领域1, 领域2]`
TAG_RE = re.compile(r'`\[([^\]]*)\]`')
MULTI_DOMAIN_RE = re.compile(r'^\[(.*)\]$')

def parse_line(line):
    """解析单行记忆，返回 (text, scale, time, eval, domains) 或 None"""
    line = line.strip()
    if not line.startswith('- '):
        return None

    m = TAG_RE.search(line)
    if not m:
        return None

    tag_str = m.group(1)
    text = line[:m.start()].replace('- ', '').strip()

    parts = tag_str.split('|')

    if len(parts) == 4:
        # 格式1: [规模|时间|评价|领域]
        scale = parts[0].strip() or None
        time = parts[1].strip() or None
        eval_ = parts[2].strip() or None
        domains = [d.strip() for d in parts[3].split() if d.strip()]
        if not domains:
            return None
        return text, scale, time, eval_, domains

    elif len(parts) == 1 and ',' in tag_str:
        # 格式2: [领域1, 领域2]（只有领域）
        domains = [d.strip() for d in tag_str.split(',') if d.strip()]
        if not domains:
            return None
        return text, None, None, None, domains

    return None


def import_all():
    # 加载模型（WSL 缓存里的 bge）
    print("加载 bge-base-zh-v1.5 ...")
    model = SentenceTransformer('BAAI/bge-base-zh-v1.5', local_files_only=True)

    db = MemoryDB(DB_PATH, embed_model=model)
    print(f"数据库: {DB_PATH}")

    total = 0
    skipped = 0

    # 遍历伪造记忆目录
    for root, dirs, files in os.walk(FAKE_MEMORY_DIR):
        for fname in sorted(files):
            if not fname.endswith('.md'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, encoding='utf-8') as f:
                lines = f.readlines()

            file_count = 0
            dup_count = 0
            for ln in lines:
                parsed = parse_line(ln)
                if parsed is None:
                    continue
                text, scale, time, eval_, domains = parsed
                # 一条记忆有多个领域 → 存多条（每个领域一条）
                for dom in domains:
                    inserted = db.add_memory(text, dom, scale, time, eval_)
                    if inserted:
                        file_count += 1
                    else:
                        dup_count += 1
                    total += 1

            print(f"  {fname}: +{file_count} 条 (跳过重复 {dup_count})")
            db.conn.commit()

    print(f"\n完成: 共导入 {total} 条记忆 (含多领域拆分)")
    print(f"数据库总计: {db.count()} 条")
    db.close()


if __name__ == '__main__':
    import_all()
