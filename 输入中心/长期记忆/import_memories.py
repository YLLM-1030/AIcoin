# import_memories.py — 批量导入伪造记忆 .md → memories.db
# 用法: source ~/mini_coin_venv/bin/activate
#       python3 /mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/import_memories.py
#       python3 import_memories.py --all     # 全扫整个伪造记忆目录（重建用，慎用）
#
# 增量导入机制（2026-08-11）：
#   - 默认只扫 _待导入/ 目录（不再全扫 → 不会误扫 BERT 训练数据等其他文件）
#   - 导入成功的 .md 自动移到 _已导入/（归档，防止重复扫描）
#   - 新记忆 = 丢文件进 _待导入/ → 跑一次即入

import os
import re
import sys
import time
import shutil

# 强制离线（必须在 import sentence_transformers 之前设置）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from memory_db import MemoryDB

# ── 路径配置（WSL 格式，因为在 WSL 里跑）──
BASE = "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/长期记忆文件内容存放"
FAKE_MEMORY_DIR = BASE + "/伪造记忆"          # 全扫模式根目录（--all）
INBOX_DIR = BASE + "/_待导入"                 # 增量导入：新记忆放这里
DONE_DIR = BASE + "/_已导入"                  # 导入成功后归档
DB_PATH = "/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆/memories.db"

# 标准 10 领域（train_bert.py 定义）+ 允许的其他领域
VALID_DOMAINS = {'哲学/思辨', '经济/商业', '文化/教育', '文学/艺术', '历史/怀旧',
                 '科技/代码', '游戏/娱乐', '生活/日常', '社交/关系', '健康/医疗'}

# ── 解析一条记忆行 ──
# 格式: - 记忆内容  `[规模|时间|评价|领域]`
# 或:   - 记忆内容  `[领域1, 领域2]`
TAG_RE = re.compile(r'`\[([^\]]*)\]`')

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


def check_format(fpath, lines):
    """格式校验：统计可解析行 + 非标准 domain，返回 (ok_lines, bad_lines, bad_domains)"""
    ok = 0; bad = 0; bad_domains = set()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        p = parse_line(ln)
        if p is None:
            bad += 1
            continue
        ok += 1
        for d in p[4]:
            if d not in VALID_DOMAINS:
                bad_domains.add(d)
    return ok, bad, bad_domains


def import_dir(src_dir, archive=True):
    """递归导入目录下所有 .md，导入成功的移动到 _已导入/"""
    total = 0; file_count_total = 0
    for root, dirs, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.endswith('.md'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, encoding='utf-8') as f:
                lines = f.readlines()

            # 格式校验（防 BERT 训练格式等非 db 格式文件误入）
            ok, bad, bad_domains = check_format(fpath, lines)
            if ok == 0:
                print(f"  ⚠️ {fname}: 0 条可解析（{bad} 行无效）— 跳过，可能是非记忆文件")
                continue
            if bad_domains:
                print(f"  ⚠️ {fname}: 非标准领域 {bad_domains} — 继续导入（可后续修正）")

            file_count = 0; dup_count = 0
            for ln in lines:
                parsed = parse_line(ln)
                if parsed is None:
                    continue
                text, scale, time, eval_, domains = parsed
                for dom in domains:
                    inserted = db.add_memory(text, dom, scale, time, eval_)
                    if inserted:
                        file_count += 1
                    else:
                        dup_count += 1
                    total += 1

            print(f"  {fname}: +{file_count} 条 (跳过重复 {dup_count})")
            file_count_total += file_count

            # 导入成功 → 归档到 _已导入/（仅增量模式；--all 不归档保留原文件）
            if archive and file_count > 0:
                os.makedirs(DONE_DIR, exist_ok=True)
                dest = os.path.join(DONE_DIR, fname)
                # 同名冲突加序号
                if os.path.exists(dest):
                    base, ext = os.path.splitext(fname)
                    dest = os.path.join(DONE_DIR, f"{base}_{int(time.time())}{ext}")
                try:
                    shutil.move(fpath, dest)
                    print(f"    → 已归档: _已导入/{os.path.basename(dest)}")
                except Exception as e:
                    print(f"    ⚠️ 归档失败: {e}")
            db.conn.commit()
    return file_count_total, total


def import_all():
    global db
    # 加载模型（WSL 缓存里的 bge）
    print("加载 bge-base-zh-v1.5 ...")
    model = SentenceTransformer('BAAI/bge-base-zh-v1.5', local_files_only=True)

    db = MemoryDB(DB_PATH, embed_model=model)
    print(f"数据库: {DB_PATH}")

    # 增量模式：只扫 _待导入/
    os.makedirs(INBOX_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)
    inbox_files = [f for f in os.listdir(INBOX_DIR) if f.endswith('.md')]
    if inbox_files:
        print(f"[增量] 扫描 _待导入/ ({len(inbox_files)} 个文件)...")
        added, total = import_dir(INBOX_DIR, archive=True)
        print(f"[增量] +{added} 条（处理 {total} 条含重复）")
    else:
        print("[增量] _待导入/ 为空，无新记忆")
        added, total = 0, 0

    # --all：全扫整个伪造记忆目录（重建用，忽略归档目录）
    if '--all' in sys.argv:
        print("[全扫] 扫描整个伪造记忆目录（重建）...")
        a2, t2 = import_dir(FAKE_MEMORY_DIR, archive=False)
        added += a2; total += t2

    print(f"\n完成: 本次 +{added} 条")
    print(f"数据库总计: {db.count()} 条")
    db.close()


if __name__ == '__main__':
    import_all()
