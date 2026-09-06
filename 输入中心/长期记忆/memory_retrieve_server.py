# memory_retrieve_server.py — 长期记忆检索 HTTP 服务（独立进程）
# 端口 18774，供 pipeline_server.py 调用
# 流程：POST /retrieve {l0, thought, action} → L0.5 → BERT分类 → 重心位移 → 走一步 → 标签过滤 → embedding top1
# 返回 4 块结构化数据供页面显示
# 运行（WSL）：source ~/mini_coin_venv/bin/activate && python3 memory_retrieve_server.py

import os
import sys
import json
import threading

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE = '/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/输入中心/长期记忆'
sys.path.insert(0, BASE)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from train_bert import MultiTaskClassifier, DOMAIN_LABELS, SCALE_LABELS, TIME_LABELS, EVAL_LABELS
from memory_db import MemoryDB

MODEL_PATH = '/home/autogram-coin/coin_models/chinese-roberta-wwm-ext'
SAVE_DIR = '/home/autogram-coin/coin_models/domain_classifier'
DB_PATH = BASE + '/memories.db'
MAX_LEN = 128
PORT = 18774

print("加载 BERT 分类器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = MultiTaskClassifier(MODEL_PATH, len(DOMAIN_LABELS), len(SCALE_LABELS),
                            len(TIME_LABELS), len(EVAL_LABELS))
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model.pt'), map_location='cuda'))
model.cuda().eval()

print("加载 bge embedding...")
embed = SentenceTransformer('BAAI/bge-base-zh-v1.5', local_files_only=True)

print("连接记忆库...")
db = MemoryDB(DB_PATH, embed)
print(f"记忆库共 {db.count()} 条")

# ── 属性坐标轴（重心位移用）──
AXES = {
    '规模': {'普通': 0.0, '中等': 0.5, '重大': 1.0},
    '时间': {'过去': -1.0, '现在': 0.0, '未来': 1.0},
    '评价': {'我不喜欢': -1.0, '无': 0.0, '我喜欢': 1.0},
}
ORDERS = {
    '规模': ['普通', '中等', '重大'],
    '时间': ['过去', '现在', '未来'],
    '评价': ['我不喜欢', '无', '我喜欢'],
}
DIM_LABELS = {'规模': SCALE_LABELS, '时间': TIME_LABELS, '评价': EVAL_LABELS}
DIM_NAMES = ['规模', '时间', '评价']


def classify(text):
    """返回 {维度: {标签: 概率}}"""
    enc = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True,
                    return_tensors='pt')
    with torch.no_grad():
        d, s, t, e = model(enc['input_ids'].cuda(), enc['attention_mask'].cuda())

    def soft(labels, x):
        p = torch.softmax(x[0], dim=-1)
        return {lb: float(v) for lb, v in zip(labels, p.tolist())}

    return {
        '领域': soft(DOMAIN_LABELS, d),
        '规模': soft(SCALE_LABELS, s),
        '时间': soft(TIME_LABELS, t),
        '评价': soft(EVAL_LABELS, e),
    }


def top1(probs):
    return max(probs.items(), key=lambda kv: kv[1])


def centroid(probs, axis):
    return sum(p * axis[lb] for lb, p in probs.items())


def step_label(base_label, delta, order):
    """全量标签沿 delta 方向走一级；|delta| 太小则不动"""
    if abs(delta) < 0.05:
        return base_label, False
    i = order.index(base_label)
    j = max(0, min(len(order) - 1, i + (1 if delta > 0 else -1)))
    return order[j], j != i


def format_probs(probs, top_k=4):
    """把 {标签:概率} 转成 [{label, prob}] 排序列表"""
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"label": lb, "prob": round(p * 100, 1)} for lb, p in items]


def retrieve(user_text, thought, action):
    """
    输入：别人说 + 硬币想法 + 硬币动作
    返回 4 块结构化数据：
      l0:     原始输入
      l05:    L0.5 第一视角转写
      labels: 标签计算过程（全量分类/硬币分类/重心位移/主移动/走一步/目标空间）
      hit:    检索结果（命中空间 + 候选列表）
    """
    coin_text = f"{thought}。{action}" if thought else action
    full_text = f'{user_text}。{coin_text}'
    l05 = f"{user_text}\n我想：{thought}\n{action}"

    pf = classify(full_text)
    pc = classify(coin_text)

    # 完整对话分类
    full_cls = {}
    for dim in ['领域', '规模', '时间', '评价']:
        lb, p = top1(pf[dim])
        full_cls[dim] = {"top": lb, "prob": round(p * 100, 1), "all": format_probs(pf[dim])}

    # 硬币回复分类
    coin_cls = {}
    for dim in ['领域', '规模', '时间', '评价']:
        lb, p = top1(pc[dim])
        coin_cls[dim] = {"top": lb, "prob": round(p * 100, 1), "all": format_probs(pc[dim])}

    # 重心位移
    delta_info = {}
    delta = {}
    # 时间/规模轴强度减半：这两个轴分类易波动，减半避免它们抢走主移动
    # 评价轴（真正的"硬币态度"信号）保持全量
    DAMP_DIMS = {'时间': 0.5, '规模': 0.5}
    for dim in DIM_NAMES:
        c_full = centroid(pf[dim], AXES[dim])
        c_coin = centroid(pc[dim], AXES[dim])
        d = c_coin - c_full
        if dim in DAMP_DIMS:
            d = d * DAMP_DIMS[dim]  # 减半
        delta[dim] = d
        delta_info[dim] = {"full": round(c_full, 2), "coin": round(c_coin, 2), "delta": round(d, 2)}

    main_dim = max(delta, key=lambda d: abs(delta[d]))
    # |Δ| < 0.05 视为无显著移动（避免 0 位移被随机选成主移动）
    if abs(delta[main_dim]) < 0.05:
        main_info = {"dim": None, "delta": round(delta[main_dim], 2), "dir": "无显著移动"}
    else:
        main_info = {"dim": main_dim, "delta": round(delta[main_dim], 2),
                     "dir": "正方向" if delta[main_dim] > 0 else "负方向"}

    # 走一步：只沿【主移动轴】走一步（|Δ| 最大的那个维度），其他轴保持全量对话的分类
    # 领域始终用全量对话的（用户话题）
    domain, _ = top1(pf['领域'])
    target = {'领域': domain}
    steps = []
    for dim in DIM_NAMES:
        base, _ = top1(pf[dim])
        if main_info["dim"] is not None and dim == main_dim:
            stepped, moved = step_label(base, delta[dim], ORDERS[dim])
            target[dim] = stepped
            steps.append({"dim": dim, "base": base, "stepped": stepped, "moved": moved})
        else:
            target[dim] = base  # 非主移动轴：不动
            steps.append({"dim": dim, "base": base, "stepped": base, "moved": False})

    # SQLite 过滤：严格 → 宽松
    combos = [
        [('domain', target['领域']), ('scale', target['规模']),
         ('time', target['时间']), ('eval', target['评价'])],
        [('domain', target['领域']), ('scale', target['规模']), ('time', target['时间'])],
        [('domain', target['领域']), ('scale', target['规模'])],
        [('domain', target['领域'])],
        [],
    ]
    hit = None
    for combo in combos:
        kwargs = {k: v for k, v in combo}
        res = db.search(full_text, top_k=3, **kwargs)
        if res:
            hit = {
                "space": combo if combo else "（全部记忆）",
                "candidates": [{"id": mid, "text": mtext, "sim": round(cos, 4)}
                               for mid, mtext, cos in res],
            }
            break
    if not hit:
        hit = {"space": None, "candidates": []}

    return {
        "l0": {"user": user_text, "thought": thought, "action": action},
        "l05": l05,
        "labels": {
            "full_cls": full_cls,
            "coin_cls": coin_cls,
            "delta": delta_info,
            "main": main_info,
            "steps": steps,
            "target": target,
        },
        "hit": hit,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "memories": db.count()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/retrieve":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body)
        except Exception:
            self._json(400, {"error": "bad json"})
            return
        user_text = data.get("user", "").strip()
        thought = data.get("thought", "").strip()
        action = data.get("action", "").strip()
        if not user_text:
            self._json(400, {"error": "empty user"})
            return
        try:
            result = retrieve(user_text, thought, action)
            self._json(200, result)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(ex)})


if __name__ == "__main__":
    print(f"长期记忆检索服务 → http://localhost:{PORT}")
    print(f"POST /retrieve  {{user, thought, action}}")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.serve_forever()
