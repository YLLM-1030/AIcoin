"""
pipeline_server.py — 输入管线管理中心
可视化编辑 prompt 拼装，输出完整格式化 prompt
"""
import json, threading, os, datetime, re, time, signal, sys
import numpy as np
from http.server import SimpleHTTPRequestHandler, BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer
import requests

# ── 可复用端口服务器 (Ctrl+C 不残留) ──
class ReusableTCPServer(ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

# ── 内置桥 (18000-18009) ──
_source_stores = {}  # {port: {source_name: text}}
_store_lock = threading.Lock()

class BridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/source_last":
            port = int(self.server.server_address[1])
            with _store_lock:
                data = _source_stores.get(port, {})
            self._json(200, data)
            return
        self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        port = int(self.server.server_address[1])
        try:
            parsed = json.loads(body)
            source = parsed.get("source", "")
            text = parsed.get("text", "")
            if source and text:
                with _store_lock:
                    if port not in _source_stores:
                        _source_stores[port] = {}
                    _source_stores[port][source] = text[:5000]
                # 弹幕输入到达（端口=弹幕模块 endpoint）→ 触发 reply，与对话一致
                if _is_danmaku_port(port):
                    threading.Thread(target=danmaku_reply, daemon=True).start()
        except Exception:
            pass
        self._json(200, {"ok": True})

    def _json(self, code, data):
        b = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(b))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    log_message = lambda *a: None

BRIDGE_URL = "http://localhost:18771"
BRIDGE_API = "http://localhost:18771/source_last"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
RETRIEVE_API = "http://localhost:18774/retrieve"   # 长期记忆检索服务
RETRIEVE_HEALTH = "http://localhost:18774/health"

# 长期记忆检索结果（最新一轮），供页面显示
last_retrieval = None
last_retrieval_lock = threading.Lock()

# ── 模块定义 ──
# 每个模块：id, label, format_start, format_end, enabled, editable, content
MODULES = [
    {"id": "system",  "label": "系统提示", "enabled": True, "editable": True,
     "format_start": "<|im_start|>system\n", "format_end": "<|im_end|>",
     "content": """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""},

    {"id": "memory",  "label": "记忆", "enabled": True, "editable": True,
     "format_start": "<|im_start|>记忆\n", "format_end": "<|im_end|>",
     "sources": [{"name": "静态记忆", "active": True}],
     "content": "我记得拉姆是我的创造者，是我的父亲。"},

    {"id": "summary", "label": "历史摘要", "enabled": False, "editable": True,
     "format_start": "<|im_start|>摘要\n", "format_end": "<|im_end|>",
     "sources": [{"name": "自动生成", "active": True}],
     "content": "（自动生成）"},

    {"id": "chat",    "label": "对话输入", "enabled": True, "editable": True,
     "format_start": "<|im_start|>输入\n", "format_end": "<|im_end|>",
     "sources": [{"name": "拉姆", "active": True}]},

    {"id": "danmaku", "label": "弹幕输入", "enabled": False, "editable": True,
     "format_start": "<|im_start|>弹幕\n", "format_end": "<|im_end|>",
     "sources": [{"name": "弹幕源", "active": False}],
     "content": "弹幕[{source}]：{text}"},

    {"id": "vision",  "label": "视觉输入", "enabled": False, "editable": True,
     "format_start": "<|im_start|>视觉\n", "format_end": "<|im_end|>",
     "sources": [{"name": "视觉源", "active": False}],
     "content": "（暂未接入）"},

    {"id": "tool",    "label": "工具返回", "enabled": False, "editable": True,
     "format_start": "<|im_start|>工具\n", "format_end": "<|im_end|>",
     "sources": [{"name": "工具源", "active": False}],
     "content": "（暂未接入）"},

    {"id": "emotion", "label": "情绪指引", "enabled": False, "editable": True,
     "format_start": "<|im_start|>情绪\n", "format_end": "<|im_end|>",
     "sources": [{"name": "FACS", "active": False}],
     "content": "（暂未接入）"},

    {"id": "prompt",  "label": "提示词", "enabled": False, "editable": True,
     "format_start": "<|im_start|>提示词\n", "format_end": "<|im_end|>",
     "content": "请用可爱的语气回复"},
]

pipeline_lock = threading.Lock()

# 自动化配置（前端「自动化」分页可编辑，持久化到 pipeline_config.json）
automation = {"sys_interval": 30, "sys_prompts": ["你可以继续往下说"], "sys_bridge": "18002"}

def save_config(backup=False):
    global CONFIG_PATH
    path = CONFIG_PATH
    if backup:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CONFIG_PATH.replace(".json", f"_{ts}.json")
    with pipeline_lock:
        data = {"pipeline": pipeline, "automation": automation}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def load_config():
    global pipeline, automation
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            automation.update(data.get("automation", {}))
            saved_modules = data.get("pipeline", [])
            saved = {m["id"]: m for m in saved_modules}
            # 按保存的顺序重建，保留用户拖拽后的顺序
            merged = []
            seen = set()
            for sm in saved_modules:
                mid = sm["id"]
                if mid in seen: continue
                seen.add(mid)
                if mid in saved:
                    # 以默认模块为模板，用保存值覆盖
                    default = None
                    for dm in MODULES:
                        if dm["id"] == mid: default = dm; break
                    if default:
                        mc = dict(default)
                        mc.update({k: sm[k] for k in sm if k in mc})
                        merged.append(mc)
                    else:
                        merged.append(dict(sm))
            # 补漏：新增的默认模块
            for dm in MODULES:
                if dm["id"] not in seen:
                    merged.append(dict(dm))
            pipeline.clear()
            pipeline.extend(merged)
            print(f"[配置] 已从 {CONFIG_PATH} 加载 (顺序已保留)")
        except Exception as e:
            print(f"[配置] 加载失败: {e}")

pipeline = [dict(m) for m in MODULES]
load_config()
# ── system 提示词 API/本地 双向切换状态 (08-12 拉姆) ──
_sys_prompt_backup = None      # 切到 API 前备份的本地 system content（切回用）
_sys_prompt_is_api = False     # 当前 system content 是否为 API 提示词
LLM_BRIDGE_PORT = 18771        # LLM 桥（后端模式切换转发目标）
history = []  # [("user", text), ("assistant", reply)]
last_reply = ""
last_reply_box = [""]  # 可修改容器，避免闭包global问题
last_built_prompt = ""
PROMPT_BRIDGE_PORT = 18000  # 拼装成品 POST 目标端口
RETRIEVAL_PORT = 18001     # 检索命中 POST 目标端口
RETRIEVAL_TOP_N = 1         # 发送几条命中
_retrieval_cache = ""       # daemon 缓存, 检索空时不覆盖桥
exec_queue = []              # 执行队列 (搜索/结束)
express_queue = []            # 表达队列 (回复/表情)
error_queue = []              # 报错队列 (独立, 不堵塞其他)
express_paused = False         # 表达队列暂停
exec_paused = False            # 执行队列暂停
error_paused = False           # 报错队列暂停
ERROR_INTERVAL = 3             # 报错间隔(秒)
EXPRESS_BRIDGE_PORT = 18004   # 表达队列完成信号桥
EXEC_BRIDGE_PORT = 18003       # 执行队列桥 (预留)

def _push_reply_to_queues(reply):
    """解析LLM回复中的工具, 推入对应队列（坏文本已由 _sanitize_reply 清洗）"""
    import re
    for m in re.finditer(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[TOOL\]', reply, re.DOTALL):
        cmd, arg = m.group(1), m.group(2).strip()
        item = {"cmd": cmd, "arg": arg, "time": datetime.datetime.now().strftime("%H:%M:%S"), "status": "等待"}
        # 非法表情兜底（2026-08-09）：不在 6 种标准 → 丢弃，防表达队列卡死（VAD 桥不认脏表情 → 轮询超时）
        # 覆盖所有入口（_do_send 清洗过 + /inject /chat 未清洗路径）
        if cmd == "表情" and arg not in VALID_EMOTIONS:
            print(f"[队列] 丢弃非法表情: {arg}")
            continue
        if cmd in ("回复", "表情"):
            express_queue.append(item)
        elif cmd == "报错":
            error_queue.append(item)
        elif cmd in ("搜索", "结束"):
            exec_queue.append(item)

# ── 记忆层级 ──
mem_l1 = []  # L1 段落摘要列表，元素为 (文本, ppl)
mem_l2 = []  # L2 会话摘要列表
mem_l3 = []  # L3 日常摘要列表，元素为 (文本, 开始时间, 结束时间)
mem_l4 = []  # L4 永久记忆列表，元素为 (文本, 类型, 时间)
l1_processed_idx = 0  # 已处理到历史第几条
SUMMARY_MODEL_URL = "http://localhost:18773/completion"

# 复读检测白名单：语气词/拟声词（单字），重复是正常表达不判复读
REPEAT_WHITELIST = set("喵哈呜嗯啊哦嘿啦呀呢吧嘛哇嘤咯噜啾汪咩哼唉哎嘻嘻嗷嗷咿噢哦吼诶欸")

# 合法表情（与系统提示一致）；grammar 掉线时的清洗层兜底
VALID_EMOTIONS = {"认真", "wink", "开心", "spin", "转头", "担心"}

# 工具格式 grammar（[^\[\]] 转义写法，减少 llama.cpp 解析失败概率）
GRAMMAR_TOOL = ("root ::= item*\nitem ::= [^\\[] | \"[\" ( [^Tt] | \"T\" [^Oo] | \"TO\" \"O\" \"L\" ( \":\" tool_args \"]\" | \"]\" ) )\n"
                "tool_args ::= \"表情\" \" \"? (\"认真\"|\"wink\"|\"开心\"|\"spin\"|\"转头\"|\"担心\") \" \"? \"[TOOL]\" | "
                "(\"回复\"|\"搜索\"|\"报错\"|\"结束\") \" \"? [^\\]]* \" \"? \"[TOOL]\"")


def _strip_repeat(text):
    """删除复读（自回归从某 token 开始复读即危险，连第一份一起删光）。
    白名单语气词（喵/哈/呜/啊…）组成的复读放行——拟声词重复是正常表达"""
    n = len(text)
    # 尾巴复读：尾部片段重复 ≥4 次 → 从复读开始整段删除
    for L in range(1, 11):
        if n < L * 4: continue
        tail = text[-L:]
        if all(ch in REPEAT_WHITELIST for ch in tail):
            continue                          # 语气词白名单 → 放行
        count, pos = 0, n - L
        while pos >= 0 and text[pos:pos + L] == tail:
            count += 1; pos -= L
        if count >= 4:
            return text[:pos + L]
    # 6x 任意重复：删掉重复块
    for L in range(2, 100):
        need = 6 if L <= 4 else 2
        if n < L * need: continue
        for start in range(0, min(L, n - L * need + 1)):
            chunk = text[start:start + L]
            if all(ch in REPEAT_WHITELIST for ch in chunk):
                continue
            ok = True
            for k in range(1, need):
                if text[start + k * L: start + (k + 1) * L] != chunk:
                    ok = False; break
            if ok:
                return text[:start] + text[start + need * L:]
    return text


# 未闭合/残片工具兜底：整个替换成报错工具（拉姆 2026-08-11：重拳，不补闭合不删除，保留错误信号）
BAD_TOOL_REPLACE = "[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL]"

# 报错惯性兜底（2026-08-11，拉姆）：把报错当回复用 = 坏文本，在下桥清洗层替换（防错误内容进上文误导后续）
# 上轮主动报错 + 本轮又报错 + 内容非标准 → 本轮偷换成回复（下一轮自由）
_last_round_error = False
ERROR_STD_TEXT = "有没有人告诉拉姆我的ai出问题了"   # 系统提示里的标准报错文案

# ── 钻牛角尖检测（2026-08-11，拉姆）：回复与上一轮有 ≥4 字公共片段, 连续第3轮同一片段 → 替换成 wink ──
_prev_reply_text = ""      # 上一轮回复文本（用于比较）
_dup_state = {"phrase": None, "streak": 0}   # 正在追踪的重复短语 + 连续命中轮数

def _lcs_substring(a, b, min_len=4):
    """a、b 的最长公共连续子串（从长到短滑窗）；>=min_len 返回，否则 None。纯语气词子串不算"""
    if not a or not b:
        return None
    for L in range(min(len(a), len(b)), min_len - 1, -1):
        seen = set()
        for i in range(len(a) - L + 1):
            sub = a[i:i + L]
            if sub in seen:
                continue
            seen.add(sub)
            if sub in b and not all(ch in REPEAT_WHITELIST for ch in sub):
                return sub
    return None

def _check_drill(arg):
    """与上一轮回复比较: 有 ≥4 字公共片段 → 计入连续轮数; 连续第3轮同一片段 → 触发(返回True)并重置"""
    global _prev_reply_text, _dup_state
    phrase = _lcs_substring(arg, _prev_reply_text) if _prev_reply_text else None
    if phrase:
        if _dup_state["phrase"] == phrase:
            _dup_state["streak"] += 1
        else:
            _dup_state = {"phrase": phrase, "streak": 1}
        _prev_reply_text = arg
        if _dup_state["streak"] >= 2:     # 轮2 streak=1, 轮3 streak=2 → 触发
            _dup_state = {"phrase": None, "streak": 0}
            _prev_reply_text = ""
            return True
        return False
    _dup_state = {"phrase": None, "streak": 0}
    _prev_reply_text = arg
    return False


def _sanitize_reply(reply):
    """LLM 回复下桥清洗（下桥后第一步，history/上文记忆/检索记忆全部基于干净回复）：
    ① 删复读（从复读开始整段删）
    ② [TOOL:xxx] 打开后，arg 内任何 [ 开头但非 [TOOL] 闭合的结构 = 非法 → 删到第一个合法 [TOOL] 前
    ③ 未闭合（[TOOL:xxx 无正确 [TOOL] 闭合）→ 整个替换成报错工具（不补闭合）
    ④ 孤立 [TOOL]（无对应 [TOOL:xxx]）→ 删除；孤立 [ 残片（[OL]/[OL 等非 TOOL 开头）→ 替换成报错工具
    ⑤ 非法表情（不在 6 种标准）→ 整个工具丢弃
    ⑥ 报错惯性（2026-08-11）：上轮主动报错 + 本轮报错 + 内容非标准 → 偷换成回复（报错当回复用=坏文本，防进上文）"""
    global _last_round_error
    used_error = False   # 本轮主动报错（清洗兜底产生的报错不计，避免清洗自身触发惯性）
    if not reply:
        return reply
    reply = _strip_repeat(reply)
    out = []
    i, n = 0, len(reply)
    open_cmd = None
    while i < n:
        m = re.search(r'\[TOOL:(\w+)\]', reply[i:])
        if not m:
            rest = reply[i:]
            if open_cmd is not None:
                # ③ 未闭合工具（[TOOL:xxx 无正确 [TOOL] 闭合）→ 整个替换成报错工具
                out.pop()   # 去掉已 append 的 [TOOL:xxx]
                out.append(BAD_TOOL_REPLACE)
            else:
                # ④ 普通文本：删孤立 [TOOL]；[ 开头非 TOOL 的残片（[OL]/[OL/[等）→ 替换成报错工具
                rest = re.sub(r'\[TOOL\]', '', rest)
                rest = re.sub(r'\[(?!TOOL)[^\[\]]*\]?', BAD_TOOL_REPLACE, rest)
                out.append(rest)
            break
        start = i + m.start()
        out.append(reply[i:start])            # 工具前文本（thought / 闭合后文本）
        cmd = m.group(1)
        out.append(f"[TOOL:{cmd}]")
        i = start + len(f"[TOOL:{cmd}]")
        open_cmd = cmd
        seg = reply[i:]
        c = re.search(r'\[TOOL\]', seg)
        if c:
            arg_raw = _strip_repeat(seg[:c.start()])   # arg 内复读也删（闭合工具场景）
            # ② arg 内任何 [ 开头（非闭合）→ 非法 → 删到闭合前
            bad = arg_raw.find('[')
            arg = arg_raw[:bad] if bad >= 0 else arg_raw
            if cmd == "表情" and arg.strip() not in VALID_EMOTIONS:
                out.pop()   # ⑤ 非法表情（不在 6 种）→ 整个工具丢弃（grammar 掉线兜底）
            else:
                # ⑥ 报错惯性兜底：上轮主动报错 + 本轮报错 + 内容非标准 → 偷换成回复（防坏文本进上文）
                if cmd == "报错" and _last_round_error and arg.strip() != ERROR_STD_TEXT:
                    print(f"[防惯性] 连续报错非标准 → 偷换成回复: {arg[:24]}")
                    cmd = "回复"
                    out[-1] = "[TOOL:回复]"   # 替换刚 append 的 [TOOL:报错]
                if cmd == "报错":
                    used_error = True
                # ⑦ 钻牛角尖（2026-08-11）：回复与上轮有 ≥4 字公共片段连续3轮 → 替换成 wink（打断钻牛角尖）
                if cmd == "回复" and arg.strip() and _check_drill(arg.strip()):
                    print(f"[防牛角尖] 连续3轮重复短语 → 替换 wink | 原回复: {arg[:24]}")
                    cmd = "表情"
                    arg = "wink"
                    out[-1] = "[TOOL:表情]"   # 替换刚 append 的 [TOOL:回复]
                out.append(arg)
                out.append("[TOOL]")
            i += c.end()
            open_cmd = None
        else:
            # ③ 未闭合工具（[TOOL:xxx 后面没有 [TOOL] 闭合）→ 整个替换成报错工具
            out.pop()   # 去掉已 append 的 [TOOL:xxx]
            out.append(BAD_TOOL_REPLACE)
            break
    _last_round_error = used_error
    return "".join(out).strip()


def parse_reply(reply):
    """解析模型输出，拆分为内心思考、回复、工具调用"""
    thought = ""
    tool_reply = ""
    tools = []
    import re
    # 找第一个工具调用位置
    first_tool = re.search(r'\[TOOL:', reply)
    if first_tool:
        thought = reply[:first_tool.start()].strip()
        rest = reply[first_tool.start():]
    else:
        thought = reply
        rest = ""
    # 提取所有工具
    tool_reply = thought  # 默认：如果没有 [TOOL:回复]，thought 就是"回答"
    for tm in re.finditer(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[TOOL\]', rest, re.DOTALL):
        cmd = tm.group(1)
        arg = tm.group(2).strip()
        if cmd == "回复":
            tool_reply = arg
        else:
            tools.append({"cmd": cmd, "arg": arg})
    return thought, tool_reply, tools

L1_PROMPT = "用第一人称把一段对话压缩成一句话。只压缩，不解释。\n\n"
L2_PROMPT = "你是一个记忆管理器。找到以下对话中最重要的一个主题，并且说出谁支持这个主题，谁反对这个主题。你只需要输出主题，以及支持者和反对者。不需要说明支持的理由，也不用输出其他东西。\n格式：**主题**：xx **支持者**：xx **反对者**：xx。每项在一行。\n\n"
L3_PROMPT = "通过我参与的几条闲聊的摘要，判断我和谁进行了闲聊，然后输出：\"我\"和哪些人进行了闲聊。此外不需要解释也不用输出其他的\n\n"

# TODO: 第一阶段 LLM 判断"说的是不是我" + 替换称呼（硬币你→我等）
# 当前只做纯代码的格式化：标记来源名、清理工具标记

def preprocess_l1(user_text, thought, tool_reply, src_names):
    """预处理 L1 输入：标记来源名、清理格式"""
    # 1. 标记来源名
    # 无需替换 user_text 中的名字，来源名已在 format 中处理
    # 2. 清理想法中的来源名（改为"有人说"）+ 工具标记
    for name in src_names:
        thought = re.sub(re.escape(name) + r'说', '有人说', thought)
    thought = re.sub(r'\[TOOL:\w+\][^]]*\[TOOL\]', '', thought).strip()
    # 3. 清理回复中的工具标记
    tool_desc = re.sub(r'\[TOOL:\w+\][^]]*\[TOOL\]', '', tool_reply).strip()
    return user_text, thought, tool_desc

def summarize(text, prompt_template, max_tokens=300):
    """调用摘要模型，返回 (摘要文本, ppl分数)"""
    full_prompt = f"<|im_start|>system\n{prompt_template}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>"
    try:
        r = requests.post(SUMMARY_MODEL_URL, json={
            "prompt": full_prompt, "n_predict": max_tokens,
            "temperature": 0.3, "n_probs": 1,
            "stop": ["<|im_end|>", "\n\n\n"]
        }, timeout=60)  # 60s timeout for slower generations
        data = r.json()
        content = data.get("content", "").strip()
        # 从 token probs 计算 PPL
        comps = data.get("completion_probabilities", [])
        if comps and content:
            logprobs = []
            for cp in comps:
                lp = cp.get("logprob")
                if lp is not None:
                    logprobs.append(lp)
            ppl = float(np.exp(-np.mean(logprobs))) if logprobs else 0.0
            print(f"[PPL调试] comps数量={len(comps)}, logprobs数量={len(logprobs)}, 均值={np.mean(logprobs) if logprobs else 0:.4f}, ppl={ppl:.4f}")
            if not logprobs or ppl < 1.5:
                print(f"[PPL原始] {str(comps[:5])}")
        else:
            ppl = 0.0
    except Exception as e:
        print(f"[摘要错误] {e}")
        content = ""
        ppl = 0.0
    # 空内容重试一次
    if not content:
        print(f"[摘要空响应] 重试...")
        try:
            r2 = requests.post(SUMMARY_MODEL_URL, json={
                "prompt": full_prompt, "n_predict": max_tokens,
                "temperature": 0.3, "n_probs": 1,
                "stop": ["<|im_end|>", "\n\n\n"]
            }, timeout=60)
            content = r2.json().get("content", "").strip()
        except Exception as e2:
            print(f"[摘要重试错误] {e2}")
    return content, ppl

def calc_ppl(text):
    """算一段文本本身的 PPL（无上下文，纯文本）"""
    try:
        r = requests.post(SUMMARY_MODEL_URL, json={
            "prompt": text, "n_predict": 0, "n_probs": 1
        }, timeout=30)
        data = r.json()
        comps = data.get("completion_probabilities", [])
        if comps:
            logprobs = [cp.get("logprob") for cp in comps if cp.get("logprob") is not None]
            if logprobs:
                ppl = float(np.exp(-np.mean(logprobs)))
                print(f"[PPL纯文本] {text[:30]}... → {ppl:.2f}")
                return ppl
        return 0.0
    except Exception as e:
        print(f"[PPL错误] {e}")
        return 0.0

def build_l05(user_raw, reply_full):
    """
    把一轮对话（L0）预处理成 L0.5 第一视角转写。
    返回 (user_part, thought, action, l05_text) 或 None
    """
    thought, tool_reply, tools = parse_reply(reply_full)

    # 工具调用 → 动作描述
    action_lines = []
    for t in tools:
        cmd = t["cmd"]
        arg = t["arg"]
        if cmd == "回复":
            action_lines.append(f"我说：{arg}")
        elif cmd == "报错":
            action_lines.append("我故意报错")
        elif cmd == "搜索":
            action_lines.append(f"我搜索：{arg}")
        elif cmd == "表情" or cmd == "结束":
            continue
        elif cmd == "结束":
            action_lines.append("我等一下再回应")
    if not action_lines and tool_reply:
        action_lines.append(f"我说：{tool_reply}")
    action_desc = "\n".join(action_lines) if action_lines else "（沉默）"

    lines = user_raw.split("\n")
    src_names = []
    raw_inputs = []
    for line in lines:
        line = line.strip()
        m = re.match(r'(?:对话|弹幕)\[(.+?)\]：(.*)', line)
        if m:
            src_names.append(m.group(1))
            raw_inputs.append((m.group(1), m.group(2).strip()))
        elif line:
            raw_inputs.append(("", line))

    processed = []
    final_thought = re.sub(r'\[TOOL:\w+\][^]]*\[TOOL\]', '', thought).strip()
    final_tool = action_desc
    for name, txt in raw_inputs:
        if name:
            ut, th, tl = preprocess_l1(txt, thought, tool_reply, [name])
            processed.append(f"{name}对我说：{ut}")
            final_thought = th
        else:
            processed.append(txt)

    user_part = "\n".join(processed)
    l05_text = f"{user_part}\n我想：{final_thought}\n{final_tool}"
    return user_part, final_thought, final_tool, l05_text


def trigger_retrieval():
    """
    长期记忆检索：处理【最新一轮】（不等待，实时）。
    LLM 回复完成后调用，把 L0.5 POST 给检索服务，结果存 last_retrieval。
    L0/L0.5 本地即时生成（不依赖检索服务）；标签计算和命中依赖 18774。
    """
    global last_retrieval
    try:
        if len(history) < 2:
            return
        # 取最新一轮（最后两条）
        user_raw = history[-2][1]
        reply_full = history[-1][1]
        built = build_l05(user_raw, reply_full)
        if not built:
            return
        user_part, thought, action, l05_text = built

        # 本地先填 L0/L0.5（检索服务挂了这两块也显示）
        with last_retrieval_lock:
            last_retrieval = {
                "l0": {"user": user_part, "thought": thought, "action": action},
                "l05": l05_text,
                "labels": None,
                "hit": None,
                "error": None,
            }

        payload = json.dumps({
            "user": user_part,
            "thought": thought,
            "action": action,
        }, ensure_ascii=False).encode("utf-8")
        r = requests.post(RETRIEVE_API, data=payload,
                          headers={"Content-Type": "application/json"}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            with last_retrieval_lock:
                last_retrieval = data
            hit_text = ''
            cands = data.get('hit', {}).get('candidates', [])
            if cands:
                hit_text = cands[0].get('text', '')[:40]
            print(f"[检索] 完成 → 命中: {hit_text}")
        else:
            with last_retrieval_lock:
                last_retrieval["error"] = f"服务返回 {r.status_code}: {r.text[:80]}"
            print(f"[检索] 服务返回 {r.status_code}: {r.text[:100]}")
    except requests.exceptions.ConnectionError:
        with last_retrieval_lock:
            if last_retrieval:
                last_retrieval["error"] = "检索服务离线（18774）"
        print("[检索] 服务离线（18774），L0/L0.5 已本地生成")
    except Exception as e:
        with last_retrieval_lock:
            if last_retrieval:
                last_retrieval["error"] = str(e)
        print(f"[检索] 错误: {e}")


def trigger_summarize():
    """检查是否需要触发摘要：L0→L1, L1满10→L2"""
    global mem_l1, mem_l2, history, l1_processed_idx
    # 超过5轮（10条）且有未处理的老轮次时才触发
    if len(history) > 10 and l1_processed_idx < len(history):
        # 取最先未处理的那一轮（user + assistant）
        user_raw = history[l1_processed_idx][1]
        reply_full = history[l1_processed_idx + 1][1]
        thought, tool_reply, tools = parse_reply(reply_full)

        # 把工具调用转为动作描述
        action_lines = []
        for t in tools:
            cmd = t["cmd"]
            arg = t["arg"]
            if cmd == "回复":
                action_lines.append(f"我说：{arg}")
            elif cmd == "报错":
                action_lines.append("我故意报错")
            elif cmd == "搜索":
                action_lines.append(f"我搜索：{arg}")
            elif cmd == "表情" or cmd == "结束":
                continue  # 不写进摘要
            elif cmd == "结束":
                action_lines.append("我等一下再回应")
        if not action_lines and tool_reply:
            action_lines.append(f"我说：{tool_reply}")
        action_desc = "\n".join(action_lines) if action_lines else "（沉默）"

        lines = user_raw.split("\n")
        src_names = []
        raw_inputs = []
        for line in lines:
            line = line.strip()
            m = re.match(r'(?:对话|弹幕)\[(.+?)\]：(.*)', line)
            if m:
                src_names.append(m.group(1))
                raw_inputs.append((m.group(1), m.group(2).strip()))
            elif line:
                raw_inputs.append(("", line))

        # 对每个来源做预处理
        processed = []
        final_thought = re.sub(r'\[TOOL:\w+\][^]]*\[TOOL\]', '', thought).strip()
        final_tool = action_desc
        for name, txt in raw_inputs:
            if name:
                ut, th, tl = preprocess_l1(txt, thought, tool_reply, [name])
                processed.append(f"{name}对我说：{ut}")
                final_thought = th
            else:
                processed.append(txt)

        user_part = "\n".join(processed)

        # L1 摘要 + PPL
        input_text = f"{user_part}\n我想：{final_thought}\n{final_tool}\n<think>\n</think>\n开始生成压缩内容："
        print(f"[L1处理] idx={l1_processed_idx} 输入前80字={input_text[:80]}")
        summary, _ = summarize(input_text, L1_PROMPT)
        if summary:
            raw_ppl = calc_ppl(summary)  # 纯文本 PPL
            mem_l1.append((summary, raw_ppl))
            print(f"[L1] {summary} (raw_ppl={raw_ppl:.2f})")
            # L1→L4：极高PPL直接进永久记忆
            if raw_ppl > 15:
                ts = datetime.datetime.now().strftime("%m-%d %H:%M")
                mem_l4.append((summary, "重要事件", ts))
                print(f"[L1→L4] 极高PPL条目已存储")
        l1_processed_idx += 2  # 标记已处理

        # L1→L2：满6条时合并最老3条，PPL<5的丢弃
        if len(mem_l1) >= 6:
            batch = mem_l1[:3]  # 取最老3条
            filtered = [(s, p) for s, p in batch if p >= 5.0]
            print(f"[L2触发] {len(batch)}条中 {len(filtered)}条通过PPL过滤")
            if filtered:
                merge_text = "\n".join(f"- {s}" for s, p in filtered) + "\n合并："
                l2, _ = summarize(merge_text, L2_PROMPT, max_tokens=500)
                if l2:
                    ts = datetime.datetime.now().strftime("%m-%d %H:%M")
                    mem_l2.append((l2, ts))
                    print(f"[L2] {l2}")
            # 移除已处理的3条（最老的）
            mem_l1 = mem_l1[3:]

        return True
    return False



def call_deepseek(prompt, system="", max_tokens=128):
    """调用 DeepSeek Chat API"""
    DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_KEY = os.environ.get("MINICOIN_LLM_API_KEY", "")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(DEEPSEEK_API, json={
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }, headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"}, timeout=30)
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[DeepSeek错误] {e}")
        return ""

def trigger_l3():
    """独立触发 L2→L3 合并"""
    global mem_l2, mem_l3, mem_l4
    if len(mem_l2) >= 6:
        batch_l2 = mem_l2[:3]  # 取最老3条
        start_t = batch_l2[0][1]
        end_t = batch_l2[-1][1]
        start_t = batch_l2[0][1]
        end_t = batch_l2[-1][1]
        merge_text = "\n".join(f"- {s}" for s, t in batch_l2)

        # 调4B
        print(f"[L3输入]\n{L3_PROMPT}{merge_text}")
        l3, _ = summarize(merge_text, L3_PROMPT, max_tokens=128)
        if not l3:
            l3 = "我和其他人进行了闲聊"
        ts = f"{start_t}~{end_t}"
        mem_l3.append((l3, start_t, end_t))
        mem_l4.append((l3, "日常", ts))
        print(f"[L3] {l3}")
        mem_l2 = mem_l2[3:]  # 移除最老的3条


def _export_l4():
    """导出 L4 永久记忆到 md（时间戳命名 → 长期记忆/未整理长期记忆/）"""
    if not mem_l4:
        print("[L4导出] L4 为空，跳过")
        return ""
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "长期记忆", "未整理长期记忆")
    os.makedirs(export_dir, exist_ok=True)
    now = datetime.datetime.now()
    path = os.path.join(export_dir, f"L4_{now.strftime('%Y%m%d_%H%M%S')}.md")
    lines = [f"# 未整理长期记忆 - {now.strftime('%Y-%m-%d %H:%M')}",
             f"共 {len(mem_l4)} 条", ""]
    for s, tp, t in mem_l4:
        lines.append(f"- [{tp}] {s} ({t})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[L4导出] {len(mem_l4)} 条 → {path}")
    return path


def _l4_daily_export_loop():
    """每天 00:00 自动导出一次 L4（00:00 分钟窗口 + 跨天防重）"""
    last = ""
    while True:
        time.sleep(30)
        now = datetime.datetime.now()
        day = now.strftime("%Y-%m-%d")
        if now.hour == 0 and now.minute == 0 and day != last:
            try:
                _export_l4()
            except Exception as e:
                print(f"[L4导出] 失败: {e}")
            last = day


def _build_final_context():
    """拼装成品：话题摘要(L2) + 最近摘要(L1) + 对话历史(L0)。
    实时读 mem_l2 / mem_l1 / history，页面预览与推桥共用同一份，防止两处漂移"""
    all_ctx = [{"role": "输入" if r == "user" else "硬币", "content": c[:500]} for r, c in history]
    ctx_lines = []
    if mem_l2:
        ctx_lines.append("[话题摘要]")
        for s, t in mem_l2[-2:]:   # 最新 2 条话题摘要（2026-08-11 修正：原 [:2] 永远显示最老2条）
            one_line = s.replace('\n', ' ').replace('  ', ' ')
            ctx_lines.append(f" {one_line};")
    if mem_l1:
        ctx_lines.append("[最近摘要]")
        for i, (s, p) in enumerate(mem_l1):
            end = ";" if i < len(mem_l1) - 1 else ""
            ctx_lines.append(f" {s}{end}")
    ctx_lines.append("[对话历史]")
    ctx_lines.append("")
    for c in all_ctx[-10:]:   # 与页面 L0(mem_l0) 同源同法：最近10条原文直显，不配对（2026-08-11 修复乱拼）
        ctx_lines.append(c["content"][:500])
        ctx_lines.append("")
    return "\n".join(ctx_lines)

def build_prompt(input_text):
    """按管线拼装完整 prompt"""
    global last_built_prompt
    parts = []
    with pipeline_lock:
        active = [m for m in pipeline if m["enabled"]]

    for m in active:
        show_src = m.get('show_src', True)   # 模块配置：是否显示源名（默认显示）
        if m["id"] == "chat":
            # chat 多源：谁有内容谁出现（对话[源名]：内容），没内容的源跳过
            lines = []
            for s in m.get("sources", []):
                if not s.get("active"):
                    continue
                ep = str(s.get("endpoint", "") or "")
                if ep.isdigit():
                    ep = f"http://localhost:{ep}"
                if not ep:
                    continue
                sname = s.get("name", "")
                try:
                    r = requests.get(f"{ep}/source_last", timeout=2)
                    if r.status_code == 200:
                        data = r.json()
                        if isinstance(data, dict):
                            txt = next((v for v in data.values() if v and v != "wait"), "")
                            if txt:
                                lines.append(f"对话[{sname}]：{txt}")
                except Exception:
                    pass
            content = "\n".join(lines) if lines else ""
        elif m["id"] == "danmaku":
            src_name, txt = _read_from_source(m)
            if txt:
                content = f"{src_name}\n{txt}" if show_src else txt
            else:
                content = ""
        elif m["id"] == "summary" or m["id"] == "memory" or m["id"] == "tool":
            src_name, content = _read_from_source(m)
            if content and src_name and show_src:
                content = f"{src_name}\n{content}"
            if not content and m["id"] == "summary" and len(history) > 8:
                content = f"以上共有{len(history)//2}轮对话"
            if not content and m["id"] == "memory":
                content = m.get("content", "")
        else:
            content = m.get("content", "")

        if not content:
            continue
        if content:
            # 桥/输入内容前统一加换行
            if m["id"] in ("summary", "memory", "chat", "danmaku"):
                content = "\n" + content
            parts.append(f"{m['format_start']}{content}{m['format_end']}")

    parts.append("<|im_start|>硬币\n")
    last_built_prompt = "\n".join(parts)
    return last_built_prompt


def _source_read(port, name=""):
    """读源：从背后桥取第一个非空 text（wait 视为无内容）。兼容内置桥/外部桥"""
    ep = str(port)
    if ep.isdigit():
        ep = f"http://localhost:{ep}"
    try:
        r = requests.get(f"{ep}/source_last", timeout=2)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return next((v for v in data.values() if v and v != "wait"), "")
    except Exception:
        pass
    return ""


def _source_clear(port, name=""):
    """清源：清空背后桥内容置 wait（消费一次后避免反复触发）。
    内置桥(18000-18009)同进程直清内存；外部桥读全量后所有字段置 wait + 放 wait 哨兵"""
    p = str(port)
    if p.isdigit() and 18000 <= int(p) <= 18009:
        with _store_lock:
            d = _source_stores.get(int(p), {})
            for k in list(d.keys()):
                d[k] = "wait"
    else:
        try:
            # 外部桥：读全量，所有字段清成 wait（防 source 名错配残留，如 ASR 写 asr / 模块源名拉姆）
            r = requests.get(f"http://localhost:{p}/source_last", timeout=2).json() or {}
            for k in list(r.keys()):
                requests.post(f"http://localhost:{p}/text",
                              json={"source": k, "text": "wait"}, timeout=2)
            # 全清后放 wait 哨兵（桥为空时兜底，语义与内置桥一致）
            requests.post(f"http://localhost:{p}/text",
                          json={"source": name or "wait", "text": "wait"}, timeout=2)
        except Exception:
            try:
                requests.post(f"http://localhost:{p}/text",
                              json={"source": name or "wait", "text": "wait"}, timeout=2)
            except Exception:
                pass


def _read_from_source(module):
    """从模块 source 的桥读内容。兜底源：不论桥上 source 名是什么，取第一个非空 text（wait 视为无内容）；
    name 只是显示标签（显示源名时用），不是过滤条件"""
    if not module.get("sources"):
        return "", ""
    for src in module["sources"]:
        if not src.get("active"):
            continue
        ep = str(src.get("endpoint", "") or "")
        if not ep:
            continue
        name = src.get("name", "")
        txt = _source_read(ep, name)
        if txt:
            return name, txt
    return "", ""


def _is_danmaku_port(port):
    """端口是否为弹幕模块（enabled 且 active source）的 endpoint"""
    with pipeline_lock:
        for m in pipeline:
            if m.get("id") == "danmaku" and m.get("enabled"):
                for s in m.get("sources", []):
                    if s.get("active") and str(s.get("endpoint", "")) == str(port):
                        return True
    return False


def danmaku_reply():
    """弹幕输入到达 → 标记积压（user 轮由 _do_send 成功后从桥记录，保证 L0 只含 LLM 真实看见的输入）"""
    try:
        _mark_pending()
    except Exception as e:
        print(f"[弹幕输入] {e}")


# ── LLM 发送状态机：表达队列门控 + 发送后强制冷却 + 积压合并 + 系统源注入 ──
# 冷却逻辑（2026-08-09 改，双门控）：
#   1. 表达队列门控：express_queue 非空（TTS/表情播放中）→ 压住，播完才放行
#   2. 发送后强制冷却：_do_send 完成后 SEND_POST_COOLDOWN 秒内不处理积压
#      ——覆盖"LLM 推理返回→表达队列填充"的窗口，防止推理期间新输入
#        被 _pending_send 误清 / 乱序；2s 一般不卡对话，表达队列也够填充了
_llm_busy = False       # LLM 请求在途（_do_send 内置位/解锁，防御性保留）
_pending_send = False    # 有积压输入待发送
_pending_lock = threading.Lock()
_last_send_ts = 0.0      # 上次发送时间戳（强制冷却 + 静默间隔判断）
SEND_POST_COOLDOWN = 2.0 # 发送后强制冷却秒数


def _sys_source_active():
    """注入桥是否有 chat 模块的 active 源在使用（系统源关闭则停止注入）"""
    bridge = str(automation.get("sys_bridge", ""))
    with pipeline_lock:
        for m in pipeline:
            if m.get("id") == "chat":
                for s in m.get("sources", []):
                    if s.get("active") and str(s.get("endpoint", "")) == bridge:
                        return True
    return False


def _inject_system_prompt():
    """静默达到间隔 → 随机选一条提示 → POST 到自动化配置的注入桥（与源名解耦）"""
    import random as _r
    port = str(automation.get("sys_bridge", ""))
    if not port:
        return
    prompts = automation.get("sys_prompts") or ["你可以继续往下说"]
    text = _r.choice(prompts)
    try:
        requests.post(f"http://localhost:{port}/text",
                      json={"source": "系统", "text": text}, timeout=2)
        print(f"[自动化] 注入到桥 {port}: {text}")
    except Exception as e:
        print(f"[自动化] 注入失败: {e}")


def _mark_pending():
    """输入到达 → 标记积压（冷却期内只拼 prompt，压住发送信号）"""
    global _pending_send
    with _pending_lock:
        _pending_send = True


def _check_trigger_sources():
    """轮询检测（ASR 驱动）：任一触发源桥有非空内容（wait 视为空）→ 标记积压。
    幂等操作（只置 bool），user 轮由 _do_send 成功后统一记录 → 无重复记录问题"""
    for port, name in _trigger_sources():
        if _source_read(port, name):
            _mark_pending()
            return


def _write_chat_bridge(source, text):
    """手动对话输入：写入 chat 模块对应源名的桥（与 ASR 写桥同路径，由检测触发）"""
    with pipeline_lock:
        for m in pipeline:
            if m.get("id") != "chat":
                continue
            for s in m.get("sources", []):
                if not s.get("active"):
                    continue
                if source and s.get("name") != source:
                    continue
                ep = str(s.get("endpoint", "") or "")
                if ep.isdigit():
                    ep = f"http://localhost:{ep}"
                if not ep:
                    continue
                try:
                    requests.post(f"{ep}/text",
                                  json={"source": s.get("name", "拉姆"), "text": text}, timeout=2)
                except Exception:
                    pass
                return


def _trigger_sources():
    """触发源列表 [(port, name)]：弹幕模块源 + chat 全部源 + 自动化注入源"""
    srcs = []
    with pipeline_lock:
        for m in pipeline:
            if m.get("id") in ("danmaku", "chat"):
                for s in m.get("sources", []):
                    if s.get("active") and s.get("endpoint"):
                        srcs.append((str(s.get("endpoint")), s.get("name", "")))
    ab = str(automation.get("sys_bridge", ""))
    if ab and not any(p == ab for p, _ in srcs):
        srcs.append((ab, "系统"))
    return srcs


def _consume_trigger_sources():
    """push 成功后清空所有触发源（清桥置 wait），避免反复触发。无论这次谁触发的都全清"""
    for port, name in _trigger_sources():
        _source_clear(port, name)
    print("[消费] 触发源已清空")


def _do_send():
    """统一发送：build_prompt + 发 LLM + 回复处理（对话/弹幕/系统共用同一路径）
    L0 哲学：只有 LLM 处理成功的输入才记录 user 轮（清桥前从触发源桥读）
    门控：_llm_busy 置位期间 _send_loop 压住——LLM 推理+回复进队列完成前不发下一条"""
    global _llm_busy
    _llm_busy = True
    try:
        prompt = build_prompt("")
        payload = json.dumps({
            "prompt": prompt, "n_predict": 500, "temperature": 0.9,
            "grammar": GRAMMAR_TOOL,
            "logit_bias": [[1930, 10.0], [3925, -10.0]]
        }).encode()
        print(f"[LLM请求] {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} prompt_len={len(prompt)} 尾部40字={prompt[-40:]!r}")
        r = requests.post(BRIDGE_URL, data=payload,
                          headers={"Content-Type": "application/json"}, timeout=60)
        reply = r.json().get("content", "")
        print(f"[LLM回复-原始] {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} len={len(reply)} 前800字={reply[:800]!r}")
        reply = _sanitize_reply(reply)   # 下桥清洗第一步：删复读 + 工具闭合/嵌套清理（后续全用干净回复）
        if not reply:
            # 空回复（模型 EOS 直接空 / 响应无 content）→ 判失败：丢这一轮
            # 不记录 L0、不进队列，清触发源桥（输入当不存在，防污染）；下一轮输入正常触发
            print("[LLM回复] 空回复，丢弃本轮")
            _consume_trigger_sources()
            return
        last_reply_box[0] = reply
        _push_reply_to_queues(reply)
        # 输出侧 VAD：LLM 回复 → FACS /output（驱动"我说我/我说你"表情）
        # 方案 B：thought（[TOOL: 前部分）+ [TOOL:回复] 的纯文本 arg；无回复工具则用全文
        try:
            _vad_text = reply.split('[TOOL:')[0].strip()
            _vm = re.search(r'\[TOOL:回复\]\s*(.*?)\s*\[TOOL\]', reply, re.DOTALL)
            if _vm:
                _vad_text = (_vad_text + ' ' + _vm.group(1).strip()).strip()
            if _vad_text:
                requests.post("http://localhost:18768/output",
                              json={"text": _vad_text}, timeout=2)
        except Exception:
            pass
        # —— user 轮：LLM 真实看见的输入（发送成功、清桥前，从触发源桥读）——
        user_parts = []
        with pipeline_lock:
            for m in pipeline:
                if m.get("id") == "chat":
                    for s in m.get("sources", []):
                        if s.get("active"):
                            txt = _source_read(str(s.get("endpoint", "")), s.get("name", ""))
                            if txt:
                                user_parts.append(f"对话[{s.get('name')}]：{txt}")
                elif m.get("id") == "danmaku":
                    _, dm = _read_from_source(m)
                    if dm:
                        user_parts.append(dm)
        if user_parts:
            history.append(("user", "\n".join(user_parts)))
        history.append(("assistant", reply))
        print(f"[LLM回复] {reply[:60]}")
        trigger_summarize()
        trigger_retrieval()
        _consume_trigger_sources()   # push 成功 → 清空所有触发源桥（弹幕/拉姆/系统）
    finally:
        _llm_busy = False   # 无论成功/空回复/异常都解锁（空回复无队列内容 → 立即放行下一条）


def _send_loop():
    """发送 daemon：表达队列清空 + 无进行中 LLM → 有积压则发送；静默 30s 且系统源激活 → 注入系统提示"""
    import time as _t
    global _pending_send, _last_send_ts
    while True:
        _t.sleep(0.5)
        now = _t.time()
        _check_trigger_sources()   # ASR 驱动：触发源桥非空 → 标记积压（幂等）
        with _pending_lock:
            pending = _pending_send
        if _llm_busy or express_queue:
            continue  # LLM 推理中 / 上一条还没说完（表达队列未清空）→ 压住发送
        if now - _last_send_ts < SEND_POST_COOLDOWN:
            continue  # 发送后强制冷却：刚发完 2s 内不处理积压（覆盖 LLM 推理返回→表达队列填充的窗口）
        if pending:
            try:
                _do_send()
            except Exception as e:
                print(f"[发送错误] {e}")
                _consume_trigger_sources()   # 失败 → 丢这一轮（输入当不存在，防污染）
            with _pending_lock:
                _pending_send = False
            _last_send_ts = _t.time()
        elif now - _last_send_ts >= float(automation.get("sys_interval", 30)):
            # 静默达到间隔 → 系统源仍激活才注入（关闭则跳过，等源重新打开后立即触发）
            if _sys_source_active():
                _inject_system_prompt()
                _last_send_ts = now          # 防连续注入
                with _pending_lock:
                    _pending_send = True


def _probe_bridge(port):
    """探测桥连通性：GET /source_last 通=up"""
    try:
        r = requests.get(f"http://localhost:{port}/source_last", timeout=1)
        return "up" if r.status_code == 200 else "down"
    except Exception:
        return "down"

def _get_bridge_backend():
    """查询 LLM 桥当前后端模式 (api/local)，桥不可达返回 unknown"""
    try:
        r = requests.get(f"http://localhost:{LLM_BRIDGE_PORT}/api/backend", timeout=2)
        d = r.json()
        return d.get("backend", "unknown") if r.status_code == 200 else "unknown"
    except Exception:
        return "unknown"


def _get_bridges():
    """收集所有桥（内置/系统/模块配置）+ 占用 + 连通状态"""
    sys_bridges = [
        (18771, "LLM 桥", ["对话回复", "弹幕回复"]),
        (18773, "摘要模型", ["历史摘要"]),
        (18774, "检索服务", ["长期记忆检索"]),
        (18775, "搜索工具", ["工具搜索"]),
        (8000,  "TTS 桥", ["表达队列→语音"]),
        (18000, "拼装成品桥", ["拼装成品推送"]),
        (18003, "执行队列桥", ["执行队列 done 信号"]),
        (18004, "表达队列桥", ["表达队列 done 信号"]),
    ]
    bridges = {}
    for port, name, occ in sys_bridges:
        bridges[str(port)] = {"port": port, "name": name, "type": "system", "occupier": list(occ)}
    for p in range(18000, 18010):   # 内置桥（管线启动的）
        k = str(p)
        if k not in bridges:
            bridges[k] = {"port": p, "name": "内置桥", "type": "builtin", "occupier": []}
    # 模块 source 占用（动态扫配置）
    with pipeline_lock:
        for m in pipeline:
            for s in m.get("sources", []):
                ep = str(s.get("endpoint", "") or "")
                if ep.isdigit() and len(ep) <= 5:
                    if ep not in bridges:
                        bridges[ep] = {"port": int(ep), "name": "配置桥", "type": "module", "occupier": []}
                    label = m.get("label", m.get("id", "?"))
                    if label not in bridges[ep]["occupier"]:
                        bridges[ep]["occupier"].append(label)
    # 自动化注入桥占用标注
    ab = str(automation.get("sys_bridge", ""))
    if ab.isdigit() and len(ab) <= 5:
        if ab not in bridges:
            bridges[ab] = {"port": int(ab), "name": "配置桥", "type": "module", "occupier": []}
        if "自动化·系统提示" not in bridges[ab]["occupier"]:
            bridges[ab]["occupier"].append("自动化·系统提示")
    result = list(bridges.values())
    result.sort(key=lambda b: b["port"])
    # 并发探测连通
    probes = {}
    def _probe(b):
        probes[b["port"]] = _probe_bridge(b["port"])
    ts = [threading.Thread(target=_probe, args=(b,)) for b in result]
    for t in ts: t.start()
    for t in ts: t.join()
    for b in result:
        b["status"] = probes.get(b["port"], "down")
    return result


def get_timeline(input_text):
    """生成调试用 timeline"""
    lines = []
    with pipeline_lock:
        active = [m for m in pipeline if m["enabled"]]
    for m in active:
        label = m["label"]
        content = m.get("content", "")
        fmt = m["format_start"] + "..." + m["format_end"]
        if m["id"] == "chat" or m["id"] == "danmaku":
            srcs = m.get("sources", [{"name": "拉姆"}])
            src_name = srcs[0]["name"] if srcs else "拉姆"
            if m["id"] == "chat":
                content = f"对话[{src_name}]：{input_text or '(空)'}"
            else:
                content = f"弹幕[{src_name}]：{input_text or '(空)'}"
        elif m["id"] == "summary":
            content = f"{len(history)//2}轮对话" if len(history) > 8 else "(不足)"
        elif m["id"] == "memory" or m["id"] == "tool":
            content = m.get("content","")[:60]
        elif not content:
            content = "(暂无)"
        lines.append(f"{label}:\n  格式: {fmt}\n  内容: {content}")
    return "\n".join(lines)


# ── HTTP ──
class PipelineHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global PROMPT_BRIDGE_PORT, RETRIEVAL_PORT, RETRIEVAL_TOP_N
        if self.path == "/":
            self._html(200, PAGE_HTML)
        elif self.path == "/modules":
            with pipeline_lock:
                try:
                    r = requests.get(BRIDGE_API, timeout=2)
                    sl = r.json() if r.status_code == 200 else {}
                except:
                    sl = {}
                self._json(200, {"modules": pipeline, "source_last": sl, "prompt_port": PROMPT_BRIDGE_PORT, "built_prompt": last_built_prompt,
                                  "retrieval_top_n": RETRIEVAL_TOP_N, "retrieval_port": RETRIEVAL_PORT,
                                  "express_paused": express_paused, "exec_paused": exec_paused, "error_paused": error_paused,
                                  "express_bridge_port": EXPRESS_BRIDGE_PORT, "exec_bridge_port": EXEC_BRIDGE_PORT,
                                  "sys_prompt_is_api": _sys_prompt_is_api, "bridge_backend": _get_bridge_backend()})
        elif self.path == "/bridges":
            self._json(200, {"bridges": _get_bridges(),
                             "checked_at": datetime.datetime.now().strftime("%H:%M:%S")})
        elif self.path == "/automation":
            self._json(200, {"automation": automation})
        elif self.path == "/output":
            # 解析最近回复中的工具调用
            raw = last_reply_box[0]
            tools = []
            import re
            for m in re.finditer(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[TOOL\]', raw, re.DOTALL):
                tools.append({"cmd": m.group(1), "arg": m.group(2).strip()})
            # 完整历史：最近 10 条(5轮) + 旧历史
            all_ctx = []
            for role, content in history:
                all_ctx.append({"role": "输入" if role == "user" else "硬币", "content": content[:600]})
            total = len(history) // 2
            recent = all_ctx[-10:] if len(all_ctx) >= 10 else all_ctx
            old_history = all_ctx[:-10] if len(all_ctx) > 10 else []
            # 记忆层级
            mem_l0 = all_ctx[-10:] if len(all_ctx) >= 10 else all_ctx
            # 拼装成品：复用共用函数（页面预览与推桥同源，防漂移）
            fc = _build_final_context()
            self._json(200, {
                "raw": raw,
                "tools": tools,
                "recent": recent,
                "old_history": old_history,
                "total_rounds": total,
                "mem_l0": mem_l0,
                "mem_l1": [s for s, p in mem_l1],
                "mem_l1_ppl": [round(p, 2) for s, p in mem_l1],
                "mem_l2": [s for s, t in mem_l2[-2:]],
                "mem_l2_time": [t for s, t in mem_l2[-2:]],
                "mem_l3": [s for s, t1, t2 in mem_l3],
                "mem_l3_time": [f"{t1}~{t2}" for s, t1, t2 in mem_l3],
                "mem_l4": [f"[{tp}] {s} ({t})" for s, tp, t in mem_l4],
                "final_context": fc or "（暂无）",
                "retrieval": last_retrieval,
                "exec_queue": exec_queue,
                "error_queue": error_queue,
                "express_queue": express_queue,
            })
        elif self.path == "/api/export_l4":
            try:
                path = _export_l4()
                self._json(200, {"ok": True, "path": path, "count": len(mem_l4)})
            except Exception as e:
                self._json(200, {"ok": False, "error": str(e)})
        elif self.path == "/config/save":
            path = save_config(backup=True)
            self._json(200, {"ok": True, "path": path})
        elif self.path == "/config/port":
            PROMPT_BRIDGE_PORT = int(data.get("port", 18000))
            self._json(200, {"ok": True, "port": PROMPT_BRIDGE_PORT})
        elif self.path == "/config/load":
            if os.path.exists(CONFIG_PATH):
                load_config()
                with pipeline_lock:
                    self._json(200, {"ok": True, "modules": pipeline})
            else:
                self._json(404, {"error": "no config file"})
        elif self.path == "/timeline":
            # 从最近输入构建 timeline（无新输入时用空）
            txt = self._last_input if hasattr(self, '_last_input') else ""
            self._json(200, {"timeline": get_timeline(txt), "prompt": last_built_prompt})
        else:
            self.send_error(404)

    def do_POST(self):
        global PROMPT_BRIDGE_PORT, RETRIEVAL_PORT, RETRIEVAL_TOP_N, express_paused, exec_paused, error_paused, EXPRESS_BRIDGE_PORT, EXEC_BRIDGE_PORT, ERROR_INTERVAL
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        data = json.loads(body) if body else {}

        if self.path == "/modules/order":
            new_order = data.get("order", [])
            with pipeline_lock:
                old = {m["id"]: m for m in pipeline}
                pipeline.clear()
                for mid in new_order:
                    if mid in old:
                        pipeline.append(old[mid])
                # 补漏
                for m in MODULES:
                    if m["id"] not in old:
                        pipeline.append(dict(m))
            save_config(backup=False)
            self._json(200, {"ok": True})

        elif self.path == "/modules/toggle":
            mid = data.get("id", "")
            en = data.get("enabled", True)
            with pipeline_lock:
                for m in pipeline:
                    if m["id"] == mid:
                        m["enabled"] = en
            save_config(backup=False)
            self._json(200, {"ok": True})

        elif self.path == "/modules/content":
            mid = data.get("id", "")
            field = data.get("field", "content")
            value = data.get("value", "")
            with pipeline_lock:
                for m in pipeline:
                    if m["id"] == mid and m.get("editable"):
                        try:
                            m[field] = json.loads(value) if field == "sources" else value
                        except:
                            m[field] = value
            save_config(backup=False)
            self._json(200, {"ok": True})

        elif self.path == "/modules/apply_api_prompt":
            # 08-12 拉姆: 系统提示 双向切换——API 提示词(读桌面 txt) ↔ 本地提示词(切前备份)
            global _sys_prompt_backup, _sys_prompt_is_api
            try:
                api_path = "/mnt/c/Users/Autogram-coin/Desktop/api模式提示词。.txt"
                with pipeline_lock:
                    for m in pipeline:
                        if m["id"] == "system":
                            if _sys_prompt_is_api:
                                # 切回本地：用备份（无备份则保持现状并报错）
                                if _sys_prompt_backup is None:
                                    self._json(500, {"error": "无本地提示词备份，无法切回"})
                                    return
                                m["content"] = _sys_prompt_backup
                                _sys_prompt_is_api = False
                                save_config(backup=False)
                                self._json(200, {"ok": True, "is_api": False, "content": m["content"][:60]})
                                return
                            # 切到 API：先备份当前，再读 txt 覆盖
                            _sys_prompt_backup = m.get("content", "")
                            if not os.path.exists(api_path):
                                self._json(500, {"error": f"提示词文件不存在: {api_path}"})
                                return
                            with open(api_path, encoding="utf-8") as f:
                                txt = f.read().strip()
                            if not txt:
                                self._json(500, {"error": "提示词文件为空"})
                                return
                            m["content"] = txt
                            _sys_prompt_is_api = True
                            save_config(backup=False)
                            self._json(200, {"ok": True, "is_api": True, "content": txt[:60]})
                            return
                self._json(500, {"error": "system 模块未找到"})
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif self.path == "/api/bridge_mode":
            # 08-12 拉姆: 页面一键切换 LLM 桥后端模式 (api/local)，转发到桥
            try:
                mode = str(data.get("mode", "") or "").strip().lower()
                if mode in ("api", "local"):
                    r = requests.post(f"http://localhost:{LLM_BRIDGE_PORT}/api/backend",
                                      json={"mode": mode}, timeout=5)
                    self._json(200, r.json())
                else:
                    r = requests.get(f"http://localhost:{LLM_BRIDGE_PORT}/api/backend", timeout=5)
                    self._json(200, r.json())
            except Exception as e:
                self._json(500, {"error": f"桥不可达: {e}"})

        elif self.path == "/automation":
            # 保存自动化配置（静默间隔 / 提示列表 / 注入桥）
            with pipeline_lock:
                if "sys_interval" in data:
                    try: automation["sys_interval"] = float(data["sys_interval"])
                    except Exception: pass
                if "sys_prompts" in data and isinstance(data["sys_prompts"], list):
                    automation["sys_prompts"] = [str(p) for p in data["sys_prompts"]]
                if "sys_bridge" in data:
                    automation["sys_bridge"] = str(data["sys_bridge"])
            save_config(backup=False)
            self._json(200, {"ok": True, "automation": automation})

        elif self.path in ("/asr", "/text"):
            text = data.get("text", "")
            source = data.get("source", "")
            if text:
                if not source:
                    with pipeline_lock:
                        for m in pipeline:
                            if m["id"] == "chat" and m.get("sources"):
                                for s in m["sources"]:
                                    if s.get("active") and s.get("name"):
                                        source = s["name"]
                                        break
                                break
                    if not source: source = "拉姆"
                self._last_input = text
                # 写入 chat 源桥（与 ASR 写桥同路径，由 _check_trigger_sources 检测触发）
                _write_chat_bridge(source, text)
                _mark_pending()   # 对话输入到达 → 标记积压，由 _send_loop 冷却后统一发送
            self._json(200, {"ok": True})

        elif self.path == "/inject":
            user_text = data.get("user", "")
            assistant_text = data.get("assistant", "")
            if user_text and assistant_text:
                self._last_input = user_text
                history.append(("user", user_text))
                history.append(("assistant", assistant_text))
                last_reply_box[0] = assistant_text
                _push_reply_to_queues(assistant_text)
                trigger_summarize()
                trigger_retrieval()   # 长期记忆检索：处理最新一轮
            self._json(200, {"ok": True})

        elif self.path == "/queue/pause_express":
            express_paused = not express_paused
            self._json(200, {"ok": True, "paused": express_paused})
        elif self.path == "/queue/pause_exec":
            exec_paused = not exec_paused
            self._json(200, {"ok": True, "paused": exec_paused})
        elif self.path == "/queue/pause_error":
            error_paused = not error_paused
            self._json(200, {"ok": True, "paused": error_paused})
        elif self.path == "/config/express_bridge":
            EXPRESS_BRIDGE_PORT = int(data.get("port", 18004))
            self._json(200, {"ok": True, "port": EXPRESS_BRIDGE_PORT})
        elif self.path == "/config/exec_bridge":
            EXEC_BRIDGE_PORT = int(data.get("port", 18003))
            self._json(200, {"ok": True, "port": EXEC_BRIDGE_PORT})
        elif self.path == "/config/error_interval":
            ERROR_INTERVAL = int(data.get("interval", 3))
            self._json(200, {"ok": True, "interval": ERROR_INTERVAL})

        elif self.path == "/inject_l2":
            texts = data.get("texts", [])
            if texts:
                now = datetime.datetime.now().strftime("%m-%d %H:%M")
                for t in texts:
                    mem_l2.append((t, now))
                print(f"[手动注入] {len(texts)}条L2")
                trigger_l3()
            self._json(200, {"ok": True})

        elif self.path == "/inject_l4":
            texts = data.get("texts", [])
            tp = str(data.get("type", "手动") or "手动")
            if texts:
                now = datetime.datetime.now().strftime("%m-%d %H:%M")
                for t in texts:
                    mem_l4.append((t, tp, now))
                print(f"[手动注入] {len(texts)}条L4(type={tp})")
            self._json(200, {"ok": True, "count": len(texts)})

        elif self.path == "/chat":
            text = data.get("text", "")
            if not text:
                self._json(400, {"error": "empty"})
                return
            self._last_input = text
            prompt = build_prompt(text)
            payload = json.dumps({
                "prompt": prompt, "n_predict": 500, "temperature": 0.9,
                "grammar": GRAMMAR_TOOL,
                "logit_bias": [[1930, 10.0], [3925, -10.0]]
            }).encode()
            try:
                r = requests.post(BRIDGE_URL, data=payload,
                                  headers={"Content-Type": "application/json"}, timeout=60)
                reply = r.json().get("content", "")
            except Exception as e:
                reply = f"[错误] {e}"

            last_reply_box[0] = reply
            _push_reply_to_queues(reply)
            history.append(("user", text))
            history.append(("assistant", reply))
            trigger_summarize()
            trigger_retrieval()   # 长期记忆检索：处理最新一轮
            self._json(200, {"reply": reply, "prompt": prompt})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code, data):
        b = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(b))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _html(self, code, html):
        b = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(b))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


PAGE_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>mini_coin 管线</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#8b949e;--green:#3fb950;--blue:#58a6ff;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font:24px/1.6 sans-serif;padding:24px 24px 100px;}
h2{font-size:28px;margin-bottom:16px;}
.row{display:flex;gap:20px;margin-bottom:20px;}
.col{flex:1;}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px;}
.card h3{font-size:18px;color:var(--dim);margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px;}
.module{display:flex;align-items:center;gap:12px;background:#0d1117;border:1px solid var(--border);border-radius:10px;padding:12px 16px;margin:8px 0;cursor:grab;font-size:20px;}
.module:hover{background:#1a2030;}
.handle{color:var(--dim);cursor:grab;font-size:22px;}
.toggle{width:48px;height:24px;border-radius:12px;border:none;cursor:pointer;font-size:13px;padding:0;font-weight:bold;}
.toggle.on{background:var(--green);color:#fff;}
.toggle.off{background:var(--border);color:var(--dim);}
pre{background:#0d1117;padding:16px;border-radius:10px;font-size:16px;white-space:pre-wrap;word-break:break-all;max-height:500px;overflow-y:auto;line-height:1.5;}
.tab-bar{display:flex;gap:0;margin-bottom:16px;}
.tab{background:var(--card);border:1px solid var(--border);border-bottom:none;padding:10px 24px;cursor:pointer;font-size:18px;border-radius:8px 8px 0 0;color:var(--dim);}
.tab.active{background:var(--card);color:var(--text);border-bottom:2px solid var(--blue);}
.tab:hover{color:var(--text);}
.view{display:none;}
.view.active{display:block;}
.out-panel{display:grid;grid-template-columns:1.2fr 0.8fr;gap:16px;}
.out-full{grid-column:1/-1;}
.tool-item{background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:10px;margin:6px 0;font-size:16px;}
.tool-cmd{color:var(--blue);font-weight:bold;}
.tool-arg{color:var(--green);}
.ctx-item{padding:4px 0;font-size:15px;display:flex;gap:8px;}
.ctx-role{color:var(--dim);flex-shrink:0;}
.ctx-text{color:var(--text);}
.mem-item{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:8px;margin:4px 0;font-size:15px;color:var(--yellow);}
.mem-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.mem-grid .card{margin:0;}
.ret-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.ret-grid .card{margin:0;}
.ret-bar{height:8px;background:#0d1117;border-radius:4px;overflow:hidden;margin-top:3px;}
.ret-bar-fill{height:100%;border-radius:4px;}
.ret-tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-right:4px;}
.ret-dim{background:#161b22;border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin-bottom:6px;}
.ret-dim-title{font-weight:bold;color:var(--text);margin-bottom:4px;}
.ret-hit{background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:8px;font-size:14px;}
.ret-sim{color:var(--green);font-weight:bold;}
.input-row{display:flex;gap:12px;position:fixed;bottom:0;left:0;right:0;background:var(--card);border-top:1px solid var(--border);padding:16px 24px;}
.input-row input{flex:1;background:#0d1117;border:1px solid var(--border);border-radius:10px;padding:14px 18px;color:var(--text);font-size:20px;}
.input-row button{background:var(--blue);color:#fff;border:none;border-radius:10px;padding:14px 28px;cursor:pointer;font-size:20px;font-weight:bold;}
.content-edit{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:6px 10px;color:var(--text);font-size:16px;width:100%;margin-top:6px;resize:vertical;}
</style></head><body>
<div id="toast" style="position:fixed;top:14px;right:14px;z-index:9999;background:#161b22;color:var(--green);border:1px solid var(--border);border-radius:8px;padding:8px 14px;font-size:13px;display:none;box-shadow:0 4px 16px rgba(0,0,0,.4)"></div>
<h2>mini_coin 输入管线 <span style="font-size:14px;font-weight:normal;cursor:pointer" onclick="toggleAutoRefresh()" id="refresh-btn">🔄 自动刷新:开</span></h2>
<div class="tab-bar">
<div class="tab active" id="tab-input" onclick="switchView('input')">📥 输入视图</div>
<div class="tab" id="tab-output" onclick="switchView('output')">📤 输出视图</div>
<div class="tab" id="tab-memory" onclick="switchView('memory')">📋 上文记忆</div>
<div class="tab" id="tab-search" onclick="switchView('search')">🔍 检索记忆</div>
<div class="tab" id="tab-searchtool" onclick="switchView('searchtool')">🌐 搜索工具</div>
<div class="tab" id="tab-bridges" onclick="switchView('bridges')">🌉 桥管理</div>
<div class="tab" id="tab-automation" onclick="switchView('automation')">⚙️ 自动化</div>
<span style="flex:1"></span>
<button onclick="saveCfg()" style="background:var(--green);color:#fff;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:14px">💾 保存配置</button>
<button onclick="loadCfg()" style="background:var(--blue);color:#fff;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:14px">📂 读取配置</button>
<span id="cfg-msg" style="font-size:13px;color:var(--dim);margin-left:8px"></span>
</div>

<div id="view-input" class="view active">
<div class="row">
<div class="col"><div class="card"><h3>模块顺序（拖拽调整）</h3><div id="module-list"></div></div></div>
<div class="col"><div class="card"><h3>生成的 Prompt</h3><pre id="prompt-preview"></pre></div></div>
</div>
</div>

<div id="view-output" class="view">
<div class="out-panel">
<div class="card out-full"><h3>🔴 原始回复</h3><pre id="out-raw">（暂无）</pre></div>
<div class="card out-full"><h3>🔧 工具调用</h3><div id="out-tools"><div style="color:var(--dim)">（暂无）</div></div></div>
</div>
<div class="out-panel" style="display:flex;gap:12px">
<div class="card" style="flex:1"><h3>📋 执行队列 <span onclick="toggleQueuePause('exec')" id="pause-exec-btn" style="font-size:13px;cursor:pointer;padding:2px 10px;border-radius:4px;background:var(--bg);border:1px solid var(--border)">▶ 运行中</span> <span style="font-size:11px;color:var(--dim);cursor:pointer" onclick="toggleQueueCfg('exec')">⚙ 配置</span></h3>
  <div id="exec-cfg" style="display:none;margin-bottom:6px">
    <span style="font-size:12px;color:var(--dim)">完成信号桥 </span>
    <select id="exec-port" style="width:90px;font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px">
OPT = '<option value="18000">18000</option><option value="18001">18001</option><option value="18002">18002</option><option value="18003">18003</option><option value="18004">18004</option><option value="18005">18005</option><option value="18006">18006</option><option value="18007">18007</option><option value="18008">18008</option><option value="18009">18009</option>'
    </select>
    <button onclick="saveQueuePort('exec')" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px">保存</button>
    <span id="exec-port-ok" style="color:var(--green);font-size:12px;display:none">✓</span>
  </div>
  <pre id="queue-exec" style="font-size:12px;margin:0;min-height:60px;max-height:200px;overflow-y:auto;background:var(--bg)">（空）</pre></div>

<div class="card" style="flex:1"><h3>🔴 报错队列 <span onclick="toggleQueuePause('error')" id="pause-error-btn" style="font-size:13px;cursor:pointer;padding:2px 10px;border-radius:4px;background:var(--bg);border:1px solid var(--border)">▶ 运行中</span> <span style="font-size:11px;color:var(--dim);cursor:pointer" onclick="toggleQueueCfg('error')">⚙ 配置</span></h3>
  <div id="error-cfg" style="display:none;margin-bottom:6px">
    <span style="font-size:12px;color:var(--dim)">报错间隔 </span>
    <input id="error-interval" type="number" min="1" max="60" value="3" style="width:50px;font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px"> 秒
    <button onclick="saveErrorInterval()" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px">保存</button>
    <span id="error-interval-ok" style="color:var(--green);font-size:12px;display:none">✓</span>
  </div>
  <pre id="queue-error" style="font-size:12px;margin:0;min-height:40px;max-height:200px;overflow-y:auto;background:var(--bg)">（空）</pre></div>

<div class="card" style="flex:1"><h3>🗣 表达队列 <span onclick="toggleQueuePause('express')" id="pause-express-btn" style="font-size:13px;cursor:pointer;padding:2px 10px;border-radius:4px;background:var(--bg);border:1px solid var(--border)">▶ 运行中</span> <span style="font-size:11px;color:var(--dim);cursor:pointer" onclick="toggleQueueCfg('express')">⚙</span> <span onclick="toggleExpressMode()" id="express-mode-btn" style="font-size:11px;cursor:pointer;padding:1px 6px;border-radius:4px;background:var(--bg);border:1px solid var(--border);margin-left:4px">🔗 串行</span></h3>
  <div id="express-cfg" style="display:none;margin-bottom:6px">
    <span style="font-size:12px;color:var(--dim)">完成信号桥 </span>
    <select id="express-port" style="width:90px;font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px">
OPT = '<option value="18000">18000</option><option value="18001">18001</option><option value="18002">18002</option><option value="18003">18003</option><option value="18004">18004</option><option value="18005">18005</option><option value="18006">18006</option><option value="18007">18007</option><option value="18008">18008</option><option value="18009">18009</option>'
    </select>
    <button onclick="saveQueuePort('express')" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px">保存</button>
    <span id="express-port-ok" style="color:var(--green);font-size:12px;display:none">✓</span>
  </div>
  <pre id="queue-express" style="font-size:12px;margin:0;min-height:60px;max-height:200px;overflow-y:auto;background:var(--bg)">（空）</pre></div>
</div>
</div>

<div id="view-memory" class="view">
<div class="mem-grid">
<div class="card"><h3>L0 原文（最近5轮）</h3><div id="m-l0" style="font-size:13px;max-height:200px;overflow-y:auto"><div style="color:var(--dim)">（暂无）</div></div></div>
<div class="card"><h3>L1 段落摘要</h3><div id="m-l1" style="font-size:13px;max-height:200px;overflow-y:auto"><div style="color:var(--dim)">（暂无）</div></div></div>
<div class="card"><h3>L2 会话摘要</h3><div id="m-l2" style="font-size:13px;max-height:150px;overflow-y:auto"><div style="color:var(--dim)">（暂无）</div></div></div>
<div class="card"><h3>L3 日常摘要</h3><div id="m-l3" style="font-size:13px;max-height:150px;overflow-y:auto"><div style="color:var(--dim)">（暂无）</div></div></div>
<div class="card"><h3>L4 永久记忆 <button onclick="exportL4()" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:1px 8px;cursor:pointer;font-size:11px;margin-left:6px">📤 导出</button></h3><div id="m-l4" style="font-size:13px;max-height:150px;overflow-y:auto"><div style="color:var(--dim)">待接入向量检索</div></div></div>
<div class="card"><h3>📦 拼装成品 <span style="font-size:11px;color:var(--dim);cursor:pointer" onclick="togglePromptPort()">⚙ 配置</span></h3>
  <div id="prompt-port-cfg" style="display:none;margin-bottom:6px">
    <span style="font-size:12px;color:var(--dim)">POST到 </span>
    <select id="prompt-port" style="width:90px;font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px">
      <option value="18000">18000</option>
      <option value="18001">18001</option>
      <option value="18002">18002</option>
      <option value="18003">18003</option>
      <option value="18004">18004</option>
    </select>
    <button onclick="savePromptPort()" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px">保存</button>
    <span id="prompt-port-ok" style="color:var(--green);font-size:12px;display:none">✓</span>
  </div>
  <div id="m-final" style="font-size:13px;max-height:300px;overflow-y:auto"><div style="color:var(--dim)">（暂无）</div></div>
</div>
</div>
</div>

<div id="view-search" class="view">
<div class="ret-grid">
  <div class="card"><h3>① L0 原始对话</h3><div id="r-l0" style="font-size:13px;max-height:180px;overflow-y:auto"><div style="color:var(--dim)">等待对话...</div></div></div>
  <div class="card"><h3>② L0.5 第一视角转写</h3><div id="r-l05" style="font-size:13px;max-height:180px;overflow-y:auto;white-space:pre-wrap;font-family:monospace;line-height:1.6"><div style="color:var(--dim)">等待对话...</div></div></div>
  <div class="card"><h3>③ 标签计算</h3><div id="r-labels" style="font-size:13px;max-height:420px;overflow-y:auto"><div style="color:var(--dim)">等待对话...</div></div></div>
  <div class="card"><h3>④ 检索命中 <span style="font-size:11px;color:var(--dim);cursor:pointer" onclick="toggleRetrievalCfg()">⚙ 配置</span></h3>
  <div id="ret-cfg" style="display:none;margin-bottom:6px">
    TOP <select id="ret-top" style="font-size:12px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 4px"><option value="1">1</option><option value="2">2</option><option value="3">3</option></select>
    条 → 桥
    <select id="ret-port" style="font-size:12px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 4px;width:80px">
OPT = '<option value="18000">18000</option><option value="18001">18001</option><option value="18002">18002</option><option value="18003">18003</option><option value="18004">18004</option><option value="18005">18005</option><option value="18006">18006</option><option value="18007">18007</option><option value="18008">18008</option><option value="18009">18009</option>'
    </select>
    <button onclick="saveRetrievalCfg()" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:2px 6px;cursor:pointer;font-size:11px">保存</button>
    <span id="ret-cfg-ok" style="color:var(--green);font-size:12px;display:none">✓</span>
  </div>
  <div id="r-hit" style="font-size:13px;max-height:300px;overflow-y:auto"><div style="color:var(--dim)">等待对话...</div></div></div>
</div>
</div>

<div id="view-searchtool" class="view" style="height:calc(100vh - 130px)">
  <iframe src="http://localhost:18775" style="width:100%;height:100%;border:none;border-radius:8px;background:#0d1117"></iframe>
</div>

<div id="view-bridges" class="view">
<div class="card">
<h3>🌉 桥管理
  <button onclick="loadBridges()" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:3px 12px;cursor:pointer;font-size:13px;margin-left:10px">🔍 检测</button>
  <span id="bridge-checked" style="font-size:12px;color:var(--dim);margin-left:8px"></span>
</h3>
<div id="bridge-list" style="margin-top:8px"><div style="color:var(--dim)">点击「检测」查看所有桥的连通状态</div></div>
</div>
</div>

<div id="view-automation" class="view">
<div class="card">
<h3>⚙️ 自动化 — 系统提示</h3>
<div style="margin-top:8px;display:flex;align-items:center;gap:8px">
  <span style="font-size:13px;color:var(--dim)">静默触发间隔（秒）：</span>
  <input id="auto-interval" type="number" step="1" value="30" style="width:80px;font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:3px 8px">
  <span style="font-size:13px;color:var(--dim)">注入桥：</span>
  <select id="auto-bridge" style="font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px;width:110px"></select>
</div>
<div style="margin-top:10px;font-size:13px;color:var(--dim)">提示内容（触发后随机选一条，可多选一）：</div>
<div id="auto-prompts" style="margin-top:6px"></div>
<button onclick="addAutoPrompt()" style="background:var(--blue);color:#fff;border:none;border-radius:4px;padding:4px 12px;cursor:pointer;font-size:13px;margin-top:8px">+ 添加提示</button>
<button onclick="saveAutomation()" style="background:var(--green);color:#fff;border:none;border-radius:4px;padding:4px 14px;cursor:pointer;font-size:13px;margin-top:8px;margin-left:8px">💾 保存</button>
<span id="auto-ok" style="font-size:12px;color:var(--green);display:none;margin-left:8px">已保存</span>
</div>
</div>

<div class="input-row">
<input id="input-text" placeholder="输入消息..." onkeydown="if(event.key=='Enter')send()">
<button onclick="send()">发送</button>
</div>
<div id="modal-overlay" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);z-index:999" onclick="if(event.target===this)closeEdit()">
<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;width:600px;max-width:90vw;">
<h3 id="modal-title" style="margin-bottom:12px">编辑模块</h3>
<div id="modal-fields"></div>
<div style="display:flex;gap:8px;justify-content:flex-end;margin-top:12px">
<button onclick="closeEdit()" style="background:var(--border);color:var(--text);border:none;border-radius:8px;padding:10px 20px;font-size:16px;cursor:pointer">取消</button>
<button onclick="saveEdit()" style="background:var(--blue);color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:16px;cursor:pointer">确认</button>
</div>
</div>
</div>
<script>
function showToast(msg,color){
  var t=document.getElementById('toast');
  if(!t)return;
  t.textContent=msg;t.style.color=color||'var(--green)';t.style.display='block';
  clearTimeout(t._tm);
  t._tm=setTimeout(function(){t.style.display='none';},2000);
}
function exportL4(){
  fetch('/api/export_l4').then(function(r){return r.json();}).then(function(d){
    if(d.ok&&d.path){showToast('✓ 已导出 '+d.count+' 条 → '+d.path.split('/').pop(), null);}
    else if(!d.ok&&d.error){showToast('✗ '+d.error,'#f85149');}
    else{showToast('L4 为空，无导出','#f85149');}
  }).catch(function(){showToast('✗ 导出失败','#f85149');});
}
var modules=[],_editId=null;
function load(){fetch('/modules').then(r=>r.json()).then(function(d){
  modules=d.modules||[];window._srcLast=d.source_last||{};
  window._sysIsApi=!!d.sys_prompt_is_api; window._bridgeBackend=d.bridge_backend||'unknown';
  render();updatePreview();
}).catch(()=>document.getElementById('module-list').innerHTML='<div style="color:var(--red)">离线</div>');}
function render(){
  var c=document.getElementById('module-list');c.innerHTML='';
  modules.forEach(function(m,i){
    var d=document.createElement('div');d.className='module';d.draggable=true;
    d.dataset.index=i;
    // 手柄
    var h=document.createElement('span');h.className='handle';h.textContent='⠿';d.appendChild(h);
    // 开关
    var tb=document.createElement('button');tb.className='toggle '+(m.enabled?'on':'off');
    tb.textContent=m.enabled?'开':'关';
    tb.onclick=function(){modules[i].enabled=!modules[i].enabled;render();fetch('/modules/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:modules[i].id,enabled:modules[i].enabled})});updatePreview();};
    d.appendChild(tb);
    // 标签
    var lb=document.createElement('b');lb.textContent=m.label;d.appendChild(lb);
    // 编辑按钮（仅可编辑模块）
    if(m.editable){
      var edBtn=document.createElement('button');edBtn.textContent='编辑';
      edBtn.style.cssText='background:var(--blue);color:#fff;border:none;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:14px;margin-left:8px';
      edBtn.onclick=function(){openEdit(i);};
      d.appendChild(edBtn);
    }
    // API 模式提示词双向切换 + 桥后端切换（仅系统提示模块，08-12 拉姆）
    if(m.id==='system'){
      var apiBtn=document.createElement('button');
      apiBtn.id='apiPromptBtn';
      apiBtn.textContent=window._sysIsApi?'🔙本地提示词':'🔄API提示词';
      apiBtn.style.cssText='background:#a05ce0;color:#fff;border:none;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:14px;margin-left:8px';
      apiBtn.onclick=function(){
        fetch('/modules/apply_api_prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
          .then(function(r){return r.json();}).then(function(d){
            if(d.ok){showToast(d.is_api?'✓ 已切换 API 模式提示词':'✓ 已切回本地提示词');load();}
            else{showToast('✗ '+(d.error||'切换失败'),'#f85149');}
          }).catch(function(){showToast('✗ 切换失败(网络)','#f85149');});
      };
      d.appendChild(apiBtn);
      // 桥后端模式切换按钮
      var bkBtn=document.createElement('button');
      bkBtn.id='bridgeBackendBtn';
      bkBtn.textContent='桥:'+(window._bridgeBackend||'?');
      bkBtn.style.cssText='background:#c85a5a;color:#fff;border:none;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:14px;margin-left:8px';
      bkBtn.onclick=function(){
        var next=window._bridgeBackend==='api'?'local':'api';
        fetch('/api/bridge_mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:next})})
          .then(function(r){return r.json();}).then(function(d){
            if(d.ok){showToast('✓ 桥后端 → '+(d.backend==='api'?'API':'本地'));window._bridgeBackend=d.backend;bkBtn.textContent='桥:'+d.backend;load();}
            else{showToast('✗ '+(d.error||'切换失败'),'#f85149');}
          }).catch(function(){showToast('✗ 桥切换失败(网络)','#f85149');});
      };
      d.appendChild(bkBtn);
    }
    // 最新收到（有 sources 的模块）
    if(m.sources){
      var sl=window._srcLast||{};
      var srcNames=m.sources.map(function(s){return s.name;});
      var lastText='';
      for(var k in sl){
        if(srcNames.indexOf(k)>=0){lastText=sl[k];break;}
      }
      var lastSpan=document.createElement('span');
      lastSpan.style.cssText='color:var(--green);font-size:14px;margin-left:8px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
      lastSpan.textContent='最新: '+(lastText||'—');
      lastSpan.id='last-'+m.id;
      d.appendChild(lastSpan);
    }
    // 拖拽事件
    d.addEventListener('dragstart',function(e){e.dataTransfer.setData('text/plain',i);this.style.opacity='0.4';});
    d.addEventListener('dragend',function(){this.style.opacity='1';});
    d.addEventListener('dragover',function(e){e.preventDefault();});
    d.addEventListener('drop',function(e){
      e.preventDefault();var from=parseInt(e.dataTransfer.getData('text/plain'));var to=i;
      if(from===to)return;var item=modules.splice(from,1)[0];modules.splice(to,0,item);
      render();saveOrder();updatePreview();
    });
    c.appendChild(d);
  });
}
function openEdit(i){
  var m=modules[i];_editId=i;
  var modal=document.getElementById('modal-fields');
  modal.innerHTML='';
  // 显示源名选项（模块级）：打钩 → 源名：内容；不打钩 → 只内容
  var ssRow=document.createElement('div');ssRow.style.cssText='margin:8px 0;display:flex;align-items:center;gap:6px';
  var ssCb=document.createElement('input');ssCb.type='checkbox';ssCb.checked=m.show_src!==false;
  ssCb.onchange=function(){
    m.show_src=ssCb.checked;
    fetch('/modules/content',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:m.id,field:'show_src',value:ssCb.checked})});
  };
  ssRow.appendChild(ssCb);
  var ssLb=document.createElement('label');ssLb.style.cssText='font-size:14px;color:var(--dim)';ssLb.textContent='显示源名（打钩：源名一行+内容；不打钩：只内容）';
  ssRow.appendChild(ssLb);
  modal.appendChild(ssRow);
  // 有 sources 的模块显示源列表
  if(m.sources){
    addModalField('格式开头','format_start',m.format_start,'input');
    addModalField('格式结尾','format_end',m.format_end,'input');
    if(m.content!==undefined && m.id!=='summary' && m.id!=='danmaku'){
      addModalField('默认内容','content',m.content,'textarea');
    }
    var sl=document.createElement('div');sl.style.cssText='margin-top:8px;padding:8px;background:#0d1117;border-radius:6px';
    var title=document.createElement('div');title.style.cssText='font-size:14px;color:var(--dim);margin-bottom:6px';
    title.textContent='数据源列表';sl.appendChild(title);
    renderSrcList(sl,m.sources,m);
    modal.appendChild(sl);
  } else {
    addModalField('格式开头','format_start',m.format_start,'input');
    addModalField('格式结尾','format_end',m.format_end,'input');
    if(m.content!==undefined) addModalField('内容','content',m.content,'textarea');
  }
  document.getElementById('modal-title').textContent='编辑 '+m.label;
  document.getElementById('modal-overlay').style.display='';
}
function addModalField(label,field,value,tag){
  var modal=document.getElementById('modal-fields');
  var div=document.createElement('div');div.style.cssText='margin:8px 0';
  var lb=document.createElement('label');lb.style.cssText='font-size:14px;color:var(--dim)';lb.textContent=label;
  div.appendChild(lb);
  var el=document.createElement(tag||'input');el.className='content-edit';el.style.cssText='margin-top:2px';
  el.value=value;el.dataset.field=field;
  if(tag==='textarea'){el.rows=3;}
  div.appendChild(el);
  modal.appendChild(div);
}
function closeEdit(){document.getElementById('modal-overlay').style.display='none';_editId=null;}
function saveEdit(){
  var i=_editId;if(i===null)return;
  var m=modules[i];
  var fields=document.getElementById('modal-fields').querySelectorAll('[data-field]');
  fields.forEach(function(el){m[el.dataset.field]=el.value;});
  // 保存到服务器
  var saves=[];
  fields.forEach(function(el){
    saves.push(fetch('/modules/content',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:m.id,field:el.dataset.field,value:el.value})}));
  });
  Promise.all(saves).then(function(){closeEdit();render();updatePreview();});
}
function renderSrcList(container,srcs,m){
  function render(){
    while(container.children.length>1)container.removeChild(container.lastChild);
    srcs.forEach(function(s,si){
      var wrap=document.createElement('div');wrap.style.cssText='background:#161b22;border:1px solid var(--border);border-radius:6px;padding:6px 8px;margin:3px 0';
      var row=document.createElement('div');row.style.cssText='display:flex;align-items:center;gap:6px';
      var dotSpan=document.createElement('span');dotSpan.textContent=s.active?'🟢':'🔴';row.appendChild(dotSpan);
      var nameSpan=document.createElement('span');nameSpan.style.cssText='flex:1;font-size:14px';nameSpan.textContent=s.name;row.appendChild(nameSpan);
      wrap.appendChild(row);
      // 最新收到
      var lastLine=document.createElement('div');lastLine.style.cssText='font-size:12px;color:var(--green);margin:2px 0 4px 14px;';
      lastLine.setAttribute('data-src-last', s.name);
      lastLine.textContent='最新: '+((window._srcLast||{})[s.name]||'(等待中)');
      wrap.appendChild(lastLine);
      var editArea=document.createElement('div');editArea.style.cssText='display:none;margin-top:6px;padding-top:6px;border-top:1px solid var(--border)';
      var a1=document.createElement('div');a1.style.cssText='font-size:12px;color:var(--dim)';a1.textContent='名称：';editArea.appendChild(a1);
      var ni=document.createElement('input');ni.className='content-edit';ni.style.cssText='font-size:13px';ni.value=s.name;
      ni.onchange=function(){s.name=this.value;save(fbSpan);nameSpan.textContent=s.name;};editArea.appendChild(ni);
      var a2=document.createElement('div');a2.style.cssText='font-size:12px;color:var(--dim);margin-top:4px';a2.textContent='桥地址：';editArea.appendChild(a2);
      var sel=document.createElement('select');sel.style.cssText='font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px;width:150px';
      var _opts=['18000','18001','18002','18003','18004','18005','18006','18007','18008','18009','自定义...'];
      var _has=false;
      _opts.forEach(function(v){var o=document.createElement('option');o.value=v;o.textContent=v;if(v===s.endpoint){o.selected=true;_has=true;}sel.appendChild(o);});
      if(!_has&&s.endpoint){var o=document.createElement('option');o.value=s.endpoint;o.textContent=s.endpoint;o.selected=true;sel.appendChild(o);}
      if(!s.endpoint)sel.value='18000';
      sel.onchange=function(){if(this.value==='自定义...'){ei.style.display='';s.endpoint=ei.value||'';}else{s.endpoint=this.value;ei.style.display='none';}save(fbSpan);};
      editArea.appendChild(sel);
      var ei=document.createElement('input');ei.className='content-edit';ei.style.cssText='font-size:13px;margin-top:4px;display:'+(sel.value==='自定义...'?'':'none');ei.value=(sel.value==='自定义...'&&s.endpoint? s.endpoint:'');ei.placeholder='输入桥端口或URL';
      ei.onchange=function(){s.endpoint=this.value;save(fbSpan);};editArea.appendChild(ei);
      var svRow=document.createElement('div');svRow.style.cssText='margin-top:8px;display:flex;align-items:center;gap:8px';
      var svBtn=document.createElement('button');svBtn.textContent='保存';svBtn.style.cssText='background:var(--blue);color:#fff;border:none;border-radius:4px;padding:2px 12px;cursor:pointer;font-size:12px';
      svBtn.onclick=function(){save(fbSpan);};
      var fbSpan=document.createElement('span');fbSpan.style.cssText='font-size:14px;font-weight:bold;color:var(--green);display:none';
      svRow.appendChild(svBtn);svRow.appendChild(fbSpan);editArea.appendChild(svRow);
      wrap.appendChild(editArea);
      var edBtn=document.createElement('button');edBtn.textContent='编辑';
      edBtn.style.cssText='background:var(--blue);color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px';
      edBtn.onclick=function(){var v=editArea.style.display;editArea.style.display=v==='none'?'block':'none';edBtn.textContent=v==='none'?'收起':'编辑';};
      row.appendChild(edBtn);
      var togBtn=document.createElement('button');togBtn.textContent=s.active?'开':'关';
      togBtn.style.cssText='background:'+(s.active?'var(--green)':'var(--border)')+';color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px;min-width:28px';
      togBtn.onclick=function(){s.active=!s.active;togBtn.textContent=s.active?'开':'关';togBtn.style.background=s.active?'var(--green)':'var(--border)';dotSpan.textContent=s.active?'🟢':'🔴';save(null);};
      row.appendChild(togBtn);
      var delBtn=document.createElement('button');delBtn.textContent='✕';
      delBtn.style.cssText='background:#f85149;color:#fff;border:none;border-radius:4px;padding:2px 6px;cursor:pointer;font-size:12px';
      delBtn.onclick=function(){srcs.splice(si,1);save(null);render();};
      row.appendChild(delBtn);
      container.appendChild(wrap);
    });
    var addBtn=document.createElement('button');addBtn.textContent='+ 新增数据源';
    addBtn.style.cssText='background:var(--blue);color:#fff;border:none;border-radius:4px;padding:6px 12px;cursor:pointer;font-size:13px;margin-top:4px';
    addBtn.onclick=function(){srcs.push({name:'新源',endpoint:'',active:true});save(null);render();};
    container.appendChild(addBtn);
  }
  function save(fb){
    m.sources=srcs;
    if(fb){fb.style.display='inline';fb.textContent='…';}
    fetch('/modules/content',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:m.id,field:'sources',value:JSON.stringify(srcs)})}).then(function(r){
      if(fb){
        fb.textContent=r.ok?'✓':'✗';
        fb.style.color=r.ok?'var(--green)':'#f85149';
        setTimeout(function(){fb.style.display='none';},2000);
      }else{
        showToast(r.ok?'✓ 已保存':'✗ 保存失败', r.ok?null:'#f85149');
      }
    }).catch(function(){
      if(fb){fb.textContent='✗';fb.style.color='#f85149';setTimeout(function(){fb.style.display='none';},2000);}
      else{showToast('✗ 保存失败','#f85149');}
    });
  }
  window._editSrcRefresh=null;
  render();
}
function saveOrder(){fetch('/modules/order',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({order:modules.map(function(m){return m.id})})});}
function updatePreview(){
  // 直接从服务端取真实拼装成品
  fetch('/modules').then(r=>r.json()).then(function(d){
    if(d.built_prompt)document.getElementById('prompt-preview').textContent=d.built_prompt;
  });
}
function esc(s){return(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function switchView(name){
  document.getElementById('tab-input').className='tab'+(name==='input'?' active':'');
  document.getElementById('tab-output').className='tab'+(name==='output'?' active':'');
  document.getElementById('tab-memory').className='tab'+(name==='memory'?' active':'');
  document.getElementById('tab-search').className='tab'+(name==='search'?' active':'');
  document.getElementById('tab-searchtool').className='tab'+(name==='searchtool'?' active':'');
  document.getElementById('tab-bridges').className='tab'+(name==='bridges'?' active':'');
  document.getElementById('tab-automation').className='tab'+(name==='automation'?' active':'');
  document.getElementById('view-input').className='view'+(name==='input'?' active':'');
  document.getElementById('view-output').className='view'+(name==='output'?' active':'');
  document.getElementById('view-memory').className='view'+(name==='memory'?' active':'');
  document.getElementById('view-search').className='view'+(name==='search'?' active':'');
  document.getElementById('view-searchtool').className='view'+(name==='searchtool'?' active':'');
  document.getElementById('view-bridges').className='view'+(name==='bridges'?' active':'');
  document.getElementById('view-automation').className='view'+(name==='automation'?' active':'');
  if(name==='output')refreshOutput();
  if(name==='memory')refreshMemory();
  if(name==='search')refreshMemory();  // 检索 tab 复用同一个数据源
  if(name==='bridges')loadBridges();
  if(name==='automation')loadAutomation();
}
var _autoPrompts = [];
function loadAutomation(){
  fetch('/automation').then(function(r){return r.json();}).then(function(d){
    var a = d.automation || {};
    document.getElementById('auto-interval').value = a.sys_interval || 30;
    _autoPrompts = (a.sys_prompts||[]).slice();
    renderAutoPrompts();
    // 填充桥下拉
    var sel = document.getElementById('auto-bridge');
    if(!sel.options.length){
      ['18000','18001','18002','18003','18004','18005','18006','18007','18008','18009'].forEach(function(v){
        var o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);
      });
    }
    sel.value = String(a.sys_bridge || '18002');
  });
}
function renderAutoPrompts(){
  var c = document.getElementById('auto-prompts'); c.innerHTML='';
  _autoPrompts.forEach(function(p, i){
    var wrap=document.createElement('div');wrap.style.cssText='display:flex;gap:6px;margin:4px 0';
    var inp=document.createElement('input');inp.className='content-edit';inp.value=p;
    inp.style.cssText='flex:1;font-size:13px;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 8px';
    inp.onchange=function(){_autoPrompts[i]=this.value;};
    wrap.appendChild(inp);
    var del=document.createElement('button');del.textContent='✕';
    del.style.cssText='background:#f85149;color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px';
    del.onclick=function(){_autoPrompts.splice(i,1);renderAutoPrompts();};
    wrap.appendChild(del);
    c.appendChild(wrap);
  });
}
function addAutoPrompt(){
  _autoPrompts.push(''); renderAutoPrompts();
  var els=document.querySelectorAll('#auto-prompts input');
  if(els.length) els[els.length-1].focus();
}
function saveAutomation(){
  fetch('/automation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    sys_interval: parseFloat(document.getElementById('auto-interval').value)||30,
    sys_bridge: document.getElementById('auto-bridge').value,
    sys_prompts: _autoPrompts.filter(function(s){return s.trim();})
  })}).then(function(r){return r.json();}).then(function(){
    var ok=document.getElementById('auto-ok');ok.style.display='inline';
    setTimeout(function(){ok.style.display='none';},2000);
  });
}

function loadBridges(){
  var list=document.getElementById('bridge-list');
  list.innerHTML='<div style="color:var(--dim)">检测中...</div>';
  fetch('/bridges').then(function(r){return r.json();}).then(function(d){
    document.getElementById('bridge-checked').textContent='最后查询: '+d.checked_at;
    var html='<table style="width:100%;border-collapse:collapse;font-size:14px">';
    html+='<tr style="color:var(--dim);text-align:left">'
      +'<th style="padding:6px;border-bottom:1px solid var(--border)">状态</th>'
      +'<th style="padding:6px;border-bottom:1px solid var(--border)">端口</th>'
      +'<th style="padding:6px;border-bottom:1px solid var(--border)">名称</th>'
      +'<th style="padding:6px;border-bottom:1px solid var(--border)">类型</th>'
      +'<th style="padding:6px;border-bottom:1px solid var(--border)">占用</th></tr>';
    (d.bridges||[]).forEach(function(b){
      var up=b.status==='up';
      var dot=up?'<span style="color:#3fb950">●</span>':'<span style="color:#f85149">●</span>';
      var st=up?'<span style="color:#3fb950">连通</span>':'<span style="color:#f85149">断开</span>';
      var type=b.type==='builtin'?'内置':(b.type==='system'?'系统':(b.type==='module'?'模块配置':b.type));
      html+='<tr><td style="padding:6px;border-bottom:1px solid var(--border)">'+dot+' '+st+'</td>'
        +'<td style="padding:6px;border-bottom:1px solid var(--border)">'+b.port+'</td>'
        +'<td style="padding:6px;border-bottom:1px solid var(--border)">'+esc(b.name)+'</td>'
        +'<td style="padding:6px;border-bottom:1px solid var(--border)">'+type+'</td>'
        +'<td style="padding:6px;border-bottom:1px solid var(--border)">'+esc((b.occupier||[]).join('、')||'—')+'</td></tr>';
    });
    html+='</table>';
    list.innerHTML=html;
  }).catch(function(){list.innerHTML='<div style="color:var(--red)">检测失败</div>';});
}

function refreshOutput(){
  fetch('/output').then(function(r){return r.json();}).then(function(d){
    document.getElementById('out-raw').textContent=d.raw||'（暂无）';
    var tc=document.getElementById('out-tools');
    if(d.tools&&d.tools.length){
      var h='';
      d.tools.forEach(function(t){h+='<div class="tool-item"><span class="tool-cmd">['+t.cmd+']</span> <span class="tool-arg">'+esc(t.arg)+'</span></div>';});
      tc.innerHTML=h;
    }else{tc.innerHTML='<div style="color:var(--dim)">（无工具调用）</div>';}
    // 队列
    var eq=d.exec_queue||[];
    var erq=d.error_queue||[];
    var xq=d.express_queue||[];
    document.getElementById('queue-exec').textContent=eq.length?eq.map(function(q){return '['+q.time+'] 搜索: '+q.arg;}).join('\\n'):'（空）';
    document.getElementById('queue-error').textContent=erq.length?erq.map(function(q){return '['+q.time+'] '+q.status+' '+q.cmd+': '+q.arg;}).join('\\n'):'（空）';
    document.getElementById('queue-express').textContent=xq.length?xq.map(function(q){return '['+q.time+'] '+q.status+' '+q.cmd+': '+q.arg;}).join('\\n'):'（空）';
  });
}
function refreshMemory(){
  fetch('/output').then(function(r){return r.json();}).then(function(d){
    // L0: 原文
    var l0=d.mem_l0||[];
    document.getElementById('m-l0').innerHTML=l0.length
      ? l0.map(function(c){return '<div class="ctx-item"><span class="ctx-role">['+c.role+']</span><span class="ctx-text">'+esc(c.content)+'</span></div>';}).join('')
      : '<div style="color:var(--dim)">（暂无）</div>';
    // L1~L4
    ['l1','l2','l3','l4'].forEach(function(lv){
      var items=d['mem_'+lv]||[];
      var ppls=(d['mem_'+lv+'_ppl'])||[];
      var times=(d['mem_'+lv+'_time'])||[];
      document.getElementById('m-'+lv).innerHTML=items.length
        ? items.map(function(x,i){
            var ppl=typeof ppls[i]!=='undefined'?' <span style="color:var(--dim);font-size:11px">ppl='+ppls[i]+'</span>':'';
            var tm=times[i]?' <span style="color:var(--blue);font-size:11px">['+times[i]+']</span>':'';
            return '<div style="padding:2px 0;font-size:13px;color:var(--green)">• '+tm+esc(x)+ppl+'</div>';
          }).join('')
        : '<div style="color:var(--dim)">（'+(lv==='l4'?'待接入向量检索':'暂无')+'）</div>';
    });
    // 拼装成品
    document.getElementById('m-final').innerHTML='<pre style="font-size:13px;margin:0;padding:8px;background:#0d1117;border-radius:6px">'+esc(d.final_context||'（暂无）')+'</pre>';
    // 检索 4 块
    refreshRetrieval(d.retrieval);
  });
}
function refreshRetrieval(rv){
  // rv 可能为 null（还没处理过任何一轮）
  var l0=rv&&rv.l0||{};
  var l05=rv&&rv.l05||'';
  var lb=rv&&rv.labels||null;
  var hit=rv&&rv.hit||null;
  var err=rv&&rv.error||null;
  // ① L0
  var l0Html='';
  if(l0.user)l0Html+='<div class="ctx-item"><span class="ctx-role">[输入]</span><span class="ctx-text">'+esc(l0.user)+'</span></div>';
  if(l0.thought)l0Html+='<div class="ctx-item"><span class="ctx-role">[想]</span><span class="ctx-text">'+esc(l0.thought)+'</span></div>';
  if(l0.action)l0Html+='<div class="ctx-item"><span class="ctx-role">[说]</span><span class="ctx-text">'+esc(l0.action)+'</span></div>';
  document.getElementById('r-l0').innerHTML=l0Html||'<div style="color:var(--dim)">等待对话...</div>';
  // ② L0.5（本地生成，不依赖服务）
  document.getElementById('r-l05').innerHTML=l05
    ? esc(l05)
    : '<div style="color:var(--dim)">等待对话...</div>';
  // 服务错误提示
  var errHtml=err?'<div style="color:var(--red);font-size:12px;margin-bottom:8px">⚠ '+esc(err)+'</div>':'';
  // ③ 标签计算（紧凑单页布局）
  var h=errHtml;
  if(!lb){
    h+='<div style="color:var(--dim)">等待检索服务返回...</div>';
  }else{
  // ── 表格：行=4属性，列=全量/硬币/Δ/主移动 ──
  var dims=['领域','规模','时间','评价'];
  var tl=lb.full_cls||{},cl=lb.coin_cls||{},dl=lb.delta||{};
  var mainDim=lb.main?lb.main.dim:null, mainDelta=lb.main?lb.main.delta:0;
  var hasMain=(mainDim&&Math.abs(mainDelta)>=0.05);
  h+='<table style="width:100%;border-collapse:collapse;font-size:12px">';
  h+='<tr style="color:var(--dim)"><th style="text-align:left;padding:3px 6px">属性</th><th style="text-align:left;padding:3px 6px">总体对话</th><th style="text-align:left;padding:3px 6px">仅硬币</th><th style="text-align:left;padding:3px 6px">Δ</th><th style="text-align:left;padding:3px 6px">主移动</th></tr>';
  dims.forEach(function(dm){
    var t=tl[dm]||{},c=cl[dm]||{};
    var dd=dl[dm]||{};
    var isMain=(dm===mainDim&&hasMain);
    h+='<tr style="border-top:1px solid var(--border)">';
    h+='<td style="padding:4px 6px;color:var(--dim);white-space:nowrap">'+dm+'</td>';
    h+='<td style="padding:4px 6px"><b>'+esc(t.top||'?')+'</b> <span style="color:var(--dim)">'+t.prob+'%</span></td>';
    h+='<td style="padding:4px 6px"><b>'+esc(c.top||'?')+'</b> <span style="color:var(--dim)">'+c.prob+'%</span></td>';
    h+='<td style="padding:4px 6px">'+((dd.delta>=0?'+':'')+dd.delta)+'</td>';
    h+='<td style="padding:4px 6px">'+(isMain?'<span style="color:var(--yellow)">← '+esc(lb.main?lb.main.dir:'')+' (|Δ|='+Math.abs(mainDelta).toFixed(2)+')</span>':(dm==='领域'?'—':'<span style="color:var(--dim)">—</span>'))+'</td>';
    h+='</tr>';
  });
  h+='</table>';
  if(!hasMain){
    h+='<div style="color:var(--dim);font-size:11px;margin-top:4px">（无显著移动：|Δ| 全部 < 0.05）</div>';
  }
  // 目标空间（一行标签）
  var tg=lb.target||{};
  h+='<div style="margin:8px 0;font-weight:bold;color:var(--blue)">目标空间</div>';
  h+='<div>';
  h+='<span class="ret-tag" style="background:var(--blue);color:#fff">领域:'+esc(tg['领域']||'?')+'</span>';
  ['规模','时间','评价'].forEach(function(dm){
    h+='<span class="ret-tag" style="background:var(--green);color:#fff">'+dm+':'+esc(tg[dm]||'?')+'</span>';
  });
  h+='</div>';
  }
  document.getElementById('r-labels').innerHTML=h;
  // ④ 检索命中
  var hh=errHtml;
  if(!hit){
    hh+='<div style="color:var(--dim)">等待检索服务返回...</div>';
  }else if(hit.candidates&&hit.candidates.length){
    hh+='<div style="color:var(--dim);font-size:12px;margin-bottom:6px">命中空间: '+esc(JSON.stringify(hit.space))+'</div>';
    hit.candidates.forEach(function(c,i){
      hh+='<div class="ret-hit">';
      hh+='<div style="margin-bottom:4px"><span class="ret-sim">sim='+c.sim+'</span> <span style="color:var(--dim);font-size:12px">[id:'+c.id+']</span>'+(i===0?' <span style="color:var(--yellow);font-size:11px">← TOP1</span>':'')+'</div>';
      hh+='<div style="color:var(--text)">'+esc(c.text)+'</div>';
      hh+='</div>';
    });
  }else{
    hh+='<div style="color:var(--dim)">无命中</div>';
  }
  document.getElementById('r-hit').innerHTML=hh;
}
function renderDim(name,c){
  if(!c)return'';
  var h='<div class="ret-dim"><div class="ret-dim-title">'+name+': '+esc(c.top)+' ('+c.prob+'%)</div>';
  (c.all||[]).forEach(function(t){
    var w=t.prob;
    var col=w>=50?'var(--green)':(w>=20?'var(--yellow)':'var(--red)');
    h+='<div style="font-size:11px;color:var(--dim)">'+esc(t.label)+' '+t.prob+'%</div>';
    h+='<div class="ret-bar"><div class="ret-bar-fill" style="width:'+w+'%;background:'+col+'"></div></div>';
  });
  h+='</div>';
  return h;
}
function renderDimCompact(name,c){
  if(!c)return'';
  var h='<div style="margin-bottom:6px"><div style="font-size:12px;color:var(--dim)">'+name+': <b style="color:var(--text)">'+esc(c.top)+'</b> ('+c.prob+'%)</div>';
  h+='<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:3px">';
  (c.all||[]).forEach(function(t){
    var w=t.prob;
    var col=w>=50?'var(--green)':(w>=20?'var(--yellow)':'var(--red)');
    h+='<div style="flex:1 1 48%;min-width:70px;background:#0d1117;border-radius:4px;padding:2px 4px">';
    h+='<div style="font-size:10px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+esc(t.label)+' '+t.prob+'%</div>';
    h+='<div class="ret-bar" style="height:3px;margin-top:1px"><div class="ret-bar-fill" style="width:'+w+'%;background:'+col+'"></div></div>';
    h+='</div>';
  });
  h+='</div></div>';
  return h;
}
function saveCfg(){
  var m=document.getElementById('cfg-msg');m.textContent='保存中...';
  fetch('/config/save').then(function(r){return r.json();}).then(function(d){
    var fn=d.path.split('/').pop()||d.path;m.textContent='已保存 → '+fn;setTimeout(function(){m.textContent='';},4000);
  }).catch(function(){m.textContent='保存失败';});
}
function loadCfg(){
  var m=document.getElementById('cfg-msg');m.textContent='加载中...';
  fetch('/config/load').then(function(r){return r.json();}).then(function(d){
    if(d.ok){modules=d.modules||[];render();updatePreview();m.textContent='已加载配置';setTimeout(function(){m.textContent='';},3000);}
    else{m.textContent='加载失败';}
  }).catch(function(){m.textContent='加载失败';});
}
function send(){
  var t=document.getElementById('input-text').value;if(!t)return;
  document.getElementById('input-text').value='';
  fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})}).then(r=>r.json()).then(function(d){
    var pre=document.getElementById('prompt-preview');
    pre.textContent=d.prompt||d.error;
    load();
  });
}
load();
var _autoRefresh=true;
setInterval(function(){
  if(!_autoRefresh)return;
  fetch('/modules').then(function(r){return r.json();}).then(function(d){
    window._srcLast=d.source_last||{};
    // 刷新 prompt 预览
    if(d.built_prompt)document.getElementById('prompt-preview').textContent=d.built_prompt;
    // 更新主页面最新收到
    modules.forEach(function(m){
      if(!m.sources)return;
      var sp=document.getElementById('last-'+m.id);
      if(!sp)return;
      var sl=window._srcLast||{};
      var srcNames=m.sources.map(function(s){return s.name;});
      var lastText='';
      for(var k in sl){if(srcNames.indexOf(k)>=0){lastText=sl[k];break;}}
      sp.textContent='最新: '+(lastText||'—');
    });
    // 刷新编辑框中的最新收到
    var ed=document.getElementById('modal-overlay');
    if(ed.style.display!=='none'){
      ed.querySelectorAll('[data-src-last]').forEach(function(el){
        var name=el.getAttribute('data-src-last');
        el.textContent='最新: '+((window._srcLast||{})[name]||'(等待中)');
      });
    }
    // 如果输出视图或记忆视图激活，自动刷新
    if(document.getElementById('view-output').className.indexOf('active')>=0)refreshOutput();
    if(document.getElementById('view-memory').className.indexOf('active')>=0)refreshMemory();
    if(document.getElementById('view-search').className.indexOf('active')>=0)refreshMemory();
  });
},3000);
function toggleAutoRefresh(){
  _autoRefresh=!_autoRefresh;
  document.getElementById('refresh-btn').textContent=_autoRefresh?'🔄 自动刷新:开':'⏸ 自动刷新:关';
}
function togglePromptPort(){
  var d=document.getElementById('prompt-port-cfg');
  d.style.display=d.style.display==='none'?'block':'none';
}
function toggleQueueCfg(queue){
  var el=document.getElementById(queue+'-cfg');
  el.style.display=el.style.display==='none'?'block':'none';
}
function saveQueuePort(queue){
  var port=parseInt(document.getElementById(queue+'-port').value);
  var ep=queue==='express'?'express_bridge':'exec_bridge';
  fetch('/config/'+ep,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({port:port})})
  .then(function(r){return r.json();}).then(function(d){
    var ok=document.getElementById(queue+'-port-ok');
    ok.style.display='';setTimeout(function(){ok.style.display='none';},2000);
  });
}
function saveErrorInterval(){
  var v=parseInt(document.getElementById('error-interval').value)||3;
  fetch('/config/error_interval',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({interval:v})})
  .then(function(r){return r.json();}).then(function(d){
    var ok=document.getElementById('error-interval-ok');
    ok.style.display='';setTimeout(function(){ok.style.display='none';},2000);
  });
}
var _expressMode = 'serial';  // serial|parallel
function toggleExpressMode(){
  _expressMode = _expressMode === 'serial' ? 'parallel' : 'serial';
  var btn = document.getElementById('express-mode-btn');
  btn.textContent = _expressMode === 'serial' ? '🔗 串行' : '⚡ 并行';
}
function toggleQueuePause(queue){
  var ep='pause_express';
  if(queue==='exec') ep='pause_exec';
  if(queue==='error') ep='pause_error';
  fetch('/queue/'+ep,{method:'POST'}).then(r=>r.json()).then(function(d){
    _setPauseBtn(queue, d.paused);
  });
}
function _setPauseBtn(queue, paused){
  var btn=document.getElementById('pause-'+queue+'-btn');
  if(paused){btn.textContent='⏸ 已暂停';btn.style.color='var(--red)';btn.style.borderColor='var(--red)';}
  else{btn.textContent='▶ 运行中';btn.style.color='var(--green)';btn.style.borderColor='var(--border)';}
}
(function(){
  fetch('/modules').then(r=>r.json()).then(function(d){
    _setPauseBtn('express', d.express_paused);
    _setPauseBtn('exec', d.exec_paused);
    _setPauseBtn('error', d.error_paused);
    if(d.express_bridge_port) document.getElementById('express-port').value = String(d.express_bridge_port);
    if(d.exec_bridge_port) document.getElementById('exec-port').value = String(d.exec_bridge_port);
  });
})();{
  var v=parseInt(document.getElementById('prompt-port').value)||18000;
  fetch('/config/port',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({port:v})})
  .then(function(r){return r.json();}).then(function(d){
    var ok=document.getElementById('prompt-port-ok');
    ok.style.display='';setTimeout(function(){ok.style.display='none';},2000);
  });
}
(function(){
  fetch('/modules').then(r=>r.json()).then(function(d){
    if(d.prompt_port){
      var sel=document.getElementById('prompt-port');
      var v=String(d.prompt_port);
      for(var i=0;i<sel.options.length;i++){if(sel.options[i].value===v){sel.value=v;return;}}
      // 不在预设列表里, 添加一个选项
      sel.innerHTML+='<option value="'+v+'" selected>'+v+'</option>';
    }
  });
})();
function toggleRetrievalCfg(){
  var d=document.getElementById('ret-cfg');
  d.style.display=d.style.display==='none'?'block':'none';
}
function saveRetrievalCfg(){
  var t=document.getElementById('ret-top').value;
  var p=document.getElementById('ret-port').value;
  fetch('/config/retrieval',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({top_n:parseInt(t),port:parseInt(p)})})
  .then(function(r){return r.json();}).then(function(d){
    document.getElementById('ret-cfg-ok').style.display='';setTimeout(function(){document.getElementById('ret-cfg-ok').style.display='none';},2000);
  });
}
(function(){
  fetch('/modules').then(r=>r.json()).then(function(d){
    if(d.retrieval_top_n)document.getElementById('ret-top').value=d.retrieval_top_n;
    if(d.retrieval_port)document.getElementById('ret-port').value=String(d.retrieval_port);
  });
})();
</script></body></html>"""

if __name__ == "__main__":
    # 启动内置桥 18000-18004
    BRIDGE_PORTS = [18000, 18001, 18002, 18003, 18004, 18005, 18006, 18007, 18008, 18009]
    for bp in BRIDGE_PORTS:
        try:
            bs = ReusableTCPServer(("0.0.0.0", bp), BridgeHandler)
            t = threading.Thread(target=bs.serve_forever, daemon=True)
            t.start()
            print(f"  桥 {bp} 已启动")
        except OSError:
            print(f"  桥 {bp} 已被占用, 跳过")
            continue
    # 后台线程: 持续推拼装成品到桥
    def _push_loop():
        import time
        time.sleep(1)
        while True:
            time.sleep(1)
            try:
                # 拼装成品 = final_context（话题摘要L2 + 最近摘要L1 + 对话历史L0，与页面预览同源）
                fc = _build_final_context()
                requests.post(f"http://localhost:{PROMPT_BRIDGE_PORT}/text",
                              json={"source": "上文记忆拼装成品", "text": fc}, timeout=3)
                build_prompt("[...]")
                # 检索命中 daemon 推桥 (跟上面一模一样)
                hit_block = ""
                if last_retrieval:
                    cands = last_retrieval.get("hit", {}).get("candidates", [])
                    texts = [c.get("text", "") for c in cands[:RETRIEVAL_TOP_N] if c.get("text")]
                    hit_block = "\n---\n".join(texts)
                if hit_block:
                    _retrieval_cache = hit_block
                if _retrieval_cache:
                    requests.post(f"http://localhost:{RETRIEVAL_PORT}/text",
                                  json={"source": "长期记忆检索", "text": _retrieval_cache}, timeout=3)
            except Exception:
                pass
    build_prompt("[...]")
    threading.Thread(target=_send_loop, daemon=True).start()   # LLM 发送状态机（冷却+积压+系统注入）
    threading.Thread(target=_l4_daily_export_loop, daemon=True).start()   # 每天00:00导出L4永久记忆
    threading.Thread(target=_push_loop, daemon=True).start()
    # 表达队列消费线程
    def _consume_express():
        import time as _time
        while True:
            _time.sleep(0.1)
            if express_paused or not express_queue:
                continue
            item = express_queue[0]
            if item["status"] == "等待":
                item["status"] = "执行中"
                try:
                    import time as _t
                    if item["cmd"] == "回复":
                        requests.post("http://localhost:8000/api/queue_text",
                            json={"text": item["arg"]}, timeout=30)
                        print(f"[TTS] 排队完成, 轮询桥{EXPRESS_BRIDGE_PORT}...")
                        for _ in range(600):
                            _time.sleep(0.1)
                            try:
                                sl = requests.get(f"http://localhost:{EXPRESS_BRIDGE_PORT}/source_last", timeout=2).json()
                                ttd = sl.get("tts_done", "MISSING")
                                if ttd == "done":
                                    print(f"[TTS] 桥{EXPRESS_BRIDGE_PORT}检测到 tts_done=done!")
                                    break
                                if _ == 0: print(f"[TTS] 等待done, 当前桥内容: {sl}")
                            except: pass
                        print(f"[TTS] 播放完成: {item['arg'][:30]}")
                        # 消费完立刻清桥，进入等待状态
                        try: requests.post(f"http://localhost:{EXPRESS_BRIDGE_PORT}/text", json={"source":"tts_done","text":"wait"}, timeout=2)
                        except: pass
                    elif item["cmd"] in ("表情", "报错", "结束"):
                        requests.post("http://localhost:18770/push",
                            json={"type": "emotion", "cmd": item["cmd"], "arg": item["arg"]}, timeout=5)
                        print(f"[表情] 推送完成, 轮询桥{EXPRESS_BRIDGE_PORT}...")
                        for _ in range(600):
                            _time.sleep(0.1)
                            try:
                                sl = requests.get(f"http://localhost:{EXPRESS_BRIDGE_PORT}/source_last", timeout=2).json()
                                if sl.get("tts_done") == "done":
                                    print(f"[表情] 桥{EXPRESS_BRIDGE_PORT}检测到 tts_done=done!")
                                    break
                            except: pass
                        print(f"[表情] 完成: {item['arg']}")
                        try: requests.post(f"http://localhost:{EXPRESS_BRIDGE_PORT}/text", json={"source":"tts_done","text":"wait"}, timeout=2)
                        except: pass
                except Exception as e:
                    print(f"[表达队列] 执行失败: {e}")
                item["status"] = "完成"
                express_queue.pop(0)
            _time.sleep(0.3)
    threading.Thread(target=_consume_express, daemon=True).start()

    # 执行队列消费者：搜索/报错/结束
    def _consume_exec():
        import time
        while True:
            time.sleep(0.1)
            if exec_paused or not exec_queue:
                continue
            item = exec_queue[0]
            if item["status"] == "等待":
                item["status"] = "执行中"
                try:
                    if item["cmd"] == "搜索":
                        print(f"[搜索] 查询: {item['arg']}")
                        try:
                            requests.post("http://localhost:18775/api/queue_search",
                                json={"query": item["arg"]}, timeout=5)
                        except Exception as e:
                            print(f"[搜索] 推送失败: {e}")
                        print(f"[搜索] 轮询桥{EXEC_BRIDGE_PORT}...")
                        for _ in range(600):
                            time.sleep(0.1)
                            try:
                                sl = requests.get(f"http://localhost:{EXEC_BRIDGE_PORT}/source_last", timeout=2).json()
                                if sl.get("tts_done") == "done":
                                    print(f"[搜索] done!")
                                    break
                            except: pass
                        try: requests.post(f"http://localhost:{EXEC_BRIDGE_PORT}/text", json={"source":"tts_done","text":"wait"}, timeout=2)
                        except: pass
                    elif item["cmd"] == "结束":
                        print(f"[执行队列] 结束: {item['arg']}")
                except Exception as e:
                    print(f"[执行队列] 失败: {e}")
                item["status"] = "完成"
                exec_queue.pop(0)
            time.sleep(0.3)
    threading.Thread(target=_consume_exec, daemon=True).start()

    # 报错队列消费者: 播放 → 等待间隔 → 下一个
    ERROR_WAV = '特殊语音/报错语音.wav'
    TTS_CACHE = '/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/asr+2d8.2确认/配置文件/frontend/tts_cache'
    def _consume_error():
        import shutil, time
        while True:
            time.sleep(0.1)
            if error_paused or not error_queue:
                continue
            item = error_queue[0]
            if item["status"] == "等待":
                item["status"] = "执行中"
                try:
                    src = '/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/asr+2d8.2确认/配置文件/frontend/' + ERROR_WAV
                    fname = 'error_' + str(int(time.time()*1000)) + '.wav'
                    shutil.copy(src, os.path.join(TTS_CACHE, fname))
                    requests.post("http://127.0.0.1:18770/push",
                        json={"type":"speak","audio_url":"tts_cache/" + fname}, timeout=3)
                    print(f"[报错] 播放, {ERROR_INTERVAL}s后继续...")
                    time.sleep(ERROR_INTERVAL)
                except Exception as e:
                    print(f"[报错队列] 失败: {e}")
                item["status"] = "完成"
                error_queue.pop(0)
            time.sleep(0.3)
    threading.Thread(target=_consume_error, daemon=True).start()
    # 启动管线中心
    port = 18772
    srv = ReusableTCPServer(("0.0.0.0", port), PipelineHandler)
    print(f"管线管理中心 → http://localhost:{port}")
    srv.serve_forever()
