"""
mini_coin v10.4 — UIA 桌面感知 + 结构化输入中心

核心架构:
  - 感知层: Windows UIA (desktop_diff.py) 读取窗口控件树 + 哈希 diff
  - 对话层: qwen3:8b 常驻显存 + 传入 chat history
  - 场景层: [当前场景] 段动态注入，包含窗口/输入/任务变化
  - 情绪层: logprobs 律度驱动 TTS + Live2D

依赖:
  - Bridge 服务器: localhost:8090 (server.py, 皮套控制)
  - Ollama: localhost:11434 (LLM)
  - 模块: context_cache.py, uia 感知层
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time

import requests
from flask import Flask, request, jsonify, Response, stream_with_context

# ── 项目根目录加入 Python path (重组后跨目录 import) ──
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ── 情绪系统 ──
from emotion.state_machine import build_guidance
from emotion import detector as emotion_detector
import threading as _threading


_last_emotion_measure_time = 0
EMOTION_MEASURE_INTERVAL = 30  # 秒

def _async_emotion_measure(user_text: str):
    """后台线程：触发情绪检测，最少间隔30秒"""
    global _last_emotion_measure_time
    now = time.time()
    if now - _last_emotion_measure_time < EMOTION_MEASURE_INTERVAL:
        return  # 压制频繁触发
    _last_emotion_measure_time = now
    try:
        emotion_detector.measure(user_text)
    except Exception as e:
        print(f"[情绪] 测量失败: {e}", flush=True)

def _async_task_analysis(user_text: str, ai_reply: str):
    """后台线程：调用 task-analyzer v2，全量覆盖笔记本状态"""
    try:
        scene = ctx.get_scene()
        tasks = ctx.get_tasks()
        context = ctx.get_context_items()
        
        _log_task(f"开始分析 (场景:{scene}, 任务:{len(tasks)}条, 上下文:{len(context)}条)")
        result = analyze(scene, tasks, context, user_text, ai_reply)
        
        if result is None:
            _log_task("分析失败，跳过")
            return
        
        _log_task(f"结果: scene={result['scene'][:30]}, tasks={result['tasks']}, context={result['context']}")
        
        # 全量覆盖
        ctx.set_scene(result["scene"])
        ctx.set_tasks(result["tasks"])
        ctx.set_context_items(result["context"])
        
        _log_task("笔记本已更新")
        
        # 🔍 提取关键词 → 向量检索 → 缓存结果供下一轮注入
        keywords = result.get("keywords", [])
        if keywords and isinstance(keywords, list) and any(k.strip() for k in keywords):
            try:
                all_hits = []
                seen = set()
                for kw in keywords:
                    kw = kw.strip()
                    if not kw or kw in seen:
                        continue
                    seen.add(kw)
                    hits = mv.search(kw, top_k=3)
                    for h in hits:
                        text = h.get("text", "").strip()
                        score = h.get("score", 0)
                        if text and score >= 0.35:
                            dedup_key = text[:80]
                            if dedup_key not in seen:
                                seen.add(dedup_key)
                                all_hits.append((text, score))
                
                # 按相关度降序，取 top 5
                all_hits.sort(key=lambda x: x[1], reverse=True)
                top = all_hits[:5]
                
                if top:
                    lines = [f"[记忆 · 来自关键词: {'/'.join(keywords[:3])}]"]
                    for text, score in top:
                        lines.append(f"  [{score:.2f}] {text}")
                    retrieval_text = "\n".join(lines)
                    ctx.set_memory_retrieval(retrieval_text)
                    _log_task(f"🔍 记忆检索: {len(top)} 条 (关键词: {keywords})")
                else:
                    ctx.set_memory_retrieval("")
                    _log_task(f"🔍 记忆检索: 无结果 (关键词: {keywords})")
            except Exception as e:
                _log_task(f"❌ 记忆检索失败: {e}")
                ctx.set_memory_retrieval("")
        else:
            ctx.set_memory_retrieval("")
            _log_task(f"⚠️ 未收到 keywords (原始结果: {result.get('keywords', '字段缺失')})")
    except Exception as e:
        _log_task(f"异常: {e}")


# ── v10.6: faster-whisper 本地 STT ──
_whisper_model = None
_whisper_lock = threading.Lock()

def _get_whisper_model():
    """懒加载 faster-whisper 模型（线程安全，CPU int8）"""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        from faster_whisper import WhisperModel
        print("[STT] 加载 faster-whisper small (CPU int8)...")
        _whisper_model = WhisperModel("Systran/faster-whisper-small", device="cpu", compute_type="int8")
        print("[STT] faster-whisper 模型加载完成")
        return _whisper_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_cache import ContextCache
from task_manager import analyze, warmup as warmup_task_analyzer

# ── 简易文件日志（TaskAnalyzer 调试用，WSL /tmp 目录） ──
_TASK_LOG = "/tmp/task_analyzer.log"
def _log_task(msg: str):
    """写 [TaskAnalyzer] 日志到文件 + stdout"""
    print(msg, flush=True)
    try:
        with open(_TASK_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass

# 启动时写一行，确认新代码已加载
_log_task("🟢 v10_server 已启动 (关键词检索版)")

# 记忆向量引擎 — 从项目根目录的 memory/ 包导入
from memory.vector import MemoryVector

# ── 配置 ──────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
BRIDGE_URL = "http://localhost:8090"
V10_PORT = 8091

LLM_MODEL = "minicoin_r1"

SOUL_PROMPT_FILE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "SOUL_mini_prompt.md"))

# 自主模式配置
HEARTBEAT_INTERVAL = 60.0      # 心跳间隔（秒）— 每分钟检查一次
DANMAKU_URGENT_THRESHOLD = 5   # 弹幕紧急触发阈值

app = Flask(__name__)

# ── v10.2: CORS 支持（允许浏览器跨端口请求）────
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ── 全局状态 ──────────────────────────────────────

ctx = ContextCache()
mv = MemoryVector()  # 向量记忆引擎

# ── 感知收集器 (四条流 → 每日 JSONL) ──────────────
from memory.collector import get_collector
_collector = get_collector()

# ── UIA 场景感知 ──────────────────────────────
# 视觉感知已迁移至 desktop_perception.py / desktop_diff.py
# 旧的 VL 截图/OCR 管线已封存至 legacy_vision_pipeline.py

# ── 辅助函数 ──────────────────────────────────────

LONG_TERM_MEMORY_FILE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "long_term_memory.md"))

def load_soul_prompt() -> str:
    """加载 SOUL prompt + 长期记忆 + 抽象标签"""
    try:
        with open(SOUL_PROMPT_FILE, "r", encoding="utf-8") as f:
            soul = f.read()
        
        parts = [soul]
        
        # 注入静态长期记忆
        if os.path.exists(LONG_TERM_MEMORY_FILE):
            with open(LONG_TERM_MEMORY_FILE, "r", encoding="utf-8") as f:
                parts.append("## 长期记忆\n\n" + f.read())
        
        # 注入 API 生成的抽象标签
        memory_dir = os.path.join(_PROJ_ROOT, "data", "memory")
        abstract_memories = _load_abstract_labels(memory_dir)
        if abstract_memories:
            parts.append("## 抽象知识库\n\n" + abstract_memories)
        
        return "\n\n---\n\n".join(parts)
    except FileNotFoundError:
        return "你是 mini_coin，一个AI桌面伙伴。用中文回复。"


def _load_abstract_labels(memory_dir: str) -> str:
    """加载所有 summary.json 中的抽象标签和永久知识"""
    lines = []
    try:
        for fname in sorted(os.listdir(memory_dir)):
            if not fname.endswith("_summary.json"):
                continue
            with open(os.path.join(memory_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            for label in data.get("abstract_labels", []):
                name = label.get("label", "")
                matched = len(label.get("matched", []))
                lines.append(f"- [{label.get('level','?')}] {name} (证据:{matched}条)")
            for pk in data.get("permanent_knowledge", []):
                content = pk.get("content", "")[:150]
                lines.append(f"- [永久知识] {content}")
    except Exception as e:
        print(f"[Memory] 加载抽象标签失败: {e}")
    return "\n".join(lines[:50])  # 最多50条，控制token



def parse_emotion(text: str) -> dict:
    """从 LLM 输出中提取情绪标签 (+ 记忆标签) 和纯文本回复"""
    m = re.search(r'\[emotion:(\w+)\]', text)
    emotion = m.group(1) if m else "neutral"

    # 提取 [remember:...] 内联记忆判决（方案B：主LLM顺手判断，零额外API调用）
    rm = re.search(r'\[remember:(.+?)\]', text)
    remember_summary = rm.group(1).strip() if rm else None

    # 清理所有标签，留纯文本用于展示/TTS
    clean = re.sub(r'\[emotion:\w+\]\s*', '', text)
    clean = re.sub(r'\[remember:.+?\]\s*', '', clean).strip()

    result = {"emotion": emotion, "reply": clean}
    if remember_summary:
        result["remember_summary"] = remember_summary
    return result


def puppet_emotion(emotion: str):
    """通过 Bridge HTTP API 切换皮套表情 + 触发自主神经动作"""
    try:
        requests.post(f"{BRIDGE_URL}/emotion",
                      json={"emotion": emotion}, timeout=2)
    except Exception as e:
        print(f"[v10] 皮套表情失败: {e}")
    # 异步触发自主神经系统（不阻塞）
    threading.Thread(target=trigger_autonomic, args=(emotion,), daemon=True).start()


def trigger_autonomic(emotion: str):
    """调用自主神经小模型 → 获取动作 → 发送到 3D puppet"""
    try:
        r = requests.post("http://127.0.0.1:8092/action",
                         json={"emotion": emotion}, timeout=3)
        data = r.json()
        action = data.get("action", "")
        if action and action != "none":
            print(f"[v10] 自主神经: {emotion} → {action}")
            requests.post(f"{BRIDGE_URL}/puppet_action",
                         json={"action": action}, timeout=2)
    except Exception as e:
        pass  # 自主神经不可用时静默


# ── 情绪 → TTS 语音参数映射（edge-tts rate/volume/pitch）──
EMOTION_TTS_PARAMS = {
    "happy":     {"rate": "+5%",   "volume": "+5%",  "pitch": "+8Hz"},
    "sad":       {"rate": "-8%",   "volume": "-5%",  "pitch": "-10Hz"},
    "surprise":  {"rate": "+5%",   "volume": "+8%",  "pitch": "+10Hz"},
    "angry":     {"rate": "+3%",   "volume": "+10%", "pitch": "+5Hz"},
    "fear":      {"rate": "+8%",   "volume": "+3%",  "pitch": "+8Hz"},
    "think":     {"rate": "-3%",   "volume": "+0%",  "pitch": "-3Hz"},
    "panic":     {"rate": "+8%",   "volume": "+5%",  "pitch": "+8Hz"},
    "shy":       {"rate": "-5%",   "volume": "-3%",  "pitch": "-5Hz"},
    "obey":      {"rate": "-5%",   "volume": "-3%",  "pitch": "-5Hz"},
    "neutral":   {"rate": "+0%",   "volume": "+0%",  "pitch": "+0Hz"},
}

def puppet_speak(text: str, duration_ms: int = None, emotion: str = "neutral"):
    """Edge TTS 生成 mp3（带情绪参数）→ 发送音频 URL 到前端。前端播完自动删。"""
    # 收集: AI 输出
    _collector.record_ai(text)
    
    import asyncio, hashlib
    import edge_tts
    tts_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "tts_cache"))
    os.makedirs(tts_dir, exist_ok=True)
    voice = "zh-CN-XiaoyiNeural"
    
    # 根据情绪选 TTS 参数
    tts_params = EMOTION_TTS_PARAMS.get(emotion, EMOTION_TTS_PARAMS["neutral"])
    cache_key = hashlib.md5((text + voice + str(tts_params)).encode()).hexdigest()[:12]
    filename = f"{cache_key}.mp3"
    audio_path = os.path.join(tts_dir, filename)
    
    async def _gen():
        await edge_tts.Communicate(
            text, voice,
            rate=tts_params["rate"],
            volume=tts_params["volume"],
            pitch=tts_params["pitch"],
        ).save(audio_path)
    
    try:
        if not os.path.exists(audio_path):
            asyncio.run(_gen())
        real_dur_ms = max(2000, int(len(text) / 4 * 1000) + 500)
    except Exception as e:
        print(f"[TTS] 生成失败: {e}, 回退文本模式")
        real_dur_ms = max(2000, len(text) * 120)
        requests.post(f"{BRIDGE_URL}/speak",
                      json={"text": text, "duration_ms": real_dur_ms}, timeout=2)
        return
    
    audio_url = f"tts_cache/{filename}"
    requests.post(f"{BRIDGE_URL}/speak",
                  json={"text": text, "audio_url": audio_url, "duration_ms": real_dur_ms}, timeout=2)


def puppet_params(params: dict):
    """注入 Live2D 原始参数"""
    try:
        requests.post(f"{BRIDGE_URL}/params", json=params, timeout=1)
    except Exception as e:
        print(f"[v10] 皮套参数失败: {e}")


def puppet_perception_event(event_type: str, data: dict = None):
    """通过 Bridge 广播感知事件给前端（用于即时皮套反应）"""
    try:
        requests.post(f"{BRIDGE_URL}/perception_event",
                      json={"type": event_type, "data": data or {}}, timeout=1)
    except Exception:
        pass  # 如果 bridge 不支持 /perception_event（旧版），静默失败


def call_ollama_chat(prompt: str, history: list = None) -> str:
    """调用 Ollama LLM（qwen3:8b）"""
    messages = [{"role": "system", "content": load_soul_prompt()}]
    if history:
        for h in history[-20:]:  # 最多保留 20 轮历史
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "num_predict": 2048,
            "repeat_penalty": 1.15,
            "repeat_last_n": 64,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.3,
            "top_p": 0.9,
        },
        "keep_alive": -1,
    }, timeout=120)

    full = resp.json()["message"]["content"]
    # 提取 </think> 之后的内容（保留实际回复，跳过 Qwen3 推理块）
    think_end = full.rfind('</think>')
    if think_end != -1:
        clean = full[think_end + 8:].strip()
    else:
        clean = full.strip()
    clean = clean.strip()
    # 如果剥离后为空（全部是 think），返回空让上层跳过
    if not clean:
        print("[LLM] 回复全部是 think 块，返回空")
    return clean



# ── 记忆向量辅助函数 ──────────────────────────────

def _load_memory_context(user_query: str) -> str:
    """
    搜索与当前查询相关的历史记忆，返回格式化的 prompt 文本。
    异步安全：search() 内部加锁。
    """
    if not user_query or not user_query.strip():
        return ""

    try:
        results = mv.search(user_query, top_k=3)
        if not results:
            return ""

        lines = []
        for r in results:
            lines.append(f"我记得：{r['text']}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Memory] search 失败: {e}")
        return ""


def _archive_memory(user_msg: str, assistant_reply: str, visual_context: str, summary: str = None):
    """
    后台线程：归档本轮对话记忆。
    - 如果有 summary（主LLM已内联判断）→ 直接 embed+store，零额外API调用
    - 如果没有 summary → 回退到 mv.judge_and_store（调qwen3:8b判断，兼容旧行为）
    不阻塞 /chat 响应。
    """
    try:
        if summary:
            # 方案B：主LLM已判断，直接存
            mid = mv.store(summary, metadata={
                "user_msg": user_msg[:100],
                "reply": assistant_reply[:100],
                "context": visual_context[:100] if visual_context else "",
            })
            if mid:
                print(f"[Memory] ✅ 已归档: {summary[:60]}")
            return

        # 回退：skill LLM 判断（兼容心跳/旧版调用）
        result = mv.judge_and_store(user_msg, assistant_reply, visual_context)
        if result.get("stored"):
            print(f"[Memory] ✅ 已归档: {result.get('summary', '')[:60]}")
        else:
            decision = result.get("decision", "?")
            if decision != "forget":
                print(f"[Memory] ⏭️ 跳过 ({decision})")
    except Exception as e:
        print(f"[Memory] archive 失败: {e}")


# ── 场景配置（手动 JSON）─────────────────────────

SCENE_CONFIG_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "scene_config.json")
)
_scene_cache = {"scene": "", "mtime": 0}

def _load_scene_config() -> str:
    """读取手动场景配置，返回格式化的 [场景] 行。缓存按 mtime 失效。"""
    try:
        mtime = os.path.getmtime(SCENE_CONFIG_FILE)
        if mtime == _scene_cache["mtime"] and _scene_cache["scene"]:
            return _scene_cache["scene"]
        with open(SCENE_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        scene_text = data.get("scene", "").strip()
        if scene_text:
            result = f"[场景] {scene_text}"
        else:
            result = ""
        _scene_cache["scene"] = result
        _scene_cache["mtime"] = mtime
        return result
    except Exception as e:
        print(f"[Scene] 读取失败: {e}")
        return _scene_cache.get("scene", "")


# ── Tool Calling ──────────────────────────────────

TOOL_CALL_RE = re.compile(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[/TOOL\]', re.DOTALL)
perception_paused = threading.Event()

def pause_perception():
    pass  # VL 已封存，不再需要

def resume_perception():
    pass


# ── 对话历史摘要生成 ──────────────────────────────

def _build_history_summary(ctx: "ContextCache") -> str:
    """
    从 ContextCache 提取上文摘要：对话历史 + 弹幕摘要 + 工具返回。
    按时间排序，标明来源（拉姆说/我说/弹幕/工具）。
    """
    events = []  # [(timestamp, formatted_line)]

    # 用户发言 → "拉姆说xxx"
    for t, txt in ctx._data.get("user_msg_history", []):
        short = txt[:60] + ("..." if len(txt) > 60 else "")
        events.append((t, f"拉姆说{short}"))

    # AI自发言 → "我说xxx"
    for t, txt in ctx._data.get("self_utterance_history", []):
        short = txt[:60] + ("..." if len(txt) > 60 else "")
        events.append((t, f"我说{short}"))

    # 工具返回 → "我做了xxx"
    for t, td in ctx._data.get("tool_result_history", []):
        result_text = td.get("result", "")
        if not result_text:
            continue
        first_line = result_text.split('\n')[0][:80]
        char_count = len(result_text)
        summary = ContextCache._summarize_tool(
            td.get("tool", "?"), td.get("args", ""),
            result_text, first_line, char_count
        )
        events.append((t, summary))

    # 弹幕摘要 → 前3条，"xx说"格式
    dm_queue = ctx._data.get("danmaku_queue", [])
    for d in dm_queue[:3]:
        text_short = d["text"][:40] + ("..." if len(d["text"]) > 40 else "")
        events.append((0, f"{d['user']}说{text_short}"))

    # 按时间排序
    events.sort(key=lambda e: e[0])

    # 取最近 10 条
    recent = events[-10:] if len(events) > 10 else events

    lines = [f"  {txt}" for _, txt in recent]
    if not lines:
        return ""

    return "对话历史：\n" + "\n".join(lines)


# ── PowerShell 键鼠操作（WSL 兼容）──
# 问题: Add-Type 编译需要干净环境块，WSL 的巨型 env 会导致 PowerShell 报错
# 方案: 启动时写 .ps1 文件预编译，后续直接调脚本避免 Add-Type 开销


# ── 颜色分类工具 ──────────────────────────────────

def execute_tool(name: str, args: str) -> str:
    """执行 tool 调用，返回结果文本注入 LLM"""
    args = args.strip()

    if name == "eye_search":
        return "eye_search 已迁移至 UIA 感知层 (desktop_diff.py)，待接入。"

    elif name == "eye_fullscan_api":
        return "eye_fullscan_api 已迁移至 UIA 感知层，待接入。"

    elif name == "hand_click":
        # ⚠️ 已禁用：避免与人类操作者抢鼠标
        return "hand_click 已禁用（保护模式）"

    elif name == "hand_type":
        return "hand_type 已禁用（保护模式）"

    elif name == "hand_scroll":
        return "hand_scroll 已禁用（保护模式）"

    elif name == "hand_press":
        return "hand_press 已禁用（保护模式）"

    elif name == "read_file":
        # 读取本地文件或列出目录
        # 用法: [TOOL:read_file] path/to/file  [/TOOL]
        if not args:
            return "错误: 需要指定文件路径"
        try:
            import os as _os
            filepath = args.strip().strip('"').strip("'")
            # Windows 路径 → WSL 路径转换
            if len(filepath) >= 2 and filepath[1] == ':':
                drive = filepath[0].lower()
                rest = filepath[2:].replace('\\', '/')
                filepath = f"/mnt/{drive}{rest}"
            if not _os.path.isabs(filepath):
                filepath = _os.path.join(_os.path.dirname(__file__), filepath)
            filepath = _os.path.normpath(filepath)
            
            if not _os.path.exists(filepath):
                return f"路径不存在: {filepath}，请检查拼写。"
            
            # 如果是目录 → 列出内容
            if _os.path.isdir(filepath):
                items = _os.listdir(filepath)
                dirs = [f"  [DIR]  {d}/" for d in sorted(items) if _os.path.isdir(_os.path.join(filepath, d))]
                files = [f"  [FILE] {f}" for f in sorted(items) if _os.path.isfile(_os.path.join(filepath, f))]
                lines = [f"目录 '{filepath}' 包含 {len(dirs)} 个子目录, {len(files)} 个文件:\n"]
                if dirs:
                    lines.extend(dirs[:20])
                    if len(dirs) > 20:
                        lines.append(f"  ... 还有 {len(dirs)-20} 个子目录")
                if files:
                    lines.extend(files[:20])
                    if len(files) > 20:
                        lines.append(f"  ... 还有 {len(files)-20} 个文件")
                if not dirs and not files:
                    lines.append("  (空目录)")
                return "\n".join(lines)
            
            # 文件大小检查
            if _os.path.getsize(filepath) > 10 * 1024 * 1024:
                return f"文件过大 ({_os.path.getsize(filepath)/1024/1024:.0f}MB)，超过10MB上限"
            
            # 处理特殊文件格式
            ext = _os.path.splitext(filepath)[1].lower()
            
            if ext == ".docx":
                # Word 2007+ → zip 解压 XML 提取
                try:
                    import zipfile as _zf, xml.etree.ElementTree as _ET
                    with _zf.ZipFile(filepath) as z:
                        xml_content = z.read("word/document.xml")
                    tree = _ET.fromstring(xml_content)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    texts = []
                    for p in tree.iter(f"{{{ns['w']}}}p"):
                        line = "".join(t.text or "" for t in p.iter(f"{{{ns['w']}}}t"))
                        if line.strip():
                            texts.append(line.strip())
                    content = "\n".join(texts)
                    if len(content) > 50000:
                        content = content[:50000] + "\n\n[已截断前 50000 字符]"
                    return f"文件内容 ({filepath}, .docx 提取 {len(texts)} 段):\n{content}"
                except Exception as e:
                    return f"无法解析 .docx 文件: {e}。尝试用 Word 打开后另存为 .txt。"
            
            if ext == ".doc":
                # Word 97-2003 → 二进制提取 UTF-16LE 文本
                try:
                    with open(filepath, "rb") as f:
                        data = f.read()
                    text = data.decode("utf-16-le", errors="ignore")
                    import re as _re
                    chinese = _re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\n\r]+", text)
                    content = "".join(chinese)
                    lines = [l for l in content.split("\n") if len(l.strip()) > 5]
                    content = "\n".join(lines)
                    if len(content) > 50000:
                        content = content[:50000] + "\n\n[已截断前 50000 字符]"
                    if len(content) < 50:
                        return f"无法从 .doc 文件提取有意义文本 ({filepath})。文件可能是加密的或格式不标准。"
                    return f"文件内容 ({filepath}, .doc 提取 {len(lines)} 行):\n{content}"
                except Exception as e:
                    return f"无法解析 .doc 文件: {e}。尝试用 Word 打开后另存为 .txt。"
            
            # 普通文本文件
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(50000)
            total = _os.path.getsize(filepath)
            if len(content) >= 50000:
                more = f"\n\n[文件总大小 {total} 字节, 已截断前 50000 字符]"
            else:
                more = ""
            lines = content.count("\n") + 1
            return f"文件内容 ({filepath}, {lines} 行):\n{content}{more}"
        except UnicodeDecodeError:
            return f"无法读取: {filepath} 不是文本文件（可能是二进制格式如 .doc/.exe）"
        except PermissionError:
            return f"权限不足: 无法访问 {filepath}。文件可能被其他程序占用。"
        except Exception as e:
            return f"read_file 失败: {e}"

    elif name == "web_search":
        # 秘塔搜索引擎
        # 用法: [TOOL:web_search] 搜索关键词 [/TOOL]
        if not args:
            return "错误: 需要指定搜索关键词"
        try:
            from web_search import web_search
            return web_search(args.strip(), count=5)
        except Exception as e:
            return f"web_search 失败: {e}"

    elif name == "task_done":
        resume_perception()
        return "任务完成，glance 已恢复。"

    elif name == "suspend":
        # 悬置一条上下文到 [当前状态]
        # 用法: [TOOL:suspend] 创造者让我叫他小明 [/TOOL]
        if not args:
            return "用法: suspend 要记住的内容"
        ctx.add_suspended(args.strip())
        return f"已悬置: {args.strip()}"

    elif name == "unsuspend":
        # 取消悬置
        # 用法: [TOOL:unsuspend] 小明 [/TOOL]
        if not args:
            return "用法: unsuspend 关键词 (移除包含该词的悬置项)"
        n = ctx.remove_suspended(args.strip())
        return f"移除了 {n} 条悬置项" if n else f"未找到包含 '{args.strip()}' 的悬置项"

    elif name == "list_suspended":
        # 列出所有悬置项
        items = ctx.list_suspended()
        if items:
            return "当前悬置:\n" + "\n".join(f"  - {t}" for t in items)
        return "无悬置上下文"

    elif name == "clear_suspended":
        ctx.clear_suspended()
        return "所有悬置已清除"

    elif name == "reset_vision":
        return "reset_vision 已废弃（VL 管线已封存）"

    elif name in ("mini_coin_help", "mini_coin_mail"):
        return f"{name} 暂未接入（WSL 迁移后待重做）。请直接告诉用户你目前做不到这件事。"

    else:
        return f"未知 tool: {name}"


# ── API 端点 ──────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """
    对话入口 — 统一输入总线

    请求: {"text": "用户说的话", "history": [...]}
    响应: {"reply": "...", "emotion": "happy"}
    
    自动注入 visual上下文 + 弹幕批次到 prompt 中。
    用户说话时标记 last_user_msg_time。
    """
    data = request.get_json(force=True)
    user_text = data.get("text", "") or data.get("message", "") or ""
    history = data.get("history", [])

    # 标记用户说话了
    if user_text.strip():
        ctx.mark_user_spoke()
        ctx.add_user_msg(user_text)  # 记录到统一时间线
        # 收集: 用户输入
        _collector.record_user(user_text)

    # 🔒 暂停感知循环 — 避免 VL 推理与 Ollama LLM 抢 GPU
    pause_perception()

    # 构建 prompt
    prompt_parts = []

    # 0. 场景描述（手动配置）
    scene_line = _load_scene_config()
    if scene_line:
        prompt_parts.append(scene_line)

    # 情绪行为指引
    emotion_guidance = build_guidance()
    ctx.set_emotion_guidance(emotion_guidance or "")

    # 1. 对话历史摘要（广义场景的一部分：上文 + 弹幕摘要 + 工具返回）
    history_summary = _build_history_summary(ctx)
    if history_summary:
        prompt_parts.append(history_summary)

    # 2. 弹幕（当前轮，精确）
    danmaku_text = ctx.get_danmaku_text()
    if danmaku_text:
        prompt_parts.append(danmaku_text)

    # 3. 用户消息
    if user_text.strip():
        prompt_parts.append(f"[对话] 拉姆: {user_text}")
    else:
        # 没有用户消息时（心跳触发），给 LLM 一个自主上下文
        offline_sec = ctx.get_offline_seconds()
        if offline_sec > 60:
            offline_min = int(offline_sec / 60)
            prompt_parts.append(f"[系统状态] 主人离线 {offline_min} 分钟。你处于自主模式。")
        prompt_parts.append("当前没有人在跟你说话。根据弹幕和视觉状态，自主决定是否行动。")

    # 4. 记忆上下文（放在用户消息后面，作为"我记得"的自然补充）
    if user_text.strip():
        memory_context = _load_memory_context(user_text)
        if memory_context:
            prompt_parts.append(memory_context)
        kw_memory = ctx.get_memory_retrieval()
        if kw_memory:
            prompt_parts.append(kw_memory)

    # 如果什么上下文都没有，返回空
    if not prompt_parts:
        resume_perception()
        return jsonify({"reply": "", "emotion": "neutral", "skipped": True})

    prompt = "\n\n".join(prompt_parts)

    # 记录本轮 prompt 快照（调试回溯）
    ctx.record_prompt(prompt[:500], danmaku_text, trigger="chat")

    # ── Tool Calling 循环 ──
    max_tool_rounds = 10
    tool_history = []  # 每次请求独立，不继承历史
    final_reply = ""
    final_emotion = "neutral"

    for round_i in range(max_tool_rounds):
        # 合并原始历史 + 本轮工具调用记录，让 LLM 看到完整的"我说了什么 → 工具返回了什么"链条
        merged_history = history + tool_history
        llm_output = call_ollama_chat(prompt, merged_history)
        result = parse_emotion(llm_output)

        # 检测 tool 调用
        tool_matches = TOOL_CALL_RE.findall(llm_output)
        if tool_matches:
            if round_i == 0:
                pause_perception()  # 第一轮 tool call 时暂停 glance

            non_tool_text = TOOL_CALL_RE.sub("", llm_output).strip()
            if non_tool_text:
                tool_history.append({"role": "assistant", "content": non_tool_text})

            for tool_name, tool_args in tool_matches:
                print(f"[ToolCall] 执行 {tool_name}: {tool_args[:80]}")
                tool_result = execute_tool(tool_name, tool_args)

                # 工具结果入输入中心 → 下次 get_context_for_prompt() 自动包含
                ctx.add_tool_result(tool_name, tool_args, tool_result)

                tool_history.append({
                    "role": "assistant",
                    "content": f"[TOOL:{tool_name}] {tool_args} [/TOOL]"
                })
                tool_history.append({
                    "role": "system",
                    "content": f"tool 返回: {tool_result}\n继续思考。如果需要继续操作，用 [TOOL:xxx] 格式。完成任务后回复用户。"
                })

            # 注入当前画面作为背景参考（不要复述画面，专注 tool 结果）
            vision_ctx = ctx.get_context_for_prompt()
            if vision_ctx:
                vision_brief = vision_ctx.split("\n")[0] if "\n" in vision_ctx else vision_ctx[:100]
                prompt = f"Tool 返回: {tool_result}\n\n（背景: {vision_brief}）\n专注 tool 结果，不要描述画面。完成任务直接回复用户。"
            else:
                prompt = f"Tool 返回: {tool_result}\n\n根据返回结果继续。完成任务直接回复用户。"
            continue

        # 没有 tool call → 这是最终回复
        final_reply = result["reply"]
        final_emotion, emotion_intensity = emotion_detector.get_top_emotion()  # logprobs 律度，替换旧的 [emotion:xxx] 标签
        if round_i > 0:
            resume_perception()
        else:
            resume_perception()  # 非 tool 路径也需恢复
        break

    # 如果全是 tool call 没有最终回复
    if not final_reply.strip():
        resume_perception()
        return jsonify({"reply": "操作已完成", "emotion": "neutral"})

    # 清空已消耗的弹幕
    ctx.get_danmaku_and_clear()

    # 清理回复 (剥离 TOOL/think/emotion/remember 标签)
    final_reply = re.sub(r'\[TOOL:.*?\[/TOOL\]\s*', '', final_reply, flags=re.DOTALL)
    final_reply = re.sub(r'<think>.*?</think>', '', final_reply, flags=re.DOTALL)
    final_reply = re.sub(r'\[emotion:\w+\]\s*', '', final_reply)
    final_reply = re.sub(r'\[remember:.+?\]\s*', '', final_reply).strip()

    # 驱动皮套
    puppet_emotion(final_emotion)
    time.sleep(0.15)
    puppet_speak(final_reply, emotion=final_emotion)
    # 记录自发言：让 LLM 下次知道自己说过什么
    ctx.add_self_utterance(final_reply)

    # 异步归档记忆（后台线程，不阻塞响应）
    if user_text.strip():
        vision_ctx = ctx.get_context_for_prompt()
        summary = result.get("remember_summary")
        threading.Thread(
            target=_archive_memory,
            args=(user_text, final_reply, vision_ctx, summary),
            daemon=True,
        ).start()

    # 异步情绪测量（后台线程，不阻塞响应）
    _threading.Thread(
        target=_async_emotion_measure,
        args=(user_text,),
        daemon=True,
    ).start()

    # 异步任务分析（task-analyzer 模型，不阻塞响应）
    if user_text.strip():
        _threading.Thread(
            target=_async_task_analysis,
            args=(user_text, final_reply),
            daemon=True,
        ).start()

    return jsonify({
        "reply": final_reply,
        "emotion": final_emotion,
        "raw": llm_output,
    })


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """
    流式对话入口 — SSE (Server-Sent Events)
    
    请求: {"text": "用户说的话"}
    响应: text/event-stream
    
    事件类型:
    - data: {"type":"token","text":"..."}    逐 token 流
    - data: {"type":"emotion","emotion":"happy"}  情绪标签
    - data: {"type":"done","reply":"...","emotion":"happy"}  完成
    - data: {"type":"error","message":"..."}  错误
    """
    data = request.get_json(force=True)
    user_text = data.get("text", "") or data.get("message", "") or ""

    if not user_text.strip():
        return jsonify({"reply": "", "emotion": "neutral", "skipped": True})

    ctx.mark_user_spoke()
    ctx.add_user_msg(user_text)

    # 🔒 暂停感知循环 — 避免 VL 推理与 Ollama LLM 抢 GPU
    pause_perception()

    # 构建 prompt（和 /chat 逻辑一致）
    prompt_parts = []

    # 0. 场景描述（手动配置）
    scene_line = _load_scene_config()
    if scene_line:
        prompt_parts.append(scene_line)

    emotion_guidance = build_guidance()
    ctx.set_emotion_guidance(emotion_guidance or "")
    
    # 对话历史摘要（广义场景的一部分）
    history_summary = _build_history_summary(ctx)
    if history_summary:
        prompt_parts.append(history_summary)

    danmaku_text = ctx.get_danmaku_text()
    if danmaku_text:
        prompt_parts.append(danmaku_text)

    prompt_parts.append(f"[对话] 拉姆: {user_text}")
    memory_context = _load_memory_context(user_text)
    if memory_context:
        prompt_parts.append(memory_context)
    kw_memory = ctx.get_memory_retrieval()
    if kw_memory:
        prompt_parts.append(kw_memory)
    prompt = "\n\n".join(prompt_parts)

    def generate():
        """SSE 生成器：支持工具调用循环"""
        try:
            tool_history = []
            max_rounds = 5
            final_emotion = "neutral"
            final_reply = ""
        
            for rnd in range(max_rounds):
                messages = [{"role": "system", "content": load_soul_prompt()}]
                for h in tool_history:
                    messages.append(h)
                messages.append({"role": "user", "content": prompt if rnd == 0 else "根据 tool 返回结果决定下一步。"})
            
                full_reply = ""
            
                try:
                    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
                        "model": LLM_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {"temperature": 0.7, "num_predict": 2048},
                        "keep_alive": -1,
                    }, stream=True, timeout=30)
                
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line: continue
                        try: chunk = json.loads(line)
                        except: continue
                        if chunk.get("done"): break
                        token = chunk.get("message", {}).get("content", "")
                        if not token: continue
                        full_reply += token
                        yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
                
                    # 只用 logprobs 律度（不用 LLM 内联 [emotion:xxx] 标签）
                    final_emotion = "neutral"
                    _detected, _ = emotion_detector.get_top_emotion()
                    if _detected != "neutral":
                        final_emotion = _detected
                
                    # 检测 tool 调用
                    tool_matches = TOOL_CALL_RE.findall(full_reply)
                    if tool_matches:
                        for tool_name, tool_args in tool_matches:
                            yield f"data: {json.dumps({'type': 'tool_start', 'name': tool_name}, ensure_ascii=False)}\n\n"
                            tool_result = execute_tool(tool_name, tool_args)
                            # 工具结果入输入中心
                            ctx.add_tool_result(tool_name, tool_args, tool_result)
                            tool_history.append({"role": "assistant", "content": f"[TOOL:{tool_name}] {tool_args} [/TOOL]"})
                            tool_history.append({"role": "system", "content": f"tool 返回: {tool_result}\n继续思考。如果需要继续操作，用 [TOOL:xxx] 格式。完成任务后回复用户。"})
                            yield f"data: {json.dumps({'type': 'tool_end', 'name': tool_name}, ensure_ascii=False)}\n\n"
                        continue  # 继续下一轮
                
                    # 没有 tool → 最终回复
                    final_reply = full_reply
                    break
                
                except Exception as e:
                    print(f"[SSE] 错误: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
                    return
        
        
            if not final_reply:
                final_reply = "操作已完成"
        
            # 清理输出
            clean = re.sub(r'\[TOOL:.*?\[/TOOL\]\s*', '', final_reply, flags=re.DOTALL)
            clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL)
            clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL)
            clean = re.sub(r'\[emotion:\w+\]\s*', '', clean)
            clean = re.sub(r'\[remember:.+?\]\s*', '', clean).strip()
        
            # 清空弹幕
            ctx.get_danmaku_and_clear()

            # 驱动皮套
            if clean:
                puppet_emotion(final_emotion)
                time.sleep(0.1)
                puppet_speak(clean, emotion=final_emotion)
                ctx.add_self_utterance(clean)

            # 异步归档记忆
            if user_text.strip():
                rm = re.search(r'\[remember:(.+?)\]', final_reply)
                remember_summary = rm.group(1).strip() if rm else None
                vision_ctx = ctx.get_context_for_prompt()
                threading.Thread(
                    target=_archive_memory,
                    args=(user_text, clean, vision_ctx, remember_summary),
                    daemon=True,
                ).start()

                # 异步情绪测量
                _threading.Thread(
                    target=_async_emotion_measure,
                    args=(user_text,),
                    daemon=True,
                ).start()

                # 异步任务分析（task-analyzer 模型，不阻塞响应）
                _threading.Thread(
                    target=_async_task_analysis,
                    args=(user_text, clean),
                    daemon=True,
                ).start()

            yield f"data: {json.dumps({'type': 'done', 'reply': clean, 'emotion': final_emotion}, ensure_ascii=False)}\n\n"
        finally:
            resume_perception()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.route("/stt", methods=["POST"])
def stt_transcribe():
    """
    语音转文字 — faster-whisper 本地 CPU 推理
    
    请求: multipart/form-data, audio 字段为 WAV 音频文件
    响应: {"text": "识别结果", "success": true}
    
    首次调用会加载模型（~10s），后续调用 <1s。
    """
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "缺少 audio 字段"}), 400
    
    audio_file = request.files["audio"]
    print(f"[STT] 收到音频: {audio_file.filename}, content_type={audio_file.content_type}")
    
    # 保存为临时文件（faster-whisper 需要文件路径）
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        audio_file.save(tmp.name)
        tmp.close()
        file_size = os.path.getsize(tmp.name)
        print(f"[STT] 临时文件: {tmp.name} ({file_size} bytes)")
        
        if file_size < 1000:
            print(f"[STT] 音频文件太小，跳过 ({file_size} bytes)")
            return jsonify({"success": False, "error": "音频太短"})
        
        # 调用 faster-whisper
        model = _get_whisper_model()
        segments, info = model.transcribe(tmp.name, language="zh", beam_size=5)
        
        text = " ".join(seg.text.strip() for seg in segments)
        print(f"[STT] 识别结果 ({info.language}, p={info.language_probability:.2f}): {text[:60]}")
        
        return jsonify({"success": True, "text": text})
    except Exception as e:
        print(f"[STT] 识别失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass




@app.route("/context", methods=["GET"])
def get_context():
    """查询当前视觉上下文"""
    return jsonify({
        "prompt_context": ctx.get_context_for_prompt(),
        "danmaku": ctx.get_danmaku_text(),
        "danmaku_count": ctx.get_danmaku_count(),
        "offline_seconds": round(ctx.get_offline_seconds(), 1),
        "emotion_guidance": ctx.get_emotion_guidance(),
        "stats": ctx.stats(),
    })


@app.route("/debug_prompt", methods=["GET"])
def debug_prompt():
    """调试：查看当前 ContextCache 全量状态 + 最近N轮 prompt 快照"""
    return jsonify({
        "danmaku_text": ctx.get_danmaku_text(),
        "danmaku_count": ctx.get_danmaku_count(),
        "timeline": ctx.get_context_for_prompt(),
        "stats": ctx.stats(),
        "offline_seconds": round(ctx.get_offline_seconds(), 1),
        "history": ctx.get_prompt_history(),
    })


# ── 输入总线排序 ──
@app.route("/pipeline/order", methods=["GET", "POST"])
def pipeline_order():
    """查询或设置输入总线各 section 的输出顺序"""
    if request.method == "GET":
        return jsonify({"order": list(ctx.SECTION_ORDER)})
    data = request.get_json(force=True)
    new_order = data.get("order")
    if not new_order or not isinstance(new_order, list) or len(new_order) != len(ctx.SECTION_ORDER):
        return jsonify({"error": "invalid order", "expected": ctx.SECTION_ORDER}), 400
    ctx.SECTION_ORDER = new_order
    ctx.save_pipeline_config()
    return jsonify({"ok": True, "order": new_order})


@app.route("/danmaku", methods=["POST"])
def danmaku():
    """
    弹幕接收端点 — 由 danmaku_listener 推送

    请求: {"user": "用户名", "text": "弹幕内容", "type": "danmaku|super_chat|guard", "score": 0}
    响应: {"ok": true, "count": 3, "urgent": false}
    
    弹幕写入 ContextCache.danmaku_queue，LLM 在 /chat 时自动读取。
    写入后检查是否达到紧急阈值，返回 urgent 标志供 danmaku_listener 判断是否主动触发推理。
    """
    data = request.get_json(force=True)
    user = data.get("user", "未知")
    text = data.get("text", "")
    msg_type = data.get("type", "danmaku")
    score = data.get("score", 0)

    if not text.strip():
        return jsonify({"ok": False, "error": "empty text"}), 400

    ctx.add_danmaku(user, text, score, msg_type)
    
    # 收集: 弹幕输入
    _collector.record_danmaku(f"[{user}] {text}")
    
    count = ctx.get_danmaku_count()
    urgent = ctx.is_danmaku_urgent(DANMAKU_URGENT_THRESHOLD)

    return jsonify({"ok": True, "count": count, "urgent": urgent})


@app.route("/status", methods=["GET"])
def status():
    """v10 服务状态"""
    offline_sec = ctx.get_offline_seconds()
    return jsonify({
        "v10": "10.6",
        "llm_model": LLM_MODEL,
        "vision": "UIA Windows Desktop Perception",
        "cache_stats": ctx.stats(),
        "danmaku_count": ctx.get_danmaku_count(),
        "offline_seconds": round(offline_sec, 1),
        "autonomous": offline_sec > HEARTBEAT_INTERVAL,
    })


@app.route("/action", methods=["POST"])
def action():
    """直接操控皮套（调试用）"""
    data = request.get_json(force=True)
    action_type = data.get("type", "")

    if action_type == "emotion":
        puppet_emotion(data.get("emotion", "neutral"))
    elif action_type == "speak":
        puppet_speak(data.get("text", ""), data.get("duration_ms"), emotion=data.get("emotion", "neutral"))
    elif action_type == "params":
        puppet_params(data.get("params", {}))
    elif action_type == "reset":
        puppet_params({})  # Bridge 的 /reset 处理
        try:
            requests.post(f"{BRIDGE_URL}/reset", timeout=1)
        except Exception:
            pass
    else:
        return jsonify({"error": f"unknown action type: {action_type}"}), 400

    return jsonify({"ok": True})




# ── 自主心跳循环（后台线程）─────────────────────────

def heartbeat_loop():
    """
    自主心跳循环：每 HEARTBEAT_INTERVAL 秒检查是否需要自主触发推理。

    触发条件（满足任一即触发）：
      1. 弹幕队列非空 AND 用户离线
      2. 沉默超时 — 60s 没说话时轻推一下
    """
    print(f"[Heartbeat] 心跳循环已启动 (interval={HEARTBEAT_INTERVAL}s)")

    last_periodic_trigger = 0
    PERIODIC_INTERVAL = 30.0
    SPEAK_COOLDOWN = 0.0     # 说话冷却已取消，靠 HEARTBEAT_INTERVAL 控制频率
    SILENCE_TIMEOUT = 60.0   # 超过 60s 没说话 → 轻推一下（保持时间线活跃）

    while True:
        try:
            time.sleep(HEARTBEAT_INTERVAL)
            
            offline = ctx.get_offline_seconds()
            danmaku_count = ctx.get_danmaku_count()
            silence_age = ctx.get_last_self_utterance_age()

            # 基础发言冷却——防止模型自己停不下来
            if silence_age is not None and silence_age < SPEAK_COOLDOWN:
                continue

            should_trigger = False
            trigger_reason = ""

            # 弹幕 → 触发
            if danmaku_count > 0:
                urgent = ctx.is_danmaku_urgent(DANMAKU_URGENT_THRESHOLD)
                should_trigger = urgent or (offline > HEARTBEAT_INTERVAL)
                trigger_reason = "danmaku" if should_trigger else ""
            # 沉默太久了 → 轻推（不管什么模式，至少每分钟说一句保持时间戳）
            elif silence_age is not None and silence_age > SILENCE_TIMEOUT and time.time() - last_periodic_trigger > PERIODIC_INTERVAL:
                should_trigger = True
                trigger_reason = "silence_timeout"
                last_periodic_trigger = time.time()

            if not should_trigger:
                continue

            print(f"[Heartbeat] 触发自主推理 ({trigger_reason}, danmaku={danmaku_count})")

            prompt_parts = []

            # 0. 场景描述（手动配置）
            scene_line = _load_scene_config()
            if scene_line:
                prompt_parts.append(scene_line)

            emotion_guidance = build_guidance()
            ctx.set_emotion_guidance(emotion_guidance or "")

            # 对话历史摘要（广义场景的一部分）
            history_summary = _build_history_summary(ctx)
            if history_summary:
                prompt_parts.append(history_summary)

            danmaku_text = ctx.get_danmaku_text()

            # 1. 弹幕（如果有）
            if danmaku_text:
                prompt_parts.append(danmaku_text)

            # 对话历史摘要
            history_summary = _build_history_summary(ctx)
            if history_summary:
                prompt_parts.append(history_summary)

            # 记忆（放在弹幕/用户消息后面，作为"我记得"的补充）
            if danmaku_text:
                memory_context = _load_memory_context(danmaku_text)
                if memory_context:
                    prompt_parts.append(memory_context)
            kw_memory = ctx.get_memory_retrieval()
            if kw_memory:
                prompt_parts.append(kw_memory)

            # 给 LLM 足够的决策信息
            offline_sec = ctx.get_offline_seconds()
            silence_sec = ctx.get_last_self_utterance_age()
            
            status_parts = []
            status_parts.append(f"创造者上次发言: {offline_sec:.0f}s前" if offline_sec < 60 else f"创造者 {int(offline_sec/60)} 分钟没说话了")
            if silence_sec is not None:
                status_parts.append(f"你上次发言: {silence_sec:.0f}s前")
            status_parts.append("决定原则: 根据和创造者的对话节奏，判断该说什么。")
            status_parts.append("如果你不确定该不该开口 → 沉默。TLDR: when in doubt, shut up。")
            prompt_parts.append("[决策信息] " + " | ".join(status_parts))

            prompt = "\n\n".join(prompt_parts)

            # DEBUG: 记录发送给 LLM 的视觉上下文
            if vision_context:
                print(f"[Heartbeat→LLM] 视觉上下文 ({len(vision_context)}字): {vision_context[:300]}")

            # 记录本轮 prompt 快照（调试回溯）
            ctx.record_prompt("heartbeat: " + "\n".join(prompt_parts[-3:]) if len(prompt_parts) > 3 else "\n".join(prompt_parts), trigger=trigger_reason)

            # 调用 LLM — 暂停感知循环确保 GPU 独占
            pause_perception()
            llm_output = call_ollama_chat(prompt, [])
            resume_perception()
            result = parse_emotion(llm_output)

            # 清空已消耗的弹幕
            ctx.get_danmaku_and_clear()

            if not result["reply"].strip():
                print("[Heartbeat] LLM 跳过（无回复）")
                continue

            # 过滤工具调用（心跳不处理工具，直接传给用户会乱）
            import re as _re
            clean_reply = _re.sub(r'\[TOOL:[^\]]*\].*?\[/TOOL\]', '', result["reply"], flags=_re.DOTALL).strip()

            # 驱动皮套 — 只走 logprobs 测量，不下坠到 LLM 内联标签
            _final_em, _ = emotion_detector.get_top_emotion()
            puppet_emotion(_final_em)
            time.sleep(0.15)
            puppet_speak(clean_reply, emotion=_final_em)
            ctx.add_self_utterance(clean_reply)

            # 异步归档记忆 — 弹幕互动也是记忆来源
            if danmaku_text:
                summary = result.get("remember_summary")
                threading.Thread(
                    target=_archive_memory,
                    args=(danmaku_text, result["reply"], vision_context if vision_context else "", summary),
                    daemon=True,
                ).start()

                # 异步情绪测量
                _threading.Thread(
                    target=_async_emotion_measure,
                    args=(danmaku_text,),
                    daemon=True,
                ).start()

            # 记忆维护 (压缩+提取, 非阻塞)

            print(f"[Heartbeat] ✨ {clean_reply[:60]}...")

        except Exception as e:
            import traceback
            print(f"[Heartbeat] 错误: {e}")
            resume_perception()  # 确保无论如何都恢复感知循环


# ── 感知主循环（后台线程）─────────────────────────

def _auto_process_memory():
    """启动时检查并处理未完成的每日 JSONL（含向量库写入）"""
    try:
        memory_dir = os.path.join(_PROJ_ROOT, "data", "memory")
        for fname in sorted(os.listdir(memory_dir)):
            if not fname.endswith(".jsonl") or fname.startswith("wechat_"):
                continue
            date = fname.replace(".jsonl", "")
            summary_file = os.path.join(memory_dir, f"{date}_summary.json")
            if os.path.exists(summary_file):
                continue  # 已处理
            # 跳过今天（还在收集中）
            today = time.strftime("%Y-%m-%d")
            if date == today:
                continue
            print(f"[Memory] 自动处理 {date} ({os.path.getsize(os.path.join(memory_dir, fname))} bytes)...")
            try:
                from memory.api import process_date
                process_date(date)
            except Exception as e:
                print(f"[Memory] 自动处理 {date} 失败: {e}")
    except Exception as e:
        print(f"[Memory] 自动处理异常: {e}")

# ── 启动 ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(" mini_coin v10.6 — 本地 STT (faster-whisper)")
    print("=" * 55)
    print(f" LLM:   {LLM_MODEL} @ {OLLAMA_URL}")
    print(f" 视觉:   UIA Windows Desktop Perception (desktop_diff.py)")
    print(f" STT:   faster-whisper small (CPU int8)")
    print(f" 皮套:   {BRIDGE_URL}")
    print(f" 本服:   http://localhost:{V10_PORT}")
    print()
    print(" 端点:")
    print(f"   POST /chat        — 对话 (<500ms)")
    print(f"   POST /chat_stream — 流式对话 (SSE)")
    print(f"   POST /stt         — 语音转文字 (本地 CPU)")
    print(f"   POST /danmaku     — 弹幕推送")
    print(f"   GET  /context     — 视觉上下文")
    print(f"   GET  /status      — 运行状态")
    print(f"   POST /action      — 皮套调试")
    print()

    # ── 情绪基线初始化 ──
    print(" [情绪] 测量基线 (中性上下文)...")
    try:
        emotion_detector.init_baseline()
        print(" [情绪] 基线就绪")
    except Exception as e:
        print(f" [情绪] 基线测量失败 (情绪系统降级): {e}")

    # ── 任务分析器预热（后台，不阻塞启动）──
    _threading.Thread(target=warmup_task_analyzer, daemon=True).start()

    # ── 启动服务 ──
    print()

    # 加载 pipeline 持久化配置
    ctx.load_pipeline_config()
    print(f" Pipeline: {ctx.SECTION_ORDER}")

    print(f" 后台线程:")
    print(f"   heartbeat_loop  — 自主决策 ({HEARTBEAT_INTERVAL}s)")
    print("=" * 55)

    # 自动处理未完成的每日记忆
    threading.Thread(target=_auto_process_memory, daemon=True).start()


    # 启动自主心跳循环
    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    # 启动 Flask
    app.run(host="127.0.0.1", port=V10_PORT, debug=False, threaded=True)
