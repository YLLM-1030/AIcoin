"""
mini_coin v10.3.1 — 双模型并行异步 + 向量记忆系统 (方案B: 主LLM内联判断)

核心突破: 双模型常驻显存 + 双通道感知(Fast+Deep) + 零 IO 上下文缓存 + 向量记忆
v10.1 新增: 统一输入总线 — 弹幕 + 视觉 + 语音平等注入 ContextCache, LLM自主决策
v10.3 新增: 向量记忆系统 — nomic-embed-text embedding + 语义搜索 + 自动归档
v10.3.1 优化: 记忆判断从独立LLM调用改为主LLM内联 [remember:...] — 零额外GPU调用

架构:
  ┌─ /chat (端口 8091) ─────────────────────────────┐
  │  对话线程: qwen3:8b 常驻显存，<500ms 立即回复      │
  │  自动注入 ContextCache 中的视觉上下文 + 弹幕         │
  │  解析 [emotion:xxx] 标签 → 驱动皮套                 │
  ├─ /vision (异步) ─────────────────────────────────┤
  │  视觉线程: qwen2.5vl-test 常驻显存，2-4s 深度分析    │
  │  结果写入 ContextCache，不阻塞对话                   │
  ├─ /fast (CPU) ───────────────────────────────────┤
  │  FastPerception: <100ms 画面变化/颜色/字幕检测       │
  │  触发即时皮套反应（不经过 LLM）                      │
  ├─ /danmaku (接收) ───────────────────────────────┤
  │  弹幕管道写入 ContextCache.danmaku_queue            │
  ├─ perception_loop (后台) ─────────────────────────┘
  │  每秒: 截图 → FastPerception → 有变化触发 VL 深度分析
  └─ heartbeat_loop (后台) ──────────────────────────┘
     每 8s 检查: 弹幕+视觉 → 自主触发 LLM 推理

依赖:
  - Bridge 服务器: localhost:8090 (server.py, 皮套控制)
  - Ollama: localhost:11434 (LLM + VL 模型)
  - 模块: context_cache.py, fast_perception.py

启动:
  python v10_server.py
  (需先启动 server.py 和 Ollama)
"""

import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from io import BytesIO

import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from PIL import Image

# ── 项目根目录加入 Python path (重组后跨目录 import) ──
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ── llama-cpp-python 视觉后端（替代 Ollama VL）──
_vision_llamacpp = None
_vision_llamacpp_lock = threading.Lock()

def _get_vision_model():
    """懒加载 llama-cpp-python 视觉模型（线程安全，CUDA）"""
    global _vision_llamacpp
    if _vision_llamacpp is not None:
        return _vision_llamacpp
    with _vision_llamacpp_lock:
        if _vision_llamacpp is not None:
            return _vision_llamacpp
        if not USE_LLAMACPP:
            return None
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        print(f"[Vision] 加载 Qwen2.5-VL-3B (全速)...")
        _vision_llamacpp = Llama(
            model_path=LLAMACPP_MODEL_PATH,
            chat_handler=Qwen25VLChatHandler(clip_model_path=LLAMACPP_MMPROJ_PATH),
            n_ctx=4096,
            n_gpu_layers=-1,        # 全层 GPU，3B 模型 ~2GB，不减速
            verbose=False,
        )
        print("[Vision] llama-cpp-python 模型加载完成")
        return _vision_llamacpp

def _call_llamacpp_vision(img_b64: str, prompt: str, max_tokens: int = 300) -> str:
    """用 llama-cpp-python 做视觉推理"""
    model = _get_vision_model()
    if model is None:
        return '{"error": "llama-cpp model not available"}'
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": '描述这是个什么UI元素（输入法/微信图标/按钮），列出上面可见的文字内容。只输出JSON: {"desc":"UI描述和文字","x":像素X,"y":像素Y}'},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    content = response["choices"][0]["message"]["content"]
    # 去markdown包装
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()
    return content

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

# 确保能导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_cache import ContextCache
from fast_perception import FastPerception

# 记忆向量引擎 — 从项目根目录的 memory/ 包导入
from memory.vector import MemoryVector

# ── 配置 ──────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
BRIDGE_URL = "http://localhost:8090"
V10_PORT = 8091

LLM_MODEL = "mini_coin-v7:latest"          # P4S4_v7 17阶段微调主脑（对话 + 弹幕回复）
VISION_ENABLED = True

# ── llama-cpp-python 视觉后端配置 ─────────────────
LLAMACPP_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), "models", "qwen2.5-vl-3b",
    "Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
)
LLAMACPP_MMPROJ_PATH = os.path.join(
    os.path.expanduser("~"), "models", "qwen2.5-vl-3b",
    "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"
)
USE_LLAMACPP = os.path.exists(LLAMACPP_MODEL_PATH) and os.path.exists(LLAMACPP_MMPROJ_PATH)  # mmproj 可选

# ── 云端 VL API 配置（fullscan 兜底）─────────────
CLOUD_VL_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
CLOUD_VL_MODEL = "qwen-vl-max"  # 或 qwen-vl-plus
CLOUD_VL_ENABLED = bool(CLOUD_VL_API_KEY)

SOUL_PROMPT_FILE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "SOUL_mini_prompt.md"))

# 感知循环配置
PERCEPTION_INTERVAL = 1.0      # Fast 层每秒跑一次
DEEP_VISION_INTERVAL = 1.5     # VL 深度分析最小间隔（秒）— 打字等小变化更快响应
DEEP_VISION_MAX_INTERVAL = 5.0 # VL 深度分析最大间隔（秒）

# 自主模式配置
HEARTBEAT_INTERVAL = 30.0      # 心跳间隔（秒）— 每半分钟检查一次
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
fast_eye = FastPerception(ocr_enabled=True)
mv = MemoryVector()  # 向量记忆引擎

# ── 感知收集器 (四条流 → 每日 JSONL) ──────────────
from memory.collector import get_collector
_collector = get_collector()

last_deep_time = 0.0

_vl_processing = False    # VL 处理中标志: True=正在分析, 跳过新截图
_scanning_busy = False    # DINO+VL 串行锁: 一个 cycle 完成前不开始新的

_perception_running = True  # perception_loop 停止标志

# ── 统一截图函数（PowerShell GDI 原生，兼容 RDP/多显示器）──

def capture_screen() -> Image.Image | None:
    """统一截图：PowerShell GDI → PIL Image
    加固版：20s 超时 + 最多 3 次重试 + 超时后检查文件是否已写入
    返回 None 表示截图彻底失败（调用方需自行处理降级）"""
    import subprocess, os as _os
    tmp_win = "C:\\Users\\Autogram-coin\\AppData\\Local\\Temp\\mini_coin_cap.jpg"
    tmp_wsl = "/mnt/c/Users/Autogram-coin/AppData/Local/Temp/mini_coin_cap.jpg"
    ps_cmd = (
        'Add-Type -AssemblyName System.Drawing,System.Windows.Forms;'
        '$ErrorActionPreference="Stop";'
        '$s=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;'
        '$b=New-Object System.Drawing.Bitmap($s.Width,$s.Height);'
        '$g=[System.Drawing.Graphics]::FromImage($b);'
        '$g.CopyFromScreen(0,0,0,0,$s.Size);'
        f'try{{$b.Save("{tmp_win}",[System.Drawing.Imaging.ImageFormat]::Jpeg)}}catch{{Write-Error "SAVE_FAILED: $_"}};'
        '$g.Dispose();$b.Dispose()'
    )
    ps_args = ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-Command", ps_cmd]

    MAX_RETRIES = 3
    CAPTURE_TIMEOUT = 20  # 1920x1080 GDI 在 WSL 跨层调用需要较长时间

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 清理旧文件，避免读到上一次的截图
            if _os.path.exists(tmp_wsl):
                _os.unlink(tmp_wsl)
            result = subprocess.run(ps_args, capture_output=True,
                           timeout=CAPTURE_TIMEOUT, env={'PATH': ''})
            if _os.path.exists(tmp_wsl):
                return Image.open(tmp_wsl)
            else:
                stderr_text = result.stderr.decode("utf-8", errors="replace").strip()[:200] if result.stderr else "(empty)"
                print(f"[Screen] PS 完成但文件未生成 (attempt {attempt}/{MAX_RETRIES}) stderr: {stderr_text}")
        except subprocess.TimeoutExpired:
            # 超时后 PS 进程可能仍在运行并最终写入文件
            print(f"[Screen] PS 超时 ({CAPTURE_TIMEOUT}s) attempt {attempt}/{MAX_RETRIES}")
            if _os.path.exists(tmp_wsl):
                # 超时但文件已被 PS 后台写入 → 直接用它
                try:
                    return Image.open(tmp_wsl)
                except Exception:
                    pass  # 文件可能不完整，继续重试
        except FileNotFoundError:
            print(f"[Screen] PowerShell 未找到 (attempt {attempt}/{MAX_RETRIES})")
            return None  # 不可恢复的错误
        except Exception as e:
            print(f"[Screen] PS 截图异常 (attempt {attempt}/{MAX_RETRIES}): {e}")

    print("[Screen] 截图彻底失败 (3 次尝试均失败)")
    return None


# ── EyeSystem: GNDINO 九宫格常驻扫描 + 自动 focus ──

_eye_system = None
_eye_lock = threading.Lock()

def _get_eye() -> "EyeSystem":
    global _eye_system
    if _eye_system is None:
        with _eye_lock:
            if _eye_system is not None:
                return _eye_system
            from groundingdino.util.inference import load_model, predict, load_image
            import groundingdino
            CFG = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")
            WGT = os.path.join(os.path.expanduser("~"), "models", "grounding-dino", "groundingdino_swint_ogc.pth")
            _eye_system = _EyeSystem(load_model(CFG, WGT), predict, load_image)
            print("[EyeSystem] GNDINO 初始化完成")
    return _eye_system


class _EyeSystem:
    """GNDINO 全屏检测 → bbox + 类别 + 置信度。作为注意力算法，持续观察变化区域。"""
    LOOP_CATS = "icon . button . window . text . input . menu . taskbar . notification"

    def __init__(self, model, predict_fn, load_image_fn):
        self.model = model
        self.predict = predict_fn
        self.load_image = load_image_fn

    def scan(self, pil_img):
        """
        全屏 GNDINO (0.2s) → 返回所有检测元素。
        """
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        pil_img.save(tmp.name, quality=85)
        try:
            _, tens = self.load_image(tmp.name)
        finally:
            os.unlink(tmp.name)
        
        boxes, logits, phrases = self.predict(
            self.model, tens, self.LOOP_CATS, box_threshold=0.25, text_threshold=0.20)
        
        elements = []
        w, h = pil_img.size
        for box, conf, phrase in zip(boxes, logits, phrases):
            # GNDINO 返回归一化 cxcywh: [center_x, center_y, width, height] (0-1)
            # 转为像素 xyxy: [x1, y1, x2, y2]
            cx_norm, cy_norm, bw_norm, bh_norm = box[0], box[1], box[2], box[3]
            cx = int(cx_norm * w)
            cy = int(cy_norm * h)
            x1 = int((cx_norm - bw_norm/2) * w)
            y1 = int((cy_norm - bh_norm/2) * h)
            x2 = int((cx_norm + bw_norm/2) * w)
            y2 = int((cy_norm + bh_norm/2) * h)
            bbox = [x1, y1, x2, y2]
            elements.append({
                "label": phrase,
                "bbox": bbox,
                "center": [cx, cy],
                "confidence": float(conf),
            })
        elements.sort(key=lambda e: -e["confidence"])
        return elements


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
        return "你是 mini_coin，一个AI桌面伙伴。用中文回复。回复前用 [emotion:xxx] 标记表情。"


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


def pil_to_base64(img: Image.Image, max_size=(1920, 1080)) -> str:
    """PIL Image → base64 JPEG"""
    img = img.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


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


def puppet_speak(text: str, duration_ms: int = None):
    """Edge TTS 生成 mp3 → 发送音频 URL 到前端。前端播完自动删。"""
    # 收集: AI 输出
    _collector.record_ai(text)
    
    import asyncio, hashlib
    import edge_tts
    tts_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "tts_cache"))
    os.makedirs(tts_dir, exist_ok=True)
    voice = "zh-CN-XiaoyiNeural"
    cache_key = hashlib.md5((text + voice).encode()).hexdigest()[:12]
    filename = f"{cache_key}.mp3"
    audio_path = os.path.join(tts_dir, filename)
    
    async def _gen():
        await edge_tts.Communicate(text, voice).save(audio_path)
    
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
        "options": {"temperature": 0.7, "num_predict": 2048},
        "keep_alive": -1,
    }, timeout=30)

    full = resp.json()["message"]["content"]
    # 跳过 Qwen3 <think> 块（处理闭合和未闭合两种情况）
    clean = re.sub(r'<think>.*?</think>', '', full, flags=re.DOTALL)
    # 如果 think 标签没闭合（num_predict 不够），删掉从 <think> 到末尾
    clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL)
    clean = clean.strip()
    # 如果剥离后为空（全部是 think），返回空让上层跳过
    if not clean:
        print("[LLM] 回复全部是 think 块，返回空")
    return clean


# ── 云端 VL API (阿里百炼 DashScope) ──

FULLSCAN_CLOUD_PROMPT = """你是一个桌面视觉助手。{task}

请用中文 JSON 回答（只输出 JSON）：
{{
  "natural_desc": "对屏幕内容的总体描述",
  "elements": [
    {{"label": "元素名", "bbox_2d": [x1, y1, x2, y2], "confidence": 0.95}}
  ]
}}
bbox_2d 使用绝对像素坐标。不要遗漏任何细节。"""


def call_cloud_vl(img_b64: str, prompt: str) -> str:
    """调用云端 VL 模型做 fullscan"""
    if not CLOUD_VL_ENABLED:
        return '{"error": "CLOUD_VL not configured (set DASHSCOPE_API_KEY)"}'
    resp = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {CLOUD_VL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": CLOUD_VL_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]
            }],
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        timeout=60,
    )
    result = resp.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


# ── 旧版兼容（保持向后兼容，内部调用 glance）───


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
            lines.append(f"[记忆] {r['text']}  (相关度:{r['score']:.2f})")
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


# ── Tool Calling ──────────────────────────────────

TOOL_CALL_RE = re.compile(r'\[TOOL:(\w+)\]\s*(.*?)\s*\[/TOOL\]', re.DOTALL)
perception_paused = threading.Event()  # 为 True 时暂停 glance


def pause_perception():
    """暂停 glance 循环，防止 focus 期间被覆盖"""
    perception_paused.set()
    print("[ToolCall] 暂停 glance 循环")


def resume_perception():
    """恢复 glance 循环"""
    perception_paused.clear()
    print("[ToolCall] 恢复 glance 循环")


# ── PowerShell 键鼠操作（WSL 兼容）──
# 问题: Add-Type 编译需要干净环境块，WSL 的巨型 env 会导致 PowerShell 报错
# 方案: 启动时写 .ps1 文件预编译，后续直接调脚本避免 Add-Type 开销


# ── 颜色分类工具 ──────────────────────────────────

def _classify_color(arr):
    """
    输入 numpy 数组 (H,W,3) RGB，返回颜色名称。
    用中心 60% 区域 + HSV 色相分档，模糊覆盖所有明暗变体。
    支持: 红/橙/黄/绿/蓝/紫/白/黑/灰
    """
    import numpy as np
    h, w = arr.shape[:2]
    # 中心 60% 采样，去掉边缘杂色
    mh, mw = int(h * 0.2), int(w * 0.2)
    center = arr[mh:h-mh, mw:w-mw]
    if center.size == 0:
        center = arr
    mean_rgb = center.mean(axis=(0, 1))

    # 亮度极值 → 白/黑
    brightness = mean_rgb.mean()
    if brightness > 225: return "白"
    if brightness < 40:  return "黑"

    # 低饱和度 → 灰
    max_c, min_c = mean_rgb.max(), mean_rgb.min()
    if max_c - min_c < 20:
        return "灰"

    # RGB→HSV 色相近似 (不依赖 cv2)
    r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return "灰"
    if mx == r:
        h = 60 * (g - b) / (mx - mn)
    elif mx == g:
        h = 60 * (b - r) / (mx - mn) + 120
    else:
        h = 60 * (r - g) / (mx - mn) + 240
    if h < 0:
        h += 360

    # 色相 → 颜色 (宽范围模糊匹配)
    if h < 20 or h >= 340:   return "红"
    if 20 <= h < 50:         return "橙"
    if 50 <= h < 80:         return "黄"
    if 80 <= h < 170:        return "绿"
    if 170 <= h < 260:       return "蓝"
    if 260 <= h < 340:       return "紫"

def _eye_search(screenshot, args: str) -> str:
    """
    统一视觉检索: color 绿 | look x,y | find 关键词
    颜色模式: GNDINO 全扫 → 颜色筛选 → VL 批量描述 → 一次性返回
    """
    args = args.strip()
    if not args:
        return "用法: color 绿 | look x1,y1,x2,y2 | find 关键词"

    w, h = screenshot.size
    now = time.time()

    # ── color 模式 ──
    if args.startswith("color "):
        target_color = args[6:].strip()
        if not target_color:
            return "请指定颜色，如: color 绿"

        eye = _get_eye()
        elements = eye.scan(screenshot)
        if not elements:
            return "未检测到任何UI元素"

        # 颜色筛选
        import numpy as np
        matched = []
        for e in elements:
            x1, y1, x2, y2 = e["bbox"]
            try:
                crop_arr = np.array(screenshot.crop((x1, y1, x2, y2)))
                color_name = _classify_color(crop_arr)
                if color_name == target_color:
                    matched.append(e)
            except Exception:
                continue

        if not matched:
            # 列出检测到的所有颜色帮助调试
            colors_found = set()
            for e in elements[:20]:
                try:
                    crop_arr = np.array(screenshot.crop(tuple(e["bbox"])))
                    colors_found.add(_classify_color(crop_arr))
                except Exception:
                    pass
            hint = f" (检测到的颜色: {', '.join(sorted(colors_found))})" if colors_found else ""
            return f"未找到{target_color}色元素{hint}"

        total = len(matched)
        print(f"[EyeSearch] {target_color}色: {total}/{len(elements)} 个匹配")

        # 逐个 VL 描述
        results = []
        VL_TIMEOUT = 20.0
        for i, e in enumerate(matched):
            if time.time() - now > VL_TIMEOUT:
                break
            x1, y1, x2, y2 = e["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            MIN_CROP = 160
            if x2 - x1 < MIN_CROP:
                hw = MIN_CROP // 2
                x1, x2 = max(0, cx - hw), min(w, cx + hw)
            if y2 - y1 < MIN_CROP:
                hh = MIN_CROP // 2
                y1, y2 = max(0, cy - hh), min(h, cy + hh)
            try:
                sub = screenshot.crop((x1, y1, x2, y2))
                b64 = pil_to_base64(sub)
                raw = _call_llamacpp_vision(b64,
                    f"截图区域({x1},{y1})到({x2},{y2})，全屏{w}x{h}。列出UI类型和可见文字。",
                    max_tokens=80)
                raw = raw.strip()
                import json as _json
                try:
                    data = _json.loads(raw)
                    desc = data.get("desc", e["label"])
                    ox = int(data.get("x", cx))
                    oy = int(data.get("y", cy))
                except Exception:
                    desc = e["label"]
                    ox, oy = cx, cy

                h_pos = "左" if ox < w // 3 else ("右" if ox > w * 2 // 3 else "中")
                v_pos = "上" if oy < h // 3 else ("下" if oy > h * 2 // 3 else "中")
                results.append(f"[{v_pos}{h_pos}]({ox},{oy}){desc}")
                print(f"[EyeSearch] {i+1}/{total} {desc} ({ox},{oy})")
            except Exception as ex:
                print(f"[EyeSearch] VL fail @ {cx},{cy}: {ex}")
                results.append(f"[?]({cx},{cy}){e['label']}")

        done = len(results)
        status = "" if done == total else f" ({done}/{total})"
        return f"{target_color}色元素{status}:\n" + "\n".join(results)

    # ── look 模式 ──
    elif args.startswith("look "):
        coords = args[5:].strip()
        try:
            parts = [int(x.strip()) for x in coords.replace(",", " ").split()]
            if len(parts) == 2:
                # 单点 → 224×224 区域
                cx, cy = parts[0], parts[1]
                hw = 112
                x1, y1, x2, y2 = max(0, cx-hw), max(0, cy-hw), min(w, cx+hw), min(h, cy+hw)
            elif len(parts) == 4:
                x1, y1, x2, y2 = parts
            else:
                return "look: 需要 x,y 或 x1,y1,x2,y2"
            sub = screenshot.crop((x1, y1, x2, y2))
            b64 = pil_to_base64(sub)
            desc = _call_llamacpp_vision(b64,
                f"截图区域({x1},{y1})到({x2},{y2})，全屏{w}x{h}。列出UI类型和所有可见文字。用中文。",
                max_tokens=200)
            return f"({x1},{y1})-({x2},{y2}): {desc.strip()}"
        except Exception as e:
            return f"look 失败: {e}"

    # ── find 模式 ──
    elif args.startswith("find "):
        query = args[5:].strip()
        if not query:
            return "find: 需要搜索关键词"
        eye = _get_eye()
        import os as _os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        screenshot.save(tmp.name, quality=85)
        try:
            _, tens = eye.load_image(tmp.name)
        finally:
            _os.unlink(tmp.name)
        boxes, logits, phrases = eye.predict(
            eye.model, tens, query, box_threshold=0.30, text_threshold=0.20)
        if len(boxes) == 0:
            return f"未找到 '{query}'"
        results = []
        for b, l, p in zip(boxes, logits, phrases):
            cx = int(b[0] * w)
            cy = int(b[1] * h)
            results.append(f"({cx},{cy}) {p} conf={float(l):.0%}")
        results.sort(key=lambda x: -float(x.split("conf=")[-1].rstrip("%")))
        return f"find '{query}':\n" + "\n".join(results[:5])

    else:
        return "未知模式。用法: color 绿 | look x,y | find 关键词"


def execute_tool(name: str, args: str) -> str:
    """执行 tool 调用，返回结果文本注入 LLM"""
    args = args.strip()

    if name == "eye_search":
        screenshot = capture_screen()
        if screenshot is None:
            return "截图失败，请稍后重试。"
        return _eye_search(screenshot, args)

    elif name == "eye_fullscan_api":
        task_desc = args.strip() if args else "仔细扫描整个屏幕，列出所有可见的窗口、按钮、图标、输入框、菜单项、通知等UI元素，并给出它们的绝对像素坐标。"
        try:
            screenshot = capture_screen()
            if screenshot is None:
                return "截图失败，无法执行全屏扫描。"
            img_b64 = pil_to_base64(screenshot)
            print(f"[CloudVL] 调用 {CLOUD_VL_MODEL}...")
            t0 = time.time()
            result_text = call_cloud_vl(img_b64, FULLSCAN_CLOUD_PROMPT.format(task=task_desc))
            elapsed = time.time() - t0
            print(f"[CloudVL] {elapsed:.1f}s")
            # 同时存到缓存
            ctx._data["last_cloud_vision"] = result_text
            return f"云端VL fullscan完成 ({elapsed:.1f}s):\n{result_text[:1500]}"
        except Exception as e:
            return f"eye_fullscan_api 失败: {e}"

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
        # 旧记忆层

    # 构建 prompt：弹幕 + 视觉上下文 + 用户消息
    prompt_parts = []

    # 0. 记忆上下文
    if user_text.strip():
        # 向量记忆 (旧引擎)
        memory_context = _load_memory_context(user_text)
        if memory_context:
            prompt_parts.append(memory_context)

    # 1. 弹幕（如果有）
    danmaku_text = ctx.get_danmaku_text()
    if danmaku_text:
        prompt_parts.append(f"[弹幕·待回复]\n{danmaku_text}")

    # 2. 视觉上下文
    vision_context = ctx.get_context_for_prompt()
    if vision_context:
        prompt_parts.append(vision_context)

    # 3. 用户消息
    if user_text.strip():
        prompt_parts.append(f"用户说：{user_text}")
    else:
        # 没有用户消息时（心跳触发），给 LLM 一个自主上下文
        offline_sec = ctx.get_offline_seconds()
        if offline_sec > 60:
            offline_min = int(offline_sec / 60)
            prompt_parts.append(f"[系统状态] 主人离线 {offline_min} 分钟。你处于自主模式。")
        prompt_parts.append("当前没有人在跟你说话。根据弹幕和视觉状态，自主决定是否行动。")

    # 如果什么上下文都没有，返回空
    if not prompt_parts:
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
        llm_output = call_ollama_chat(prompt)  # 不传 history，统一时间线 (ContextCache) 已包含所有上下文
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
        final_emotion = result["emotion"]
        if round_i > 0:
            resume_perception()
        break

    # 如果全是 tool call 没有最终回复
    if not final_reply.strip():
        resume_perception()
        return jsonify({"reply": "操作已完成", "emotion": "neutral"})

    # 清空已消耗的弹幕
    ctx.get_danmaku_and_clear()

    # 驱动皮套
    puppet_emotion(final_emotion)
    time.sleep(0.15)
    puppet_speak(final_reply)
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

    # 构建 prompt（和 /chat 逻辑一致）
    prompt_parts = []
    memory_context = _load_memory_context(user_text)
    if memory_context:
        prompt_parts.append(memory_context)
    danmaku_text = ctx.get_danmaku_text()
    if danmaku_text:
        prompt_parts.append(f"[弹幕·待回复]\n{danmaku_text}")
    vision_context = ctx.get_context_for_prompt()
    if vision_context:
        prompt_parts.append(vision_context)
    prompt_parts.append(f"用户说：{user_text}")
    prompt = "\n\n".join(prompt_parts)

    def generate():
        """SSE 生成器：支持工具调用循环"""
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
            emotion = "neutral"
            emotion_found = False
            
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
                    if not emotion_found:
                        m = re.search(r'\[emotion:(\w+)\]', full_reply)
                        if m:
                            emotion = m.group(1)
                            emotion_found = True
                            yield f"data: {json.dumps({'type': 'emotion', 'emotion': emotion}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"
                
                final_emotion = emotion
                
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
            puppet_speak(clean)
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

        yield f"data: {json.dumps({'type': 'done', 'reply': clean, 'emotion': final_emotion}, ensure_ascii=False)}\n\n"

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




def _merge_cells_to_rects(changed_cells, cell_w, cell_h, max_regions=3):
    """合并相邻变化格子为矩形区域。返回 [(x1,y1,x2,y2), ...]"""
    if not changed_cells:
        return []
    # 构建变化格子的网格坐标
    grid = set()
    for cx, cy, x1, y1, x2, y2 in changed_cells:
        gx = (x1 + x2) // (2 * cell_w)
        gy = (y1 + y2) // (2 * cell_h)
        grid.add((int(gx), int(gy)))
    if not grid:
        return []
    visited = set()
    regions = []
    for start in grid:
        if start in visited:
            continue
        stack = [start]
        comp = set()
        while stack:
            g = stack.pop()
            if g in visited or g not in grid:
                continue
            visited.add(g)
            comp.add(g)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nb = (g[0]+dx, g[1]+dy)
                if nb in grid and nb not in visited:
                    stack.append(nb)
        if comp:
            min_gx = min(g[0] for g in comp)
            max_gx = max(g[0] for g in comp)
            min_gy = min(g[1] for g in comp)
            max_gy = max(g[1] for g in comp)
            x1 = min_gx * cell_w
            y1 = min_gy * cell_h
            x2 = (max_gx + 1) * cell_w
            y2 = (max_gy + 1) * cell_h
            regions.append((x1, y1, x2, y2))
    regions.sort(key=lambda r: -(r[2]-r[0])*(r[3]-r[1]))
    return regions[:max_regions]


def _eye_scan(pil_screenshot, fast_result=None):
    """FP网格变化 → DINO+VL串行 → batch/timeout 推送"""
    global _scanning_busy
    if _scanning_busy:
        return  # 上一轮 DINO+VL 还没跑完
    _scanning_busy = True
    try:
        # GNDINO 仍跑（给 eye_find 工具用），但不驱动 VL 裁剪
        eye = _get_eye()
        elements = eye.scan(pil_screenshot)
        ctx._data["last_focus_elements"] = elements[:20]

        # VL 只跟 FastPerception 网格走
        if not fast_result:
            return
        changed_cells = fast_result.get("changed_cells", [])
        if not changed_cells:
            return

        w, h = pil_screenshot.size
        cell_w, cell_h = w // 18, h // 12
        regions = _merge_cells_to_rects(changed_cells, cell_w, cell_h)


        if not regions:
            return

        # last_grid 标记视觉已就绪（首次设置后保持 True）
        if not ctx._data.get("last_grid"):
            ctx._data["last_grid"] = True

        global _vl_processing

        if USE_LLAMACPP and not _vl_processing:
            x1, y1, x2, y2 = regions[0]

            # bbox 面积上限：超过画面 50% → 跳过
            area_ratio = (x2-x1)*(y2-y1) / (w*h)
            if area_ratio <= 0.5:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                try:
                    bw, bh = x2-x1, y2-y1
                    MIN_CROP = 224
                    if bw < MIN_CROP:
                        hw = MIN_CROP // 2
                        x1, x2 = max(0, cx-hw), min(w, cx+hw)
                    if bh < MIN_CROP:
                        hh = MIN_CROP // 2
                        y1, y2 = max(0, cy-hh), min(h, cy+hh)
                    crop = pil_screenshot.crop((x1, y1, x2, y2))

                    crop_b64 = pil_to_base64(crop)
                    _vl_processing = True

                    desc = _call_llamacpp_vision(crop_b64,
                        f"截图区域({x1},{y1})到({x2},{y2})，全屏{w}x{h}。列出UI类型和可见文字。",
                        max_tokens=100)
                    desc = desc.strip()

                    # 解析 JSON 输出
                    import json as _json
                    try:
                        data = _json.loads(desc)
                        vldesc = data.get("desc", "")
                        obj_x = int(data.get("x", cx))
                        obj_y = int(data.get("y", cy))
                    except Exception:
                        vldesc = desc
                        obj_x, obj_y = cx, cy

                    # 根据绝对坐标推断屏幕方位
                    h_pos = "左" if obj_x < w // 3 else ("右" if obj_x > w * 2 // 3 else "中")
                    v_pos = "上" if obj_y < h // 3 else ("下" if obj_y > h * 2 // 3 else "中")
                    zone = v_pos + h_pos

                    if vldesc and len(vldesc) > 1:
                        # 直接推送到 ContextCache
                        result = {
                            "natural_desc": f"[{zone}] ({obj_x},{obj_y}) {vldesc}",
                            "scene": "desktop",
                            "notable": vldesc,
                            "position": {"x": obj_x, "y": obj_y, "zone": zone},
                        }
                        ctx.update_deep_vision(result)
                        ctx._data["last_vision_update"] = now
                        ctx._data["urgent_visual"] = True
                        last_deep_time = now
                        print(f"[Eye] VL push: {zone}({obj_x},{obj_y}) {vldesc}")
                        _vl_processing = False
                except Exception as e:
                    _vl_processing = False
                    print(f"[Eye] VL 推理异常: {e}")

        # 暂存 DINO 元素（给 eye_find 工具用）
        ctx._data["_pending_dino"] = [{"label": r['label'], "bbox_2d": r['bbox'], "center": r['center'], "confidence": r['confidence']} for r in elements[:10]]
    except Exception as e:
        print(f"[Eye] ❌ {e}")
    finally:
        _scanning_busy = False  # 🔓 DINO+VL 完成，允许下一轮


@app.route("/fast", methods=["POST"])
def fast():
    """
    快速感知入口 — <100ms CPU 检测

    请求: {"image": "base64..."} 或自动截图
    响应: {"scene_changed": bool, "is_dark": bool, ...}
    """
    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image")

    if img_b64:
        img = Image.open(BytesIO(base64.b64decode(img_b64)))
    else:
        img = capture_screen()
        if img is None:
            return jsonify({"error": "screenshot_failed", "scene_changed": False}), 503

    result = fast_eye.analyze(img)
    ctx.update_fast_check(result)

    # 如果有新字幕，写入缓存
    if result.get("subtitle"):
        ctx.add_subtitle(result["subtitle"])

    return jsonify(result)


@app.route("/context", methods=["GET"])
def get_context():
    """查询当前视觉上下文"""
    return jsonify({
        "prompt_context": ctx.get_context_for_prompt(),
        "danmaku": ctx.get_danmaku_text(),
        "danmaku_count": ctx.get_danmaku_count(),
        "offline_seconds": round(ctx.get_offline_seconds(), 1),
        "stats": ctx.stats(),
        "fast_stats": fast_eye.stats,
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
        "vision": "GNDINO + cloud VL",
        "cache_stats": ctx.stats(),
        "fast_stats": fast_eye.stats,
        "danmaku_count": ctx.get_danmaku_count(),
        "offline_seconds": round(offline_sec, 1),
        "autonomous": offline_sec > HEARTBEAT_INTERVAL,
        "last_deep_vision_ago": round(time.time() - last_deep_time, 1) if last_deep_time else None,
        "vl_processing": _vl_processing,
        "scanning_busy": _scanning_busy,
    })


@app.route("/reset_vision", methods=["POST"])
def reset_vision():
    """一键重启视觉系统：解锁死锁、重置状态、强制下一轮立即触发 VL"""
    global _vl_processing, _scanning_busy, last_deep_time

    _vl_processing = False
    _scanning_busy = False
    last_deep_time = 0.0

    fast_eye.reset()
    ctx._data["last_grid"] = None
    ctx._data.pop("last_vision_update", None)

    print("[Vision] 视觉系统已手动重置（/reset_vision）")
    return jsonify({"ok": True, "msg": "视觉系统已重置：锁已释放、FP基线已清、下次循环立即触发VL"})


@app.route("/action", methods=["POST"])
def action():
    """直接操控皮套（调试用）"""
    data = request.get_json(force=True)
    action_type = data.get("type", "")

    if action_type == "emotion":
        puppet_emotion(data.get("emotion", "neutral"))
    elif action_type == "speak":
        puppet_speak(data.get("text", ""), data.get("duration_ms"))
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
      1. 视觉变化 (urgent_visual flag) — 画面有动静
      2. 弹幕队列非空 AND 用户离线
      3. 首次扫描完成 — 让 mini_coin 描述初始画面
      4. 定期视觉检查 (每30s) — 有视觉上下文 + 用户离线时主动评论
    """
    print(f"[Heartbeat] 心跳循环已启动 (interval={HEARTBEAT_INTERVAL}s)")

    last_visual_trigger = 0  # 防止重复触发
    last_periodic_trigger = 0
    first_scan_triggered = False
    PERIODIC_INTERVAL = 30.0
    VISUAL_COOLDOWN = 15.0   # 视觉触发后 15s 内不再重复触发
    VISION_MAX_AGE = 300.0   # 视觉数据超过 5 分钟视为过期
    SPEAK_COOLDOWN = 30.0    # 说话后30秒内心跳不触发
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

            # 视觉触发：重大变化立即触发（最小2s CD）
            visual_urgent = ctx._data.get("urgent_visual", False)
            should_visual = visual_urgent and time.time() - last_visual_trigger > 2.0
            if should_visual:
                ctx._data["urgent_visual"] = False
                last_visual_trigger = time.time()

            has_vision = ctx._data.get("last_grid") is not None
            last_update = ctx._data.get("last_vision_update", 0)
            vision_age = time.time() - last_update if last_update else 999
            if has_vision and vision_age > VISION_MAX_AGE:
                has_vision = False

            should_trigger = False
            trigger_reason = ""

            # 首次扫描 → 总是触发（描述初始画面）
            if has_vision and not first_scan_triggered:
                first_scan_triggered = True
                should_trigger = True
                trigger_reason = "first_scan"
            # 视觉重大变化 → 触发
            elif should_visual and time.time() - last_periodic_trigger > VISUAL_COOLDOWN:
                should_trigger = True
                trigger_reason = "visual_change"
                last_periodic_trigger = time.time()
            # 弹幕 → 触发
            elif danmaku_count > 0:
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

            if trigger_reason == "visual_change":
                prompt_parts.append("[系统状态] 画面上出现了变化。你根据上下文判断：创造者在不在电脑前？这变化值得说吗？")
            elif trigger_reason == "silence_timeout":
                prompt_parts.append("[系统状态] 你已经超过1分钟没说话了。保持时间感——随便说点什么都行，一句就好。也可以只发个时间戳。")
            
            danmaku_text = ctx.get_danmaku_text()

            # 0. 记忆上下文
            if danmaku_text:
                memory_context = _load_memory_context(danmaku_text)
                if memory_context:
                    prompt_parts.append(memory_context)

            # 1. 弹幕（如果有）
            if danmaku_text:
                prompt_parts.append(f"[弹幕·待回复]\n{danmaku_text}")

            vision_context = ctx.get_context_for_prompt()
            if vision_context:
                prompt_parts.append(vision_context)

            # 给 LLM 足够的决策信息
            offline_sec = ctx.get_offline_seconds()
            silence_sec = ctx.get_last_self_utterance_age()
            
            status_parts = []
            status_parts.append(f"创造者上次发言: {offline_sec:.0f}s前" if offline_sec < 60 else f"创造者 {int(offline_sec/60)} 分钟没说话了")
            if silence_sec is not None:
                status_parts.append(f"你上次发言: {silence_sec:.0f}s前")
            status_parts.append("决定原则: 根据时间线和画面判断创造者在不在电脑前、值不值得说话。")
            status_parts.append("如果你不确定该不该开口 → 沉默。TLDR: when in doubt, shut up。")
            prompt_parts.append("[决策信息] " + " | ".join(status_parts))

            prompt = "\n\n".join(prompt_parts)

            # DEBUG: 记录发送给 LLM 的视觉上下文
            if vision_context:
                print(f"[Heartbeat→LLM] 视觉上下文 ({len(vision_context)}字): {vision_context[:300]}")

            # 记录本轮 prompt 快照（调试回溯）
            ctx.record_prompt("heartbeat: " + "\n".join(prompt_parts[-3:]) if len(prompt_parts) > 3 else "\n".join(prompt_parts), trigger=trigger_reason)

            # 调用 LLM
            llm_output = call_ollama_chat(prompt, [])
            result = parse_emotion(llm_output)

            # 清空已消耗的弹幕
            ctx.get_danmaku_and_clear()

            if not result["reply"].strip():
                print("[Heartbeat] LLM 跳过（无回复）")
                continue

            # 过滤工具调用（心跳不处理工具，直接传给用户会乱）
            import re as _re
            clean_reply = _re.sub(r'\[TOOL:[^\]]*\].*?\[/TOOL\]', '', result["reply"], flags=_re.DOTALL).strip()

            # 驱动皮套
            puppet_emotion(result["emotion"])
            time.sleep(0.15)
            puppet_speak(clean_reply)
            ctx.add_self_utterance(clean_reply)

            # 异步归档记忆 — 弹幕互动也是记忆来源
            if danmaku_text:
                summary = result.get("remember_summary")
                threading.Thread(
                    target=_archive_memory,
                    args=(danmaku_text, result["reply"], vision_context if vision_context else "", summary),
                    daemon=True,
                ).start()

            # 记忆维护 (压缩+提取, 非阻塞)

            print(f"[Heartbeat] ✨ {clean_reply[:60]}...")

        except Exception as e:
            import traceback
            print(f"[Heartbeat] 错误: {e}")


# ── 感知主循环（后台线程）─────────────────────────

def perception_loop():
    """
    后台循环：每秒截屏 → FastPerception → 有变化时触发深度分析 + 即时皮套反应
    
    不会阻塞 /chat 对话，运行在独立线程中。
    """
    global last_deep_time

    print("[Perception] 感知循环已启动 (interval=1s)")
    
    # 启动时预加载视觉模型（避免首次调用时延迟）
    if VISION_ENABLED:
        try:
            _get_eye()
            print("[EyeSystem] GNDINO 预加载完成")
        except Exception as e:
            print(f"[EyeSystem] 预加载失败: {e}")
    
    round_count = 0
    consecutive_screen_fails = 0
    MAX_SCREEN_FAILS_BEFORE_WARN = 10

    while True:
        if not _perception_running:
            print("[Perception] 收到停止信号，退出循环")
            break
        try:
            time.sleep(PERCEPTION_INTERVAL)
            round_count += 1

            # 1. 截图（加固版：失败返回 None 而非抛异常）
            screenshot = capture_screen()
            if screenshot is None:
                consecutive_screen_fails += 1
                if consecutive_screen_fails == 1:
                    print("[Perception] 截图失败，跳过本轮（视觉数据保持上次有效值）")
                elif consecutive_screen_fails >= MAX_SCREEN_FAILS_BEFORE_WARN:
                    print(f"[Perception] ⚠️ 连续 {consecutive_screen_fails} 次截图失败，视觉数据可能已过期")
                continue  # 跳过本轮，不更新任何视觉状态

            consecutive_screen_fails = 0  # 截图成功，重置计数

            # 2. Fast 检测
            fast_result = fast_eye.analyze(screenshot)
            ctx.update_fast_check(fast_result)

            # 3. 字幕检测 → 写入缓存
            if fast_result.get("subtitle"):
                ctx.add_subtitle(fast_result["subtitle"])

            # 4. 即时皮套反应（不经过 LLM！）
            _trigger_instant_reactions(fast_result)

            # 5. 判断是否需要触发 VL 深度分析
            now = time.time()
            time_since_deep = now - last_deep_time

            should_deep = False
            if fast_result["scene_changed"] and time_since_deep > DEEP_VISION_INTERVAL:
                should_deep = True
            elif time_since_deep > DEEP_VISION_MAX_INTERVAL:
                should_deep = True  # 太久没分析，强制刷新

            if should_deep and VISION_ENABLED:
                if perception_paused.is_set():
                    print("[Perception] auto-scan 暂停 (tool calling)，跳过")
                else:
                    _eye_scan(screenshot, fast_result)

        except Exception as e:
            print(f"[Perception] 循环错误 (round {round_count}): {e}")
            # 非截图类异常不应该无限循环，短暂休眠避免 CPU 100%
            time.sleep(1)


def _trigger_instant_reactions(fast_result: dict):
    """
    根据 FastPerception 结果触发即时皮套反应。
    优先级: 低（仅在 idle 时表现），不干扰对话。
    """
    triggers = []

    if fast_result.get("scene_changed"):
        triggers.append("scene_change")

    if fast_result.get("is_red"):
        triggers.append("red_tone")

    if fast_result.get("is_dark"):
        triggers.append("dark")

    if not triggers:
        return

    # 通过 Bridge 广播感知事件（新版支持）或回退到直接调 emotion
    try:
        puppet_perception_event("perception_trigger", {
            "triggers": triggers,
            "is_dark": fast_result.get("is_dark", False),
            "is_red": fast_result.get("is_red", False),
        })
    except Exception:
        pass

    # 回退：直接调皮套表情（旧版 bridge 兼容）
    if "scene_change" in triggers or "red_tone" in triggers:
        puppet_emotion("surprise")
        # 前倾参数（表示关注）
        puppet_params({
            "ParamBodyAngleX": -10,
            "ParamEyeLOpen": 1.0,
            "ParamEyeROpen": 1.0,
        })
    elif "dark" in triggers:
        puppet_emotion("think")
        puppet_params({
            "ParamBodyAngleX": 8,  # 前倾
        })


# ── 记忆自动处理 ──────────────────────────────────

def _auto_process_memory():
    """启动时检查并处理未完成的每日 JSONL"""
    try:
        memory_dir = os.path.join(_PROJ_ROOT, "data", "memory")
        for fname in sorted(os.listdir(memory_dir)):
            if not fname.endswith(".jsonl") or fname.startswith("wechat_"):
                continue
            date = fname.replace(".jsonl", "")
            summary_file = os.path.join(memory_dir, f"{date}_summary.json")
            if os.path.exists(summary_file):
                continue  # 已处理
            # 只处理昨天的（今天的还在收集中）
            yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
            if date != yesterday:
                continue
            print(f"[Memory] 自动处理 {date} ({os.path.getsize(os.path.join(memory_dir, fname))} bytes)")
            try:
                from memory.api import process_date
                process_date(date)
            except Exception as e:
                print(f"[Memory] 自动处理失败: {e}")
    except Exception as e:
        print(f"[Memory] 自动处理异常: {e}")

# ── 启动 ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(" mini_coin v10.6 — 本地 STT (faster-whisper)")
    print("=" * 55)
    print(f" LLM:   {LLM_MODEL} @ {OLLAMA_URL}")
    print(f" 视觉:   GNDINO 九宫格 + auto-focus + 云端 VL fullscan")
    print(f" STT:   faster-whisper small (CPU int8)")
    print(f" 皮套:   {BRIDGE_URL}")
    print(f" 本服:   http://localhost:{V10_PORT}")
    print()
    print(" 端点:")
    print(f"   POST /chat        — 对话 (<500ms)")
    print(f"   POST /chat_stream — 流式对话 (SSE)")
    print(f"   POST /stt         — 语音转文字 (本地 CPU)")
    print(f"   POST /vision      — 异步视觉 (202)")
    print(f"   POST /fast        — 快速感知 (<100ms)")
    print(f"   POST /danmaku     — 弹幕推送")
    print(f"   GET  /context     — 视觉上下文")
    print(f"   GET  /status      — 运行状态")
    print(f"   POST /action      — 皮套调试")
    print()
    print(f" 后台线程:")
    print(f"   perception_loop — Fast感知 ({PERCEPTION_INTERVAL}s)")
    print(f"   heartbeat_loop  — 自主决策 ({HEARTBEAT_INTERVAL}s)")
    print("=" * 55)

    # 自动处理未完成的每日记忆
    threading.Thread(target=_auto_process_memory, daemon=True).start()

    # 启动感知循环
    perception_thread = threading.Thread(target=perception_loop, daemon=True)
    perception_thread.start()

    # 启动自主心跳循环
    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    # 启动 Flask
    app.run(host="127.0.0.1", port=V10_PORT, debug=False, threaded=True)
