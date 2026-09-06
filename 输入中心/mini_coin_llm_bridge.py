"""
mini_coin_llm_bridge.py — 透明代理 + ASR 文本接收
- POST /completion → 转发模型（本地 llama.cpp 或 DeepSeek API，2026-08-12 新增 API 模式）
- POST /text → 接收 ASR 文本（存到 source_last）+ 转发模型
- GET  / → 监控页面
- GET  /source_last → 返回各源最新文本
"""
import json
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer
import requests

# ── 模型后端配置 (2026-08-12) ──
BACKEND = "api"                       # 当前后端: "api"=DeepSeek / "local"=本地 llama.cpp（运行时可用 POST /api/backend 切换）
LOCAL_API = "http://localhost:8080/completion"   # 本地 llama-server
API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "sk-a7ceee1e88444f799a2f857b323bf4a3"
API_MODEL = "deepseek-chat"           # DeepSeek V4 Flash（如模型名不同改这里）
API_TIMEOUT = 55                      # 略小于管线 timeout=60，让桥先返回空 → 管线判空丢轮

_last_prompt = "（暂无）"
_last_reply = "（暂无）"
source_last = {}  # {source_name: last_text}


def _call_model(payload, raw_body):
    """转发模型：api 模式把 llama.cpp payload 转 OpenAI 格式；local 模式原样转发。
    返回 (reply_str, err_msg)；API 失败返回空串 + 错误信息（管线判空丢轮，不中断）"""
    if BACKEND == "api":
        prompt = payload.get("prompt", "") or ""
        n_predict = int(payload.get("n_predict", 500) or 500)
        temperature = payload.get("temperature", 0.9)
        api_payload = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": temperature,
            "stream": False,
        }
        # grammar/logit_bias 是 llama.cpp 专属，API 不支持 → 丢弃（格式约束靠管线 _sanitize_reply 兜底）
        try:
            r = requests.post(API_URL, json=api_payload,
                              headers={"Authorization": f"Bearer {API_KEY}",
                                       "Content-Type": "application/json"},
                              timeout=API_TIMEOUT)
            data = r.json()
            reply = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            if not reply:
                err = data.get("error", {}).get("message", "空回复") if isinstance(data, dict) else str(data)
                return "", f"[API错误] {err}"
            return reply, ""
        except Exception as e:
            return "", f"[API错误] {e}"
    else:
        # 本地 llama.cpp /completion：原样转发
        r = requests.post(LOCAL_API, data=raw_body,
                          headers={"Content-Type": "application/json"}, timeout=60)
        data = r.json()
        return data.get("content", ""), ""


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global BACKEND
        if self.path == "/source_last":
            self._json(200, source_last)
            return
        if self.path == "/api/backend":   # 08-12: 查询当前后端模式
            self._json(200, {"backend": BACKEND, "model": API_MODEL if BACKEND == "api" else "local-llama"})
            return
        # 监控页面
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="2"><title>LLM桥</title>
<style>body{{background:#111;color:#eee;font-family:monospace;padding:20px}}
.box{{background:#1a1a2e;padding:15px;margin:10px 0;border-radius:8px}}
pre{{white-space:pre-wrap;word-break:break-all}}
.in{{border-left:3px solid #4fc3f7}} .out{{border-left:3px solid #81c784}}
.lb{{color:#888;font-size:12px}}
</style></head><body>
<h2>LLM桥</h2>
<div class="box in"><div class="lb">▶ 输入</div><pre style="color:#4fc3f7">{_last_prompt}</pre></div>
<div class="box out"><div class="lb">◀ 输出</div><pre style="color:#81c784">{_last_reply}</pre></div>
</body></html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global _last_prompt, _last_reply, source_last, BACKEND
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            parsed = json.loads(body)

            # POST /api/backend → 切换后端模式 (08-12: api/local 运行时切换，不用改代码重启)
            if self.path == "/api/backend":
                mode = str(parsed.get("mode", "") or "").strip().lower()
                if mode in ("api", "local"):
                    BACKEND = mode
                    print(f"[桥] 后端切换: {mode}")
                self._json(200, {"ok": True, "backend": BACKEND,
                                 "model": API_MODEL if BACKEND == "api" else "local-llama"})
                return

            # POST /text → ASR 文本（存储 + 转发）
            if self.path == "/text":
                text = parsed.get("text", "")
                source = parsed.get("source", "") or "拉姆"
                if text:
                    source_last[source] = text[:80]
                self._json(200, {"ok": True, "source_last": source_last})
                return

            # POST / → 转发模型（本地/API 由 _call_model 分流）
            prompt_raw = parsed.get("prompt", "") or ""
            _last_prompt = prompt_raw[:2000]
            reply, err = _call_model(parsed, body)
            if err:
                _last_reply = err[:2000]
            else:
                _last_reply = reply[:2000]
            self._json(200, {"content": reply})
        except Exception as e:
            _last_reply = f"[错误] {e}"
            self._json(500, {"error": str(e)})

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    log_message = lambda *a: None


if __name__ == "__main__":
    print(f"LLM 桥 → http://localhost:18771 (后端: {'DeepSeek API ' + API_MODEL if BACKEND == 'api' else LOCAL_API}，POST /api/backend 可切换)")
    ThreadingTCPServer.allow_reuse_address = True
    server = ThreadingTCPServer(("0.0.0.0", 18771), ProxyHandler)
    server.serve_forever()
