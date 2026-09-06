"""
dashboard_server.py — 迷你硬币 输入中心 后端
WebSocket 端口 19767（接收 ASR/弹幕推送）
HTTP 端口 8091（前端页面 + API 代理）
"""
import asyncio, json, os, requests
import websockets
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

LLAMA_API = "http://localhost:8080/v1/chat/completions"
WS_PORT = 19767
HTTP_PORT = 8091
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── WebSocket：推送 ASR/弹幕 到前端 ──
connected = set()

async def ws_handler(ws):
    connected.add(ws)
    try:
        async for msg in ws:
            data = json.loads(msg)
            # 后端也可以接受推送（例如从 ASR 模块转发）
            pass
    finally:
        connected.remove(ws)

async def broadcast(msg):
    if connected:
        await asyncio.gather(*[ws.send(json.dumps(msg, ensure_ascii=False)) for ws in connected])

# ── HTTP：前端 + API 代理 ──
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "":
            self.path = "/dashboard.html"
        elif path.startswith("/api/"):
            self.send_error(404)
            return
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/chat":
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            asyncio.run_coroutine_threadsafe(broadcast({"type": "speak", "text": text}), loop)
            self.send_json({"ok": True})
        elif path == "/api/asr":
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            asyncio.run_coroutine_threadsafe(broadcast({"type": "asr", "text": text}), loop)
            self.send_json({"ok": True})
        elif path == "/api/dm":
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")
            asyncio.run_coroutine_threadsafe(broadcast({"type": "dm", "text": text}), loop)
            self.send_json({"ok": True})
        else:
            self.send_error(404)

    def send_json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # 安静

# ── 启动 ──
async def main():
    global loop
    loop = asyncio.get_running_loop()

    # HTTP
    os.chdir(SCRIPT_DIR)
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[*] 仪表盘: http://localhost:{HTTP_PORT}")

    # WebSocket
    async with websockets.serve(ws_handler, "0.0.0.0", WS_PORT):
        print(f"[*] WebSocket: ws://localhost:{WS_PORT}")

        # 保持运行
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
