"""
vad_bridge.py — 最小版本: HTTP /push → WebSocket 广播
端口: 18769 (前端HTML + WS), 18770 (API)
"""
import asyncio, json, sys, os, pathlib, mimetypes, urllib.parse
from websockets.http11 import Response, Headers

FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend"

ws_clients = set()


async def _serve_static(conn, request):
    """返回静态文件, WebSocket升级请求返回None"""
    if request.headers.get("upgrade", "").lower() == "websocket":
        return None
    path = urllib.parse.unquote(request.path)
    if path == '/':
        path = '/puppet_live2d_v2.html'
    filepath = (FRONTEND_DIR / path.lstrip('/')).resolve()
    if str(filepath).startswith(str(FRONTEND_DIR.resolve())) and filepath.is_file():
        mime, _ = mimetypes.guess_type(str(filepath))
        body = filepath.read_bytes()
        h = Headers([('Content-Type', mime or 'application/octet-stream'),
                     ('Content-Length', str(len(body)))])
        return Response(200, "OK", h, body)
    return Response(404, "Not Found", Headers(), b"")


async def ws_handler(websocket):
    ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass  # 只收不发
    finally:
        ws_clients.discard(websocket)


async def _broadcast(payload: dict):
    if not ws_clients:
        return
    msg = json.dumps(payload, ensure_ascii=False)
    await asyncio.gather(
        *(c.send(msg) for c in ws_clients),
        return_exceptions=True,
    )


async def handle_http(reader, writer):
    """HTTP API: POST /push"""
    try:
        data = await reader.read(65536)
        raw = data.decode('utf-8', errors='replace')
        lines = raw.split('\r\n')
        first = lines[0] if lines else ''
        body_start = raw.find('\r\n\r\n')
        body = raw[body_start + 4:] if body_start > 0 else ''

        if first.startswith('POST /push'):
            payload = json.loads(body) if body else {}
            print(f"[BRIDGE] 收到: {json.dumps(payload, ensure_ascii=False)}")
            await _broadcast(payload)
            resp = json.dumps({"ok": True}).encode()
        elif first.startswith('GET /status'):
            resp = json.dumps({
                "ok": True, "clients": len(ws_clients)
            }).encode()
        else:
            resp = json.dumps({"error": "not found"}).encode()

        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Access-Control-Allow-Origin: *\r\n"
            b"Content-Length: " + str(len(resp)).encode() + b"\r\n\r\n" + resp
        )
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()


async def main():
    port = 18769

    from websockets.asyncio.server import serve as ws_serve

    print(f"╔══════════════════════════════════════╗")
    print(f"║  VAD Bridge — 最小版本              ║")
    print(f"║  http://localhost:{port}   (前端+WS)  ║")
    print(f"║  API → POST http://localhost:18770/push ║")
    print(f"╚══════════════════════════════════════╝")

    async with ws_serve(ws_handler, "0.0.0.0", port,
                        process_request=_serve_static) as ws_server:
        print(f"[服务] 已启动: http://0.0.0.0:{port}")
        api_server = await asyncio.start_server(handle_http, "0.0.0.0", 18770)
        print(f"[API] 已启动: http://0.0.0.0:18770")
        async with api_server:
            await api_server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
