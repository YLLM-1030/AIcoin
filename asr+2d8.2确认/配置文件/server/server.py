"""
mini_coin Live2D Bridge Server v3.1
v10 扩展: 新增 /perception_event 和 /broadcast 端点支持感知事件广播
"""
import json, math, os, time, asyncio, threading, queue
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import websockets

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))
WS_CLIENTS = set()
MSG_QUEUE = queue.Queue()  # 线程安全消息队列

EMOTION_MAP = {
    "neutral":  {"ParamAngleX":0,"ParamAngleY":0,"ParamAngleZ":0,"ParamBodyAngleX":0,"ParamBodyAngleY":0,"ParamBodyAngleZ":0,"ParamMouthSmile":0,"ParamMouthOpenY":0,"ParamEyeLOpen":1,"ParamEyeROpen":1,"ParamBrowLY":0,"ParamBrowRY":0,"ParamBrowLAngle":0,"ParamBrowRAngle":0,"ParamCheek":0,"ParamEyeBallX":0,"ParamEyeBallY":0,"Param11":0,"Param12":0,"Param23":0},
    "angry":    {"ParamAngleZ":-15,"ParamBrowLY":0,"ParamBrowRY":0,"ParamBrowLAngle":-1,"ParamBrowRAngle":1,"ParamEyeLOpen":0.6,"ParamEyeROpen":0.6,"ParamBodyAngleX":-5,"ParamMouthForm":0.4,"ParamMouthOpenY":0},
    "surprise": {"ParamEyeLOpen":1,"ParamEyeROpen":1,"ParamMouthOpenY":0.9,"ParamBrowLY":1,"ParamBrowRY":1,"ParamAngleY":-3},
    "happy":    {"ParamCheek":0.5,"ParamEyeLSmile":1,"ParamEyeRSmile":1,"ParamEyeLOpen":0.3,"ParamEyeROpen":0.3,"ParamMouthOpenY":0.2},
    "think":    {"ParamAngleZ":12,"ParamBrowLY":0.4,"ParamBrowRY":0,"ParamMouthOpenY":0},
    "panic":    {"Param11":1,"Param12":1,"ParamMouthOpenY":0.4,"ParamEyeLOpen":0.9,"ParamEyeROpen":0.9,"ParamBrowLY":0.8},
    "shy":      {"ParamCheek":0.6,"ParamAngleZ":8,"ParamEyeLOpen":0.4,"ParamEyeROpen":0.4,"ParamBrowLY":0.2,"ParamMouthOpenY":0},
    "obey":     {"Param23":1,"ParamCheek":0.3,"ParamEyeLOpen":0.7,"ParamEyeROpen":0.7,"ParamBrowLY":0.3,"ParamAngleZ":-5},
}

class BridgeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)
    def log_message(self, f, *a): pass

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length)
        body = json.loads(raw.decode('utf-8')) if length > 0 else {}
        resp = {"ok": True}

        if path == '/params':
            MSG_QUEUE.put({"type":"params","source":"user","params":body})
        elif path == '/speak':
            MSG_QUEUE.put({
                "type":"speak",
                "text":body.get("text",""),
                "duration":body.get("duration_ms",3000),
                "audio_url":body.get("audio_url",""),  # v11: Edge TTS mp3
            })
            resp["text"] = body.get("text","")[:30]
        elif path == '/emotion':
            e = body.get("emotion","neutral")
            if e in EMOTION_MAP:
                MSG_QUEUE.put({"type":"params","source":"user","params":EMOTION_MAP[e]})
            # 同时广播 emotion 给 3D 自主神经使用
            MSG_QUEUE.put({"type":"emotion","emotion":e})
        elif path == '/puppet_action':
            MSG_QUEUE.put({"type":"puppet_action","action":body.get("action","")})
            resp["action"] = body.get("action","")
        elif path == '/reset':
            MSG_QUEUE.put({"type":"params","source":"user","params":{
                "ParamBodyAngleX":0,"ParamBodyAngleY":0,"ParamBodyAngleZ":0,
                "ParamMouthSmile":0,"ParamMouthOpenY":0,
                "ParamEyeLOpen":1,"ParamEyeROpen":1,
                "ParamBrowLY":0,"ParamBrowRY":0,"ParamCheek":0,
            }})
        elif path == '/perception_event':
            # v10: 接收感知事件并广播到所有 WebSocket 客户端
            event_type = body.get("type","unknown")
            event_data = body.get("data",{})
            MSG_QUEUE.put({"type":"perception_event","event":event_type,"data":event_data})
            resp["event"] = event_type
        elif path == '/broadcast':
            # v10: 通用广播端点，直接推消息到 WebSocket
            MSG_QUEUE.put(body)
            resp["msg_type"] = body.get("type","unknown")
        elif path == '/tts_done':
            # v11: 前端播放完毕，清理 tts_cache 文件
            url = raw.decode('utf-8') if raw else ""
            filename = os.path.basename(url)
            filepath = os.path.join(ROOT, "tts_cache", filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            resp["deleted"] = filename
        else:
            resp = {"error":"unknown"}

        data = json.dumps(resp, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse(self.path).path == '/status':
            data = json.dumps({"connected": len(WS_CLIENTS) > 0}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

async def ws_handler(websocket):
    WS_CLIENTS.add(websocket)
    try:
        async for _ in websocket: pass
    finally:
        WS_CLIENTS.discard(websocket)

async def broadcast(msg):
    payload = json.dumps(msg, ensure_ascii=False)
    dead = set()
    for ws in WS_CLIENTS:
        try: await ws.send(payload)
        except Exception: dead.add(ws)
    WS_CLIENTS.difference_update(dead)

async def process_queue():
    """从线程安全队列取出消息并广播"""
    while True:
        try:
            while True:
                msg = MSG_QUEUE.get_nowait()
                await broadcast(msg)
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)

async def idle_sway():
    while True:
        if WS_CLIENTS:
            t = time.time()
            await broadcast({
                "type":"params","source":"idle",
                "params":{
                    "ParamBodyAngleX":5.0*math.sin(t*0.8),
                    "ParamBodyAngleY":3.0*math.sin(t*0.6+1.0),
                    "ParamBodyAngleZ":1.5*math.sin(t*0.4+2.0),
                }
            })
        await asyncio.sleep(0.05)

async def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", 8090), BridgeHandler)
    httpd.allow_reuse_address = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print("HTTP: http://localhost:8090")
    print("  POST /params /speak /emotion /reset")
    print("  POST /perception_event /broadcast (v10)")
    print("  GET  /status")

    asyncio.create_task(idle_sway())
    asyncio.create_task(process_queue())

    async with websockets.serve(ws_handler, "127.0.0.1", 19767, reuse_address=True):
        print("WS: ws://localhost:19767")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
