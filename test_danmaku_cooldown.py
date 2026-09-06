# -*- coding: utf-8 -*-
"""弹幕发射冷却 + 待发射队列 综合测试
- 临时 18008 输出桥接收器（记录收到的发射内容+时间戳）
- 后台起 danmaku_server.py (18776)
- 注入弹幕 → 验证：秒回入队 / 冷却间隔 / 冷却实时可配 / 手动flush / 队列空=就绪
用法: python test_danmaku_cooldown.py
"""
import json, subprocess, sys, time, threading, os, urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ── 临时输出桥 18008 ──
received = []   # (ts, body_text)
class Recv(BaseHTTPRequestHandler):
    def do_POST(self):
        ln = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(ln).decode('utf-8', 'replace')
        received.append((time.time(), body))
        print(f"[18008桥] 收到: {body}")
        self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers()
        self.wfile.write(b'ok')
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers()
        self.wfile.write(b'ok')
    log_message = lambda *a: None

srv = HTTPServer(('127.0.0.1', 18008), Recv)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("[setup] 临时输出桥 18008 就绪")

# ── 启动 danmaku_server ──
dm = subprocess.Popen([PY, os.path.join(BASE, "输入中心/弹幕接收/danmaku_server.py")],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(2.5)

def post(path, data):
    req = urllib.request.Request(f"http://localhost:18776{path}",
        data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read().decode())

def get_state():
    """连 /events 读第一帧 state（SSE 长连接，读一帧就断）"""
    try:
        with urllib.request.urlopen("http://localhost:18776/events", timeout=2) as r:
            data = r.read(4096).decode('utf-8', 'replace')
        for line in data.split('\n'):
            if line.startswith('data: '):
                j = json.loads(line[6:])
                if 'pending_fire' in j:
                    return j
    except Exception as e:
        print(f"[state读取失败] {e}")
    return None

def inject(text, user='观众'):
    return post('/api/test', {'text': text, 'user': user})

# 就绪探测
for _ in range(20):
    try:
        post('/api/config', {})
        break
    except Exception:
        time.sleep(0.5)

print("=" * 60)
print("① 连续注入 2 条秒回弹幕（间隔0.5s < 冷却5s）")
print("=" * 60)
inject("主播你觉得我今天运气怎么样？")
time.sleep(0.5)
inject("主播你吃饭了吗？")
time.sleep(1.0)
st = get_state()
if st:
    print(f"   pending_fire 组数: {len(st['pending_fire'])} (期望 2)")
    for g in st['pending_fire']:
        print(f"    组: {[x['text'] for x in g]}")
    print(f"   fire_remaining: {st.get('fire_remaining', '?')}")
else:
    print("   ⚠ 无法读取状态")

print("\n等待 6s 让冷却期发射...")
time.sleep(6.0)
print(f"   18008 桥收到 {len(received)} 次发射 (期望 2)")
if received:
    for ts, b in received:
        print(f"     +{ts - received[0][0]:.1f}s  {b}")

print("=" * 60)
print("② 改冷却为 2s → 再注入 2 条秒回")
print("=" * 60)
post('/api/config', {'cooldown': 2})
cfg = post('/api/config', {})
print(f"   当前 cooldown = {cfg.get('cooldown')} (期望 2, 无需重启)")
received.clear()
inject("主播可以唱首歌吗？")
time.sleep(0.5)
inject("主播你平时喜欢什么？")
time.sleep(1.0)
print("   等待 3s...")
time.sleep(3.0)
print(f"   18008 桥收到 {len(received)} 次发射 (期望 2)")
if received:
    for ts, b in received:
        print(f"     +{ts - received[0][0]:.1f}s  {b}")

print("=" * 60)
print("③ 进弹夹（0.11~0.5）→ 手动 flush → 入队冷却发射")
print("=" * 60)
inject("主播今天天气真不错")   # ≈0.21 进 saved
inject("主播你看起来心情很好")  # ≈0.21 进 saved
time.sleep(0.5)
r = post('/api/flush', {})
print(f"   /api/flush → {r}  (期望 count=2 queued=1)")
time.sleep(1.0)
st = get_state()
if st:
    print(f"   pending_fire 组数: {len(st['pending_fire'])} (期望 1)")
print("   等待 3s 发射...")
time.sleep(3.0)
print(f"   18008 桥累计收到 {len(received)} 次发射")

print("=" * 60)
print("④ 队列空 → 就绪")
print("=" * 60)
time.sleep(3.0)
st = get_state()
if st:
    print(f"   pending_fire 组数: {len(st['pending_fire'])} (期望 0 = 就绪)")

print("=" * 60)
print("⑤ 低分弹幕 → 丢弃验证")
print("=" * 60)
inject("哈哈哈哈")
time.sleep(0.5)
st = get_state()
if st:
    print(f"   discarded 条数: {len(st['discarded'])} (期望 ≥1)")
    print(f"   pending_fire 组数: {len(st['pending_fire'])} (期望 0, 垃圾不发射)")

print("\n[清理] 关闭 danmaku_server")
dm.terminate()
srv.shutdown()
print("测试完成")
