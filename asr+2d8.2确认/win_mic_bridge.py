"""
win_mic_bridge.py — Windows 音频桥（供 WSL ASR 选择 Windows 麦克风）
运行在 Windows 侧（Windows Python + sounddevice），监听 0.0.0.0:18690。

端点：
  GET /api/devices     → Windows 录音设备列表 [{id,name,is_default}]
  GET /api/status      → 当前活动捕获会话 {dev_id: 引用数}
  GET /stream?dev=<id> → 持续 PCM 流 (16000Hz mono s16le)，供 WSL ASR 消费

依赖：Windows Python + `pip install sounddevice`
启动：python win_mic_bridge.py
"""
import json, threading, urllib.parse, time, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BYTES_PER_CHUNK = 512 * 2   # 与 WSL ASR 对齐 (512 帧 × 2 字节)

_captures = {}   # dev_id -> {'stream','buf','lock','refs'}
_cap_lock = threading.Lock()


def _devices():
    """枚举 Windows 录音设备"""
    out = []
    try:
        devs = sd.query_devices()
        default_in = None
        try:
            default_in = sd.default.device[0]
        except Exception:
            pass
        for i, d in enumerate(devs):
            if d.get('max_input_channels', 0) > 0:
                out.append({'id': i, 'name': d.get('name', ''),
                            'is_default': (i == default_in)})
    except Exception as e:
        out.append({'id': -1, 'name': f'(枚举失败: {e})', 'is_default': False})
    return out


def _get_capture(dev_id):
    """获取指定设备的捕获会话（按需创建，引用计数）"""
    with _cap_lock:
        if dev_id not in _captures:
            buf = bytearray()
            lock = threading.Lock()

            def cb(indata, frames, t, status):
                raw = indata.tobytes()
                with lock:
                    buf.extend(raw)

            try:
                st = sd.InputStream(device=dev_id, samplerate=SAMPLE_RATE,
                                    channels=CHANNELS, dtype=DTYPE, callback=cb)
                st.start()
            except Exception as e:
                raise RuntimeError(f"设备 {dev_id} 打开失败: {e}")
            _captures[dev_id] = {'stream': st, 'buf': buf, 'lock': lock, 'refs': 0}
        c = _captures[dev_id]
        c['refs'] += 1
        return c


def _release(dev_id):
    with _cap_lock:
        c = _captures.get(dev_id)
        if not c:
            return
        c['refs'] -= 1
        if c['refs'] <= 0:
            try:
                c['stream'].stop()
                c['stream'].close()
            except Exception:
                pass
            del _captures[dev_id]


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        if self.path == '/api/devices':
            self._json(200, {'devices': _devices()})
        elif self.path == '/api/status':
            with _cap_lock:
                act = {str(k): v['refs'] for k, v in _captures.items()}
            self._json(200, {'active': act})
        elif self.path.startswith('/stream'):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                dev_id = int(q.get('dev', [''])[0])
            except Exception:
                dev_id = None
            if dev_id is None:
                self._json(400, {'error': 'dev required'})
                return
            try:
                c = _get_capture(dev_id)
            except Exception as e:
                self._json(500, {'error': str(e)})
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while True:
                    with c['lock']:
                        if c['buf']:
                            data = bytes(c['buf'])
                            c['buf'].clear()
                        else:
                            data = b''
                    if data:
                        self.wfile.write(data)
                        self.wfile.flush()
                    else:
                        time.sleep(0.05)
            except Exception:
                pass
            finally:
                _release(dev_id)
        else:
            self._json(404, {})

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    print("[win_mic_bridge] 录音设备:")
    for d in _devices():
        print(f"  [{d['id']}] {d['name']}{' (默认)' if d['is_default'] else ''}")
    print(f"[win_mic_bridge] 监听 0.0.0.0:18690  (供 WSL ASR 选择 Windows 音源)")
    srv = ThreadingHTTPServer(('0.0.0.0', 18690), H)
    srv.serve_forever()
