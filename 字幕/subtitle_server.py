"""
字幕服务 — 把进 TTS 的文本实时显示到透明浏览器窗口（直播投射用）

端口: 18777
  GET /              → 字幕页面（透明底，URL 参数可配样式）
  GET /events        → SSE 推送（新字幕实时到达）
  POST /api/subtitle → 设置当前字幕 {text}，广播给所有页面
  POST /api/clear    → 清空字幕
  GET  /api/status   → 当前状态 {text, history}

页面样式参数（URL query）:
  size=64            字号 px
  color=%23ffffff    文字颜色（URL编码）
  pos=bottom|center|top   位置
  font=...           字体（可写中文如 思源黑体）
  history=1          显示上一句小字
  karaoke=1          逐字点亮（简单版）

用法: python3 subtitle_server.py
"""
import json, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

PORT = 18777
_state = {'text': '', 'ts': 0}
_history = []
_sse_clients = []          # 每个元素是一个 queue.Queue
_clients_lock = threading.Lock()

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html,body{margin:0;padding:0;background:transparent;overflow:hidden;
            font-family:'__FONT__', 'Noto Sans SC','Microsoft YaHei',sans-serif;}
  #wrap{position:fixed;left:0;right:0;top:0;bottom:0;display:flex;flex-direction:column;
        justify-content:__JUSTIFY__;align-items:center;pointer-events:none;}
  #sub{font-size:__SIZE__px;color:__COLOR__;font-weight:700;text-align:center;
       text-shadow:2px 0 0 #7fc9ff, -2px 0 0 #7fc9ff, 0 2px 0 #7fc9ff, 0 -2px 0 #7fc9ff,
                   1px 1px 0 #7fc9ff, -1px 1px 0 #7fc9ff, 1px -1px 0 #7fc9ff, -1px -1px 0 #7fc9ff,
                   0 2px 10px rgba(0,0,0,0.35);
       max-width:90vw;line-height:1.3;word-break:break-all;
       opacity:0;transform:translateY(14px);
       transition:opacity .25s ease, transform .25s ease;}
  #sub.show{opacity:1;transform:translateY(0);}
  #hist{font-size:__HSIZE__px;color:rgba(255,255,255,0.55);text-align:center;
        text-shadow:0 1px 4px rgba(0,0,0,0.6);margin-top:10px;max-width:80vw;display:__HIST__;}
  /* 字符 span：默认全亮；卡拉OK时逐字点亮 */
  #sub .char{display:inline-block;}
  #sub.kara .char{opacity:.25;transition:opacity .2s;}
  #sub.kara .char.on{opacity:1;}
</style></head><body>
<div id="wrap">
  <div id="sub"></div>
  <div id="hist"></div>
</div>
<script>
var size = parseInt(new URLSearchParams(location.search).get('size')||'64',10);
var histOn = new URLSearchParams(location.search).get('history')==='1';
var karaoke = new URLSearchParams(location.search).get('karaoke')==='1';
var lastText = '';
function render(t, immediate){
  var sub = document.getElementById('sub');
  if(!t){ sub.textContent=''; sub.className=''; return; }
  if(t === lastText){ return; }           // 同句不重播
  lastText = t;
  // 拆成字符 span（karaoke 时逐字点亮用；默认全亮）
  sub.innerHTML = t.split('').map(function(c){return '<span class="char">'+c+'</span>';}).join('');
  sub.className = karaoke ? 'show kara' : 'show';
  var spans = sub.querySelectorAll('.char');
  if(karaoke){
    var i = 0;
    var tm = setInterval(function(){
      if(i < spans.length){ spans[i].className = spans[i].className.replace(' on','') + ' on'; i++; }
      else clearInterval(tm);
    }, 60);
  }
}
var es = new EventSource('/events');
es.onmessage = function(e){
  var d = JSON.parse(e.data);
  if(d.text){ render(d.text, false); }
  if(histOn && d.hist && d.hist.length){
    document.getElementById('hist').textContent = '『' + d.hist[0] + '』';
  }
};
// 08-11 兜底: SSE 断线连续 3 次 error(约30s) → 自动刷新恢复 (不用手动点刷新; 正常时不刷新)
var errCount = 0;
es.onerror = function(){
  errCount++;
  if(errCount >= 3){ location.reload(); }
};
// 可选强制定时刷新: ?auto=1 → 每 30s reload (OBS 浏览器源后台挂久后的最后兜底)
if(new URLSearchParams(location.search).get('auto')==='1'){
  setTimeout(function(){ location.reload(); }, 30000);
}
</script></body></html>"""


def render_page(params):
    size = int(params.get('size', ['64'])[0] or 64)
    color = params.get('color', ['%23ffffff'])[0]
    color = color.replace('%23', '#')
    pos = params.get('pos', ['center'])[0]
    justify = {'bottom': 'flex-end', 'center': 'center', 'top': 'flex-start'}.get(pos, 'flex-end')
    font = params.get('font', [''])[0]
    hist = 'block' if params.get('history', [''])[0] == '1' else 'none'
    hsize = max(14, size // 2)
    return (PAGE.replace('__SIZE__', str(size))
                .replace('__COLOR__', color)
                .replace('__JUSTIFY__', justify)
                .replace('__FONT__', font)
                .replace('__HIST__', hist)
                .replace('__HSIZE__', str(hsize)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'   # 08-11 修复: EventSource 规范要求 HTTP/1.1+ (HTTP/1.0 下 Chrome 长连接不稳/断)
    server_version = 'SubtitleSrv/1.0'

    def log_message(self, *a):
        pass

    def _j(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            import urllib.parse as up
            q = up.parse_qs(up.urlsplit(self.path).query)
            body = render_page(q).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('X-Accel-Buffering', 'no')   # 防中间层缓冲 (08-11)
            self.end_headers()
            import queue
            q = queue.Queue()
            with _clients_lock:
                _sse_clients.append(q)
            try:
                # 先推当前状态
                with _clients_lock:
                    cur = _state['text']
                    hist = list(_history)
                self._send_sse({'text': cur, 'hist': hist})
                while True:
                    try:
                        msg = q.get(timeout=10)
                        self._send_sse(msg)
                    except queue.Empty:
                        self.wfile.write(b': keepalive\n\n')
                        self.wfile.flush()
            except Exception:
                pass
            finally:
                with _clients_lock:
                    if q in _sse_clients:
                        _sse_clients.remove(q)
        elif self.path == '/api/status':
            with _clients_lock:
                self._j(200, {'text': _state['text'], 'history': list(_history)})
        else:
            self._j(404, {'error': 'not found'})

    def _send_sse(self, msg):
        data = json.dumps(msg, ensure_ascii=False)
        self.wfile.write(('data: ' + data + '\n\n').encode('utf-8'))
        self.wfile.flush()

    def do_POST(self):
        global _history   # 08-11 修复根因: _history = _history[:5] 重新绑定 → 需 global, 否则 insert() 抛 UnboundLocalError (POST 崩溃=字幕永不更新)
        import urllib.parse as up
        import traceback
        try:
            ln = int(self.headers.get('Content-Length', 0))
            d = json.loads(self.rfile.read(ln) or b'{}')
        except Exception:
            d = {}
        try:
            if self.path == '/api/subtitle':
                text = str(d.get('text', '')).strip()
                with _clients_lock:
                    _state['text'] = text
                    _state['ts'] = time.time()
                    if text:
                        _history.insert(0, text)
                        _history = _history[:5]
                    clients = list(_sse_clients)
                print(f"[字幕] {text[:40]}" if text else "[字幕] 清空")
                msg = {'text': text, 'hist': list(_history)}
                for q in clients:
                    try:
                        q.put_nowait(msg)
                    except Exception:
                        pass
                self._j(200, {'ok': True, 'text': text})
            elif self.path == '/api/clear':
                with _clients_lock:
                    _state['text'] = ''
                    _state['ts'] = 0
                    clients = list(_sse_clients)
                for q in clients:
                    try:
                        q.put_nowait({'text': '', 'hist': []})
                    except Exception:
                        pass
                self._j(200, {'ok': True})
            else:
                self._j(404, {'error': 'not found'})
        except Exception:
            # 08-11 修复: 处理异常必须回响应, 不能裸断连 (断连=客户端看到 Empty reply, 字幕永不更新)
            traceback.print_exc()
            try:
                self._j(500, {'error': 'server error'})
            except Exception:
                pass


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == '__main__':
    print(f'字幕服务 → http://localhost:{PORT}')
    print(f'字幕页面 → http://localhost:{PORT}/?size=64&pos=bottom&history=1')
    Server(('0.0.0.0', PORT), Handler).serve_forever()
