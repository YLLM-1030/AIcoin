"""
B站弹幕采集器 — WS 独立线程 + 桥 19000
用法: python blc_collector.py <房间ID> [--mode open_live|wbi]
  open_live = B站直播开放平台（合法稳定，默认）: 需配置 OPEN_LIVE_* 常量
  wbi       = 旧版自写 WS + WBI 签名（不稳定，fallback）"""
import asyncio, os, json, struct, hashlib, hmac, uuid, datetime, time, urllib.parse, threading, zlib
from http.server import HTTPServer, BaseHTTPRequestHandler
import aiohttp, brotli

ROOM_ID = 22387248
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
HEADER = struct.Struct('>I2H2I')
WBI_TABLE = [46,47,18,2,53,8,23,32,15,50,10,31,58,3,45,35,27,43,5,49,33,9,42,19,29,28,14,39,12,38,41,13]

# ── B站直播开放平台配置（2026-08-11 申请通过，合法稳定监听）──
OPEN_LIVE_AK_ID = 'OMherZJ6SPbDw1DMUC9hUz18'
OPEN_LIVE_AK_SECRET = 'McOpgSDBCjrXgXKvhxLGz7eKtptNFe'
OPEN_LIVE_APP_ID = 1786938510612          # 项目ID「弹幕读取器」
OPEN_LIVE_CODE = ''                        # 默认身份码（会被同目录 blc_config.json 覆盖；页面可改）
OPEN_LIVE_HOST = 'https://live-open.biliapi.com'
_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blc_config.json')

_store = {}; _store_lock = threading.Lock()
_store['弹幕'] = 'wait'                               # 读后清空哨兵：wait=无新消息
_current_room = 0; _room_lock = threading.Lock()

def _store_line(line):
    """存一行：从 wait 或上次未读内容追加，等 danmaku_server 用 /take 读走"""
    with _store_lock:
        cur = _store.get('弹幕', 'wait')
        if cur == 'wait': cur = ''
        _store['弹幕'] = (cur + line + '\n')[-5000:]

# ── 身份码配置（可运行时修改，页面 19000 输入）──
def _load_cfg():
    try:
        with open(_CFG_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _get_code():
    """身份码：blc_config.json 优先，其次 OPEN_LIVE_CODE 默认值"""
    return (_load_cfg().get('code') or OPEN_LIVE_CODE or '').strip()

def _save_cfg(code):
    with open(_CFG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'code': code.strip()}, f, ensure_ascii=False)

class BH(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/source_last': self._j(200, dict(_store))
        elif self.path == '/config': self._j(200, {'code': _get_code()})
        elif self.path == '/take':
            # 原子「返回弹幕 + 置 wait」：单消费者读后清空，无竞态
            with _store_lock:
                dm = _store.get('弹幕', 'wait')
                _store['弹幕'] = 'wait'
            self._j(200, {'弹幕': dm})
        elif self.path == '/' or self.path.startswith('/?'):
            with _room_lock: r = _current_room
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            html = CTRL_PAGE % r
            b = html.encode()
            self.send_header('Content-Length', len(b))
            self.end_headers()
            self.wfile.write(b)
        else: self._j(404, {})
    def do_POST(self):
        d=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
        if self.path == '/text':
            s,t=d.get('source',''),d.get('text','')
            if s and t:
                for line in t.split('\n'):      # 弹幕只走这一条路（读后清空）
                    line = line.strip()
                    if line:
                        _store_line(line)
                if s != '弹幕':                 # 非弹幕 source 保留覆盖（给 /source_last 兼容）
                    with _store_lock: _store[s]=t
            self._j(200, {"ok":True})
        elif self.path == '/switch':
            url_or_id = d.get('room','').strip()
            if not url_or_id:
                self._j(400, {"error":"需要房间号或URL"})
                return
            # 提取房间ID
            import re as _re
            m = _re.search(r'(\d{3,})', url_or_id)
            if not m:
                self._j(400, {"error":"未识别房间号"})
                return
            new_rid = int(m.group(1))
            with _room_lock: 
                global _current_room
                _current_room = new_rid
            self._j(200, {"ok":True, "room":new_rid})
        elif self.path == '/config':
            code = d.get('code', '').strip()
            if code:
                _save_cfg(code)
                print(f'[配置] 身份码已更新: {code[:4]}...{code[-2:]}（自动重连生效）')
            self._j(200, {"ok":True, "code":code})
        else:
            self._j(404, {})
    def _j(self,c,d):
        b=json.dumps(d,ensure_ascii=False).encode()
        self.send_response(c); self.send_header('Content-Type','application/json')
        self.send_header('Access-Control-Allow-Origin','*'); self.send_header('Content-Length',len(b))
        self.end_headers(); self.wfile.write(b)
    def do_OPTIONS(self): self.send_response(200); self.end_headers()

_wbi_key=''
def _wbi_sign(params):
    if not _wbi_key: return params
    p={**params,'wts':str(int(datetime.datetime.now().timestamp()))}
    p={k:''.join(ch for ch in str(p[k]) if ch not in "!'()*") for k in sorted(p)}
    w_rid=hashlib.md5((urllib.parse.urlencode(p)+_wbi_key).encode()).hexdigest()
    return {**params,'wts':p['wts'],'w_rid':w_rid}

def _pack(op,body,ver=1):
    if isinstance(body,dict): body=json.dumps(body).encode()
    elif isinstance(body,str): body=body.encode()
    h=HEADER.pack(HEADER.size+len(body),HEADER.size,ver,op,1)
    return h+body

def _parse(data):
    off=0; results=[]
    while off<len(data):
        try: l,hl,ver,op,seq=HEADER.unpack_from(data,off)
        except: break
        body=data[off+hl:off+l]; off+=l
        if op==5 and body:
            if ver==3: body=brotli.decompress(body); results.extend(_parse(body))
            elif ver==2: body=zlib.decompress(body); results.extend(_parse(body))
            elif ver==0: results.append(json.loads(body.decode()))
    return results

# ── 开放平台（open_live）签名与接口 ──
def _open_live_sign(body_str):
    """B站开放平台 HTTP 签名：x-bili- 头字典序拼接 + HMAC-SHA256"""
    md5 = hashlib.md5(body_str.encode()).hexdigest()
    ts = str(int(time.time()))
    nonce = str(uuid.uuid4())
    m = {
        'x-bili-accesskeyid': OPEN_LIVE_AK_ID,
        'x-bili-content-md5': md5,
        'x-bili-signature-method': 'HMAC-SHA256',
        'x-bili-signature-nonce': nonce,
        'x-bili-signature-version': '1.0',
        'x-bili-timestamp': ts,
    }
    s = '\n'.join(f'{k}:{m[k]}' for k in sorted(m))
    auth = hmac.new(OPEN_LIVE_AK_SECRET.encode(), s.encode(), hashlib.sha256).hexdigest()
    return {**m, 'Authorization': auth, 'Content-Type': 'application/json', 'Accept': 'application/json'}

def _open_live_start(code):
    """POST /v2/app/start → (wss_link, auth_body, game_id)"""
    import requests
    body = json.dumps({'code': code, 'app_id': OPEN_LIVE_APP_ID}, ensure_ascii=False)
    r = requests.post(f'{OPEN_LIVE_HOST}/v2/app/start', headers=_open_live_sign(body),
                      data=body.encode(), timeout=10, verify=False)
    d = r.json()
    if d.get('code') != 0:
        raise Exception(f"start失败 code={d.get('code')} msg={d.get('message')}")
    dd = d['data']
    wi = dd['websocket_info']
    url = wi['wss_link'][0] if isinstance(wi['wss_link'], list) else wi['wss_link']
    return url, wi['auth_body'], str(dd['game_info']['game_id'])

def _open_live_heartbeat(game_id):
    """POST /v2/app/heartbeat：180s 无心跳自动关闭，20s 一次"""
    import requests
    body = json.dumps({'game_id': game_id})
    r = requests.post(f'{OPEN_LIVE_HOST}/v2/app/heartbeat', headers=_open_live_sign(body),
                      data=body.encode(), timeout=10, verify=False)
    return r.json().get('code') == 0

# ── WS in dedicated thread ──
def _handle_msg(entry):
    """统一消息处理：WBI 弹幕/SC + 开放平台 DM/SC"""
    c = entry.get('cmd', '')
    if c.startswith('DANMU_MSG'):
        info = entry.get('info', [])
        u = info[2][1] if len(info) > 2 and len(info[2]) > 1 else '?'
        t = info[1] if len(info) > 1 else ''
        if t:
            print(f'[弹幕] {u}: {t}')
            _store_line(f'[{u}] {t}')
    elif c == 'SUPER_CHAT_MESSAGE':
        d2 = entry.get('data', {})
        u, m, p = d2.get('user_info', {}).get('uname', '?'), d2.get('message', ''), d2.get('price', 0)
        if u and m:
            print(f'[SC] Y{p} {u}: {m}')
            _store_line(f'[SC Y{p}] {u}: {m}')
    elif c == 'LIVE_OPEN_PLATFORM_DM':
        dd = entry.get('data', {})
        u, t = dd.get('uname', '?'), dd.get('msg', '')
        if t:
            print(f'[弹幕] {u}: {t}')
            _store_line(f'[{u}] {t}')
    elif c == 'LIVE_OPEN_PLATFORM_SUPER_CHAT':
        dd = entry.get('data', {})
        u, m, p = dd.get('uname', '?'), dd.get('message', ''), dd.get('rmb', 0)
        if u and m:
            print(f'[SC] Y{p} {u}: {m}')
            _store_line(f'[SC Y{p}] {u}: {m}')


def _ws_thread(initial_room, use_open_live=False):
    import websocket
    global _wbi_key, _current_room
    with _room_lock: _current_room = initial_room
    room_id = initial_room
    while True:
        # 检查是否要切换房间
        with _room_lock:
            if _current_room != room_id:
                room_id = _current_room
                print(f'切换房间 → {room_id}')
        try:
            if use_open_live:
                code = _get_code()
                if not code:
                    print('[开放平台] ⚠️ 身份码未配置！请到控制面板 http://localhost:19000 输入身份码')
                    time.sleep(10); continue
                url, auth_body, game_id = _open_live_start(code)
                rr = room_id
                conn_code = code
                print(f'[开放平台] start成功 game_id={game_id}')
                hb_stop = threading.Event()
                def _app_hb():
                    while not hb_stop.is_set():
                        time.sleep(20)
                        try:
                            if not _open_live_heartbeat(game_id):
                                print('[应用心跳] code!=0，可能已过期')
                        except Exception as e:
                            print(f'[应用心跳] 异常: {e}')
                threading.Thread(target=_app_hb, daemon=True).start()
            else:
                async def _init_all():
                    global _wbi_key
                    async with aiohttp.ClientSession() as s:
                        r=await s.get('https://api.live.bilibili.com/room/v1/Room/get_info',
                            headers={'User-Agent':USER_AGENT},params={'room_id':room_id})
                        d=await r.json(); rr=d['data']['room_id']
                        r2=await s.get('https://api.bilibili.com/x/web-interface/nav',headers={'User-Agent':USER_AGENT})
                        d2=await r2.json(); wi=d2['data']['wbi_img']
                        ik=wi['img_url'].rpartition('/')[2].partition('.')[0]
                        sk=wi['sub_url'].rpartition('/')[2].partition('.')[0]
                        _wbi_key=''.join((ik+sk)[i] for i in WBI_TABLE if i<len(ik+sk))
                        r3=await s.get('https://api.live.bilibili.com/xlive/web-room/v1/index/getDanmuInfo',
                            headers={'User-Agent':USER_AGENT}, params=_wbi_sign({'id':rr,'type':0}))
                        d3=await r3.json()
                        if d3.get('code')!=0: raise Exception(f"code={d3.get('code')}")
                        host=d3['data']['host_list'][0]; token=d3['data']['token']
                        return rr,f"wss://{host['host']}:{host['wss_port']}/sub",token
                rr,url,token=asyncio.run(_init_all())
                print(f'认证 token 长度: {len(token)}, host: {url}')
                game_id = None; hb_stop = None
        except Exception as e:
            print(f'API失败: {e}, 5s后重试...')
            time.sleep(5); continue

        ws=websocket.create_connection(url,timeout=10)
        if use_open_live:
            ws.send(_pack(7, auth_body).decode('latin1'), opcode=websocket.ABNF.OPCODE_BINARY)
            print(f'[开放平台] 已连接（鉴权发送完成）')
        else:
            ws.send(_pack(7,json.dumps({'uid':0,'roomid':rr,'protover':3,'platform':'web','type':2,'key':token})).decode('latin1'),opcode=websocket.ABNF.OPCODE_BINARY)
            print(f'已连接房间 {rr}')
        last_hb=time.time(); last_check=time.time()
        try:
            while True:
                ws.settimeout(1)
                try: msg=ws.recv()
                except Exception: msg=None
                # 每3秒检查是否换房间 / 身份码变化
                if time.time()-last_check>3:
                    with _room_lock:
                        if _current_room != room_id:
                            raise Exception('room_changed')
                    if use_open_live and _get_code() != conn_code:
                        print('[开放平台] 身份码已更新，重连...')
                        raise Exception('code_changed')
                    last_check=time.time()
                if msg and isinstance(msg,bytes):
                    for entry in _parse(msg):
                        _handle_msg(entry)
                if time.time()-last_hb>30:
                    with _room_lock:
                        if _current_room != room_id:
                            print(f'切换房间 → {_current_room}')
                            raise Exception('room_changed')
                    ws.send(_pack(2,b'{}').decode('latin1'),opcode=websocket.ABNF.OPCODE_BINARY)
                    last_hb=time.time()
        except Exception as e:
            print(f'WS断开: {e}')
            ws.close()
            if use_open_live and hb_stop:
                hb_stop.set()

# ── 控制页面 ──
CTRL_PAGE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>弹幕采集器</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui;background:#0d1117;color:#c9d1d9;display:flex;justify-content:center;padding-top:60px}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;width:420px}
h2{font-size:20px;margin-bottom:16px}
.row{display:flex;gap:8px;margin-bottom:12px}
input{flex:1;font-size:14px;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:8px 12px}
button{padding:8px 16px;background:#1a7f37;color:#fff;border:none;border-radius:6px;font-size:14px;cursor:pointer}
button:hover{background:#209043}
.info{color:#8b949e;font-size:13px;margin-top:8px}
.info b{color:#58a6ff}
.links{margin-top:12px;font-size:13px}
.links a{color:#58a6ff;margin-right:12px}
</style></head><body>
<div class="card">
<h2>🎬 弹幕采集器</h2>
<div class="row">
  <input id="code-input" placeholder="开放平台身份码 (play-live.bilibili.com 右下角)" autofocus
    onkeydown="if(event.key==='Enter')saveCode()">
  <button onclick="saveCode()">保存身份码</button>
</div>
<div class="row">
  <input id="room-input" placeholder="B站直播间链接 或 房间号..." autofocus
    onkeydown="if(event.key==='Enter')switchRoom()">
  <button onclick="switchRoom()">切换</button>
</div>
<div class="info">
  当前房间: <b>%s</b>
  <br>弹幕桥: <b>19000</b> (消费者 /take 轮询)
</div>
<div class="links">
  <a href="http://localhost:18776" target="_blank">弹幕接收页 →</a>
</div>
</div>
<script>
function saveCode(){
  var v = document.getElementById('code-input').value.trim();
  if(!v) return;
  fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({code: v})
  }).then(function(r){ return r.json(); }).then(function(d){
    if(d.ok){ alert('身份码已保存: '+d.code); }
    else alert('保存失败: '+(d.error||'?'))
  });
}
fetch('/config').then(function(r){ return r.json(); }).then(function(d){
  if(d.code) document.getElementById('code-input').value = d.code;
});
function switchRoom(){
  var v = document.getElementById('room-input').value.trim();
  if(!v) return;
  fetch('/switch', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({room: v})
  }).then(function(r){ return r.json(); }).then(function(d){
    if(d.ok){ alert('已切换: '+d.room); location.reload(); }
    else alert('失败: '+(d.error||'?'))
  });
}
</script>
</body></html>"""

if __name__=='__main__':
    import sys
    rid=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else ROOM_ID
    mode = 'open_live'                              # 默认开放平台（合法稳定）
    if '--mode' in sys.argv:
        mode = sys.argv[sys.argv.index('--mode')+1]
    use_open_live = (mode == 'open_live')
    print(f'模式: {mode}（{"开放平台·合法" if use_open_live else "WBI·不稳定fallback"}）')
    threading.Thread(target=_ws_thread, args=(rid, use_open_live), daemon=True).start()
    from socketserver import ThreadingMixIn
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): daemon_threads = True
    print(f'弹幕桥 → http://localhost:19000')
    print(f'控制面板 → http://localhost:19000')
    ThreadedHTTPServer(('0.0.0.0',19000),BH).serve_forever()
