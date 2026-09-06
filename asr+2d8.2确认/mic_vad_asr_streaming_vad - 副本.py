"""
mic_vad_asr_streaming_vad.py — 流式 ASR + VAD（听觉编码器 v2：多输入源）
parec 版（绕过 PortAudio，直接 PulseAudio 管道读取）。

能力：
- 自带 6 个桥 (18680-18685)：POST /text 写文本、GET /source_last 读文本（与 pipeline 内置桥同模式）
- 管理页 (18686)：可创建多个输入源，每个源独立配置 音频设备(source) + 输出桥(out_bridge)
- 每个输入源：独立 parec 进程 + 独立 VAD 状态 + 独立识别流，共享同一个 paraformer 模型
- 识别文本 → 写该源自己的输出桥 {"source":"asr","text":"..."}
- VAD/FACS (18768)：保持单通道共享，所有源的识别结果都推（多源共享表情引擎）
- 配置持久化：asr_config.json {"inputs":[{id,name,source,out_bridge},...]}，改配置热生效（自动重启对应 parec）
"""
import sys, os, time, math, threading, argparse, json, subprocess, urllib.request as _ureq
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

SAMPLE_RATE = 16000
CHUNK = 512
PREBUFFER_MS = 300

CHUNK_SIZE = [0, 10, 5]
CHUNK_STRIDE = CHUNK_SIZE[1] * 960
STREAMING_EVERY_N = CHUNK_STRIDE // CHUNK
BYTES_PER_CHUNK = CHUNK * 2
_prebuffer_chunks = math.ceil(PREBUFFER_MS / (CHUNK / SAMPLE_RATE * 1000))

FACS_URL = "http://127.0.0.1:18768"
ASR_BRIDGE_PORTS = [18680, 18681, 18682, 18683, 18684, 18685]   # ASR 自带桥
ASR_MGMT_PORT = 18686                                            # ASR 管理页
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_config.json")

DEFAULT_INPUTS = [{"id": "main", "name": "我的麦克风", "source": "@DEFAULT_SOURCE@", "out_bridge": 18680}]

def _load_config():
    """读配置：{"inputs":[{id,name,source,out_bridge}]}；兼容旧版 {source,out_bridge,source_name}"""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("inputs"), list) and d["inputs"]:
            return d["inputs"]
        if isinstance(d, dict) and d.get("source"):
            return [{"id": "main", "name": "我的麦克风",
                     "source": d.get("source"), "out_bridge": int(d.get("out_bridge", 18680))}]
    except Exception as e:
        print(f"[配置] 读取失败用默认: {e}")
    return [dict(DEFAULT_INPUTS[0])]

def _save_config(inputs):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"inputs": inputs}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[配置] 保存失败: {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=6000.0)
    p.add_argument("--silence", type=float, default=0.8)
    p.add_argument("--source", type=str, default="")
    return p.parse_args()

def rms_energy(frame):
    return float(np.sqrt(np.mean(frame.astype(np.float32)**2)))

def _post_json(url, data, timeout=1):
    try:
        body = json.dumps(data, ensure_ascii=False).encode()
        req = _ureq.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
        _ureq.urlopen(req, timeout=timeout)
    except Exception:
        pass

def _push_vad(text):
    print(f"  [VAD] -> {text[:40]}", flush=True)
    _post_json(f"{FACS_URL}/facs", {"text": text})

def _push_silent():
    try:
        _ureq.urlopen(_ureq.Request(f"{FACS_URL}/silent", method='POST'), timeout=1)
    except Exception as e:
        print(f"  [silent ERR] {e}")

def _push_asr_bridge(text, port):
    """识别文本 → 指定输出桥（source 名固定 asr；管线侧是兜底源，源名由管线模块配置决定）"""
    _post_json(f"http://127.0.0.1:{port}/text", {"source": "asr", "text": text})

try:
    from funasr import AutoModel
except ImportError:
    print("[!] pip install funasr"); sys.exit(1)

def load_streaming_model():
    print("[*] 加载 paraformer-zh-streaming..."); sys.stdout.flush()
    t0 = time.time()
    m = AutoModel(model="paraformer-zh-streaming", disable_update=True)
    print(f"[+] ({time.time()-t0:.0f}s)")
    return m

class StreamRecognizer:
    def __init__(self, m):
        self.model = m; self.cache = {}; self.final_text = ""; self.lock = threading.Lock()
    def feed(self, a, is_final=False):
        with self.lock:
            try:
                r = self.model.generate(input=a, cache=self.cache, is_final=is_final,
                    chunk_size=CHUNK_SIZE, encoder_chunk_look_back=4, decoder_chunk_look_back=1)
                t = r[0].get("text", "").strip() if r else ""
                new = t[len(self.final_text):] if t.startswith(self.final_text) else t
                if t.startswith(self.final_text):
                    self.final_text = t
                else:
                    self.final_text += new
                return new
            except Exception:
                return ""

# ── 多输入源运行时 ──
_sources = {}          # id -> source 运行时对象
_sources_lock = threading.Lock()
_model = None          # 共享 paraformer 模型
_threshold = 6000.0
_silence_samples = int(SAMPLE_RATE * 0.8)
MIN_TRIGGER_HITS = 4

def _make_source(c):
    return {
        'id': c['id'], 'name': c.get('name', c['id']),
        'source': c['source'], 'out_bridge': int(c.get('out_bridge', 18680)),
        'proc': None, 'rec': None,
        'ring': deque(maxlen=_prebuffer_chunks),
        'state': {'recording': False, 'chunks': [], 'silence_count': 0, 'trigger_hits': 0,
                  'voice_frames': 0, 'segment_count': 0, 'feed_count': 0},
    }

def _reset_state(s):
    s['state'] = {'recording': False, 'chunks': [], 'silence_count': 0, 'trigger_hits': 0,
                  'voice_frames': 0, 'segment_count': 0, 'feed_count': 0}
    s['rec'] = None
    s['ring'] = deque(maxlen=_prebuffer_chunks)

def _start_source_parec(s):
    """启动该源 parec（已存在先杀 = 热切换）"""
    if s['proc']:
        try:
            s['proc'].kill()
        except Exception:
            pass
    s['proc'] = subprocess.Popen(
        ["parec", "--device=" + str(s['source']), "--format=s16le",
         "--rate=" + str(SAMPLE_RATE), "--channels=1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    _reset_state(s)
    print(f"[*] parec 启动 ({s['name']}/{s['id']}, source={s['source']})", flush=True)

def _apply_config(cfgs):
    """按配置增删改输入源（幂等：无变化不动作）"""
    ids = [c['id'] for c in cfgs]
    with _sources_lock:
        for sid in list(_sources):
            if sid not in ids:
                s = _sources.pop(sid)
                try:
                    s['proc'].kill()
                except Exception:
                    pass
                print(f"[配置] 移除输入源: {sid} ({s['name']})")
        for c in cfgs:
            cid = c['id']
            if cid in _sources:
                s = _sources[cid]
                if str(s['source']) != str(c['source']):
                    s['source'] = c['source']
                    _start_source_parec(s)
                s['name'] = c.get('name', cid)
                s['out_bridge'] = int(c.get('out_bridge', 18680))
            else:
                s = _make_source(c)
                _sources[cid] = s
                _start_source_parec(s)
                threading.Thread(target=audio_thread, args=(cid,), daemon=True).start()
                print(f"[配置] 新增输入源: {cid} ({c.get('name', '')})")

def _vad_timer():
    """VAD/FACS 保持单通道共享：所有 recording 源的实时文本都推 FACS"""
    while True:
        time.sleep(0.6)
        with _sources_lock:
            srcs = list(_sources.values())
        pushed = False
        for s in srcs:
            if s['state']['recording'] and s['rec']:
                try:
                    with s['rec'].lock:
                        text = s['rec'].final_text
                except Exception:
                    text = ""
                if text and text.strip():
                    _push_vad(text.strip())
                    pushed = True
        if not pushed:
            _push_silent()

# ── 自带桥（与 pipeline 内置桥同模式）──
_asr_stores = {}
_asr_store_lock = threading.Lock()

class ReusableTCPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class AsrBridgeHandler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass
    def do_GET(self):
        if self.path == "/source_last":
            port = self.server.server_address[1]
            with _asr_store_lock:
                data = _asr_stores.get(port, {})
            self._json(200, data)
            return
        self._json(404, {})
    def do_POST(self):
        if self.path == "/text":
            try:
                ln = int(self.headers.get("Content-Length", 0) or 0)
                d = json.loads(self.rfile.read(ln) or b"{}")
            except Exception:
                d = {}
            s, t = d.get("source", ""), d.get("text", "")
            if s and t:
                port = self.server.server_address[1]
                with _asr_store_lock:
                    if port not in _asr_stores:
                        _asr_stores[port] = {}
                    _asr_stores[port][s] = t[:5000]
            self._json(200, {"ok": True})
            return
        self._json(404, {})
    def log_message(self, *a):
        pass

# ── 音源枚举 ──
def _list_sources():
    try:
        out = subprocess.run(["pactl", "list", "sources", "short"],
                             capture_output=True, text=True, timeout=3)
        items = []
        for ln in out.stdout.strip().split("\n"):
            parts = [p for p in ln.split("\t") if p]
            if len(parts) >= 2:
                items.append({"name": parts[1], "desc": " ".join(parts[2:])})
        return items
    except Exception as e:
        return [{"name": "@DEFAULT_SOURCE@", "desc": f"(枚举失败: {e})"}]

# ── 管理服务（18686）──
PAGE_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>ASR 听觉编码器</title>
<style>
body{background:#0d1117;color:#c9d1d9;font-family:system-ui;margin:0;padding:24px}
h1{font-size:18px;color:#58a6ff}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px;max-width:680px}
.src{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;margin-bottom:10px}
.src h3{margin:0 0 8px;font-size:14px;color:#58a6ff}
label{display:block;font-size:12px;color:#8b949e;margin:8px 0 3px}
select,input[type=text],input[type=number]{width:100%;background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:5px;font-size:13px;box-sizing:border-box}
button{background:#238636;color:#fff;border:none;border-radius:4px;padding:8px 16px;font-size:13px;cursor:pointer;margin-top:10px;margin-right:6px}
button:hover{background:#2ea043}
button.del{background:#da3633}.del:hover{background:#f85149}
button.add{background:#1f6feb}.add:hover{background:#388bfd}
#status{font-size:12px;color:#8b949e;margin-top:10px}
.ok{color:#3fb950}.bad{color:#f85149}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:6px 8px;border-bottom:1px solid #21262d;text-align:left;color:#8b949e}
td:first-child{color:#c9d1d9}
</style></head><body>
<h1>🎙 ASR 听觉编码器</h1>
<div class="card">
  <h1 style="font-size:15px">🎛 输入源（每个源：音频设备 → 输出桥）</h1>
  <div id="inputs"></div>
  <button class="add" onclick="addSrc()">＋ 添加输入源</button>
  <button onclick="save()">保存并生效</button>
  <div id="status"></div>
</div>
<div class="card">
  <h1 style="font-size:15px">📡 自带桥</h1>
  <table><tr><th>端口</th><th>占用</th></tr><tbody id="bridges"></tbody></table>
</div>
<script>
var BRIDGES = [18680,18681,18682,18683,18684,18685];
var inputs = [];
function loadSources(cb){
  fetch('/api/sources').then(r=>r.json()).then(d=>{
    window._sources = d.sources||[];
    cb&&cb();
  });
}
function renderInputs(){
  var box=document.getElementById('inputs'); box.innerHTML='';
  inputs.forEach(function(inp,idx){
    var d=document.createElement('div'); d.className='src';
    var srcOpts='';
    (window._sources||[]).forEach(function(s){
      srcOpts+='<option value="'+s.name+'"'+(s.name===inp.source?' selected':'')+'>'+s.name+' ('+s.desc+')</option>';
    });
    var brOpts='';
    BRIDGES.forEach(function(p){
      brOpts+='<option value="'+p+'"'+(String(p)===String(inp.out_bridge)?' selected':'')+'>ASR桥 '+p+'</option>';
    });
    d.innerHTML=
      '<h3>输入源 '+(idx+1)+'</h3>'+
      '<label>名称（仅前端标识）</label><input type="text" value="'+(inp.name||'')+'" onchange="inputs['+idx+'].name=this.value">'+
      '<label>音频设备</label><select onchange="inputs['+idx+'].source=this.value">'+srcOpts+'</select>'+
      '<label>输出桥</label><select onchange="inputs['+idx+'].out_bridge=parseInt(this.value)">'+brOpts+'</select>'+
      '<br><button class="del" onclick="delSrc('+idx+')">删除</button>';
    box.appendChild(d);
  });
}
function addSrc(){
  inputs.push({id:'src'+Date.now(),name:'新输入源',source:'@DEFAULT_SOURCE@',out_bridge:18680});
  renderInputs();
}
function delSrc(idx){ inputs.splice(idx,1); renderInputs(); }
function save(){
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({inputs:inputs.map(function(i){return {id:i.id,name:i.name,source:i.source,out_bridge:parseInt(i.out_bridge)||18680};})})})
    .then(r=>r.json()).then(function(){
      document.getElementById('status').innerHTML='<span class="ok">✔ 已保存并生效</span> '+new Date().toLocaleTimeString();
    }).catch(function(e){
      document.getElementById('status').innerHTML='<span class="bad">✘ 保存失败: '+e+'</span>';
    });
}
function loadBridges(){
  fetch('/api/bridges').then(r=>r.json()).then(d=>{
    var tb=document.getElementById('bridges'); tb.innerHTML='';
    (d.ports||[]).forEach(function(p){
      fetch('http://localhost:'+p+'/source_last').then(r=>r.json()).then(data=>{
        var keys=Object.keys(data||{});
        var tr=document.createElement('tr');
        tr.innerHTML='<td>'+p+'</td><td>'+(keys.length?keys.join(', '):'—')+'</td>';
        tb.appendChild(tr);
      }).catch(function(){});
    });
  });
}
function init(){
  loadSources(function(){
    fetch('/api/config').then(r=>r.json()).then(function(d){
      inputs=(d.inputs||[]).map(function(i){return {id:i.id,name:i.name||'',source:i.source,out_bridge:parseInt(i.out_bridge)||18680};});
      if(!inputs.length) inputs.push({id:'main',name:'我的麦克风',source:'@DEFAULT_SOURCE@',out_bridge:18680});
      renderInputs();
    });
  });
  loadBridges();
}
init();
setInterval(loadBridges, 3000);
</script></body></html>"""

class AsrMgmtHandler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass
    def _html(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        try:
            self.wfile.write(body.encode())
        except Exception:
            pass
    def do_GET(self):
        if self.path == "/":
            self._html(PAGE_HTML)
        elif self.path == "/api/sources":
            self._json(200, {"sources": _list_sources()})
        elif self.path == "/api/config":
            with _sources_lock:
                inputs = [{"id": s['id'], "name": s['name'], "source": s['source'], "out_bridge": s['out_bridge']}
                          for s in _sources.values()]
            self._json(200, {"inputs": inputs})
        elif self.path == "/api/bridges":
            self._json(200, {"ports": ASR_BRIDGE_PORTS})
        else:
            self._json(404, {})
    def do_POST(self):
        if self.path == "/api/config":
            try:
                ln = int(self.headers.get("Content-Length", 0) or 0)
                d = json.loads(self.rfile.read(ln) or b"{}")
            except Exception:
                d = {}
            raw = d.get("inputs")
            if isinstance(raw, list) and raw:
                cfgs = []
                for i, x in enumerate(raw):
                    cfgs.append({
                        "id": str(x.get("id") or f"src{i}"),
                        "name": str(x.get("name") or f"输入源{i+1}"),
                        "source": str(x.get("source") or "@DEFAULT_SOURCE@"),
                        "out_bridge": int(x.get("out_bridge") or 18680),
                    })
                _save_config(cfgs)
                _apply_config(cfgs)
                self._json(200, {"ok": True, "inputs": cfgs})
                return
            self._json(400, {"error": "inputs required"})
            return
        self._json(404, {})
    def log_message(self, *a):
        pass

def _config_watcher():
    """每秒检查配置文件：输入源增删/音源变化 → 应用（外部改文件也生效）"""
    while True:
        time.sleep(1)
        cfgs = _load_config()
        _apply_config(cfgs)

def main():
    global _model, _threshold, _silence_samples
    args = parse_args()
    _threshold = args.threshold
    _silence_samples = int(SAMPLE_RATE * args.silence)

    cfgs = _load_config()
    if args.source:                     # 命令行 --source 覆盖第一个源（兼容旧启动）
        cfgs[0]['source'] = args.source

    _model = load_streaming_model()     # 共享模型

    # 起自带桥 18680-18685
    for p in ASR_BRIDGE_PORTS:
        try:
            bs = ReusableTCPServer(("127.0.0.1", p), AsrBridgeHandler)
            threading.Thread(target=bs.serve_forever, daemon=True).start()
            print(f"[桥] {p} 启动")
        except Exception as e:
            print(f"[桥] {p} 启动失败: {e}")
    # 起管理页 18686
    try:
        mgmt = ReusableTCPServer(("127.0.0.1", ASR_MGMT_PORT), AsrMgmtHandler)
        threading.Thread(target=mgmt.serve_forever, daemon=True).start()
        print(f"[管理] http://localhost:{ASR_MGMT_PORT}")
    except Exception as e:
        print(f"[管理] {ASR_MGMT_PORT} 启动失败: {e}")

    _apply_config(cfgs)

    threading.Thread(target=_vad_timer, daemon=True).start()
    threading.Thread(target=_config_watcher, daemon=True).start()

    with _sources_lock:
        n = len(_sources)
    print(f"[*] 输入源数={n}, 阈值={args.threshold}, 静音={args.silence}s")
    print(f"[*] 管理页: http://localhost:{ASR_MGMT_PORT}  |  Ctrl+C 退出, 现在说话...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] 退出")
        with _sources_lock:
            for s in _sources.values():
                try:
                    s['proc'].terminate()
                except Exception:
                    pass

def process_chunk(s, data_bytes):
    """处理一个源的一帧音频（该源线程独占）"""
    frame = np.frombuffer(data_bytes, dtype=np.int16)
    energy = rms_energy(frame)
    above = energy > _threshold
    st = s['state']; ring = s['ring']

    if not st['recording']:
        ring.append(frame.copy())
        if above:
            st['trigger_hits'] += 1
            if st['trigger_hits'] >= MIN_TRIGGER_HITS:
                st['recording'] = True
                st['chunks'] = list(ring); st['chunks'].append(frame.copy())
                st['silence_count'] = 0; st['voice_frames'] = MIN_TRIGGER_HITS
                st['trigger_hits'] = 0; st['feed_count'] = 0
                print(f"\n>>> [{s['name']}] 开始录音 <<<")
                s['rec'] = StreamRecognizer(_model)
        else:
            st['trigger_hits'] = 0
    else:
        st['chunks'].append(frame.copy())
        if above:
            st['voice_frames'] += 1; st['silence_count'] = 0
        else:
            st['silence_count'] += 1

        if len(st['chunks']) >= STREAMING_EVERY_N * (st['feed_count'] + 1):
            cs = st['feed_count'] * STREAMING_EVERY_N
            ce = (st['feed_count'] + 1) * STREAMING_EVERY_N
            audio_block = np.concatenate(st['chunks'][cs:ce]).astype(np.float32) / 32768.0
            r2 = s['rec']
            def partial_feed(a):
                new = r2.feed(a, is_final=False)
                if new and new.strip():
                    print(f"  [->] {new.strip()}", flush=True)
            threading.Thread(target=partial_feed, args=(audio_block,), daemon=True).start()
            st['feed_count'] += 1

        if st['silence_count'] >= _silence_samples // CHUNK:
            if st['silence_count'] < len(st['chunks']):
                st['chunks'] = st['chunks'][:-st['silence_count']]
            dur = len(st['chunks']) * CHUNK / SAMPLE_RATE
            if st['voice_frames'] >= MIN_TRIGGER_HITS and dur >= 0.3:
                print(f"\n>>> [{s['name']}] 录音结束 ({dur:.1f}s) <<<")
                rs = st['feed_count'] * STREAMING_EVERY_N
                if rs < len(st['chunks']):
                    rest = np.concatenate(st['chunks'][rs:]).astype(np.float32) / 32768.0
                    s['rec'].feed(rest, is_final=False)
                sid = st['segment_count']; r3 = s['rec']; sname = s['name']; oport = s['out_bridge']
                def final_asr():
                    time.sleep(0.3)
                    r3.feed(np.zeros(CHUNK_STRIDE, dtype=np.float32), is_final=True)
                    fall = r3.final_text
                    print(f">>> [ASR {sname}#{sid}] {fall}")
                    if fall:
                        _push_vad(fall)                    # VAD 保持单通道共享
                        _push_asr_bridge(fall, oport)      # 写该源自己的输出桥
                threading.Thread(target=final_asr, daemon=True).start()
                st['segment_count'] += 1
            else:
                print(f"\n>>> [{s['name']}] 忽略: 无有效人声 ({dur:.1f}s) <<<")
            st['recording'] = False; st['chunks'] = []; st['silence_count'] = 0
            st['voice_frames'] = 0; st['trigger_hits'] = 0; st['feed_count'] = 0

def audio_thread(sid):
    """一个输入源一个线程：读 parec → process_chunk"""
    while True:
        with _sources_lock:
            s = _sources.get(sid)
        if s is None:
            break
        proc = s['proc']
        if proc is None:
            time.sleep(1)
            continue
        try:
            data = proc.stdout.read(BYTES_PER_CHUNK)
        except Exception:
            data = b''
        if not data:
            time.sleep(0.3)     # proc 死了/被换，下轮拿新 proc
            continue
        process_chunk(s, data)

if __name__ == "__main__":
    main()
