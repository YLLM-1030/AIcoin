"""
搜索工具管理页面 + API 服务
端口: 18775
支持: 秘塔搜索 / DeepSeek 搜索
"""
import json
import os
import time
import urllib.request
import urllib.error
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

# ── SSE 队列 (同 TTS 模式) ──
_sse_queues = []
_sse_lock = threading.Lock()
_sse_push = None   # 最新的搜索结果, 新 SSE 连接时立即推送

DS_KEY = "sk-c584d10b539843359b74580b54de5efd"
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'search_history.json')
from web_search import web_search


def deepseek_search(query, api_key, prompt_tpl):
    """DeepSeek 联网搜索，可自定义提示词，{query} 替换为搜索词"""
    prompt = prompt_tpl.replace('{query}', query)
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "enable_search": True
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_history_entry(query, raw_result):
    history = load_history()
    history.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "result": raw_result
    })
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


PIPELINE_FILE = os.path.join(os.path.dirname(__file__), 'pipeline_searches.json')

def load_pipeline():
    if os.path.exists(PIPELINE_FILE):
        with open(PIPELINE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self._html(SEARCH_PAGE)
        elif self.path == '/api/search_history':
            self._json(200, load_history())
        elif self.path == '/api/pipeline_searches':
            self._json(200, load_pipeline())
        elif self.path == '/api/queue_sse':
            self._sse()
        else:
            self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        data = json.loads(body)

        if self.path == '/api/search':
            query = data.get('query', '')
            save = data.get('save', False)
            source = data.get('source', 'metaso')
            prompt = data.get('prompt', '请帮我搜索：{query}')
            if not query:
                self._json(400, {"error": "empty query"})
                return
            try:
                if source == 'deepseek':
                    result = deepseek_search(query, DS_KEY, prompt)
                else:
                    result = web_search(query)
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            if save:
                save_history_entry(query, result)
            self._json(200, {"result": result, "saved": save})
        elif self.path == '/api/pipeline_search':
            query = data.get('query', '')
            if not query:
                self._json(400, {"error": "empty query"})
                return
            result = web_search(query)
            entry = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "result": result
            }
            plist = load_pipeline()
            plist.append(entry)
            if len(plist) > 50:
                plist = plist[-50:]
            with open(PIPELINE_FILE, 'w', encoding='utf-8') as f:
                json.dump(plist, f, ensure_ascii=False, indent=2)
            self._json(200, {"result": result})
        elif self.path == '/api/queue_search':
            query = data.get('query', '')
            if query:
                global _sse_push
                msg = {'query': query}
                _sse_push = msg
                with _sse_lock:
                    for q in _sse_queues:
                        try: q.put_nowait(msg)
                        except: pass
                print(f"[SSE push] query={query} clients={len(_sse_queues)}")
                self._json(200, {"ok": True})
            else:
                self._json(400, {"error": "empty query"})
        else:
            self._json(404, {})

    def _json(self, code, data):
        b = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(b))
        self.end_headers()
        self.wfile.write(b)

    def _html(self, content):
        b = content.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(b))
        self.end_headers()
        self.wfile.write(b)

    def _sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        q = queue.Queue()
        with _sse_lock:
            _sse_queues.append(q)
            # 如果有缓存结果, 立即推送
            if _sse_push:
                q.put_nowait(_sse_push)
        try:
            while True:
                data = q.get()
                self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _sse_lock:
                if q in _sse_queues:
                    _sse_queues.remove(q)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()


SEARCH_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>搜索工具</title>
<style>
*{box-sizing:border-box; margin:0; padding:0}
body{font-family:system-ui,sans-serif; background:#0d1117; color:#c9d1d9; max-width:800px; margin:40px auto; padding:0 20px}
h1{font-size:18px; font-weight:600; margin-bottom:16px}
.inp-row{display:flex; gap:8px; margin-bottom:12px}
.inp-row input{flex:1; padding:8px 12px; background:#161b22; color:#c9d1d9; border:1px solid #30363d; border-radius:6px; font-size:14px}
button{padding:8px 16px; background:#185fa5; color:#fff; border:none; border-radius:6px; font-size:14px; cursor:pointer}
button:hover{background:#1a6fc4}
.save-row{display:flex; align-items:center; gap:8px; margin-bottom:12px; font-size:13px; color:#8b949e}
.save-row input[type=checkbox]{width:16px; height:16px}
#status{font-size:13px; color:#8b949e; margin-bottom:8px; min-height:20px}
#results{background:#161b22; border:1px solid #30363d; border-radius:6px; padding:12px; font-size:13px; line-height:1.6; white-space:pre-wrap; min-height:100px; max-height:500px; overflow-y:auto}
#results:empty::before{content:'（搜索结果显示在这里）'; color:#484f58}
.history-toggle{font-size:13px; color:#58a6ff; cursor:pointer; margin-left:12px}
.history-toggle:hover{text-decoration:underline}
#history-panel{display:none; margin-top:12px}
.hist-item{padding:8px; margin-bottom:8px; background:#161b22; border:1px solid #30363d; border-radius:6px; font-size:12px}
.hist-item .q{color:#58a6ff; margin-bottom:4px}
.hist-item .t{color:#484f58; font-size:11px}
.hist-item pre{white-space:pre-wrap; max-height:120px; overflow-y:auto; margin-top:4px; color:#8b949e}
</style>
</head>
<body>
<h1>🔍 搜索工具 <span class="history-toggle" onclick="toggleHistory()">📋 历史记录</span></h1>
<div class="inp-row">
  <input id="query" placeholder="输入搜索关键词..." onkeydown="if(event.key==='Enter')doSearch()">
  <select id="source" style="padding:8px 12px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;font-size:14px">
    <option value="metaso">秘塔</option>
    <option value="deepseek">DeepSeek</option>
  </select>
  <button onclick="togglePromptCfg()" style="background:#30363d;font-size:12px;padding:4px 10px">⚙</button>
  <button onclick="doSearch()">搜索</button>
</div>
<div id="prompt-cfg" style="display:none;margin-bottom:12px">
  <textarea id="ds-prompt" style="width:100%;height:80px;padding:8px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;font-size:13px;font-family:system-ui"></textarea>
  <button onclick="savePrompt()" style="margin-top:6px;font-size:12px;padding:4px 12px">保存提示词</button>
</div>
<div class="save-row">
  <input type="checkbox" id="saveMode">
  <label for="saveMode">保存搜索记录</label>
</div>
<div id="status"></div>
<div id="results"></div>
<div style="margin-top:8px;display:flex;align-items:center;gap:8px;font-size:13px;color:#8b949e">
  <span>搜索结果上桥</span>
  <select id="bridge-port" style="font-size:12px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:2px 4px;width:80px">
    <option value="">关</option>
    <option value="18000">18000</option><option value="18001">18001</option><option value="18002">18002</option><option value="18003">18003</option><option value="18004">18004</option>
    <option value="18005">18005</option><option value="18006">18006</option><option value="18007">18007</option><option value="18008">18008</option><option value="18009">18009</option>
  </select>
  <button onclick="saveBridgePort()" style="background:#30363d;color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px">保存</button>
  <span id="bridge-ok" style="color:var(--green);font-size:12px;display:none">✓</span>
</div>
<div style="margin-top:4px;display:flex;align-items:center;gap:8px;font-size:13px;color:#8b949e">
  <span>搜索完成 done</span>
  <select id="done-port" style="font-size:12px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:2px 4px;width:80px">
    <option value="">关</option>
    <option value="18000">18000</option><option value="18001">18001</option><option value="18002">18002</option><option value="18003">18003</option><option value="18004">18004</option>
    <option value="18005">18005</option><option value="18006">18006</option><option value="18007">18007</option><option value="18008">18008</option><option value="18009">18009</option>
  </select>
  <button onclick="saveDonePort()" style="background:#30363d;color:#fff;border:none;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px">保存</button>
  <span id="done-ok" style="color:var(--green);font-size:12px;display:none">✓</span>
</div>
<div style="margin-top:8px;font-size:13px;color:#58a6ff;cursor:pointer" onclick="togglePipeline()">🔗 管线搜索任务 <span id="pipe-count" style="color:#8b949e;font-size:12px"></span></div>
<div id="pipeline-panel" style="display:none;margin-top:8px" class="hist-list"></div>
<div id="history-panel">
  <h3 style="font-size:14px; margin-bottom:8px">搜索历史</h3>
  <div id="history-list"></div>
</div>
<script>
(function(){
  var p = localStorage.getItem('ds_prompt') || '请帮我搜索：{query}，返回包含标题、来源URL和摘要的搜索结果';
  document.getElementById('ds-prompt').value = p;
  var bp = localStorage.getItem('search_bridge_port') || '';
  document.getElementById('bridge-port').value = bp;
  var dp = localStorage.getItem('search_done_port') || '';
  document.getElementById('done-port').value = dp;
  // SSE: 管线推送搜索
  var es = new EventSource('/api/queue_sse');
  es.onmessage = function(e){
    var d = JSON.parse(e.data);
    if(d.query){
      document.getElementById('query').value = d.query;
      doSearch();
    }
  };
})();
function togglePromptCfg(){
  var el = document.getElementById('prompt-cfg');
  el.style.display = el.style.display==='none'?'block':'none';
}
function savePrompt(){
  localStorage.setItem('ds_prompt', document.getElementById('ds-prompt').value);
}
function saveBridgePort(){
  localStorage.setItem('search_bridge_port', document.getElementById('bridge-port').value);
  var ok=document.getElementById('bridge-ok'); ok.style.display=''; setTimeout(function(){ok.style.display='none';},2000);
}
function saveDonePort(){
  localStorage.setItem('search_done_port', document.getElementById('done-port').value);
  var ok=document.getElementById('done-ok'); ok.style.display=''; setTimeout(function(){ok.style.display='none';},2000);
}
async function doSearch(){
  var q=document.getElementById('query').value.trim();
  if(!q) return;
  document.getElementById('status').textContent='搜索中...';
  document.getElementById('results').textContent='';
  var source = document.getElementById('source').value;
  var r=await fetch('/api/search',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      query:q,
      save:document.getElementById('saveMode').checked,
      source:source,
      prompt:document.getElementById('ds-prompt').value
    })
  });
  var d=await r.json();
  if(d.error){ document.getElementById('status').textContent='错误: '+d.error; return; }
  document.getElementById('status').textContent=d.saved?'已保存':'';
  document.getElementById('results').textContent=d.result;
  // 上桥
  var bp=document.getElementById('bridge-port').value;
  console.log('[bridge] port='+bp+' text_len='+d.result.length);
  if(bp){
    fetch('http://localhost:'+bp+'/text',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({source:'搜索返回', text:d.result})
    });
  }
  var dp=document.getElementById('done-port').value;
  if(dp){
    fetch('http://localhost:'+dp+'/text',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({source:'tts_done', text:'done'})
    });
  }
}
var _histOpen=false;
async function toggleHistory(){
  _histOpen=!_histOpen;
  document.getElementById('history-panel').style.display=_histOpen?'block':'none';
  if(_histOpen){
    var r=await fetch('/api/search_history');
    var h=await r.json();
    document.getElementById('history-list').innerHTML=h.slice().reverse().map(function(x,i){
      return '<div class="hist-item"><div class="q">'+escapeHtml(x.query)+'</div><div class="t">'+x.time+'</div><pre>'+escapeHtml(x.result)+'</pre></div>';
    }).join('');
  }
}
function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
var _pipeOpen=false;
async function togglePipeline(){
  _pipeOpen=!_pipeOpen;
  document.getElementById('pipeline-panel').style.display=_pipeOpen?'block':'none';
  if(_pipeOpen){
    refreshPipeline();
  }
}
async function refreshPipeline(){
  var r=await fetch('/api/pipeline_searches');
  var h=await r.json();
  document.getElementById('pipe-count').textContent='('+h.length+')';
  document.getElementById('pipeline-panel').innerHTML=h.slice().reverse().map(function(x){
    return '<div class="hist-item"><div class="q">'+escapeHtml(x.query)+'</div><div class="t">'+x.time+'</div><pre>'+escapeHtml(x.result)+'</pre></div>';
  }).join('');
}
</script>
</body>
</html>"""


if __name__ == '__main__':
    srv = ThreadingHTTPServer(('0.0.0.0', 18775), SearchHandler)
    print('搜索工具 → http://localhost:18775')
    srv.serve_forever()
