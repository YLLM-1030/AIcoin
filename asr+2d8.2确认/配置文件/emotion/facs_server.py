"""
facs_server.py — 完整 FACS 引擎 Web 测试界面
用法: python emotion/facs_server.py
浏览器打开 http://localhost:18768
"""
import json, sys, os, math, urllib.request as _ureq, threading as _threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amygdala_detector import detect_amygdala_signals
from facs_engine import detect_speaker_emotion, detect_told_me_emotion
from vad_engine import DualVAD, map_emotions_to_au, VISIBILITY_THRESHOLD
from directional_detector import is_about_you
from directional_prototypes import SPK_TO_L_NORMAL
import output_engine   # 输出演绎引擎 (vad 共享: 输入/输出写同一个 dv + dist 计算)

dv = DualVAD()
_latest_info = {"text": "", "emotion": "", "ratio": 0}

EMO_TO_CN = {
    "happy":"开心","sad":"难过","angry":"生气","scared":"害怕",
    "surprised":"惊讶","worried":"担心","contempt":"轻蔑",
    "shy":"害羞","care":"关心","hurt":"受伤","tense":"紧张",
    "orient":"好奇","approach":"关切",
}
EMO_THR = {
    "happy":0.35,"sad":0.40,"angry":0.30,"scared":0.30,"surprised":0.30,
    "hurt":0.25,"care":0.25,"worried":0.20,"orient":0.40,
    "tense":0.35,"contempt":0.30,"approach":0.40,"shy":0.30,
}

def _ratio_one(emo, vad_val):
    if vad_val < 0.01: return 0.0
    thr = EMO_THR.get(emo, 0.4)
    if vad_val <= thr: return 0.0
    norm = (vad_val - thr) / (1.0 - thr)
    return round(min(norm, 1.0) ** 0.5, 4)

def compute_ratio(expr_layer):
    upper = expr_layer.get("upper", {}).get("emotions", {})
    lower = expr_layer.get("lower", {}).get("emotions", {})
    def best(d):
        if not d: return None, 0
        emo = max(d, key=d.get)
        return emo, d[emo]
    u_emo, u_val = best(upper)
    l_emo, l_val = best(lower)
    u_cn = EMO_TO_CN.get(u_emo, u_emo) if u_emo else "中性"
    l_cn = EMO_TO_CN.get(l_emo, l_emo) if l_emo else "中性"
    u_ratio = _ratio_one(u_emo, u_val) if u_emo else 0
    l_ratio = _ratio_one(l_emo, l_val) if l_emo else 0
    return u_cn, u_ratio, l_cn, l_ratio

def _push_bridge(emo, ratio, suppress=0.0, dist=0.0):
    """push vad_status → 前端 2D。suppress = 表达压制 0~1; dist = 注意力转移 0~1"""
    try:
        data = json.dumps({'type':'vad_status','mode':'listen','emotion':emo,'ratio':ratio,
                           'suppress':suppress,'dist':dist}).encode()
        req = _ureq.Request('http://127.0.0.1:18770/push', data=data,
            headers={'Content-Type':'application/json'}, method='POST')
        _ureq.urlopen(req, timeout=1)
    except Exception:
        pass


SHELL_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FACS 情绪引擎 · 输入/输出</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
.tabs{display:flex;gap:8px;padding:12px 20px;background:#16213e;border-bottom:1px solid #2a2a4a;align-items:center;flex-wrap:wrap}
.tabs h1{font-size:1.1em;color:#e94560;margin-right:16px}
.tab-btn{padding:8px 20px;border:1px solid #444;border-radius:8px;background:#0f0f23;color:#aaa;cursor:pointer;font-size:14px}
.tab-btn.active{background:#e94560;color:#fff;border-color:#e94560}
.tab-btn:hover{border-color:#e94560}
.hint{font-size:0.75em;color:#888;margin-left:8px}
iframe{flex:1;border:none;width:100%}
</style>
</head>
<body>
<div class="tabs">
  <h1>FACS 情绪引擎</h1>
  <button id="tab-in" class="tab-btn active" onclick="switchTab('in')">🎧 输入侧 (倾听)</button>
  <button id="tab-out" class="tab-btn" onclick="switchTab('out')">🗣 输出侧 (说话)</button>
  <span class="hint">两侧共用同一个皮层池 dv</span>
</div>
<iframe id="frame" src="/view_input"></iframe>
<script>
function switchTab(which) {
  document.getElementById('tab-in').classList.toggle('active', which==='in');
  document.getElementById('tab-out').classList.toggle('active', which==='out');
  document.getElementById('frame').src = which==='in' ? '/view_input' : '/view_output';
}
</script>
</body>
</html>"""


OUT_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>输出演绎引擎 v1</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh}
.container{max-width:1000px;margin:0 auto;padding:20px}
h1{text-align:center;color:#e94560;margin-bottom:4px;font-size:1.5em}
.sub{text-align:center;color:#888;margin-bottom:20px;font-size:0.85em}
.input-row{display:flex;gap:10px;margin-bottom:12px}
#text-input{flex:1;padding:12px 16px;border:1px solid #333;border-radius:8px;background:#16213e;color:#eee;font-size:15px;outline:none}
#text-input:focus{border-color:#e94560}
#submit-btn{padding:12px 24px;border:none;border-radius:8px;background:#e94560;color:#fff;font-size:15px;cursor:pointer}
#submit-btn:hover{background:#d63851}
#submit-btn:disabled{opacity:0.5;cursor:not-allowed}
.btn-row{display:flex;gap:6px;margin-bottom:20px;flex-wrap:wrap;align-items:center}
.btn-row button{padding:7px 14px;border:1px solid #444;border-radius:6px;background:#16213e;color:#aaa;cursor:pointer;font-size:12px}
.btn-row button:hover{border-color:#e94560;color:#eee}
.btn-row button.active{background:#e94560;color:#fff;border-color:#e94560}
#auto-status{margin-left:10px;font-size:0.8em;color:#888}
.frame-counter{font-size:0.75em;color:#888;margin-left:8px}
#results{display:flex;flex-direction:column;gap:10px}
.card{background:#16213e;border:1px solid #2a2a4a;border-radius:10px;padding:14px}
.card h3{color:#e94560;font-size:0.85em;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
.bar-row{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.bar-label{width:85px;font-size:0.78em;color:#aaa;text-align:right}
.bar-track{flex:1;height:16px;background:#0f0f23;border-radius:3px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:3px;transition:width 0.3s}
.bar-val{width:42px;font-size:0.72em;color:#888;text-align:right}
.amygdala-bar{background:linear-gradient(90deg,#ff6b35,#ff8c42)}
.cortex-bar{background:linear-gradient(90deg,#0f3460,#1a56db)}
.au-bar{background:linear-gradient(90deg,#e94560,#ff6b6b)}
.vad-bar{background:linear-gradient(90deg,#2d6a4f,#52b788)}
.params-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:4px}
.param-item{display:flex;justify-content:space-between;font-size:0.75em;padding:3px 6px;background:#0f0f23;border-radius:3px}
.param-name{color:#aaa}
.param-val{color:#ff6b6b;font-weight:bold}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.7em;margin:2px;font-weight:bold}
.tag-tense{background:rgba(255,107,53,0.2);color:#ff8c42}
.tag-orient{background:rgba(66,133,244,0.2);color:#5b9cf5}
.tag-approach{background:rgba(52,168,83,0.2);color:#5abe6e}
.tag-relax{background:rgba(233,69,96,0.2);color:#ff6b6b}
.tag-attack{background:#e94560;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.7em;font-weight:bold}
.tag-mixed{background:#ffd93d;color:#1a1a2e;padding:2px 8px;border-radius:8px;font-size:0.7em;font-weight:bold}
.champion-badge{background:#e94560;color:#fff;font-size:0.65em;padding:1px 6px;border-radius:8px;margin-left:4px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:600px){.columns{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<h1>🗣 输出演绎引擎 v1</h1>
<div class="sub">三分归属 → SPK_OUT 表(×4帧补偿) → 共享皮层池(无杏仁核) → dist转移 → 表达压制 · 连点=累加</div>

<div class="input-row">
  <input id="text-input" placeholder="硬币要说的文本, 按回车 (如: 我听说小明最近很开心)" autofocus>
  <button id="submit-btn" onclick="submitText()">测试</button>
</div>

<div id="live-bar" style="display:none;padding:8px 14px;background:#0a2a1a;border:1px solid #2a8;border-radius:8px;margin-bottom:12px;font-size:14px;color:#5abe6e">
  🎤 <span id="live-text">等待语音...</span>
  &nbsp;|&nbsp;情绪: <b id="live-emo">-</b> r=<span id="live-ratio">-</span>
</div>

<div class="btn-row">
  <button onclick="submitTextPreset('我好难过')">我好难过</button>
  <button onclick="submitTextPreset('我讨厌你')">我讨厌你</button>
  <button onclick="submitTextPreset('我喜欢你')">我喜欢你</button>
  <button onclick="submitTextPreset('你好难过')">你好难过</button>
  <button onclick="submitTextPreset('你生气了吗')">你生气</button>
  <button onclick="submitTextPreset('我听说小明最近很开心')">小明开心</button>
  <button onclick="submitTextPreset('我听说小明最近很难过')">小明难过</button>
  <button onclick="submitTextPreset('听说有人中彩票了')">中彩票</button>
  <button onclick="submitTextPreset('今天天气真好')">天气真好</button>
  <button id="auto-btn" onclick="toggleAuto()">⏯ 衰减模式</button>
  <span id="auto-status"></span>
  <span class="frame-counter" id="frame-counter"></span>
  <button onclick="resetEngine()">🔄 重置</button>
</div>

<div id="results"></div>
</div>

<script>
let autoTimer = null;
let _decayWorker = null;
let frameCount = 0;

async function submitText() {
  const text = document.getElementById('text-input').value.trim();
  if (!text) return;
  await send(text);
}

async function submitTextPreset(text) {
  document.getElementById('text-input').value = text;
  await send(text);
}

async function resetEngine() {
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  if (_decayWorker) { _decayWorker.terminate(); _decayWorker = null; }
  document.getElementById('auto-btn').classList.remove('active');
  document.getElementById('auto-status').textContent = '';
  frameCount = 0;
  document.getElementById('frame-counter').textContent = '';
  await fetch('/reset', { method: 'POST' });
  document.getElementById('results').innerHTML = '';
  document.getElementById('text-input').value = '';
  toggleAuto();   // 重置后保持常驻显示
}

function toggleAuto() {
  const btn = document.getElementById('auto-btn');
  if (_decayWorker) {
    _decayWorker.terminate();
    _decayWorker = null;
    btn.classList.remove('active');
    document.getElementById('auto-status').textContent = '已停止';
  } else {
    btn.classList.add('active');
    document.getElementById('auto-status').textContent = '衰减中...';
    // Web Worker: 后台标签页不节流
    const blob = new Blob(['setInterval(function(){postMessage("tick")},600)']);
    _decayWorker = new Worker(URL.createObjectURL(blob));
    _decayWorker.onmessage = () => sendSilent();
  }
}

async function sendSilent() {
  frameCount++;
  document.getElementById('frame-counter').textContent = '帧 #' + frameCount;
  try {
    const res = await fetch('/silent', { method: 'POST' });
    const data = await res.json();
    render(data, '(衰减帧 #' + frameCount + ')');
  } catch(e) {}
}

async function send(text) {
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  frameCount = 0;
  if (text) document.getElementById('frame-counter').textContent = '';
  try {
    const res = await fetch('/output', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if (data.error) { document.getElementById('results').innerHTML = '<div class="card" style="color:#ff6b6b">⚠ 服务端错误: ' + data.error + '</div>'; btn.disabled = false; return; }
    render(data, text);
  } catch(e) {
    document.getElementById('results').innerHTML = '<div class="card" style="color:#ff6b6b">错误: ' + e.message + '</div>';
  }
  btn.disabled = false;
}

const CN = {
  'tense':'紧张','orient':'好奇','approach':'关切','relax':'放松',
  'angry':'生气','sad':'难过','happy':'开心','scared':'害怕',
  'surprised':'惊讶','disgust':'厌恶','contempt':'轻蔑','neutral':'中性',
  'hurt':'受伤','shy':'害羞','worried':'担心',
  'amused':'被逗乐','care':'关心','fear':'恐惧',
  'face_tense':'紧张','face_orient':'好奇','face_approach':'关切',
  'face_angry':'生气','face_sad':'难过','face_happy':'开心',
  'face_scared':'害怕','face_surprised':'惊讶','face_disgust':'厌恶',
  'face_contempt':'轻蔑','face_hurt':'受伤',
  'face_shy':'害羞','face_worried':'担心','face_amused':'被逗乐',
  'face_care':'关心','face_relax':'放松',
};

function renderBar(label, val, className) {
  const display = CN[label] || label;
  const pct = Math.min(100, Math.max(0, val * 100));
  return '<div class="bar-row">'
    + '<span class="bar-label">' + display + '</span>'
    + '<div class="bar-track"><div class="bar-fill ' + className + '" style="width:' + pct + '%"></div></div>'
    + '<span class="bar-val">' + val.toFixed(3) + '</span>'
    + '</div>';
}

function renderTag(label, val, className) {
  if (val < 0.01) return '';
  return '<span class="tag ' + className + '">' + label + ':' + val.toFixed(3) + '</span>';
}

function render(data, inputText) {
  let html = '';
  
  // 三分归属
  const s = data.scores || {我:0, 你:0, 他:0};
  const subjCls = {我:'tag-me', 你:'tag-you', 他:'tag-other'}[data.subject] || '';
  const route = {我:'SPK_OUT_SELF', 你:'SPK_OUT_YOU', 他:'SPK_OUT_OTHER(纯dist)'}[data.subject] || '?';
  const coeff = {我:'×1.8', 你:'×0.8', 他:'×0.1'}[data.subject] || '';
  html += '<div class="card"><h3>📝 输出: ' + inputText + '</h3>';
  html += '<div style="font-size:0.78em;color:#888;margin-top:4px">';
  html += '说我: <b style="color:#ff6b6b">' + s.我.toFixed(4) + '</b> &nbsp;|&nbsp;';
  html += '说你: <b style="color:#ffd93d">' + s.你.toFixed(4) + '</b> &nbsp;|&nbsp;';
  html += '说别的: <b style="color:#5b9cf5">' + s.他.toFixed(4) + '</b> &nbsp;|&nbsp;';
  html += '归属: <span class="tag ' + subjCls + '">' + data.subject + '</span>'
    + ' → ' + route + ' <b>' + coeff + '</b>'
    + ' <span style="color:#888">(×' + data.frame_eq + '帧补偿)</span>';
  html += '</div></div>';

  // Layer 0 + Layer 1 并排
  html += '<div class="columns">';

  // 转移通道 dist (衰减 VAD)
  const suppressPct = Math.min(100, Math.max(0, (1 - data.mask) * 100));
  html += '<div class="card"><h3>🔵 转移通道 dist（衰减 VAD）</h3>';
  html += '<div style="font-size:2.4em;font-weight:bold;color:#0ea5e9;font-family:monospace">' + (data.dist*100).toFixed(1) + '%</div>';
  html += '<div style="font-size:1.0em;margin-top:4px">情绪表达压制: <b style="color:#ff6b6b">' + suppressPct.toFixed(0) + '%</b></div>';
  html += '<div style="font-size:0.72em;color:#888;margin-top:6px">';
  html += '增量: ' + (data.dist_delta>0 ? ('<b style="color:#0ea5e9">+' + data.dist_delta.toFixed(3) + '</b> ' + (data.dist_src||'')) : '—');
  html += '<br>衰减半衰期: ' + data.dist_hl + 's · 当前 dist 每秒衰减 ' + (100*(1-0.5**(1/data.dist_hl))).toFixed(1) + '%';
  html += '</div></div>';

  // 检测情绪
  const spk = data.speaker || {};
  html += '<div class="card"><h3>🗣 检测情绪 (0.75过滤)</h3>';
  const spkSorted = Object.entries(spk).sort((a,b)=>b[1]-a[1]);
  if (spkSorted.length === 0) html += '<div style="color:#888;font-size:0.8em">-</div>';
  for (const [k,v] of spkSorted) html += renderBar(k, v, 'cortex-bar');
  html += '</div>';

  html += '</div>'; // end columns

  // Layer 2 VAD
  const vad_a = data.vad_amygdala || {};
  const vad_b = data.vad_cortex || {};
  html += '<div class="columns">';

  html += '<div class="card"><h3>🎭 表达压制（当前压制 <b style="color:#ff6b6b">' + suppressPct.toFixed(0) + '%</b>）</h3>';
  const FA_CH = ['hurt','sad','happy','angry','scared','shy','surprised','worried','care','contempt'];
  const faAll = Object.assign({}, Object.fromEntries(FA_CH.map(c=>[c,0])), data.face||{});
  for (const [k,v] of FA_CH.map(c=>[c,faAll[c]])) html += renderBar(k, v, 'vad-bar');
  html += '</div>';

  html += '<div class="card"><h3>📊 共享皮层池 VAD</h3>';
  const info_b = data.cortex_champion ? ' (主导: ' + (CN[data.cortex_champion] || data.cortex_champion) + ')' : '';
  html += '<div style="font-size:0.75em;color:#888;margin-bottom:6px">半衰期: 少女 1min-15min' + info_b + '</div>';
  for (const [k,v] of FA_CH.map(c=>[c,vad_b[c]||0])) html += renderBar(k, v, 'vad-bar');
  html += '</div>';

  html += '</div>'; // end columns

  // 递质池负载条
  const pools = data.pools || {};
  if (Object.keys(pools).length > 0) {
    html += '<div class="card"><h3>📦 递质池负载</h3><div style="display:flex;gap:12px">';
    const poolColors = {'DA':'#ffd93d','NE':'#ff6b6b','5HT':'#5b9cf5','OXT':'#5abe6e'};
    const poolLabels = {'DA':'多巴胺','NE':'去甲肾上腺素','5HT':'血清素','OXT':'催产素'};
    for (const [pk, pv] of Object.entries(pools)) {
      const pct = Math.min(100, Math.max(0, pv * 100));
      const color = poolColors[pk] || '#888';
      html += '<div style="flex:1;text-align:center;padding:6px;background:#1a1a2e;border-radius:8px">'
        + '<div style="font-size:0.65em;color:#aaa">' + (poolLabels[pk]||pk) + '</div>'
        + '<div style="height:8px;background:#333;border-radius:4px;margin:4px 0"><div style="width:' + pct + '%;height:100%;background:' + color + ';border-radius:4px"></div></div>'
        + '<div style="font-size:0.8em;font-weight:bold;color:' + color + '">' + pct.toFixed(0) + '%</div></div>';
    }
    html += '</div></div>';
  }

  // 🆕 表情运算层 — 上下脸分别叠加
  const expr = data.expr_layer || {};
  html += '<div class="card"><h3>🎭 表情运算层</h3>';
  if (!expr.upper || Object.keys(expr.upper.emotions||{}).length === 0) {
    html += '<div style="color:#888;font-size:0.8em">未触发</div>';
  } else {
    html += '<div style="display:flex;gap:12px">';
    // 上脸
    html += '<div style="flex:1;padding:10px;background:#1a1a3d;border-radius:8px">'
      + '<b style="color:#ffd93d">⬆ 上脸 (眉/眼)</b>';
    for (const [emo, val] of Object.entries(expr.upper.emotions||{})) {
      const src = (expr.upper.sources||{})[emo] || {};
      const barW = Math.min(100, val * 100);
      html += '<div style="margin-top:8px;padding:6px;background:rgba(255,255,255,0.05);border-radius:6px">'
        + '<div style="display:flex;justify-content:space-between;font-size:0.85em">'
        + '<span style="color:#fff;font-weight:bold">' + (CN[emo]||emo) + '</span>'
        + '<span style="color:#ffd93d;font-weight:bold">' + val.toFixed(3) + '</span></div>'
        + '<div style="height:6px;background:#333;border-radius:3px;margin:4px 0"><div style="width:' + barW +'%;height:100%;background:#ffd93d;border-radius:3px"></div></div>'
        + '<div style="display:flex;gap:8px;font-size:0.72em">'
        + '<span style="color:#ff6b6b">🧠杏仁核 ' + (src['杏仁核'] ? src['杏仁核'].toFixed(3) : '—') + '</span>'
        + '<span style="color:#5b9cf5">🗣皮层 ' + (src['皮层'] ? src['皮层'].toFixed(3) : '—') + '</span>'
        + '</div></div>';
    }
    html += '</div>';
    // 下脸
    html += '<div style="flex:1;padding:10px;background:#1a2a2a;border-radius:8px">'
      + '<b style="color:#5abe6e">⬇ 下脸 (嘴)</b>';
    for (const [emo, val] of Object.entries(expr.lower.emotions||{})) {
      const src = (expr.lower.sources||{})[emo] || {};
      const barW = Math.min(100, val * 100);
      html += '<div style="margin-top:8px;padding:6px;background:rgba(255,255,255,0.05);border-radius:6px">'
        + '<div style="display:flex;justify-content:space-between;font-size:0.85em">'
        + '<span style="color:#fff;font-weight:bold">' + (CN[emo]||emo) + '</span>'
        + '<span style="color:#5abe6e;font-weight:bold">' + val.toFixed(3) + '</span></div>'
        + '<div style="height:6px;background:#333;border-radius:3px;margin:4px 0"><div style="width:' + barW +'%;height:100%;background:#5abe6e;border-radius:3px"></div></div>'
        + '<div style="display:flex;gap:8px;font-size:0.72em">'
        + '<span style="color:#ff6b6b">🧠杏仁核 ' + (src['杏仁核'] ? src['杏仁核'].toFixed(3) : '—') + '</span>'
        + '<span style="color:#5b9cf5">🗣皮层 ' + (src['皮层'] ? src['皮层'].toFixed(3) : '—') + '</span>'
        + '</div></div>';
    }
    html += '</div>';
    html += '</div>';
  }
  html += '</div>';

  // 🆕 VAD ratio (norm^0.5)
  if (data.ratio_upper !== undefined) {
    html += '<div class="card"><h3>📏 VAD ratio (norm^0.5)</h3>'
      + '<div style="display:flex;gap:16px;padding:8px 0;text-align:center">'
      + '<div style="flex:1;padding:8px;background:#1a1a3d;border-radius:8px">'
      + '<b style="color:#ffd93d">⬆ 上脸</b><br>'
      + '<span style="font-size:1.1em;color:#0af">' + (data.ratio_upper_name||'?') + '</span><br>'
      + '<b style="font-size:1.3em;color:#ffd93d">' + (data.ratio_upper||0).toFixed(4) + '</b>'
      + ' <span style="color:#888">(' + ((data.ratio_upper||0)*100).toFixed(1) + '%)</span></div>'
      + '<div style="flex:1;padding:8px;background:#1a2a2a;border-radius:8px">'
      + '<b style="color:#5abe6e">⬇ 下脸</b><br>'
      + '<span style="font-size:1.1em;color:#0af">' + (data.ratio_lower_name||'?') + '</span><br>'
      + '<b style="font-size:1.3em;color:#ffd93d">' + (data.ratio_lower||0).toFixed(4) + '</b>'
      + ' <span style="color:#888">(' + ((data.ratio_lower||0)*100).toFixed(1) + '%)</span></div></div></div>';
  }
  
  // 当前所有 VAD 值汇总
  const rawData = data.raw || {};
  if (Object.keys(rawData).length > 0 && rawData.amygdala) {
    html += '<div style="margin-top:8px;font-size:0.7em;color:#555;display:flex;flex-wrap:wrap;gap:4px">';
    for (const [s, vals] of Object.entries(rawData)) {
      for (const [ch, val] of Object.entries(vals)) {
        if (val > 0.01) {
          const tagClass = s === 'amygdala' ? 'tag-tense' : 'tag-relax';
          html += '<span class="tag ' + tagClass + '">' + ch + ':' + val.toFixed(3) + '</span>';
        }
      }
    }
    html += '</div>';
  }

  document.getElementById('results').innerHTML = html;
}

document.getElementById('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitText();
});

// ── 常驻显示: 自动启动 600ms 轮询 /silent (池状态 + dist + 压制 实时刷新) ──
toggleAuto();



</script>
</body>
</html>"""

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FACS 情绪引擎 v2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh}
.container{max-width:1000px;margin:0 auto;padding:20px}
h1{text-align:center;color:#e94560;margin-bottom:4px;font-size:1.5em}
.sub{text-align:center;color:#888;margin-bottom:20px;font-size:0.85em}
.input-row{display:flex;gap:10px;margin-bottom:12px}
#text-input{flex:1;padding:12px 16px;border:1px solid #333;border-radius:8px;background:#16213e;color:#eee;font-size:15px;outline:none}
#text-input:focus{border-color:#e94560}
#submit-btn{padding:12px 24px;border:none;border-radius:8px;background:#e94560;color:#fff;font-size:15px;cursor:pointer}
#submit-btn:hover{background:#d63851}
#submit-btn:disabled{opacity:0.5;cursor:not-allowed}
.btn-row{display:flex;gap:6px;margin-bottom:20px;flex-wrap:wrap;align-items:center}
.btn-row button{padding:7px 14px;border:1px solid #444;border-radius:6px;background:#16213e;color:#aaa;cursor:pointer;font-size:12px}
.btn-row button:hover{border-color:#e94560;color:#eee}
.btn-row button.active{background:#e94560;color:#fff;border-color:#e94560}
#auto-status{margin-left:10px;font-size:0.8em;color:#888}
.frame-counter{font-size:0.75em;color:#888;margin-left:8px}
#results{display:flex;flex-direction:column;gap:10px}
.card{background:#16213e;border:1px solid #2a2a4a;border-radius:10px;padding:14px}
.card h3{color:#e94560;font-size:0.85em;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
.bar-row{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.bar-label{width:85px;font-size:0.78em;color:#aaa;text-align:right}
.bar-track{flex:1;height:16px;background:#0f0f23;border-radius:3px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:3px;transition:width 0.3s}
.bar-val{width:42px;font-size:0.72em;color:#888;text-align:right}
.amygdala-bar{background:linear-gradient(90deg,#ff6b35,#ff8c42)}
.cortex-bar{background:linear-gradient(90deg,#0f3460,#1a56db)}
.au-bar{background:linear-gradient(90deg,#e94560,#ff6b6b)}
.vad-bar{background:linear-gradient(90deg,#2d6a4f,#52b788)}
.params-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:4px}
.param-item{display:flex;justify-content:space-between;font-size:0.75em;padding:3px 6px;background:#0f0f23;border-radius:3px}
.param-name{color:#aaa}
.param-val{color:#ff6b6b;font-weight:bold}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.7em;margin:2px;font-weight:bold}
.tag-tense{background:rgba(255,107,53,0.2);color:#ff8c42}
.tag-orient{background:rgba(66,133,244,0.2);color:#5b9cf5}
.tag-approach{background:rgba(52,168,83,0.2);color:#5abe6e}
.tag-relax{background:rgba(233,69,96,0.2);color:#ff6b6b}
.tag-attack{background:#e94560;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.7em;font-weight:bold}
.tag-mixed{background:#ffd93d;color:#1a1a2e;padding:2px 8px;border-radius:8px;font-size:0.7em;font-weight:bold}
.champion-badge{background:#e94560;color:#fff;font-size:0.65em;padding:1px 6px;border-radius:8px;margin-left:4px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:600px){.columns{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<h1>FACS 情绪引擎 v2</h1>
<div class="sub">杏仁核 → 皮层 → VAD → AU 完整链路</div>

<div class="input-row">
  <input id="text-input" placeholder="输入文字, 按回车提交 (如: 你怎么这么笨啊)" autofocus>
  <button id="submit-btn" onclick="submitText()">测试</button>
</div>

<div id="live-bar" style="display:none;padding:8px 14px;background:#0a2a1a;border:1px solid #2a8;border-radius:8px;margin-bottom:12px;font-size:14px;color:#5abe6e">
  🎤 <span id="live-text">等待语音...</span>
  &nbsp;|&nbsp;情绪: <b id="live-emo">-</b> r=<span id="live-ratio">-</span>
</div>

<div class="btn-row">
  <button onclick="submitTextPreset('你好啊')">你好啊</button>
  <button onclick="submitTextPreset('你怎么这么笨啊')">你笨蛋</button>
  <button onclick="submitTextPreset('对不起')">道歉</button>
  <button onclick="submitTextPreset('哈哈哈开玩笑的')">开玩笑</button>
  <button onclick="submitTextPreset('你好可爱')">夸奖</button>
  <button onclick="submitTextPreset('我好害怕')">恐惧</button>
  <button onclick="submitTextPreset('我好难过')">我难过</button>
  <button onclick="submitTextPreset('你太厉害了')">厉害</button>
  <button id="auto-btn" onclick="toggleAuto()">⏯ 衰减模式</button>
  <span id="auto-status"></span>
  <span class="frame-counter" id="frame-counter"></span>
  <button onclick="resetEngine()">🔄 重置</button>
</div>

<div id="results"></div>
</div>

<script>
let autoTimer = null;
let _decayWorker = null;
let frameCount = 0;

async function submitText() {
  const text = document.getElementById('text-input').value.trim();
  if (!text) return;
  await send(text);
}

async function submitTextPreset(text) {
  document.getElementById('text-input').value = text;
  await send(text);
}

async function resetEngine() {
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  if (_decayWorker) { _decayWorker.terminate(); _decayWorker = null; }
  document.getElementById('auto-btn').classList.remove('active');
  document.getElementById('auto-status').textContent = '';
  frameCount = 0;
  document.getElementById('frame-counter').textContent = '';
  await fetch('/reset', { method: 'POST' });
  document.getElementById('results').innerHTML = '';
  document.getElementById('text-input').value = '';
  toggleAuto();   // 重置后保持常驻显示
}

function toggleAuto() {
  const btn = document.getElementById('auto-btn');
  if (_decayWorker) {
    _decayWorker.terminate();
    _decayWorker = null;
    btn.classList.remove('active');
    document.getElementById('auto-status').textContent = '已停止';
  } else {
    btn.classList.add('active');
    document.getElementById('auto-status').textContent = '衰减中...';
    // Web Worker: 后台标签页不节流
    const blob = new Blob(['setInterval(function(){postMessage("tick")},600)']);
    _decayWorker = new Worker(URL.createObjectURL(blob));
    _decayWorker.onmessage = () => sendSilent();
  }
}

async function sendSilent() {
  frameCount++;
  document.getElementById('frame-counter').textContent = '帧 #' + frameCount;
  try {
    const res = await fetch('/silent', { method: 'POST' });
    const data = await res.json();
    render(data, '(衰减帧 #' + frameCount + ')');
  } catch(e) {}
}

async function send(text) {
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  frameCount = 0;
  if (text) document.getElementById('frame-counter').textContent = '';
  try {
    const res = await fetch('/facs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    render(data, text);
  } catch(e) {
    document.getElementById('results').innerHTML = '<div class="card" style="color:#ff6b6b">错误: ' + e.message + '</div>';
  }
  btn.disabled = false;
}

const CN = {
  'tense':'紧张','orient':'好奇','approach':'关切','relax':'放松',
  'angry':'生气','sad':'难过','happy':'开心','scared':'害怕',
  'surprised':'惊讶','disgust':'厌恶','contempt':'轻蔑','neutral':'中性',
  'hurt':'受伤','shy':'害羞','worried':'担心',
  'amused':'被逗乐','care':'关心','fear':'恐惧',
  'face_tense':'紧张','face_orient':'好奇','face_approach':'关切',
  'face_angry':'生气','face_sad':'难过','face_happy':'开心',
  'face_scared':'害怕','face_surprised':'惊讶','face_disgust':'厌恶',
  'face_contempt':'轻蔑','face_hurt':'受伤',
  'face_shy':'害羞','face_worried':'担心','face_amused':'被逗乐',
  'face_care':'关心','face_relax':'放松',
};

function renderBar(label, val, className) {
  const display = CN[label] || label;
  const pct = Math.min(100, Math.max(0, val * 100));
  return '<div class="bar-row">'
    + '<span class="bar-label">' + display + '</span>'
    + '<div class="bar-track"><div class="bar-fill ' + className + '" style="width:' + pct + '%"></div></div>'
    + '<span class="bar-val">' + val.toFixed(3) + '</span>'
    + '</div>';
}

function renderTag(label, val, className) {
  if (val < 0.01) return '';
  return '<span class="tag ' + className + '">' + label + ':' + val.toFixed(3) + '</span>';
}

function render(data, inputText) {
  let html = '';
  
  // 输入 + 方向分
  const dirY = data.dir_you || 0;
  const dirO = data.dir_other || 0;
  const winner = data.dir_winner || '?';
  html += '<div class="card"><h3>📝 输入: ' + inputText + '</h3>';
  html += '<div style="font-size:0.78em;color:#888;margin-top:4px">';
  html += '说你: <b style="color:#ff8c42">' + dirY.toFixed(4) + '</b> &nbsp;|&nbsp;';
  html += '说别的: <b style="color:#5b9cf5">' + dirO.toFixed(4) + '</b> &nbsp;|&nbsp;';
  html += '胜出: <b style="color:' + (winner === '说你' ? '#ff8c42' : '#5b9cf5') + '">' + winner + '</b>';
  html += ' → 递质系数: ×' + (winner === '说你' ? '1.4' : '0.8');
  html += '</div></div>';

  // Layer 0 + Layer 1 并排
  html += '<div class="columns">';

  // 杏仁核
  const amy = data.amygdala || {};
  html += '<div class="card"><h3>🧠 杏仁核 (50ms)</h3>';
  const ch_amy = data.amygdala_champion || '';
  const amySorted = Object.entries(amy).sort((a,b)=>b[1]-a[1]);
  if (amySorted.length === 0) html += '<div style="color:#888;font-size:0.8em">-</div>';
  for (const [k,v] of amySorted) {
    html += renderBar(k + ((k === ch_amy) ? ' 👑' : ''), v, 'amygdala-bar');
  }
  html += '</div>';

  // 说话者情绪
  const spk = data.speaker || {};
  html += '<div class="card"><h3>🗣 说话者 (皮层)</h3>';
  const spkSorted = Object.entries(spk).sort((a,b)=>b[1]-a[1]);
  if (spkSorted.length === 0) html += '<div style="color:#888;font-size:0.8em">-</div>';
  for (const [k,v] of spkSorted) html += renderBar(k, v, 'cortex-bar');
  html += '</div>';

  html += '</div>'; // end columns

  // Layer 2 VAD
  const vad_a = data.vad_amygdala || {};
  const vad_b = data.vad_cortex || {};
  html += '<div class="columns">';

  html += '<div class="card"><h3>📊 杏仁核VAD</h3>';
  const info_a = data.amygdala_champion ? ' (主导: ' + (CN[data.amygdala_champion] || data.amygdala_champion) + ')' : '';
  html += '<div style="font-size:0.75em;color:#888;margin-bottom:6px">半衰期: 紧张0.3s / 好奇2.5s / 关切1.0s / 放松2.0s' + info_a + '</div>';
  const vaSorted = Object.entries(vad_a).filter(([k,v])=>v>0.003).sort((a,b)=>b[1]-a[1]);
  if (vaSorted.length === 0) html += '<div style="color:#888;font-size:0.8em">-</div>';
  for (const [k,v] of vaSorted) html += renderBar(k, v, 'vad-bar');
  html += '</div>';

  html += '<div class="card"><h3>📊 皮层VAD</h3>';
  const info_b = data.cortex_champion ? ' (主导: ' + (CN[data.cortex_champion] || data.cortex_champion) + ')' : '';
  html += '<div style="font-size:0.75em;color:#888;margin-bottom:6px">半衰期: 少女 1min-15min' + info_b + '</div>';
  const vbSorted = Object.entries(vad_b).filter(([k,v])=>v>0.003).sort((a,b)=>b[1]-a[1]);
  if (vbSorted.length === 0) html += '<div style="color:#888;font-size:0.8em">-</div>';
  for (const [k,v] of vbSorted) html += renderBar(k, v, 'vad-bar');
  html += '</div>';

  html += '</div>'; // end columns

  // 递质池负载条
  const pools = data.pools || {};
  if (Object.keys(pools).length > 0) {
    html += '<div class="card"><h3>📦 递质池负载</h3><div style="display:flex;gap:12px">';
    const poolColors = {'DA':'#ffd93d','NE':'#ff6b6b','5HT':'#5b9cf5','OXT':'#5abe6e'};
    const poolLabels = {'DA':'多巴胺','NE':'去甲肾上腺素','5HT':'血清素','OXT':'催产素'};
    for (const [pk, pv] of Object.entries(pools)) {
      const pct = Math.min(100, Math.max(0, pv * 100));
      const color = poolColors[pk] || '#888';
      html += '<div style="flex:1;text-align:center;padding:6px;background:#1a1a2e;border-radius:8px">'
        + '<div style="font-size:0.65em;color:#aaa">' + (poolLabels[pk]||pk) + '</div>'
        + '<div style="height:8px;background:#333;border-radius:4px;margin:4px 0"><div style="width:' + pct + '%;height:100%;background:' + color + ';border-radius:4px"></div></div>'
        + '<div style="font-size:0.8em;font-weight:bold;color:' + color + '">' + pct.toFixed(0) + '%</div></div>';
    }
    html += '</div></div>';
  }

  // 🆕 表情运算层 — 上下脸分别叠加
  const expr = data.expr_layer || {};
  html += '<div class="card"><h3>🎭 表情运算层</h3>';
  if (!expr.upper || Object.keys(expr.upper.emotions||{}).length === 0) {
    html += '<div style="color:#888;font-size:0.8em">未触发</div>';
  } else {
    html += '<div style="display:flex;gap:12px">';
    // 上脸
    html += '<div style="flex:1;padding:10px;background:#1a1a3d;border-radius:8px">'
      + '<b style="color:#ffd93d">⬆ 上脸 (眉/眼)</b>';
    for (const [emo, val] of Object.entries(expr.upper.emotions||{})) {
      const src = (expr.upper.sources||{})[emo] || {};
      const barW = Math.min(100, val * 100);
      html += '<div style="margin-top:8px;padding:6px;background:rgba(255,255,255,0.05);border-radius:6px">'
        + '<div style="display:flex;justify-content:space-between;font-size:0.85em">'
        + '<span style="color:#fff;font-weight:bold">' + (CN[emo]||emo) + '</span>'
        + '<span style="color:#ffd93d;font-weight:bold">' + val.toFixed(3) + '</span></div>'
        + '<div style="height:6px;background:#333;border-radius:3px;margin:4px 0"><div style="width:' + barW +'%;height:100%;background:#ffd93d;border-radius:3px"></div></div>'
        + '<div style="display:flex;gap:8px;font-size:0.72em">'
        + '<span style="color:#ff6b6b">🧠杏仁核 ' + (src['杏仁核'] ? src['杏仁核'].toFixed(3) : '—') + '</span>'
        + '<span style="color:#5b9cf5">🗣皮层 ' + (src['皮层'] ? src['皮层'].toFixed(3) : '—') + '</span>'
        + '</div></div>';
    }
    html += '</div>';
    // 下脸
    html += '<div style="flex:1;padding:10px;background:#1a2a2a;border-radius:8px">'
      + '<b style="color:#5abe6e">⬇ 下脸 (嘴)</b>';
    for (const [emo, val] of Object.entries(expr.lower.emotions||{})) {
      const src = (expr.lower.sources||{})[emo] || {};
      const barW = Math.min(100, val * 100);
      html += '<div style="margin-top:8px;padding:6px;background:rgba(255,255,255,0.05);border-radius:6px">'
        + '<div style="display:flex;justify-content:space-between;font-size:0.85em">'
        + '<span style="color:#fff;font-weight:bold">' + (CN[emo]||emo) + '</span>'
        + '<span style="color:#5abe6e;font-weight:bold">' + val.toFixed(3) + '</span></div>'
        + '<div style="height:6px;background:#333;border-radius:3px;margin:4px 0"><div style="width:' + barW +'%;height:100%;background:#5abe6e;border-radius:3px"></div></div>'
        + '<div style="display:flex;gap:8px;font-size:0.72em">'
        + '<span style="color:#ff6b6b">🧠杏仁核 ' + (src['杏仁核'] ? src['杏仁核'].toFixed(3) : '—') + '</span>'
        + '<span style="color:#5b9cf5">🗣皮层 ' + (src['皮层'] ? src['皮层'].toFixed(3) : '—') + '</span>'
        + '</div></div>';
    }
    html += '</div>';
    html += '</div>';
  }
  html += '</div>';

  // 🆕 VAD ratio (norm^0.5)
  if (data.ratio_upper !== undefined) {
    html += '<div class="card"><h3>📏 VAD ratio (norm^0.5)</h3>'
      + '<div style="display:flex;gap:16px;padding:8px 0;text-align:center">'
      + '<div style="flex:1;padding:8px;background:#1a1a3d;border-radius:8px">'
      + '<b style="color:#ffd93d">⬆ 上脸</b><br>'
      + '<span style="font-size:1.1em;color:#0af">' + (data.ratio_upper_name||'?') + '</span><br>'
      + '<b style="font-size:1.3em;color:#ffd93d">' + (data.ratio_upper||0).toFixed(4) + '</b>'
      + ' <span style="color:#888">(' + ((data.ratio_upper||0)*100).toFixed(1) + '%)</span></div>'
      + '<div style="flex:1;padding:8px;background:#1a2a2a;border-radius:8px">'
      + '<b style="color:#5abe6e">⬇ 下脸</b><br>'
      + '<span style="font-size:1.1em;color:#0af">' + (data.ratio_lower_name||'?') + '</span><br>'
      + '<b style="font-size:1.3em;color:#ffd93d">' + (data.ratio_lower||0).toFixed(4) + '</b>'
      + ' <span style="color:#888">(' + ((data.ratio_lower||0)*100).toFixed(1) + '%)</span></div></div></div>';
  }
  
  // 当前所有 VAD 值汇总
  const rawData = data.raw || {};
  if (Object.keys(rawData).length > 0 && rawData.amygdala) {
    html += '<div style="margin-top:8px;font-size:0.7em;color:#555;display:flex;flex-wrap:wrap;gap:4px">';
    for (const [s, vals] of Object.entries(rawData)) {
      for (const [ch, val] of Object.entries(vals)) {
        if (val > 0.01) {
          const tagClass = s === 'amygdala' ? 'tag-tense' : 'tag-relax';
          html += '<span class="tag ' + tagClass + '">' + ch + ':' + val.toFixed(3) + '</span>';
        }
      }
    }
    html += '</div>';
  }

  document.getElementById('results').innerHTML = html;
}

document.getElementById('text-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitText();
});

// ── 语音输入轮询 ──
setInterval(async () => {
  try {
    const res = await fetch('/latest');
    const d = await res.json();
    if (!d || !d.text) return;
    document.getElementById('live-bar').style.display = 'block';
    document.getElementById('live-text').textContent = d.text;
    document.getElementById('live-emo').textContent = d.emotion || '-';
    document.getElementById('live-ratio').textContent = (d.ratio||0).toFixed(4);
    if (d.full) render(d.full, '🎤 ' + d.text);
  } catch(_) {}
}, 300);

// ── 常驻显示: 自动启动 600ms 轮询 /silent (池状态实时刷新) ──
toggleAuto();
</script>
</body>
</html>"""


class FACSHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            # 壳页面: 顶部切换 输入侧/输出侧 (共用同一个 dv)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(SHELL_HTML.encode())
        elif self.path == '/view_input':
            # 输入侧 (倾听) 调试页
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path in ('/view_output', '/out'):
            # 输出侧 (说话) 调试页 — 与输入侧共享同一个 dv
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(OUT_HTML.encode())
        elif self.path == '/latest':
            self._json(_latest_info)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global dv
        try:
            if self.path == '/facs':
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode('utf-8', errors='replace'))
                text = body.get('text', '').strip()

                if not text:
                    self._json({'error': 'empty text'})
                    return

                self._process_frame(text)

            elif self.path == '/output':
                # 输出侧: LLM 文本 → output_engine → 写进同一个 dv (vad 共享)
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode('utf-8', errors='replace'))
                text = body.get('text', '').strip()

                if not text:
                    self._json({'error': 'empty text'})
                    return

                self._process_output_frame(text)

            elif self.path == '/silent':
                self._process_silent_frame()

            elif self.path == '/reset':
                dv = DualVAD()
                output_engine.reset()   # dist 同步重置
                self._json({'ok': True})

            else:
                self.send_response(404)
                self.end_headers()

        except Exception as e:
            self._json({'error': str(e)})

    def _process_frame(self, text):
        global dv, _latest_info
        _latest_info["text"] = text
        # Layer 0: 杏仁核
        amygdala = detect_amygdala_signals(text)
        # Layer 1: 皮层 (说话者情绪)
        speaker = detect_speaker_emotion(text)
        
        # 方向检测: 冲我来的?
        atk, dir_you_sim, dir_other_sim = is_about_you(text)

        # 你说我 (atk=True): 用 told-me 原型组直接算"我的情绪", 1:1 进 VAD (无映射表无叠加)
        # 说别的 (atk=False): 老流程 — 说话者情绪 → SPK_TO_L_NORMAL 传染表
        if atk:
            speaker = detect_told_me_emotion(text)
            filtered = {k: v for k, v in speaker.items() if v > 0}
            listener = filtered
        else:
            speaker = detect_speaker_emotion(text)
            # 说话者情绪 → 听者感受 (过滤: 只留 >= 最强*0.75 的情绪)
            spk_items = sorted(speaker.items(), key=lambda x: -x[1])
            if spk_items:
                max_spk = spk_items[0][1]
                threshold = max_spk * 0.75
                filtered = {k: v for k, v in spk_items if v >= threshold}
            else:
                filtered = {}
            listener = {}
            for emo, intensity in filtered.items():
                if emo in SPK_TO_L_NORMAL:
                    for lemo, ratio in SPK_TO_L_NORMAL[emo].items():
                        listener[lemo] = round(listener.get(lemo, 0) + intensity * ratio, 4)
            for k in listener:
                listener[k] = min(1.0, listener[k])
        
        # Layer 2: VAD (输入听者感受)
        direction = "你" if atk else "别的"
        dv.update(amygdala_signals=amygdala, cortex_signals=listener, direction=direction)
        state = dv.get_state()
        face = dv.merged_for_face()
        comp = dv.get_expression_competition()
        au = map_emotions_to_au(face, comp_result=comp)
        expr_layer = dv.get_expression_layer()

        # 递质池负载
        pools = {
            'DA': round(dv.pool_sum(dv.b_channels, 'DA'), 3),
            'NE': round(dv.pool_sum(dv.b_channels, 'NE'), 3),
            '5HT': round(dv.pool_sum(dv.b_channels, '5HT'), 3),
            'OXT': round(dv.pool_sum(dv.b_channels, 'OXT'), 3),
        }
        ratio_u_cn, ratio_u_val, ratio_l_cn, ratio_l_val = compute_ratio(expr_layer)
        _latest_info["emotion"] = ratio_u_cn
        _latest_info["ratio"] = ratio_u_val
        _threading.Thread(target=lambda: _push_bridge(ratio_u_cn, ratio_u_val, dist=output_engine.dist_val), daemon=True).start()

        self._json({
            'text': text,
            'dir_you': dir_you_sim,
            'dir_other': dir_other_sim,
            'pools': pools,
            'dir_winner': '说你' if atk else '说别的',
            'amygdala': amygdala,
            'amygdala_champion': max(state['amygdala'], key=state['amygdala'].get) if state['amygdala'] else '',
            'speaker': speaker,
            'vad_amygdala': {k: round(v, 4) for k, v in state['amygdala'].items()},
            'vad_cortex': {k: round(v, 4) for k, v in state['cortex'].items()},
            'cortex_champion': max(state['cortex'], key=state['cortex'].get) if state['cortex'] else '',
            'au': au,
            'comp_amygdala': comp['amygdala'],
            'comp_cortex': comp['cortex'],
            'expr_layer': expr_layer,
            'ratio_upper_name': ratio_u_cn,
            'ratio_upper': ratio_u_val,
            'ratio_lower_name': ratio_l_cn,
            'ratio_lower': ratio_l_val,
            'raw': {
                'amygdala': {k: round(v, 4) for k, v in state['amygdala'].items()},
                'cortex': {k: round(v, 4) for k, v in state['cortex'].items()},
            }
        })

    def _process_output_frame(self, text):
        """输出侧帧: LLM 文本 → output_engine → 写进同一个 dv (共享皮层池)"""
        global dv
        out = output_engine.process(text, dv)
        # ratio 计算 (复用本文件的 compute_ratio; expr_layer 已含 dist 压制)
        ratio_u_cn, ratio_u_val, ratio_l_cn, ratio_l_val = compute_ratio(out.get('expr_layer', {}))
        out['ratio_upper_name'] = ratio_u_cn
        out['ratio_upper'] = ratio_u_val
        out['ratio_lower_name'] = ratio_l_cn
        out['ratio_lower'] = ratio_l_val
        # 说话时也上桥 (带 dist + 压制): 前端 2D 拿实时 dist
        _threading.Thread(target=lambda: _push_bridge(
            ratio_u_cn, ratio_u_val,
            suppress=round(1 - out.get('mask', 1.0), 4),
            dist=out.get('dist', 0.0)), daemon=True).start()
        self._json(out)

    def _process_silent_frame(self):
        global dv
        dv.update(amygdala_signals={}, cortex_signals={})
        output_engine.silent_tick()   # dist 也随时间衰减 (输出页衰减模式)
        state = dv.get_state()
        face = dv.merged_for_face()
        # dist 表达压制 (与输出侧同款; 池不动)
        mask = max(output_engine.MASK_FLOOR, 1 - min(1.0, output_engine.dist_val * output_engine.DIST_SUPPRESS_K))
        face = {k: v * mask for k, v in face.items()}
        comp = dv.get_expression_competition()
        au = map_emotions_to_au(face, comp_result=comp)
        expr_layer = dv.get_expression_layer()
        pools = {
            'DA': round(dv.pool_sum(dv.b_channels, 'DA'), 3),
            'NE': round(dv.pool_sum(dv.b_channels, 'NE'), 3),
            '5HT': round(dv.pool_sum(dv.b_channels, '5HT'), 3),
            'OXT': round(dv.pool_sum(dv.b_channels, 'OXT'), 3),
        }
        ratio_u_cn, ratio_u_val, ratio_l_cn, ratio_l_val = compute_ratio(expr_layer)
        _latest_info["emotion"] = ratio_u_cn
        _latest_info["ratio"] = ratio_u_val
        _threading.Thread(target=lambda: _push_bridge(ratio_u_cn, ratio_u_val, suppress=round(1-mask, 4), dist=output_engine.dist_val), daemon=True).start()

        self._json({
            'amygdala': {},
            'amygdala_champion': max(state['amygdala'], key=state['amygdala'].get) if max(state['amygdala'].values()) > 0 else '',
            'speaker': {},
            'dist': output_engine.dist_val,
            'dist_hl': output_engine.dist_hl_cur,
            'dist_delta': 0.0,
            'dist_src': '',
            'frame_eq': output_engine.OUT_FRAME_EQ,
            'mask': mask,
            'face': face,
            'vad_amygdala': {k: round(v, 4) for k, v in state['amygdala'].items()},
            'vad_cortex': {k: round(v, 4) for k, v in state['cortex'].items()},
            'cortex_champion': max(state['cortex'], key=state['cortex'].get) if max(state['cortex'].values()) > 0 else '',
            'au': au,
            'pools': pools,
            'comp_amygdala': comp['amygdala'],
            'comp_cortex': comp['cortex'],
            'expr_layer': expr_layer,
            'ratio_upper_name': ratio_u_cn,
            'ratio_upper': ratio_u_val,
            'ratio_lower_name': ratio_l_cn,
            'ratio_lower': ratio_l_val,
            'raw': {
                'amygdala': {k: round(v, 4) for k, v in state['amygdala'].items()},
                'cortex': {k: round(v, 4) for k, v in state['cortex'].items()},
            }
        })

    def _json(self, data):
        if data is not _latest_info:
            _latest_info["full"] = data
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(_sanitize_json(data), ensure_ascii=False).encode())


def _sanitize_json(obj):
    """递归清洗 inf/nan → 0: Python json.dumps 默认输出 Infinity/NaN 字面量,
    不是合法 JSON, 浏览器 JSON.parse 会崩 (如 ratio 唯一情绪时的 inf)"""
    import math
    if isinstance(obj, float):
        return 0.0 if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


if __name__ == '__main__':
    port = 18768
    print(f'FACS 引擎 v2: http://localhost:{port}')
    print('Ctrl+C 停止')
    ThreadingHTTPServer(('0.0.0.0', port), FACSHandler).serve_forever()
