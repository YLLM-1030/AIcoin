"""
output_vad_debug.py — 输出演绎引擎调试页 (facs_server.py 复制改造)
对应: 杏仁核→转移dist · 杏仁核VAD→表达压制 · 说话者→检测情绪 · 皮层VAD→共享池
用法: python output_vad_debug.py
浏览器打开 http://localhost:18773
"""
import json, sys, os, math, time as _time, urllib.request as _ureq, threading as _threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facs_engine import detect_speaker_emotion, get_embed_model
from vad_engine import DualVAD, map_emotions_to_au, VISIBILITY_THRESHOLD
import vad_engine
import numpy as np

# ── 输出侧方向系数 (patch 内存, 不动 vad_engine.py; 输入侧"你1.4/别的0.8"不受影响) ──
vad_engine.DIRECTION_COEFF["out_me"] = 1.8
vad_engine.DIRECTION_COEFF["out_you"] = 0.8
vad_engine.DIRECTION_COEFF["out_other"] = 0.1
vad_engine.DIRECTION_DECAY["out_me"] = 1.0
vad_engine.DIRECTION_DECAY["out_you"] = 1.0
vad_engine.DIRECTION_DECAY["out_other"] = 1.0

OUT_FRAME_EQ = 4   # 临时平衡系数: 输出侧无流式帧率, 一句话等效输入侧4帧

# ═══ 输出侧三池例句 ═══
ABOUT_ME = [
    "我好难过", "我很难过", "我太伤心了", "我哭了",
    "我好开心", "我太高兴了", "我心情很好",
    "我好累", "我困了", "我不舒服", "我生病了",
    "我好害怕", "我有点紧张", "我不安",
    "我好烦", "我烦死了", "我讨厌这个", "我受不了了",
    "我生气了", "我生气了哦",
    "那件事让我很难过", "这件事让我好难过", "这让我很伤心",
    "这让我很生气", "真是气死我了", "快把我气死了", "气得我直跺脚",
    "你让我好失望", "你让我很生气",
    "这让我好害怕", "把我吓了一大跳",
    "这让我好开心", "把我高兴坏了", "开心死我了",
    "这个消息让我好难过", "一想到我就难过",
    "我喜欢", "我最喜欢", "我想", "我想要", "我希望", "我觉得",
    "我打算", "我要去", "我在做",
    "我喜欢你", "我好喜欢你", "我超喜欢你", "我最喜欢你", "我才没有喜欢你",
    "我讨厌你", "我烦你", "我不想理你",
    "我相信你", "我担心你", "我心疼你", "我爱你",
]
ABOUT_YOU_OUT = [
    "你笨蛋", "你真烦", "你太过分了", "你怎么这样",
    "你走开", "你什么都做不好",
    "你好可爱", "你真棒", "你太厉害了", "你做得很好",
    "你好温柔", "你真是个小天使",
    "你要小心", "你吃饭了吗", "你能不能帮我", "你觉得呢",
    "你别跟我说话", "你离我远点",
    "你好难过", "你很难过", "你好开心", "你高兴吗",
    "你害怕了", "你生气了", "你是不是不开心", "你怎么了",
    "你没事吧", "你太累了",
]
ABOUT_OTHER_OUT = [
    "那个人真讨厌", "他好烦啊", "那个同事太过分了",
    "他是死变态", "他们太过分了", "那个客户真气人",
    "张三真的很坏", "她说话好难听",
    "今天运气好倒霉啊", "这个破天气真让人受不了",
    "今天工作太烦了", "真是烦死了没办法",
    "这也太烂了吧", "这个怎么这么难搞", "这破事情没完没了",
    "你觉得他能行吗", "你猜他考了几分", "你听说张三的事了吗",
    "你认识他吗", "他跟你说了什么", "你觉得那个人怎么样",
    "你知道他为什么生气吗", "你看见他去哪了吗",
    "你帮我问问他", "你有没有他电话",
    "我跟你说他真的很过分", "你知道吗他是故意的",
    "我听说小明最近很开心", "我听说他中彩票了",
    "听说有人中奖了", "听说那家伙出事了",
    "今天天气真好", "天气不错", "今天下雨了", "天好热啊",
]

_me_vec = _you_vec = _other_vec = None
_init_lock = _threading.Lock()


def _init_vecs():
    global _me_vec, _you_vec, _other_vec
    if _me_vec is None:
        with _init_lock:
            if _me_vec is None:
                m = get_embed_model()
                _me_vec = m.encode(ABOUT_ME).mean(axis=0)
                _you_vec = m.encode(ABOUT_YOU_OUT).mean(axis=0)
                _other_vec = m.encode(ABOUT_OTHER_OUT).mean(axis=0)


def classify_subject(text):
    _init_vecs()
    m = get_embed_model()
    tv = m.encode([text])[0]
    scores = {
        "我": float(np.dot(tv, _me_vec) / (np.linalg.norm(tv) * np.linalg.norm(_me_vec) + 1e-8)),
        "你": float(np.dot(tv, _you_vec) / (np.linalg.norm(tv) * np.linalg.norm(_you_vec) + 1e-8)),
        "他": float(np.dot(tv, _other_vec) / (np.linalg.norm(tv) * np.linalg.norm(_other_vec) + 1e-8)),
    }
    subject = max(scores, key=scores.get)
    return subject, scores


# ═══ 输出侧三张映射表 (表内=纯比例, 强度由方向系数管) ═══
SPK_OUT_SELF = {
    "sad":      {"sad": 1.0},
    "happy":    {"happy": 1.0},
    "angry":    {"angry": 1.0, "contempt": 0.3},
    "fear":     {"scared": 1.0},
    "surprise": {"surprised": 1.0},
    "disgust":  {"contempt": 1.0},
    "contempt": {"contempt": 1.0},
    "neutral":  {},
}
SPK_OUT_YOU = {
    "sad":      {"care": 0.8, "sad": 0.2},
    "happy":    {"happy": 0.7, "care": 0.3},
    "fear":     {"care": 0.8, "worried": 0.4},
    "angry":    {"scared": 0.4, "worried": 0.4},
    "surprise": {"surprised": 0.5, "happy": 0.3},
    "disgust":  {"contempt": 0.5},
    "contempt": {"contempt": 0.4, "worried": 0.2},
    "neutral":  {"care": 0.3},
}
SPK_OUT_OTHER = {}   # 讨论中: 主战场在 dist 转移

# 话题→dist 增量权重 (Berger&Milkman'12 传播力锚定: anger>awe>anxiety>surprise>positivity>sad(-0.16))
DISTRACT_WEIGHT = {"angry": 0.6, "fear": 0.6, "surprise": 0.5,
                   "disgust": 0.4, "contempt": 0.4, "sad": 0.3, "happy": 0.2, "neutral": 0.1}
DIST_HALFLIFE = {"sad": 60, "fear": 60, "angry": 60, "disgust": 50, "contempt": 50,
                 "surprise": 45, "happy": 30, "neutral": 30}
# 表达压制: mask = max(1 - k*dist, MASK_FLOOR) (Cohen'12 资源限制 + Engelhard'11 倒U下限)
DIST_SUPPRESS_K = 0.8
MASK_FLOOR = 0.15

dist_val = 0.0
dist_hl_cur = 45.0
dist_last = _time.time()
_dist_lock = _threading.Lock()


def _log(msg):
    """上线前排查日志 (正式上线时撤掉)"""
    print(f"[output_vad {_time.strftime('%H:%M:%S')}] {msg}")


def _decay_dist():
    global dist_val, dist_last
    now = _time.time()
    dt = now - dist_last
    dist_last = now
    if dt > 0 and dist_val > 0:
        dist_val *= 0.5 ** (dt / dist_hl_cur)
        if dist_val < 0.005:
            dist_val = 0.0

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

def _push_bridge(emo, ratio):
    try:
        data = json.dumps({'type':'vad_status','mode':'listen','emotion':emo,'ratio':ratio}).encode()
        req = _ureq.Request('http://127.0.0.1:18770/push', data=data,
            headers={'Content-Type':'application/json'}, method='POST')
        _ureq.urlopen(req, timeout=1)
    except Exception:
        pass

HTML = r"""<!DOCTYPE html>
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


</script>
</body>
</html>"""


class FACSHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/latest':
            self._json(_latest_info)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global dv, dist_val
        try:
            if self.path == '/output':
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode('utf-8', errors='replace'))
                text = body.get('text', '').strip()

                if not text:
                    _log("WARN 空文本")
                    self._json({'error': 'empty text'})
                    return

                self._process_frame(text)

            elif self.path == '/silent':
                self._process_silent_frame()

            elif self.path == '/reset':
                _log("重置")
                dv = DualVAD()
                dist_val = 0.0
                self._json({'ok': True})

            else:
                self.send_response(404)
                self.end_headers()

        except Exception as e:
            import traceback
            traceback.print_exc()
            _log(f"ERROR /{self.path}: {e}")
            self._json({'error': str(e)})

    def _process_frame(self, text):
        global dv, _latest_info
        _log(f"── 收到文本: [{text}]")
        _latest_info["text"] = text
        with _dist_lock:
            _decay_dist()
        # Layer 1: 检测情绪 (复用输入侧)
        speaker = detect_speaker_emotion(text)
        spk_items = sorted(speaker.items(), key=lambda x: -x[1])
        if spk_items:
            threshold = spk_items[0][1] * 0.75
            filtered = {k: v for k, v in spk_items if v >= threshold}
        else:
            filtered = {}
        # 三分归属
        subject, scores = classify_subject(text)
        _log(f"  归属: {subject} (我{scores['我']:.3f} 你{scores['你']:.3f} 他{scores['他']:.3f})")
        table = {"我": SPK_OUT_SELF, "你": SPK_OUT_YOU, "他": SPK_OUT_OTHER}[subject]
        _log(f"  检测情绪: { {k: round(v,3) for k,v in speaker.items()} } → 过滤后 { {k: round(v,3) for k,v in filtered.items()} }")
        # 查表 → 输出皮层信号 → ×4 帧补偿
        cortex = {}
        for emo, intensity in filtered.items():
            row = table.get(emo, {})
            for ch, ratio in row.items():
                cortex[ch] = cortex.get(ch, 0) + intensity * ratio
        for k in cortex:
            cortex[k] = min(1.0, cortex[k] * OUT_FRAME_EQ)
        # 写进共享皮层池 (无杏仁核)
        direction = {"我": "out_me", "你": "out_you", "他": "out_other"}[subject]
        _log(f"  皮层信号×{OUT_FRAME_EQ}: { {k: round(v,3) for k,v in cortex.items()} } → dv.update(direction={direction})")
        dv.update(amygdala_signals={}, cortex_signals=cortex, direction=direction)
        # 转移通道: 说别人 → dist 上升; 说我 → dist 清零
        global dist_val, dist_hl_cur
        with _dist_lock:
            if subject == "他":
                if filtered:
                    top_emo = max(filtered, key=filtered.get)
                    top_intensity = filtered[top_emo]
                else:
                    top_emo, top_intensity = "neutral", 0.3
                dist_delta = DISTRACT_WEIGHT.get(top_emo, 0.1) * min(1.0, top_intensity * OUT_FRAME_EQ)
                dist_hl_cur = DIST_HALFLIFE.get(top_emo, 45)
                dist_val = min(1.0, dist_val + dist_delta)
                dist_src = f"话题={top_emo} × {DISTRACT_WEIGHT.get(top_emo,0.1):.1f} → +{dist_delta:.3f}"
                _log(f"  dist: {dist_val:.3f} ({dist_src}) 半衰期{dist_hl_cur}s")
            else:
                dist_delta, dist_src = 0.0, ""
                if subject == "我":
                    dist_val = 0.0
                    _log(f"  dist: 清零 (说我)")
        # 表达压制: face = 池 × (1 - dist×k)
        state = dv.get_state()
        mask = max(MASK_FLOOR, 1 - min(1.0, dist_val * DIST_SUPPRESS_K))
        _log(f"  压制: dist={dist_val:.3f} → mask={mask:.3f} (压制{(1-mask)*100:.0f}%)")
        face = dv.merged_for_face()
        face = {k: v * mask for k, v in face.items()}
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

        self._json({
            'text': text,
            'scores': scores,
            'subject': subject,
            'table': subject,
            'filtered': filtered,
            'cortex': cortex,
            'frame_eq': OUT_FRAME_EQ,
            'dist': dist_val, 'dist_delta': dist_delta, 'dist_src': dist_src, 'dist_hl': dist_hl_cur,
            'mask': mask,
            'pools': pools,
            'speaker': speaker,
            'face': face,
            'vad_cortex': {k: round(v, 4) for k, v in state['cortex'].items()},
            'cortex_champion': max(state['cortex'], key=state['cortex'].get) if state['cortex'] else '',
            'au': au,
            'expr_layer': expr_layer,
            'ratio_upper_name': ratio_u_cn,
            'ratio_upper': ratio_u_val,
            'ratio_lower_name': ratio_l_cn,
            'ratio_lower': ratio_l_val,
            'raw': {'cortex': {k: round(v, 4) for k, v in state['cortex'].items()}}
        })

    def _process_silent_frame(self):
        global dv
        dv.update(amygdala_signals={}, cortex_signals={})
        with _dist_lock:
            _decay_dist()
        state = dv.get_state()
        face = dv.merged_for_face()
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

        mask = max(MASK_FLOOR, 1 - min(1.0, dist_val * DIST_SUPPRESS_K))
        face = {k: v * mask for k, v in face.items()}
        self._json({
            'scores': {'我': 0, '你': 0, '他': 0},
            'subject': '—', 'table': '—', 'filtered': {}, 'cortex': {},
            'frame_eq': OUT_FRAME_EQ,
            'dist': dist_val, 'dist_delta': 0, 'dist_src': '', 'dist_hl': dist_hl_cur,
            'mask': mask,
            'pools': pools,
            'speaker': {},
            'face': face,
            'vad_cortex': {k: round(v, 4) for k, v in state['cortex'].items()},
            'cortex_champion': max(state['cortex'], key=state['cortex'].get) if max(state['cortex'].values()) > 0 else '',
            'au': au,
            'expr_layer': expr_layer,
            'ratio_upper_name': ratio_u_cn,
            'ratio_upper': ratio_u_val,
            'ratio_lower_name': ratio_l_cn,
            'ratio_lower': ratio_l_val,
            'raw': {'cortex': {k: round(v, 4) for k, v in state['cortex'].items()}}
        })

    def _json(self, data):
        if data is not _latest_info:
            _latest_info["full"] = data
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())


if __name__ == '__main__':
    print('[output_vad] 预热原型向量 (encode 128条例句)...')
    try:
        _init_vecs()
        print('[output_vad] 预热完成')
    except Exception as e:
        print(f'[output_vad] 预热失败(稍后懒加载): {e}')
    port = 18773
    print(f'[output_vad] 输出演绎调试页: http://localhost:{port}')
    print('Ctrl+C 停止')
    ThreadingHTTPServer(('0.0.0.0', port), FACSHandler).serve_forever()
