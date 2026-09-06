"""
output_engine.py — 输出演绎引擎核心逻辑 (无 HTTP/HTML, dv 由调用方传入)
供 facs_server.py (皮层 VAD 共享, 主服务) 与 output_vad_debug.py (自测调试页) 共用。

设计: 引擎不持有 dv — 调用方把共享的 DualVAD 实例传进来, 输入/输出写同一个池。
      dist 状态是模块级 (进程内单例, 只在输出侧存在)。
"""
import os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facs_engine import detect_speaker_emotion, detect_self_emotion, detect_you_emotion, get_embed_model
from vad_engine import map_emotions_to_au
import vad_engine
import numpy as np

# ── 输出侧方向系数 (patch 内存, 不动 vad_engine.py; 输入侧"你1.4/别的0.8"不受影响) ──
# 只 patch 一次, 避免重复 import 时覆盖
if "out_me" not in vad_engine.DIRECTION_COEFF:
    vad_engine.DIRECTION_COEFF.update({"out_me": 1.8, "out_you": 0.8, "out_other": 0.1})
    vad_engine.DIRECTION_DECAY.update({"out_me": 1.0, "out_you": 1.0, "out_other": 1.0})

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
_init_lock = threading.Lock()


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
    """三分胜出: 情绪归属者 我/你/他 (谁分高归谁)"""
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
    "shy":      {"shy": 1.0},       # 自己说自己害羞 → 害羞 (×out_me 1.8)
    "worried":  {"worried": 1.0},   # 自己说自己担心 → 担心 (×out_me 1.8)
    "hurt":     {"hurt": 1.0},      # 自己说自己委屈 → 委屈 (×out_me 1.8)
}
# "我说你" (我对拉姆/观众说) — 1:1 表内比例, 强度由方向系数 out_you=0.8 管
SPK_OUT_YOU = {
    "angry":    {"angry": 1.0},
    "shy":      {"shy": 1.0},
    "happy":    {"happy": 1.0},
    "worried":  {"worried": 1.0},
    "fear":     {"scared": 1.0},
    "surprise": {"surprised": 1.0},
    "contempt": {"contempt": 1.0},
    "hurt":     {"hurt": 1.0},
    "sad":      {"sad": 1.0},
    "neutral":  {},
}
SPK_OUT_OTHER = {}   # 说别人: 主战场在 dist 转移

# 话题→dist 增量权重 (Berger&Milkman'12 传播力锚定)
DISTRACT_WEIGHT = {"angry": 0.6, "fear": 0.6, "surprise": 0.5,
                   "disgust": 0.4, "contempt": 0.4, "sad": 0.3, "happy": 0.2, "neutral": 0.1}
DIST_HALFLIFE = {"sad": 60, "fear": 60, "angry": 60, "disgust": 50, "contempt": 50,
                 "surprise": 45, "happy": 30, "neutral": 30}
# 表达压制: mask = max(1 - k*dist, MASK_FLOOR)
DIST_SUPPRESS_K = 0.8
MASK_FLOOR = 0.15

# ── dist 状态 (模块级单例: 输出侧独有, 进程内共享) ──
dist_val = 0.0
dist_hl_cur = 45.0
dist_last = time.time()
_dist_lock = threading.Lock()


def reset():
    """重置 dist (调试/重启用)"""
    global dist_val, dist_hl_cur
    with _dist_lock:
        dist_val = 0.0
        dist_hl_cur = 45.0


def _decay_dist():
    global dist_val, dist_last
    now = time.time()
    dt = now - dist_last
    dist_last = now
    if dt > 0 and dist_val > 0:
        dist_val *= 0.5 ** (dt / dist_hl_cur)
        if dist_val < 0.005:
            dist_val = 0.0


def silent_tick():
    """静默帧: dist 随时间衰减 (供主服务 /silent 调用)"""
    with _dist_lock:
        _decay_dist()
    return dist_val


def process(text, dv):
    """输出侧完整链路 (dv 由调用方传入 — 共享皮层池的关键)

    Returns: 供展示的 dict (scores/subject/filtered/cortex/dist/mask/face/...)
    """
    global dist_val, dist_hl_cur
    with _dist_lock:
        _decay_dist()
    # 三分归属 (先行: 决定用哪组检测/哪张表)
    subject, scores = classify_subject(text)
    # Layer 1: 检测情绪 — 按归属者选原型组
    #   "我" → self-self 新原型组 (9类, 含 shy/worried/hurt)
    #   "你" → i-to-you 新原型组 (9类, 含 shy/worried/hurt)
    #   "他" → 老 8 类 (dist 转移主战场, 不动)
    if subject == "我":
        speaker = detect_self_emotion(text)
    elif subject == "你":
        speaker = detect_you_emotion(text)
    else:
        speaker = detect_speaker_emotion(text)
    spk_items = sorted(speaker.items(), key=lambda x: -x[1])
    if spk_items:
        threshold = spk_items[0][1] * 0.75
        filtered = {k: v for k, v in spk_items if v >= threshold}
    else:
        filtered = {}
    table = {"我": SPK_OUT_SELF, "你": SPK_OUT_YOU, "他": SPK_OUT_OTHER}[subject]
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
    dv.update(amygdala_signals={}, cortex_signals=cortex, direction=direction)
    # 转移通道: 说别人 → dist 上升; 说我 → dist 清零
    dist_delta, dist_src = 0.0, ""
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
        elif subject == "我":
            dist_val = 0.0
    # 表达压制: face = 池 × mask (池不动)
    state = dv.get_state()
    mask = max(MASK_FLOOR, 1 - min(1.0, dist_val * DIST_SUPPRESS_K))
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
    return {
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
    }
