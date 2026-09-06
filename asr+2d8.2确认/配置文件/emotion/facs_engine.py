"""
facs_engine.py — 文本→FACS参数引擎 (v0: 关键词匹配, 零GPU)

管线:
  文本 → 说话者情绪(关键词) → 听者感受映射 → VAD累积/衰减 → FACS参数 → Live2D

用法(测试):
  python emotion/facs_engine.py
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ═══════════════ 第1层: 说话者情绪检测 (嵌入模型) ═══════════════

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer
import numpy as np

# 多语言模型, CPU推理, ~5ms/次
_embed_model = None
_emotion_prototypes = {}  # {emotion_name: [embedding_vectors...]}

# 每种情绪的原型句子 (TIX007/chinese-sentiment 标注数据)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_prototypes_data import PROTOTYPES as EMOTION_PROTOTYPES
from self_self_prototypes import SELF_SELF_PROTOTYPES
from i_to_you_prototypes import I_TO_YOU_PROTOTYPES
from told_me_prototypes import TOLD_ME_PROTOTYPES

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(
            'thenlper/gte-base-zh',
            local_files_only=True
        )
    return _embed_model

def _get_prototype_embedding(emotion: str) -> np.ndarray:
    if emotion not in _emotion_prototypes:
        model = get_embed_model()
        sentences = EMOTION_PROTOTYPES.get(emotion, ["..."] )
        vectors = model.encode(sentences)
        _emotion_prototypes[emotion] = np.mean(vectors, axis=0)
    return _emotion_prototypes[emotion]


def detect_speaker_emotion(text: str) -> Dict[str, float]:
    """嵌入模型检测说话者情绪 → {angry:0.8, sad:0.4, ...}"""
    model = get_embed_model()
    text_vec = model.encode([text])[0]

    sims = {}
    for emotion in EMOTION_PROTOTYPES:
        proto_vec = _get_prototype_embedding(emotion)
        sim = float(np.dot(text_vec, proto_vec) /
                    (np.linalg.norm(text_vec) * np.linalg.norm(proto_vec) + 1e-8))
        sims[emotion] = sim

    # softmax 归一化
    sim_arr = np.array(list(sims.values()))
    exp = np.exp((sim_arr - sim_arr.max()) * 5)
    soft = exp / exp.sum()

    # 文字长度置信度: 短文本低强度, 长文本高强度
    conf = min(len(text) / 10, 1.0)

    return {emotion: round(float(soft[i].item() if hasattr(soft[i], 'item') else soft[i]) * conf, 3)
            for i, emotion in enumerate(sims)}


# ═══════════════ "我说我自己" 专用检测 (self-self 原型组) ═══════════════
_self_protos = {}  # {emotion: mean_embedding_vector}

def _get_self_prototype(emotion: str) -> np.ndarray:
    if emotion not in _self_protos:
        model = get_embed_model()
        vectors = model.encode(SELF_SELF_PROTOTYPES[emotion])
        _self_protos[emotion] = np.mean(vectors, axis=0)
    return _self_protos[emotion]


def detect_self_emotion(text: str) -> Dict[str, float]:
    """
    '我说我自己' 模式检测 — 用 self-self 原型组（9 类皮层情绪）算分。
    与 detect_speaker_emotion 平行：模式判定为"自己说自己"时用这组，老组分数扔掉。
    Returns: {shy:0.8, sad:0.1, ...} 1:1 映射到 CORTEX 通道
    """
    t = text.strip()
    if not t:
        return {}
    model = get_embed_model()
    text_vec = model.encode([t])[0]

    sims = {}
    for emotion in SELF_SELF_PROTOTYPES:
        proto = _get_self_prototype(emotion)
        sim = float(np.dot(text_vec, proto) /
                    (np.linalg.norm(text_vec) * np.linalg.norm(proto) + 1e-8))
        sims[emotion] = max(0, sim)

    # softmax 归一化 (与 detect_speaker_emotion 同款)
    sim_arr = np.array(list(sims.values()))
    exp = np.exp((sim_arr - sim_arr.max()) * 5)
    soft = exp / exp.sum()
    conf = min(len(t) / 10, 1.0)
    return {emotion: round(float(soft[i]) * conf, 3)
            for i, emotion in enumerate(sims)}


# ═══════════════ "我说你" 专用检测 (i-to-you 原型组) ═══════════════
_you_protos = {}  # {emotion: mean_embedding_vector}

def _get_you_prototype(emotion: str) -> np.ndarray:
    if emotion not in _you_protos:
        model = get_embed_model()
        vectors = model.encode(I_TO_YOU_PROTOTYPES[emotion])
        _you_protos[emotion] = np.mean(vectors, axis=0)
    return _you_protos[emotion]


def detect_you_emotion(text: str) -> Dict[str, float]:
    """
    '我说你' 模式检测 — 用 i-to-you 原型组（9 类皮层情绪）算分。
    与 detect_self_emotion 平行：模式判定为"我说你"时用这组，老组分数扔掉。
    Returns: {angry:0.8, shy:0.1, ...} 1:1 映射到 CORTEX 通道
    """
    t = text.strip()
    if not t:
        return {}
    model = get_embed_model()
    text_vec = model.encode([t])[0]

    sims = {}
    for emotion in I_TO_YOU_PROTOTYPES:
        proto = _get_you_prototype(emotion)
        sim = float(np.dot(text_vec, proto) /
                    (np.linalg.norm(text_vec) * np.linalg.norm(proto) + 1e-8))
        sims[emotion] = max(0, sim)

    # softmax 归一化 (同款)
    sim_arr = np.array(list(sims.values()))
    exp = np.exp((sim_arr - sim_arr.max()) * 5)
    soft = exp / exp.sum()
    conf = min(len(t) / 10, 1.0)
    return {emotion: round(float(soft[i]) * conf, 3)
            for i, emotion in enumerate(sims)}


# ═══════════════ "你说我" 专用检测 (told-me 原型组) ═══════════════
_told_protos = {}  # {emotion: mean_embedding_vector}

def _get_told_prototype(emotion: str) -> np.ndarray:
    if emotion not in _told_protos:
        model = get_embed_model()
        vectors = model.encode(TOLD_ME_PROTOTYPES[emotion])
        _told_protos[emotion] = np.mean(vectors, axis=0)
    return _told_protos[emotion]


def detect_told_me_emotion(text: str) -> Dict[str, float]:
    """
    '你说我' 模式检测 — 用 told-me 原型组（9 类皮层情绪）算分。
    输入侧：别人对我说的话 → 直接算出"我的情绪"，1:1 进 VAD（无映射表、无叠加）。
    与 detect_speaker_emotion 平行：方向判定 is_about_you==True 时用这组，老组分数扔掉。
    Returns: {happy:0.8, shy:0.1, ...} 1:1 映射到 CORTEX 通道
    """
    t = text.strip()
    if not t:
        return {}
    model = get_embed_model()
    text_vec = model.encode([t])[0]

    sims = {}
    for emotion in TOLD_ME_PROTOTYPES:
        proto = _get_told_prototype(emotion)
        sim = float(np.dot(text_vec, proto) /
                    (np.linalg.norm(text_vec) * np.linalg.norm(proto) + 1e-8))
        sims[emotion] = max(0, sim)

    # softmax 归一化 (同款)
    sim_arr = np.array(list(sims.values()))
    exp = np.exp((sim_arr - sim_arr.max()) * 5)
    soft = exp / exp.sum()
    conf = min(len(t) / 10, 1.0)
    return {emotion: round(float(soft[i]) * conf, 3)
            for i, emotion in enumerate(sims)}


# ═══════════════ 第2层: 听者感受映射 (v0: 硬映射表) ═══════════════

# 说话者情绪 → 听者(小女生mini_coin)感受
SPEAKER_TO_LISTENER = {
    "angry":    {"hurt": 0.9, "sad": 0.5, "fear": 0.4, "confused": 0.2},
    "disgust":  {"disgusted": 0.8, "hurt": 0.4, "confused": 0.3},
    "sad":      {"sad": 0.7, "care": 0.5, "confused": 0.2},
    "happy":    {"happy": 0.8, "excited": 0.4, "shy": 0.2},
    "surprise": {"surprised": 0.9, "curious": 0.4},
    "fear":     {"fear": 0.7, "confused": 0.4, "care": 0.3},
    "contempt": {"hurt": 0.7, "sad": 0.4, "angry": 0.3},
    "neutral":  {"neutral": 0.5, "attentive": 0.3},
}


# ═══════════════ 第3层: FACS参数映射 (来自SoulLink + 我们的模型) ═══════════════

# 听者感受 → Live2D参数 (值范围 0~1, 角度范围 -30~30)
FACS_PARAMS = {
    "happy":     {"ParamEyeLSmile": 0.7, "ParamEyeRSmile": 0.7, "ParamMouthForm": 0.8,
                   "ParamCheek": 0.4, "ParamBrowLY": 0.3, "ParamBrowRY": 0.3},
    "sad":       {"ParamEyeLOpen": 0.6, "ParamEyeROpen": 0.6,
                   "ParamMouthForm": -0.5, "ParamBrowLY": -0.5, "ParamBrowRY": -0.5,
                   "ParamCheek": 0.3, "ParamBrowLAngle": -0.3, "ParamBrowRAngle": -0.3},
    "hurt":      {"ParamEyeLOpen": 0.5, "ParamEyeROpen": 0.5,
                   "ParamBrowLY": -0.6, "ParamBrowRY": -0.6, "ParamCheek": 0.6,
                   "ParamMouthForm": -0.3, "ParamBrowLAngle": -0.4, "ParamBrowRAngle": -0.4,
                   "ParamBodyAngleX": -3},   # 轻微前倾(低头)
    "surprised": {"ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
                   "ParamMouthOpenY": 0.6, "ParamBrowLY": 0.8, "ParamBrowRY": 0.8,
                   "ParamAngleZ": 5},
    "fear":      {"ParamEyeLOpen": 0.9, "ParamEyeROpen": 0.9,
                   "ParamBrowLY": 0.5, "ParamBrowRY": 0.5, "ParamMouthOpenY": 0.3,
                   "ParamBodyAngleX": 2, "ParamAngleY": -3},  # 后仰+低头
    "confused":  {"ParamBrowLY": 0.2, "ParamBrowRY": -0.1, "ParamAngleZ": 8,
                   "ParamEyeBallX": 0.3, "ParamEyeBallY": 0.2},
    "excited":   {"ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
                   "ParamEyeLSmile": 0.8, "ParamEyeRSmile": 0.8, "ParamMouthOpenY": 0.5,
                   "ParamCheek": 0.5, "ParamBrowLY": 0.5, "ParamBrowRY": 0.5},
    "shy":       {"ParamEyeLOpen": 0.7, "ParamEyeROpen": 0.7,
                   "ParamEyeLSmile": 0.4, "ParamEyeRSmile": 0.4, "ParamMouthForm": 0.3,
                   "ParamCheek": 0.8, "ParamAngleZ": -6, "ParamBodyAngleX": -2},
    "curious":   {"ParamBrowLY": 0.3, "ParamBrowRY": 0.3, "ParamAngleZ": 5,
                   "ParamEyeBallX": 0.2, "ParamBodyAngleX": -2},
    "care":      {"ParamEyeLOpen": 0.9, "ParamEyeROpen": 0.9,
                   "ParamBrowLY": 0.3, "ParamBrowRY": 0.3, "ParamBodyAngleX": -3,
                   "ParamAngleZ": 4},
    "attentive": {"ParamBodyAngleX": -2, "ParamAngleZ": 3,
                   "ParamEyeLOpen": 0.15, "ParamEyeROpen": 0.15},  # 加成
    "disgusted": {"ParamEyeLOpen": 0.7, "ParamEyeROpen": 0.7,
                   "ParamNose": 0.5, "ParamMouthForm": -0.6, "ParamBrowLY": -0.3, "ParamBrowRY": -0.3,
                   "ParamAngleZ": 3},  # 皱眉+撇嘴+歪头
    "angry":     {"ParamBrowLY": -0.5, "ParamBrowRY": -0.5,
                   "ParamMouthForm": -0.3, "ParamEyeLOpen": 0.9, "ParamEyeROpen": 0.9,
                   "ParamCheek": 0.2},  # 皱眉瞪眼(听者生气)
    "neutral":   {},
}


# ═══════════════ 第4层: VAD 衰减引擎 ═══════════════

@dataclass
class VADSlot:
    """单个情绪通道的VAD追踪"""
    value: float = 0.0             # 当前位置 0~1
    decay: float = 0.92            # 每帧衰减率
    inc_rate: float = 0.3          # 每次累加速度
    threshold_effect: float = 0.7  # 触发特效阈值

    def push(self, intensity: float):
        """收到新情绪输入时累加"""
        self.value = min(1.0, self.value + intensity * self.inc_rate)

    def tick(self):
        """每帧衰减"""
        self.value *= self.decay
        if self.value < 0.01:
            self.value = 0.0

    @property
    def triggered(self) -> bool:
        return self.value >= self.threshold_effect


class DualVADEngine:
    """双通道VAD: 输入侧(听用户说话) + 输出侧(AI说话)"""

    def __init__(self):
        self.input_slots: Dict[str, VADSlot] = {}
        self.output_slots: Dict[str, VADSlot] = {}
        self.callback_effect = None  # (name, params) 特效回调

    def _ensure_slot(self, slots: Dict[str, VADSlot], name: str):
        if name not in slots:
            slots[name] = VADSlot()
        return slots[name]

    def feed(self, text: str, channel: str = "input"):
        """喂入文本，更新VAD"""
        # 第1层: 说话者情绪
        speaker_emo = detect_speaker_emotion(text)

        # 第2层: 听者感受
        listener_feel = {}
        for semo, intensity in speaker_emo.items():
            if semo in SPEAKER_TO_LISTENER:
                for lemo, ratio in SPEAKER_TO_LISTENER[semo].items():
                    listener_feel[lemo] = max(listener_feel.get(lemo, 0), intensity * ratio)

        # 写入VAD
        slots = self.input_slots if channel == "input" else self.output_slots
        for feeling, intensity in listener_feel.items():
            slot = self._ensure_slot(slots, feeling)
            slot.push(intensity)

        # 检查特效阈值
        for name, slot in slots.items():
            if slot.triggered and self.callback_effect:
                self.callback_effect(name, slot.value)

    def tick(self, dt: float = 1/60):
        """每帧调用，衰减所有通道"""
        for slots in (self.input_slots, self.output_slots):
            for slot in slots.values():
                slot.tick()

    def get_params(self) -> Dict[str, float]:
        """合并输入+输出侧 → 最终Live2D参数"""
        merged = {}
        for slots in (self.input_slots, self.output_slots):
            for feeling, slot in slots.items():
                if feeling in FACS_PARAMS:
                    for param_id, base_val in FACS_PARAMS[feeling].items():
                        contribution = base_val * slot.value
                        if param_id in merged:
                            merged[param_id] = max(merged[param_id], contribution)
                        else:
                            merged[param_id] = contribution
        return merged

    def active_channels(self) -> Dict[str, float]:
        """返回当前活跃的情绪通道和值(用于调试)"""
        active = {}
        for slots in (self.input_slots, self.output_slots):
            for name, slot in slots.items():
                if slot.value > 0.01:
                    active[name] = round(slot.value, 3)
        return active


# ═══════════════ 测试 ═══════════════

def test_streaming():
    """模拟流式输入，逐字推进，打印每步的VAD变化"""
    print("=" * 60)
    print("FACS 引擎流式测试")
    print("=" * 60)

    engine = DualVADEngine()

    test_dialogue = [
        "你",
        "你怎么",
        "你怎么这么",
        "你怎么这么笨啊",
        "对不起",
        "对不起我错了",
        "哈哈",
        "哈哈哈开玩笑的",
        "其实你很可爱的",
    ]

    accumulated = ""
    for partial in test_dialogue:
        accumulated = partial  # 模拟流式: 每次都是完整累积文本
        print(f"\n[输入] \"{accumulated}\"", flush=True)

        # 说话者情绪
        spk = detect_speaker_emotion(accumulated)
        print(f"  说话者: {dict(sorted(spk.items(), key=lambda x:-x[1]))}", flush=True)

        # 喂入引擎
        engine.feed(accumulated, "input")
        engine.tick()

        # 活跃通道
        active = engine.active_channels()
        if active:
            sorted_active = dict(sorted(active.items(), key=lambda x: -x[1]))
            print(f"  听者VAD: {sorted_active}", flush=True)
            params = engine.get_params()
            non_zero = {k: round(v, 2) for k, v in params.items() if abs(v) > 0.01}
            if non_zero:
                print(f"  →L2D参数: {dict(sorted(non_zero.items()))}", flush=True)
        else:
            print(f"  听者VAD: (无情绪)", flush=True)

        time.sleep(0.3)  # 模拟语音节奏

    # 模拟衰减
    print(f"\n--- 沉默衰减 ---")
    for i in range(10):
        engine.tick()
        active = engine.active_channels()
        if active:
            print(f"  帧{i+1}: {dict(sorted(active.items(), key=lambda x:-x[1]))}", flush=True)
        else:
            print(f"  帧{i+1}: 归零 ✓", flush=True)
            break
        time.sleep(0.1)

    print(f"\n测试完成")

if __name__ == "__main__":
    test_streaming()
