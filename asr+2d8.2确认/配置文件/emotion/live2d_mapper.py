"""
VAD 表情运算层 → Live2D 参数映射

算法: 参数值 = 最大值 × clamp((VAD - 阈值) / (1 - 阈值))^0.5
阈值 = Calvo 2016 各情绪可见阈值, 所有参数共享。
hold参数: VAD>阈值时直接到位。
"""
import math

POWER = 0.5

def vad_to_param(vad: float, threshold: float, max_val: float) -> float:
    if vad <= threshold:
        return 0.0
    norm = (vad - threshold) / (1.0 - threshold)
    return max_val * (min(norm, 1.0) ** POWER)


EMOTION_MAP = {
    # ══════ 皮层 ══════

    "happy": {
        "threshold": 0.20,
        "params": {
            "ParamEyeLSmile":   {"max": 0.70},
            "ParamEyeRSmile":   {"max": 0.70},
            "ParamEyeLOpen":    {"max": 0.85},
            "ParamEyeROpen":    {"max": 0.85},
            "ParamMouthForm":   {"max": 2.50},
            "ParamMouthOpenY":  {"max": 0.50},
            "ParamBrowLY":      {"max": -0.50},
            "ParamBrowRY":      {"max": -0.50},
            "ParamCheek":       {"max": 1.25},
            "ParamAngleZ":      {"max": 13, "hold": True},
            "ParamBodyAngleY":  {"max": -5, "hold": True},
            "ParamBodyAngleZ":  {"max": -8, "hold": True},
        }
    },

    "sad": {
        "threshold": 0.40,
        "params": {
            "ParamBrowLY":      {"max": 1.50},
            "ParamBrowRY":      {"max": 1.50},
            "ParamBrowLAngle":  {"max": -1.50},
            "ParamBrowRAngle":  {"max": -1.50},
            "ParamBrowLForm":   {"max": 1.50},
            "ParamBrowRForm":   {"max": 1.50},
            "ParamMouthForm":   {"max": -1.50},
            "ParamEyeLOpen":    {"max": 0.70},
            "ParamEyeROpen":    {"max": 0.70},
            "ParamEyeBallY":    {"max": -0.50},
            "ParamAngleY":      {"max": 5,  "hold": True},
            "ParamBodyAngleY":  {"max": -3, "hold": True},
        }
    },

    "angry": {
        "threshold": 0.40,
        "params": {
            "ParamBrowLY":      {"max": -2.00},
            "ParamBrowRY":      {"max": -2.00},
            "ParamBrowLAngle":  {"max": 1.50},
            "ParamBrowRAngle":  {"max": 1.50},
            "ParamBrowLForm":   {"max": 2.50},
            "ParamBrowRForm":   {"max": 2.50},
            "ParamEyeLOpen":    {"max": 0.90},
            "ParamEyeROpen":    {"max": 0.90},
            "ParamMouthForm":   {"max": 1.00},
            "ParamMouthOpenY":  {"max": 0.50},
            "ParamCheek":       {"max": 1.00},
            "ParamAngleZ":      {"max": 25, "hold": True},
            "ParamBodyAngleY":  {"max": -8, "hold": True},
            "ParamBodyAngleZ":  {"max": -8, "hold": True},
        }
    },

    "surprised": {
        "threshold": 0.40,
        "params": {
            "ParamBrowLY":      {"max": 1.50},
            "ParamBrowRY":      {"max": 1.50},
            "ParamEyeLOpen":    {"max": 1.20},
            "ParamEyeROpen":    {"max": 1.20},
            "ParamMouthOpenY":  {"max": 1.00},
            "ParamAngleY":      {"max": 3,   "hold": True},
            "ParamBodyAngleY":  {"max": -3,  "hold": True},
        }
    },

    "hurt": {
        "threshold": 0.40,
        "params": {
            "ParamBrowLY":      {"max": 2.50},
            "ParamBrowRY":      {"max": 2.50},
            "ParamBrowLAngle":  {"max": 2.50},
            "ParamBrowRAngle":  {"max": 2.50},
            "ParamEyeLOpen":    {"max": 0.0},
            "ParamEyeROpen":    {"max": 0.0},
            "ParamEyeBallY":    {"max": -1.0},
            "ParamMouthForm":   {"max": -3.0},
            "ParamMouthOpenY":  {"max": 0.20},
            "ParamCheek":       {"max": 1.0},
            "ParamAngleY":      {"max": -25, "hold": True},
            "ParamAngleX":      {"max": -12, "hold": True},
            "ParamBodyAngleY":  {"max": -20, "hold": True},
        }
    },

    "care": {
        "threshold": 0.35,
        "params": {
            "ParamMouthForm":   {"max": 0.50},
            "ParamBrowLY":      {"max": -0.30},
            "ParamBrowRY":      {"max": -0.30},
            "ParamEyeLOpen":    {"max": 0.95},
            "ParamEyeROpen":    {"max": 0.95},
            "ParamEyeBallY":    {"max": -0.20},
            "ParamBodyAngleY":  {"max": -10, "hold": True},
            "ParamAngleZ":      {"max": 4,   "hold": True},
            "ParamAngleY":      {"max": 2,   "hold": True},
        }
    },

    "worried": {
        "threshold": 0.40,
        "params": {
            "ParamBrowLY":      {"max": 1.60},
            "ParamBrowRY":      {"max": 1.60},
            "ParamBrowLAngle":  {"max": -2.00},
            "ParamBrowRAngle":  {"max": -2.00},
            "ParamEyeLOpen":    {"max": 1.10},
            "ParamEyeROpen":    {"max": 1.10},
            "ParamMouthForm":   {"max": -1.00},
            "ParamMouthOpenY":  {"max": 0.30},
            "ParamAngleY":      {"max": -10, "hold": True},
            "ParamAngleX":      {"max": 8,   "hold": True},
            "ParamBodyAngleY":  {"max": -12, "hold": True},
        }
    },

    "orient": {
        "threshold": 0.40,
        "params": {
            "ParamBrowLY":      {"max": 0.90},
            "ParamBrowRY":      {"max": 0.90},
            "ParamEyeLOpen":    {"max": 1.09},
            "ParamEyeROpen":    {"max": 1.09},
            "ParamEyeBallX":    {"max": 0.30},
            "ParamMouthForm":   {"max": 0.30},
            "ParamAngleZ":      {"max": -8, "hold": True},
            "ParamBodyAngleY":  {"max": -3, "hold": True},
        }
    },

    "scared": {
        "threshold": 0.50,
        "params": {
            "ParamEyeLOpen":    {"max": 1.10},
            "ParamEyeROpen":    {"max": 1.10},
            "ParamBrowLY":      {"max": 1.00},
            "ParamBrowRY":      {"max": 1.00},
            "ParamBrowLAngle":  {"max": 1.00},
            "ParamBrowRAngle":  {"max": 1.00},
            "ParamMouthOpenY":  {"max": 0.0},
            "ParamMouthForm":   {"max": -0.50},
            "ParamAngleY":      {"max": -15, "hold": True},  # X→Y: 低头
            "ParamBodyAngleY":  {"max": 6, "hold": True},    # X→Y: 后仰
        }
    },

    "tense": {
        "threshold": 0.40,
        "params": {
            "ParamBrowLY":      {"max": -0.90},
            "ParamBrowRY":      {"max": -0.90},
            "ParamBrowLForm":   {"max": 0.90},
            "ParamBrowRForm":   {"max": 0.90},
            "ParamEyeLOpen":    {"max": 1.12},
            "ParamEyeROpen":    {"max": 1.12},
            "ParamAngleY":      {"max": 2,  "hold": True},   # X→Y: 抬头(警戒)
            "ParamBodyAngleX":  {"max": 6,  "hold": True},   # 后仰
        }
    },

    "contempt": {
        "threshold": 0.40,
        "params": {
            "ParamAngleX":      {"max": 25, "hold": True},
            "ParamAngleY":      {"max": 15, "hold": True},
            "ParamEyeBallX":    {"max": -0.80, "hold": True},
            "ParamEyeLOpen":    {"max": 0.60, "hold": True},
            "ParamEyeROpen":    {"max": 0.60, "hold": True},
            "ParamMouthForm":   {"max": 1.00, "hold": True},
            "ParamCheek":       {"max": 1.50, "hold": True},
            "ParamBrowLY":      {"max": 0.50, "hold": True},
            "ParamBrowRY":      {"max": 0.50, "hold": True},
            "ParamBodyAngleY":  {"max": -16, "hold": True},
        }
    },

    "approach": {
        "threshold": 0.35,
        "params": {
            "ParamBodyAngleY":  {"max": -10, "hold": True},
            "ParamAngleZ":      {"max": 4,   "hold": True},
            "ParamAngleY":      {"max": 2,   "hold": True},
            "ParamEyeLOpen":    {"max": 0.95},
            "ParamEyeROpen":    {"max": 0.95},
            "ParamEyeBallY":    {"max": -0.20},
            "ParamBrowLY":      {"max": -0.30},
            "ParamBrowRY":      {"max": -0.30},
            "ParamMouthForm":   {"max": 0.50},
        }
    },

    "shy": {
        "threshold": 0.35,
        "params": {
            "ParamAngleY":      {"max": -25, "hold": True},
            "ParamAngleX":      {"max": 20,  "hold": True},
            "ParamEyeLOpen":    {"max": 0.60},
            "ParamEyeROpen":    {"max": 0.60},
            "ParamEyeLSmile":   {"max": 0.60},
            "ParamEyeRSmile":   {"max": 0.60},
            "ParamEyeBallY":    {"max": -0.80},
            "ParamMouthForm":   {"max": 0.80},
            "ParamMouthOpenY":  {"max": 0.05},
            "ParamCheek":       {"max": 2.0},
            "ParamBrowLY":      {"max": -0.20},
            "ParamBrowRY":      {"max": -0.20},
            "ParamBodyAngleY":  {"max": -12, "hold": True},
        }
    },
}


def map_expression_to_live2d(expr_layer: dict) -> dict:
    active = {}
    for region in ["upper", "lower"]:
        for emo, val in expr_layer.get(region, {}).get("emotions", {}).items():
            if emo not in active or val > active[emo]:
                active[emo] = val
    if not active:
        return {}

    result = {}
    for emo, vad_val in active.items():
        mapping = EMOTION_MAP.get(emo)
        if not mapping:
            continue
        thr = mapping["threshold"]
        for param_id, cfg in mapping["params"].items():
            max_val = cfg["max"]
            if cfg.get("hold", False):
                target = max_val if vad_val > thr else 0.0
            else:
                target = vad_to_param(vad_val, thr, max_val)
            if param_id not in result or abs(target) > abs(result[param_id]):
                result[param_id] = round(target, 4)
    return result


def reset_live2d_params() -> dict:
    all_params = {}
    for emo, mapping in EMOTION_MAP.items():
        for pid in mapping["params"]:
            all_params[pid] = 0.0
    return all_params
