"""
方向检测: 说话者在说谁？ — 主语=你 vs 主语≠你
复用 gte-base-zh embedding
"""
import numpy as np
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from facs_engine import get_embed_model
from directional_prototypes import ABOUT_YOU, ABOUT_OTHER

_you_vec = None
_other_vec = None


def _init():
    global _you_vec, _other_vec
    if _you_vec is None:
        m = get_embed_model()
        _you_vec = m.encode(ABOUT_YOU).mean(axis=0)
        _other_vec = m.encode(ABOUT_OTHER).mean(axis=0)


def is_about_you(text: str) -> tuple:
    """
    Returns:
        (about_you: bool, you_score: float, other_score: float)
    """
    _init()
    m = get_embed_model()
    tv = m.encode([text])[0]
    sy = float(np.dot(tv, _you_vec) /
               (np.linalg.norm(tv) * np.linalg.norm(_you_vec) + 1e-8))
    so = float(np.dot(tv, _other_vec) /
               (np.linalg.norm(tv) * np.linalg.norm(_other_vec) + 1e-8))
    return sy > so, round(sy, 4), round(so, 4)


if __name__ == "__main__":
    tests = [
        "你笨蛋", "你让我好失望", "你好可爱", "你真棒",
        "我喜欢你", "你要小心张三是死变态",
        "我好难过", "他好烦", "那个同事太过分了",
        "张三是死变态", "今天好倒霉", "烦死了",
    ]
    for text in tests:
        about_you, sy, so = is_about_you(text)
        tag = "→ 🫵说你" if about_you else "→ 🌍说别的"
        print(f"{tag} | you={sy:.4f} other={so:.4f} | {text}")
