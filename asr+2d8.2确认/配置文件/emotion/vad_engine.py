"""
vad_engine.py — VAD衰减引擎 v3
- 神经递质池 (DA/NE/5-HT/OXT): 同池共享容量 1.0
- 预测误差输入: 有效输入 = 信号 × 池余量 × 0.8
- 方向衰减: ABOUT_YOU 正常, ABOUT_OTHER 半衰期/2
- 注意力分配: champion正常, 其他 ×4
- relax = 1.0 - max(各池总和)
"""
import numpy as np

# ═══════════════ 杏仁核参数 ═══════════════
AMYGDALA_CHANNELS = ["tense", "orient", "approach", "relax"]
AMYGDALA_HALFLIFE = {
    "tense":    0.3,
    "orient":   2.5,
    "approach": 1.0,
    "relax":    2.0,
}

# ═══════════════ 皮层参数 ═══════════════
CORTEX_CHANNELS = [
    "hurt", "sad", "happy", "angry", "scared",
    "shy", "surprised", "worried",
    "care", "contempt"
]
CORTEX_HALFLIFE = {
    "hurt":      900,    # 15 min
    "sad":       900,    # 15 min
    "happy":     600,    # 10 min
    "angry":     300,    # 5 min
    "scared":    45,     # 45s
    "shy":       180,    # 3 min
    "surprised": 60,     # 1 min
    "worried":   480,    # 8 min
    "care":      300,    # 5 min
    "contempt":  60,     # 1 min
}

# ═══════════════ 神经递质池 ═══════════════
# 同池情绪共享上限 1.0, 输入时受池余量限制
NEURO_POOLS = {
    "DA":   ["happy", "shy", "orient"],
    "NE":   ["tense", "angry", "scared", "contempt"],
    "5HT":  ["sad", "hurt", "worried"],
    "OXT":  ["care"],
    "MISC": ["surprised", "approach", "relax"],
}

# 情绪到池的反向查找
EMO_TO_POOL = {}
for pool, emos in NEURO_POOLS.items():
    for e in emos:
        EMO_TO_POOL[e] = pool

ALL_CORTEX_POOL_CHANNELS = [e for e in CORTEX_CHANNELS if e != "shy"]
# shy 也属于 DA, 但它需要来自杏仁核层, 已经在列表了

# ═══════════════ 输入/衰减参数 ═══════════════
# 帧速平衡系数: 流式帧率变化时改 k 即可 (当前 600ms/帧 → k=0.25)
_k = 0.25
INC_RATE_AMYGDALA = 1.0 * _k
INC_RATE_CORTEX   = 0.8 * _k
# 专注衰减倍数
FOCUS_MULTIPLIER = 1.0
UNFOCUS_MULTIPLIER = 0.25

# 方向对输入的放大/缩小
DIRECTION_COEFF = {
    "你":   1.4,  # 说你 → 强情绪传递
    "别的":  0.8,  # 八卦 → 弱传递
}

# 方向对衰减的影响 (半衰期乘数)
DIRECTION_DECAY = {
    "你":   1.0,  # 正常衰减
    "别的":  0.5,  # 2x衰减 (半衰期砍半)
}


class DualVAD:
    """双 VAD 系统 + 神经递质池 + 方向衰减"""

    def __init__(self):
        self.a_channels = {ch: 0.0 for ch in AMYGDALA_CHANNELS}
        self.b_channels = {ch: 0.0 for ch in CORTEX_CHANNELS}
        # 记录每条通道最近的输入方向 (影响衰减速度)
        self.a_direction = {ch: "你" for ch in AMYGDALA_CHANNELS}
        self.b_direction = {ch: "你" for ch in CORTEX_CHANNELS}

    # ── 池状态 ──
    def pool_sum(self, channels: dict, pool_name: str) -> float:
        """计算某个递质池当前总和"""
        return sum(channels.get(e, 0.0) for e in NEURO_POOLS[pool_name])

    def pool_remaining(self, channels: dict, emotion: str) -> float:
        """该情绪所属池的剩余容量 [0, 1]"""
        pool = EMO_TO_POOL.get(emotion)
        if not pool:
            return 1.0
        return max(0.0, 1.0 - self.pool_sum(channels, pool))

    # ── 核心更新 ──
    def update(self, amygdala_signals: dict, cortex_signals: dict,
               direction: str = "你", dt: float = 0.6):
        """
        更新两个 VAD 系统

        Args:
            amygdala_signals: {tense:0.4, orient:0.1, ...}
            cortex_signals:   {hurt:0.5, sad:0.2, ...}
            direction: "你" 或 "别的", 影响此帧输入情绪后续的衰减速度
            dt: 帧时长(秒)
        """
        # ── 先衰减旧值 ──
        self._decay_attention(self.a_channels, AMYGDALA_HALFLIFE, dt, self.a_direction)
        self._decay_attention(self.b_channels, CORTEX_HALFLIFE, dt, self.b_direction)

        # ── 再累加新信号 (预测误差 + 池余量限制) ──
        for ch, val in amygdala_signals.items():
            if val <= 0:
                continue
            remaining = self.pool_remaining(self.a_channels, ch)
            trans_coeff = DIRECTION_COEFF.get(direction, 1.0)
            effective = val * remaining * trans_coeff
            self.a_channels[ch] = min(1.0, self.a_channels[ch] + effective * INC_RATE_AMYGDALA)
            self.a_direction[ch] = direction  # 标记方向

        for ch, val in cortex_signals.items():
            if val <= 0:
                continue
            remaining = self.pool_remaining(self.b_channels, ch)
            trans_coeff = DIRECTION_COEFF.get(direction, 1.0)
            effective = val * remaining * trans_coeff
            self.b_channels[ch] = min(1.0, self.b_channels[ch] + effective * INC_RATE_CORTEX)
            self.b_direction[ch] = direction  # 标记方向

    # ── 衰减 (注意力分配 + 方向调整) ──
    def _decay_attention(self, channels: dict, halflife_map: dict,
                         dt: float, dir_map: dict):
        if not channels:
            return
        champion = max(channels, key=channels.get)
        for ch, val in channels.items():
            if val <= 0.005:
                channels[ch] = 0.0
                continue
            hl = halflife_map[ch]
            # 注意力乘数
            focus_mul = FOCUS_MULTIPLIER if ch == champion else UNFOCUS_MULTIPLIER
            # 方向乘数 (gossip → 2x衰减)
            dir_mul = DIRECTION_DECAY.get(dir_map.get(ch, "你"), 1.0)
            effective_hl = hl * focus_mul * dir_mul
            decay = 0.5 ** (dt / effective_hl)
            channels[ch] = val * decay

    # ── 合并供表情层使用 ──
    def merged_for_face(self) -> dict:
        """合并双路 → AU 驱动"""
        # 计算各池当前总和
        da_sum   = self.pool_sum(self.b_channels, "DA")
        ne_sum   = self.pool_sum(self.b_channels, "NE")
        s5_sum   = self.pool_sum(self.b_channels, "5HT")
        oxt_sum  = self.pool_sum(self.b_channels, "OXT")
        misc_sum = sum(self.b_channels.get(e, 0.0) for e in NEURO_POOLS["MISC"])

        max_pool = max(da_sum, ne_sum, s5_sum, oxt_sum, misc_sum)

        result = {
            "face_happy":     self.b_channels.get("happy", 0),
            "face_sad":       self.b_channels.get("sad", 0),
            "face_angry":     self.b_channels.get("angry", 0),
            "face_scared":    self.b_channels.get("scared", 0),
            "face_hurt":      self.b_channels.get("hurt", 0),
            "face_shy":       self.b_channels.get("shy", 0),
            "face_surprised": self.b_channels.get("surprised", 0),
            "face_worried":   self.b_channels.get("worried", 0),
            "face_care":      self.b_channels.get("care", 0),
            "face_contempt":  self.b_channels.get("contempt", 0),
            "face_tense":     self.a_channels.get("tense", 0),
            "face_orient":    self.a_channels.get("orient", 0),
            "face_approach":  self.a_channels.get("approach", 0),
        }
        self._pool_relax = max(0.0, 1.0 - max_pool)
        return result

    def get_state(self) -> dict:
        return {
            "amygdala": dict(self.a_channels),
            "cortex":   dict(self.b_channels),
            "direction": dict(self.b_direction),
        }

    def get_expression_competition(self) -> dict:
        return {
            "amygdala": resolve_expression_competition(
                self.a_channels, exclude={"relax"}
            ),
            "cortex": resolve_expression_competition(
                self.b_channels
            ),
        }

    def get_expression_layer(self) -> dict:
        """
        表情运算层：上下脸分别叠加杏仁核和皮层的竞争胜出情绪。
        
        Returns:
            {
                "upper": {情绪名: 总强度, "_sources": {情绪名: {杏仁核:x, 皮层:y}}},
                "lower": {同上}
            }
        """
        comp = self.get_expression_competition()
        
        def _collect(system):
            """从竞争结果中提取 (情绪名, 强度) 列表，按全脸/分脸分配"""
            c = comp.get(system, {})
            mode = c.get("mode", "none")
            items = []
            if mode == "full":
                if c.get("champion"):
                    items.append((c["champion"][0], c["champion"][1], "both"))
            elif mode == "split":
                if c.get("champion"):
                    items.append((c["champion"][0], c["champion"][1], "upper"))
                if c.get("runner_up"):
                    items.append((c["runner_up"][0], c["runner_up"][1], "lower"))
            return items
        
        def _merge(items_a, items_c, face_region):
            """合并杏仁核和皮层对同一个脸区的贡献"""
            merged = {}
            sources = {}
            for items, tag in [(items_a, "杏仁核"), (items_c, "皮层")]:
                for emo, val, region in items:
                    if region != face_region and region != "both":
                        continue
                    if emo not in merged:
                        merged[emo] = 0.0
                        sources[emo] = {}
                    merged[emo] = min(1.0, merged[emo] + val)
                    sources[emo][tag] = round(val, 3)
            # 只保留有意义的值
            result = {}
            for emo in list(merged.keys()):
                result[emo] = round(merged[emo], 4)
            return result, sources
        
        a_items = _collect("amygdala")
        c_items = _collect("cortex")
        
        upper_vals, upper_src = _merge(a_items, c_items, "upper")
        lower_vals, lower_src = _merge(a_items, c_items, "lower")
        
        return {
            "upper": {"emotions": upper_vals, "sources": upper_src},
            "lower": {"emotions": lower_vals, "sources": lower_src},
        }


# ═══════════════ VAD → AU 强度 (sigmoid 非线性映射) ═══════════════
VISIBILITY_THRESHOLD = {
    "tense":    0.40,
    "orient":   0.40,
    "happy":    0.20,
    "angry":    0.40,
    "sad":      0.40,
    "surprised":0.40,
    "scared":   0.50,
    "disgust":  0.40,
    "contempt": 0.40,
    "hurt":     0.40,
    "shy":      0.35,
    "worried":  0.40,
    "care":     0.35,
    "relax":    0.30,
}

SIGMOID_K = 12.0


def vad_to_au(vad_value: float, threshold: float, k: float = SIGMOID_K) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-k * (vad_value - threshold)))


# ═══════════════ 表情竞争决策 ═══════════════

def resolve_expression_competition(vad_channels: dict, exclude: set = set()) -> dict:
    """
    在 VAD 通道内部做表情竞争决策。
    
    Args:
        vad_channels: VAD 值字典 {emotion: intensity}
        exclude: 不参与竞争的通道 (如杏仁核的 relax)
    
    Returns:
        {
            champion: (情感名, 强度),
            runner_up: (情感名, 强度),
            ratio: champion / runner_up (无亚军时=0, 不用 inf 防 JSON 序列化崩)
            mode: "full" 或 "split"
        }
    """
    # 过滤低值 + 排除项
    valid = {k: v for k, v in vad_channels.items()
             if v >= 0.01 and k not in exclude}
    
    if not valid:
        return {"champion": None, "runner_up": None,
                "ratio": 0, "mode": "none"}
    
    sorted_items = sorted(valid.items(), key=lambda x: -x[1])
    
    champ_name, champ_val = sorted_items[0]
    if len(sorted_items) > 1:
        ru_name, ru_val = sorted_items[1]
    else:
        ru_name, ru_val = None, 0.0
    
    # 无亚军(唯一情绪) → ratio=0 (不可用 inf: json.dumps 会输出非法 JSON 的 Infinity 字面量, 前端 JSON.parse 崩)
    ratio = champ_val / ru_val if ru_val > 0 else 0.0
    # ── 倾听模式: 只用最强表情控制全脸 ──
    mode = "full"
    # ── TODO: 说话模式 ──
    # 说话时下脸由 TTS/嘴型控制, 上脸保持倾听情绪
    # mode = "split" if ratio < 2.0 else "full"
    
    return {
        "champion": (champ_name, round(champ_val, 3)),
        "runner_up": (ru_name, round(ru_val, 3)) if ru_name else None,
        "ratio": round(ratio, 2),
        "mode": mode,
    }


def map_emotions_to_au(face_channels: dict, comp_result: dict = None) -> dict:
    """
    AU 强度映射，受表情竞争决策约束。
    
    Args:
        face_channels: merged_for_face() 输出 {face_xxx: vad值}
        comp_result: get_expression_competition() 输出,
            若提供则只输出竞争中胜出的情绪
    
    Returns:
        {emotion: au_strength}
    """
    # 从竞争结果中提取允许的情绪集合
    allowed = set()
    if comp_result:
        for tag, comp in [("face_", comp_result.get("amygdala", {})),
                          ("", comp_result.get("cortex", {}))]:
            mode = comp.get("mode", "none")
            if mode == "full" and comp.get("champion"):
                allowed.add("face_" + comp["champion"][0])
            elif mode == "split":
                if comp.get("champion"):
                    allowed.add("face_" + comp["champion"][0])
                if comp.get("runner_up"):
                    allowed.add("face_" + comp["runner_up"][0])
            # 'none' → 不允许任何情绪
    
    au = {}
    for emotion, value in face_channels.items():
        if value < 0.01:
            continue
        # 如果提供了竞争结果，过滤
        if comp_result and emotion not in allowed:
            continue
        emo = emotion.replace("face_", "")
        threshold = VISIBILITY_THRESHOLD.get(emo, 0.40)
        au_val = vad_to_au(value, threshold)
        if au_val >= 0.03:
            au[emo] = round(au_val, 4)
    return au


# ═══════════════ 测试 ═══════════════
if __name__ == "__main__":
    dv = DualVAD()

    def show_emotions(title, dv):
        s = dv.get_state()
        a = {k: round(v, 3) for k, v in s["amygdala"].items() if v > 0.001}
        b = {k: round(v, 3) for k, v in s["cortex"].items() if v > 0.001}
        face = dv.merged_for_face()
        au = map_emotions_to_au(face)
        da = round(dv.pool_sum(dv.b_channels, "DA"), 3)
        ne = round(dv.pool_sum(dv.b_channels, "NE"), 3)
        s5 = round(dv.pool_sum(dv.b_channels, "5HT"), 3)
        oxt= round(dv.pool_sum(dv.b_channels, "OXT"), 3)
        rlx= round(dv._pool_relax, 3)
        # 转译成表情描述
        emoji_map = {"tense":"紧张","orient":"好奇","happy":"开心","angry":"生气","sad":"难过",
                     "scared":"害怕","hurt":"受伤","shy":"害羞",
                     "surprised":"惊讶","worried":"担心","care":"关心",
                     "contempt":"轻蔑"}
        # 去掉 relax — 无表情才是放松
        non_relax_au = {k:v for k,v in au.items() if k != "face_relax"}
        if non_relax_au:
            au_sorted = sorted(non_relax_au.items(), key=lambda x: -x[1])
            desc = []
            for k,v in au_sorted[:3]:
                cn = emoji_map.get(k.replace("face_",""), k)
                if v > 0.8:
                    desc.append(f"😠{cn}!!" if cn in ["生气","紧张"] else f"😄{cn}!!")
                elif v > 0.4:
                    desc.append(f"😐{cn}")
                else:
                    desc.append(f"🙂{cn}")
            face_desc = " | ".join(desc)
        else:
            face_desc = "😌放松 (无活跃表情)"
        print(f"\n── {title}")
        print(f"  杏仁核: {a}")
        print(f"  皮层:   {b}")
        print(f"  池: DA={da}  NE={ne}  5HT={s5}  OXT={oxt}  relax={rlx}")
        print(f"  AU: {au}")
        print(f"  🎭表情: {face_desc}")

    def frame(title, amy, ctx, direction="你"):
        for i in range(3):
            dv.update(amygdala_signals=amy, cortex_signals=ctx, direction=direction)
            show_emotions(f"{title} 帧{i+1}", dv)
        for i in range(2):
            dv.update(amygdala_signals={}, cortex_signals={})
            show_emotions(f"沉默 {i+1}", dv)

    frame("你笨蛋(说你)",
          {"tense": 0.4},
          {"hurt": 0.5, "angry": 0.3})

    frame("那个同事太坏了 说别人",
          {"tense": 0.1},
          {"contempt": 0.4, "angry": 0.2},
          direction="别的")

    frame("我心情好差 说自己",
          {"approach": 0.1, "orient": 0.1},
          {"sad": 0.6})

    frame("你好可爱 说你",
          {"relax": 0.3},
          {"happy": 0.5, "shy": 0.3})

    print("\n" + "=" * 60)
    print("最终状态")
    show_emotions("END", dv)
