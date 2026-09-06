# 情绪 → Live2D 动作映射表

> 2026-06-09 | 待逐条测试验证
>
> 设计理念：可爱/萌系风格 + 傲娇元素
> - 快乐：元气满满，眯眯眼 + 身体微前倾
> - 悲伤：低头垂眼 + 白大褂（抱住自己取暖的感觉）
> - 生气：傲娇扭头 + 脸红 + 眉毛压低（不是真凶，是娇嗔）
> - 惊讶：瞪大眼 + 头微后仰 + 蚊香眼
> - 焦虑：慌张表情 + 身体微缩 + 眼珠乱飘
> - 好奇：歪头 + 眼睛微微发亮 + 乖巧表情
> - 困惑：歪头另一边 + 眉毛不对称 + 眼珠上飘

---

## 可用参数速查

### 面部参数（每个情绪都会用）

| 参数 | 中文名 | 范围 | 说明 |
|------|--------|------|------|
| ParamAngleX | 角度 X | ±30° | **+ 左转, - 右转** (⚠️ 此模型X/Y互换, 非标准) |
| ParamAngleY | 角度 Y | ±30° | **+ 抬头, - 低头** (⚠️ 此模型X/Y互换, 非标准) |
| ParamAngleZ | 角度 Z (头部歪) | ±30° | **+ 左歪(靠左肩), - 右歪(靠右肩)** (⚠️ 此模型Z方向反转) |
| ParamEyeLOpen | 左眼开闭 | 0~1 | 1=全开 |
| ParamEyeROpen | 右眼开闭 | 0~1 | 1=全开 |
| ParamEyeLSmile | 左眼微笑 | 0~1 | 1=眯成月牙 |
| ParamEyeRSmile | 右眼微笑 | 0~1 | 1=眯成月牙 |
| ParamEyeBallX | 眼珠 X | ±1 | + 向左看, - 向右看 |
| ParamEyeBallY | 眼珠 Y | ±1 | + 向上看, - 向下看 |
| ParamBrowLY | 左眉上下 | -1~1 | + 抬眉, - 压眉 |
| ParamBrowRY | 右眉上下 | -1~1 | + 抬眉, - 压眉 |
| ParamBrowLAngle | 左眉角度 | ±30° | + 内端上挑, - 内端下压 |
| ParamBrowRAngle | 右眉角度 | ±30° | + 内端上挑, - 内端下压 |
| ParamBrowLForm | 左眉变形 | -1~1 | 形状扭曲 |
| ParamBrowRForm | 右眉变形 | -1~1 | 形状扭曲 |
| ParamMouthForm | 嘴形 | -1~1 | + 微笑, - 扁嘴 |
| ParamMouthOpenY | 嘴开闭 | 0~1 | 1=张大 |
| ParamCheek | 脸颊泛红 | 0~1 | **本模型无效果** |

### 身体参数

| 参数 | 中文名 | 范围 | 说明 |
|------|--------|------|------|
| ParamBodyAngleX | 身体旋转 X | ±30° | **+ 左转, - 右转** (⚠️ 此模型X/Y互换) |
| ParamBodyAngleY | 身体旋转 Y | ±30° | **+ 前倾, - 后仰** (⚠️ 此模型X/Y互换) |
| ParamBodyAngleZ | 身体旋转 Z | ±30° | + 右倾, - 左倾 |
| ParamBreath | 呼吸 | ±1 | 待机层自动控制 |

### 头发/物理参数

| 参数 | 说明 |
|------|------|
| ParamHairFront / Front2 | 前发摇动 |
| ParamHairSide / Side2 | 侧发摇动 |
| ParamHairBack / Back2 | 后发摇动 |
| Param3 / Param4 | 兽耳 |
| Param5 / Param6 / Param7 / Param8 | 衣服饰品 |
| Param19 / Param20 | 胸部物理 |
| Param21 / Param22 | 裙子物理 |
| Param / Param2 | 衣服物理 |

### 特殊表情配件（数值影响强度）

| 参数 | 中文名 | 实测效果 |
|------|--------|---------|
| Param10 | 蓝色眼镜 | 0=无, >0=显示蓝色眼镜 |
| Param11 | 囧表情 | 0=无, >0=囧眼+嘴型 |
| Param12 | 蚊香眼 | 0=无, >0=螺旋眼 |
| Param13 | 果冻眼 | 0=无, >0=果冻质感眼睛 |
| Param14 | 果冻眼 | (同上) |
| Param18 | **不穿**白大褂 | **1=不穿, 0=穿** (⚠️ 反向) |
| Param23 | **白眼** | 0=正常, 0.5=半白眼, 1.5=全白眼向上翻 |

---

## 7 情绪动作定义

### 1. 😊 快乐 (happy)

**角色感觉**：元气满满、想和观众分享开心的事情

| 类别 | 参数 | 值 | 说明 |
|------|------|----|------|
| 眼睛 | ParamEyeLSmile / ParamEyeRSmile | **0.7** | 眯眯眼，月牙形（核心特征） |
| 眼睛 | ParamEyeLOpen / ParamEyeROpen | **0.7** | 眼睛微闭（被脸颊推上去） |
| 眉毛 | ParamBrowLY / ParamBrowRY | **+0.3** | 眉毛微微上扬 |
| 嘴巴 | ParamMouthForm | **+0.5** | 微笑弧线 |
| 嘴巴 | ParamMouthOpenY | **0.05** | 微张嘴（呼吸感） |
| 脸颊 | ParamCheek | **0.4** | 微微脸红（健康红晕） |
| 头部 | ParamAngleZ | **+6°** | 头向右歪（卖萌经典姿势） |
| 头部 | ParamAngleX | **+3°** | 微低（亲昵感） |
| 身体 | ParamBodyAngleX | **+5°** | 身体前倾（想靠近说话） |
| 身体 | ParamBodyAngleY | **+3°** | 身体微侧 |
| 配件 | Param23 (乖巧) | **0.3** | 叠加一点乖巧感 |

**强度缩放**：
- 律度 1~3 (有点开心)：眯眯眼 0.4，歪头 3°，身体前倾 2°
- 律度 3~4 (很开心)：眯眯眼 0.7，歪头 6°，身体前倾 5°
- 律度 4+ (超级开心)：眯眯眼 0.9，歪头 10°，身体前倾 8° + 肩膀上下跳

---

### 2. 😢 悲伤 (sad)

**角色感觉**：消沉、想缩起来、需要安慰

| 类别 | 参数 | 值 | 说明 |
|------|------|----|------|
| 眼睛 | ParamEyeLOpen / ParamEyeROpen | **0.5** | 半闭眼（无精打采） |
| 眼睛 | ParamEyeLSmile / ParamEyeRSmile | **0** | 不笑 |
| 眼珠 | ParamEyeBallY | **-0.2** | 眼珠微向下看 |
| 眉毛 | ParamBrowLAngle / ParamBrowRAngle | **+15°** | 眉毛内端上挑（"八"字眉） |
| 眉毛 | ParamBrowLY / ParamBrowRY | **-0.15** | 眉头整体略压 |
| 嘴巴 | ParamMouthForm | **-0.3** | 微扁嘴 |
| 嘴巴 | ParamMouthOpenY | **-0.1** | 嘴紧闭 |
| 脸颊 | ParamCheek | **0.1** | 淡淡忧伤红晕 |
| 头部 | ParamAngleX | **-6°** | 低头 |
| 头部 | ParamAngleY | **+4°** | 头微偏（回避视线） |
| 身体 | ParamBodyAngleX | **-5°** | 身体后缩 |
| 身体 | ParamBodyAngleZ | **-3°** | 身体微向右倾（不对称的无力感） |
| 配件 | Param18 (白大褂) | **1** | 穿上白大褂 = 抱紧自己取暖 |

---

### 3. 😠 生气 (angry) — 傲娇风格

**角色感觉**：嘴上凶但脸红了 → "哼！才不是关心你呢！"

| 类别 | 参数 | 值 | 说明 |
|------|------|----|------|
| 眼睛 | ParamEyeLOpen / ParamEyeROpen | **0.8** | 眼睛睁着（瞪） |
| 眼睛 | ParamEyeLSmile / ParamEyeRSmile | **0** | 不笑 |
| 眉毛 | ParamBrowLY / ParamBrowRY | **-0.6** | 眉毛压下来 |
| 眉毛 | ParamBrowLAngle / ParamBrowRAngle | **-10°** | 眉毛内端下压（"V"字眉） |
| 嘴巴 | ParamMouthForm | **+0.1** | 嘴巴微嘟（不是真凶，是闹别扭） |
| 嘴巴 | ParamMouthOpenY | **0.03** | 微张嘴（"哼！"的嘴型） |
| 脸颊 | ParamCheek | **0.6** | 脸红！(傲娇核心：越生气越害羞) |
| 头部 | ParamAngleY | **+12°** | 头转向一边（"不看你了！"） |
| 头部 | ParamAngleZ | **-5°** | 头反向微歪 |
| 眼珠 | ParamEyeBallX | **+0.3** | 眼珠往扭头方向看（偷瞄） |
| 身体 | ParamBodyAngleY | **+8°** | 身体也转过去 |
| 身体 | ParamBodyAngleX | **-3°** | 身体后仰（"哼"的姿势） |

**傲娇核心**：生气 + 脸红 + 扭头但眼珠还在偷看 = "嘴上说不要但心里在乎"

---

### 4. 😲 惊讶 (surprise)

**角色感觉**：眼睛瞪大、脑子一片空白、说不出话

| 类别 | 参数 | 值 | 说明 |
|------|------|----|------|
| 眼睛 | ParamEyeLOpen / ParamEyeROpen | **1.0** | 眼睛全开（瞪大） |
| 眼睛 | ParamEyeLSmile / ParamEyeRSmile | **0** | 不笑 |
| 眉毛 | ParamBrowLY / ParamBrowRY | **+0.7** | 眉毛高抬 |
| 嘴巴 | ParamMouthForm | **0** | 嘴巴自然形状 |
| 嘴巴 | ParamMouthOpenY | **0.15** | 嘴微张（"诶？"） |
| 头部 | ParamAngleX | **-4°** | 头微后仰 |
| 身体 | ParamBodyAngleX | **-5°** | 身体后仰 |
| 配件 | Param12 (蚊香眼) | **0.5** | 晕眩/惊讶的蚊香眼（轻度） |

**强度缩放**：
- 律度 1~3 (有点惊讶)：眼睛 0.9，蚊香眼 0
- 律度 3~4 (很惊讶)：眼睛 1.0，蚊香眼 0.5
- 律度 4+ (超级震惊)：眼睛 1.0，蚊香眼 1.0，嘴张大 0.3

---

### 5. 😨 焦虑 (fear)

**角色感觉**：紧张、不安、想躲起来

| 类别 | 参数 | 值 | 说明 |
|------|------|----|------|
| 眼睛 | ParamEyeLOpen / ParamEyeROpen | **0.85** | 眼睛睁着（警觉） |
| 眼睛 | ParamEyeLSmile / ParamEyeRSmile | **0** | 不笑 |
| 眼珠 | ParamEyeBallX / ParamEyeBallY | **随机震动** | 眼珠左右飘忽不定 |
| 眉毛 | ParamBrowLAngle / ParamBrowRAngle | **+20°** | 眉毛内端高挑（担忧） |
| 眉毛 | ParamBrowLY / ParamBrowRY | **+0.5** | 眉毛整体抬起 |
| 嘴巴 | ParamMouthForm | **-0.1** | 嘴微紧张 |
| 嘴巴 | ParamMouthOpenY | **0.03** | 微张（紧张呼吸） |
| 头部 | ParamAngleX | **+4°** | 头微低（不敢看） |
| 头部 | ParamAngleY | **+5°** | 头微偏 |
| 身体 | ParamBodyAngleX | **-6°** | 身体后缩 |
| 身体 | ParamBodyAngleY | **-3°** | 身体微收 |
| 配件 | Param11 (慌张) | **0.5** | 慌张嘴型（轻度） |

---

### 6. 🤔 好奇 (think)

**角色感觉**：歪头思考、眼睛亮晶晶、"嗯？那是什么？"

| 类别 | 参数 | 值 | 说明 |
|------|------|----|------|
| 眼睛 | ParamEyeLOpen / ParamEyeROpen | **0.9** | 眼睛微微睁大（专注） |
| 眼睛 | ParamEyeLSmile / ParamEyeRSmile | **0.15** | 微带笑意（好奇是积极的） |
| 眼珠 | ParamEyeBallY | **+0.15** | 眼珠微向上（"想问题"） |
| 眉毛 | ParamBrowLY / ParamBrowRY | **+0.25** | 眉毛微抬 |
| 眉毛 | 不对称 | 左+0.25，右+0.1 | 单边眉微挑（"嗯？"） |
| 嘴巴 | ParamMouthForm | **+0.15** | 嘴角微翘 |
| 嘴巴 | ParamMouthOpenY | **0.03** | 微张嘴（想说又没说） |
| 头部 | ParamAngleZ | **-8°** | 头向左歪（经典好奇姿势） |
| 头部 | ParamAngleY | **-3°** | 头微转 |
| 眼珠 | ParamEyeBallX | **+0.1** | 眼珠往歪头方向（看自己在想的东西） |
| 身体 | ParamBodyAngleY | **-5°** | 身体微侧 |
| 配件 | Param23 (乖巧) | **0.2** | 一点乖巧感 |

**核心动作**：头往左歪 8° — 这是动画里最经典的"嗯？"好奇姿势。

---

### 7. 😵 困惑 (think — 与好奇共用 TTS 名但 Live2D 参数不同)

**角色感觉**：歪头（另一边）、眉毛不对称、脑子转不过弯

| 类别 | 参数 | 值 | 说明 |
|------|------|----|------|
| 眼睛 | ParamEyeLOpen / ParamEyeROpen | **0.75** | 眼睛微眯（费解） |
| 眼睛 | ParamEyeLSmile / ParamEyeRSmile | **0** | 不笑 |
| 眼珠 | ParamEyeBallY | **+0.25** | 眼珠翻向上（经典"我想想"） |
| 眉毛 | ParamBrowLY / ParamBrowRY | **-0.1** | 眉毛微压低 |
| 眉毛 | 不对称 | 左-0.3 右+0.15 | 一高一低（"啥意思？"）|
| 嘴巴 | ParamMouthForm | **-0.15** | 嘴微扁 |
| 嘴巴 | ParamMouthOpenY | **0.02** | 微张 |
| 头部 | ParamAngleZ | **+8°** | 头向**右**歪（与好奇相反） |
| 头部 | ParamAngleY | **+5°** | 头微转另一边 |
| 身体 | ParamBodyAngleZ | **+5°** | 身体微向右倾 |

**与好奇的区分**：好奇歪左边 + 眼睛亮 + 嘴角翘；困惑歪右边 + 眉毛不对称 + 眼珠上翻。

---

## 强度缩放规则（律度 → 参数幅度）

所有"值"列的数字是**最大强度**（律度 ≥ 4.0）。实际缩放：

```python
def scale_value(base_value, intensity, range_lo=1.0, range_hi=4.0):
    """律度映射到参数幅度"""
    if intensity < range_lo:
        return 0  # 不触发
    t = min(1.0, (intensity - range_lo) / (range_hi - range_lo))
    return base_value * t
```

### 快乐示例

| 律度 | 眯眯眼 | 歪头 | 身体前倾 | 脸红 |
|------|--------|------|---------|------|
| 1.2 (有点开心) | 0.14 | 1.2° | 1° | 0.08 |
| 2.0 (有点开心+) | 0.28 | 2.4° | 2° | 0.16 |
| 3.0 (开心) | 0.47 | 4° | 3.3° | 0.27 |
| 4.0 (很开心) | 0.7 | 6° | 5° | 0.4 |
| 5.0 (超开心) | 0.7 | 6° | 5° | 0.4 |

> 注：律度 > 4.0 后不再继续放大（clamp），改用**额外动作**区分（如超级开心时连续小跳）

---

## 尚未覆盖的模型能力

以下参数模型的 `.cdi3.json` 里定义了但目前没用到：

| 参数 | 可能用途 |
|------|---------|
| Param3/4 (兽耳) | 随情绪抖动（开心时立起来？） |
| Param5/6/7/8 (衣服饰品) | 待探索 |
| Param19/20 (胸X) | 呼吸联动或身体前倾时的物理 |
| Param21/22 (裙子) | 身体旋转时的物理 |
| 呆毛色块 (ParamGroup) | 随情绪摇摆 |

---

## 8. 🆕 关心 (care) — 关切合并

**角色感觉**：温柔前倾、关切注视、"你没事吧？"

**合并说明**:
- care(皮层) 和 approach(杏仁核) 是同一张脸、同一套 AU 参数
- 当前: approach 通过别名映射共享 care 的参数集, 表达式竞争层的 max() 自动处理谁更强
- 未来: 若需要更精细的叠加, 见下方"双路 VAD 合并公式"

### 双路 VAD 合并公式 (待实现)

当两个触发源(如皮层care + 杏仁核approach)同时活跃时, 可用此公式合并:

```
n1 = max(√norm_care, √norm_approach)    # 较大值
n2 = min(√norm_care, √norm_approach)    # 较小值
combined = n1 + (1 - n1) × n2
```

行为特性:
- 单路活跃 → 退化为 √norm, 等价于原公式
- 双路同时活跃 → 强度比单路高, 但永不超 1.0
- 弱信号叠加时有互相"抬升"效果
- 较大值占主体, 较小值填充剩余空间

**当前状态**: 公式已记录, 暂未实现。现用别名映射 + 表情竞争 max() 替代。

| 维度 | 参数 | 值 (满强度) | 说明 |
|------|------|------|------|
| 头部 | ParamAngleY | +3° | 微低(倾听) |
| 头部 | ParamAngleZ | +5° | 微歪(关切) |
| 身体 | ParamBodyAngleY | -6° | 明显前倾 |
| 眼睛 | ParamEyeLOpen/ROpen | 0.8 | 温柔睁着 |
| 眼睛 | ParamEyeLSmile/RSmile | 0.3 | 微带笑意 |
| 眉毛 | ParamBrowLY/RY | +0.3 | 微内压 |
| 眉毛 | ParamBrowLAngle/RAngle | +0.5° | 内端微挑 |
| 嘴巴 | ParamMouthForm | +0.8 | 温和微笑 |
| 脸颊 | ParamCheek | 0.3 | 微微泛红 |
| 配件 | Param23(乖巧) | 0.5 | 温顺感 |

**VAD驱动参数 (已实现)**:
- **映射函数**: `param = max_val × ((vad - thr) / (1 - thr))^0.5`
- 幂函数 0.5（平方根），前期陡后期缓

| 参数 | 满值(max) | 阈值(thr) | hold | VAD=0.3 | VAD=0.5 | VAD=0.8 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| ParamMouthForm | 2.40 | 0.25 | — | 0.62 | 1.39 | 2.06 |
| ParamMouthOpenY | 0.30 | 0.30 | — | 0.00 | 0.16 | 0.25 |
| ParamBrowLY/RY | -0.90 | 0.20 | — | -0.32 | -0.55 | -0.78 |
| ParamBrowLAngle/RAngle | 1.50 | 0.25 | — | 0.39 | 0.87 | 1.28 |
| ParamEyeLSmile/RSmile | 0.45 | 0.20 | — | 0.16 | 0.28 | 0.39 |
| ParamEyeBallY | -0.45 | 0.15 | — | -0.19 | -0.29 | -0.39 |
| ParamCheek | 0.90 | 0.25 | — | 0.23 | 0.52 | 0.77 |
| ParamAngleY | 3° | 0.15 | ✅ | 3 | 3 | 3 |
| ParamAngleZ | 5° | 0.15 | ✅ | 5 | 5 | 5 |
| ParamBodyAngleY | -6° | 0.15 | ✅ | -6 | -6 | -6 |

**代码位置**: `emotion/live2d_mapper.py`

**空闲动画 (待实现)**:
- 歪头微摆: multi_perlin ±3°, 无规律
- 眼珠温柔游移: random_walk ±0.15

---

## 9. 全部表情参数总表 (×3 满强度)

所有参数以 `live2d_mapper.py` 为准。thr=Calvo 2016 可见阈值。

| 表情 | thr | 面部核心 | 头部/身体 | 配件 |
|:----:|:---:|:---------|:---------|:----:|
| happy😊 | 0.20 | EyeLSmile 0.7, EyeLOpen 0.85, MouthForm 2.5, BrowLY -0.5, Cheek 1.25 | AngleZ +13, BodyAngleY -5, BodyAngleZ -8 | — |
| sad😢 | 0.40 | EyeLOpen 0.7, BrowLY 1.5, BrowLAngle -1.5, MouthForm -1.5, EyeBallY -0.5 | AngleY +5, BodyAngleY -3 | — |
| angry😠 | 0.40 | BrowLY -2.0, BrowLAngle 1.5, EyeLOpen 0.9, MouthForm 1.0, Cheek 1.0 | AngleZ +25, BodyAngleY -8, BodyAngleZ -8 | — |
| surprised😲 | 0.40 | BrowLY 1.5, EyeLOpen 1.2, MouthOpenY 1.0 | AngleY +3, BodyAngleY -3 | — |
| scared😨 | 0.50 | BrowLY 1.0, EyeLOpen 1.1, MouthForm -0.5 | AngleY -15, BodyAngleY +6 | — |
| hurt💔 | 0.40 | BrowLY 2.5, BrowLAngle 2.5, EyeLOpen 0.3, MouthForm -3.0, Cheek 1.0 | AngleY -25, AngleX -12, BodyAngleY -20 | — |
| worried😰 | 0.40 | BrowLY 1.6, BrowLAngle -2.0, EyeLOpen 1.1, MouthForm -1.0, MouthOpenY 0.3 | AngleY -10, AngleX +8, BodyAngleY -12 | — |
| contempt🙄 | 0.40 | EyeLOpen 0.6, EyeBallX -0.8, MouthForm 1.0, Cheek 1.5 | AngleY +15, AngleX +25, BodyAngleY -16 | 眼睛状态机(翻白眼/偷瞄/闭眼) |
| shy😳 | 0.35 | EyeLOpen 0.6, EyeLSmile 0.6, EyeBallY -0.8, MouthForm 0.8, Cheek 2.0 | AngleY -25, AngleX +20, BodyAngleY -12 | — |
| orient🤔 | 0.40 | BrowLY 0.9, EyeLOpen 1.09, EyeBallX 0.3, MouthForm 0.3 | AngleZ -8, BodyAngleY -3 | — |
| tense😬 | 0.40 | BrowLY -0.9, BrowLForm 0.9, EyeLOpen 1.12 | AngleY +2, BodyAngleY +6 | — |
| care💗 | 0.35 | EyeLOpen 0.95, BrowLY -0.3, EyeBallY -0.2, MouthForm 0.5 | BodyAngleY -10, AngleZ +4, AngleY +2 | — |
| approach🤗 | 0.35 | 同care (别名映射, 共享参数集) | — | — |

### 特殊动画

| 表情 | 动画 | 实现 |
|:----:|:----|:----|
| scared | 眼珠抖动(随机 ±0.3) | `startFearShake()` — setInterval 150ms |
| contempt | 翻白眼↔偷瞄↔闭眼 循环状态机 | `startContemptCycle()` — 闭眼2.2s/翻白眼+随机偷瞄, 逾规则切换 |

### 参数方向 (更正后)

| 参数 | 实际效果 | 说明 |
|:----:|:--------|:----|
| ParamAngleX + | 左转 | ⚠️ 此模型 X/Y 互换 |
| ParamAngleY + | 抬头 | ⚠️ 此模型 X/Y 互换 |
| ParamAngleZ + | 左歪 | ⚠️ 此模型 Z 方向反转 |
| ParamBodyAngleX + | 左转(侧扭) | ⚠️ 同头部 |
| ParamBodyAngleY + | 前倾 | ⚠️ 同头部 |

---

## 测试顺序建议

1. **先测快乐** — 眯眯眼 + 歪头是最直观的效果
2. **再测生气** — 验证扭头 + 脸红 + 偷瞄的傲娇感
3. **然后测悲伤** — 低头 + 白大褂
4. **惊讶** — 瞪眼
5. **恐惧** — 低头 + 眼珠抖动
6. **受伤** — 表演级委屈 (低头+闭眼+扁嘴)
7. **害羞** — 低头+脸红+憋笑
8. **轻蔑** — 翻白眼状态机
9. **好奇 vs 困惑** — 验证左右歪头的区分度
10. **关心** — 前倾 + 温柔注视 + 微笑
