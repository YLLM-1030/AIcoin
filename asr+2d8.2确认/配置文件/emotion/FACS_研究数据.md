# FACS 情绪引擎 — 研究数据汇总

> 整理于 2026-07-23

## 1. 说话者情绪检测

### 所用模型
`paraphrase-multilingual-MiniLM-L12-v2`，CPU 推理 ~5ms/次，384维。

### 原型数据来源
- **TIX007/chinese-sentiment**: joy(189), sadness(186), anger(186), fear(199), surprise(176), love(184)
- **Weibo 情绪验证词库**: disgust, contempt 的验证词汇
- 手写扩展: disgust 66句, contempt 77句, neutral 51句
- **总计**: 428句, 8分类

### 检测方法
embedding → 原型余弦距离 → softmax ×5 → × 长度置信度(min(字长/10, 1.0))

### 情绪分类体系
Ekman 7 基本情绪: angry, disgust, fear, happy, sad, surprise, contempt + neutral

---

## 2. 杏仁核层(低通路/第一反应)

### 来源: LeDoux (1996) 双通路理论
- 低通路: 丘脑→杏仁核, 12-74ms, 粗糙但即时
- 高通路: 丘脑→皮层→前额叶, ~500ms, 精确但慢
- 第一反应不由语义决定, 由信号特征决定

### 杏仁核检测特征

| 信号特征 | 反应 | 时间 |
|----------|------|------|
| 激烈/攻击性(音量高、语速快、负面词) | tense(绷紧/警戒) | 12-50ms |
| 柔和/温暖(轻声、慢速、正面词) | relax(放松/开放) | 12-50ms |
| 新异/意外(疑问句、突然变化) | orient(聚焦/歪头) | 12-50ms |
| 悲伤(低落、慢速、叹气) | approach(前倾/关切) | 50-150ms |
| 中性 | 无触发, 维持基线 | — |

### 杏仁核半衰期 (Sources: startle reflex research, orienting response research)

| 信号 | 半衰期 | 来源依据 |
|------|--------|----------|
| tense(惊跳反射) | 0.3s | 肌肉<1s恢复(Britannica), habituates 1-5 trials |
| orient(定向反应) | 2.5s | 定向反应持续3-10s(ScienceDirect) |
| approach(关切) | 1.0s | 主动社交行为, 回复快于反射 |
| relax(放松) | 2.0s | 温暖感消散较慢 |

---

## 3. 皮层层(高通路/理解后情绪)

### 来源: Hatfield 情绪感染理论(1992)

- 听者自动模仿说者表情/语调 → 生理反馈 → 真实感受
- 负面情绪传播强度 > 正面情绪(约1.3×)

### 来源: 评价理论(Lazarus, 1991; Siemer/Mauss/Gross 2007)

- 同一事件同一人可因评价方式不同产生不同情绪
- 影响因素: 批评者身份、公开/私下、是否有错、当前心情、权力差

### 被批评时四种反应类型

| 类型 | 情绪 | 触发 |
|------|------|------|
| 自责型 | guilt, shame, depression | 低自尊 |
| 反击型 | anger, retaliation | 高自尊/被冒犯 |
| 内伤型 | hurt, sadness, withdrawal | 敏感/在意评价 |
| 分析型 | curiosity, calm | 高自我认同 |

### 愤怒冰山理论

表面愤怒下藏着: hurt(被伤害), fear(害怕失去), shame(感到羞耻), powerlessness(无力感)

### 皮层情绪半衰期 (Source: EDM 2025, Nature Human Behaviour 2018)

| 情绪 | 半衰期 | 来源 |
|------|--------|------|
| confusion(困惑) | 40s | EDM 2025 |
| frustration(挫折) | 70s | EDM 2025 |
| delight(开心) | 44s | EDM 2025 |
| boredom(无聊) | 93s | EDM 2025 |
| 正面情绪(总) | ~75min | Nature 2018 (Twitter) |
| 负面情绪(总) | ~90min | Nature 2018 (Twitter) |

---

## 4. 中文语速数据

### 来源: 央视(CCTV)、百度文库研究

| 状态 | 字/秒 | 字/600ms | 场景 |
|------|--------|----------|------|
| 慢 | ≤2.5 | ≤1.5 | 悲伤、疲惫、沉思 |
| 正常 | 3-4 | 1.8-2.4 | 日常 |
| 快 | 5-7 | 3-4.2 | 兴奋、焦急、生气 |
| 极快 | ≥7 | ≥4.2 | 恐慌、暴怒 |

情绪对语速影响: 生气/焦虑时语速增加 30-40%

---

## 5. 三层情绪架构(设计)

```
输入(audio/text + 语速)
  ├─ Layer 0: 杏仁核(信号检测) → {tense, relax, orient, approach}
  │   半衰期: 0.3-2.5s
  │
  ├─ Layer 1: 皮层(embedding) → {angry, sad, happy, ...}
  │   半衰期: 40-90s (EDM 2015), 或 75-90min (Nature 2018)
  │
  └─ Layer 2: VAD叠加 → FACS → Live2D
      快慢层合并, 分别衰减
```

---

## 6. 信号检测词汇表

### 来源: DLUT 大连理工情感本体库
徐琳宏,林鸿飞,潘宇.情感词汇本体的构造[J].情报学报,2008,27(2):180-185.
27,466个中文词汇, 7大类21小类, 每词带强度(1/3/5/7/9)和极性。

| 信号 | DLUT映射 | DLUT子类 | 例词(来自本体库) |
|------|----------|----------|------|
| **tense** (威胁/绷紧) | 怒+惧 | NA(愤怒), NC(恐惧), NI(慌) | 气愤, 恼火, 大发雷霆; 胆怯, 害怕, 胆颤心惊; 慌张, 心慌, 不知所措 |
| **orient** (新异/疑问) | 惊+疑 | PC(惊奇), NL(怀疑) | 奇怪, 大吃一惊, 瞠目结舌; 多心, 生疑, 将信将疑 |
| **approach** (关切/前倾) | 哀+思 | NB(悲伤), PF(思念) | 忧伤, 悲苦, 心如刀割; 思念, 相思, 牵肠挂肚 |
| **relax** (放松/温暖) | 乐+安心+喜爱 | PA(快乐), PE(安心), PB(喜爱) | 喜悦, 欢喜, 笑眯眯; 踏实, 宽心, 定心丸; 倾慕, 宝贝, 一见钟情 |

### 词库规模
- DLUT 本体库: 27,466 词 (全量, 论文验证)
- 可从中筛选高强度词(7-9级)作为信号触发词, 中等强度(5级)作为辅助

---

## 7. 说话者→听者 第一反应映射 (待实现)

### 当前状态
- 杏仁核: tense/orient/approach/relax 已实现 (embedding 语义匹配 + ×5 softmax)
- 皮层: angry/sad/happy/fear/surprise/disgust/contempt/neutral 已实现
- VAD: 硬编码映射 (SPEAKER_TO_LISTENER)，各情绪通道独立 ×0.92 衰减

### 待完成 (基于研究成果)
1. **VAD 半衰期**: 替换 ×0.92 一刀切，用研究数据区分各情绪独立半衰期
2. **情绪感染映射**: speaker angry → listener hurt+ fear (负面放大1.3×)
3. **场景区分**: 攻击你 vs 吐槽别人 vs 自述悲伤 → 听者不同反应
4. **杏仁核+皮层合并**: 两路输出加权融合 → 统一 VAD 累积

---

## 8. 已知缺陷 (待后续修复)

### ✅ 已修复: 384维模型区分度不足
**旧**: `paraphrase-multilingual-MiniLM-L12-v2` (384维) → 4个杏仁核原型余弦距离 0.54-0.83
**新**: `thenlper/gte-base-zh` (768维, 阿里达摩院) → CMTEB 65.92, 6/6 测试全对
- 0.20GB, 768维, 中文专精
- 已删除旧 MiniLM 和 distilbert 缓存

### 已放弃的模型
- distilbert-multilingual-nli-stsb (英语数据集训练, 中文 STS 无效)
- DLUT 关键词方案 (27k词含歇后语/古语, 口语不匹配)

### 旧模型清理命令
```bash
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--distilbert-multilingual-nli-stsb-quora-ranking/
```

---

## 11. 情绪 → 面部表情 衰减层 (缺口研究)

### 问题: 内心情绪 ≠ 面部表情

成人悲伤可持续 120h, 但面部表情每 0.5-4s 就自然回落。
情绪层和表情层是两套不同时间尺度的系统。

### Ekman FACS 表情时间参数

| 阶段 | 时间 | 说明 |
|------|------|------|
| Onset (启动) | 80-260ms | 肌肉收缩, 不可控 |
| Apex (峰值) | 可持握 | 保持高位 |
| Offset (消退) | — | 肌肉回松, 非情绪驱动 |
| 总持续 | 0.5-4s | 自发表情的自然寿命 |
| 微表情 | 40-200ms | 泄漏后立即被抑制 |

### 儿童 vs 成人 情绪波

| | 成人 | 小女孩 (5-10岁) | 我们的雌小鬼 |
|------|------|------|------|
| 一波情绪 | 数小时 | 5-15 分钟 | **3-8 分钟** |
| 波形 | 缓升慢降 | 尖峰→快落→微残留 | 尖峰→快落→微残留 |
| 生理响应 | 90s | 90s (同) | 90s |
| 表情衰减 | 0.5-4s | 0.5-4s (同) | 0.5-4s |

### 表情混合 (Du et al., 2014 PNAS; Ross et al., 2016)

关键发现: 上半脸和下半脸有**独立的运动控制区**:
- **上半脸 (眉/眼)**: 辅助运动区+前扣带 (内侧), 更自动/杏仁核驱动
- **下半脸 (嘴)**: 初级运动+前运动 (腹外侧), 更可控

→ 新旧情绪重叠时:
- 上半脸倾向显示**较新的/杏仁核级**情绪
- 下半脸倾向显示**残留的/皮层级**情绪
- 或左右不对称

### 连续触发表情叠加 (缺口 — 待进一步研究)

假设场景: 用户先骂一句(tense)再夸一句(relax)
```
t=0:    "你笨蛋" → tense spike → 脸部眉毛收紧 + 嘴角下拉
t=2s:   "你真棒" → relax spike → 眉毛先松(新信号) + 嘴角还残留在下拉(旧残留)
t=3s:   嘴角慢慢转向上(新信号追上)
```
**上半脸快速切换, 下半脸滞后过渡** — 这是 Ross 2016 组件理论的预测。

### 少女情绪动力学 (Chaplin 2013 元分析 + 中国临床数据)

**来源**: 
- Chaplin & Aldao (2013) Gender Differences in Emotion Expression. Psychol Bull. 555项研究, 21,709人
- 百度健康/39健康 青春期女孩情绪临床数据

| 参数 | 值 | 来源 |
|------|------|------|
| 一波哭泣恢复 | 10-30 min | 百度健康 |
| 每日正常频率 | ≤2次/天 | 临床正常阈值 |
| 正面情绪强度 | 女孩>男孩, g=-0.28 | 元分析 |
| 外化情绪(愤怒) | 青春期女孩反超男孩 | 元分析 |
| 杏仁核/PFC比 | 杏仁核高活跃, PFC弱(25岁才成熟) | 神经发育 |
| 表情抑制能力 | 可以, 但有注意力代价 | ERP研究 |
| 激素波动 | 雌激素+皮质醇不稳 → 小事大反应 | 生理研究 |

### 表情半衰期 (少女版)

| 情绪 | 婴儿 | **少女 (8-12岁)** | 成人 | 依据 |
|------|------|------|------|------|
| 恐惧/惊吓 | ≤1min | **30-60s** | 0.7h | 儿童>成人, 但PFC开始有抑制 |
| 愤怒 | 几分钟 | **2-8min** | 2h | 青春期外化增强+恢复力 |
| 悲伤/难过 | 偏长 | **10-30min** | 120h | 百度健康: 10-30min平复 |
| 开心/喜悦 | 短暂 | **5-15min** | 35h | 女孩正面情绪高频+持久 |
| 羞耻/尴尬 | 很快 | **1-5min** | 0.5h | 少女敏感但快速恢复 |
| 兴奋 | 更短 | **2-5min** | 1.3h | 注意力转移快 |
| 厌恶 | 很短期 | **30s-2min** | 0.5h | 快速消退 |

### 情绪→表情 映射规则 (基于研究)

**默认**: 少女不隐藏情绪, 内心情绪强度 × 0.9 直接映射到面部

**特殊情况**:
1. **陌生人前** (傲娇模式): 正面情绪放大 (g=-0.28), 愤怒也被放大 (青春期外化反超)
2. **熟悉人前**: 恢复默认真实
3. **连续触发**: 上半脸(眉/眼)显示最新杏仁核信号, 下半脸(嘴)残留旧皮层情绪 (Ross 2016)
4. **冲突AU**: Smile+ Frown 不同时出现, 新信号覆盖旧信号 (AU互斥规则)
5. **表情寿命**: 0.5-4s 后自然回落, 但若皮层情绪仍≥0.3, 则重新触发

### 残留情绪阈值

基于研究推断 (少女):
- 皮层 VAD 值 < **0.15**: 面部完全中性 (无残留影响)
- 0.15-0.3: 嘴角/眉毛有微弱残留 (需要AU强度 ≥ C级才可见)
- ≥0.3: 面部持续表达, 每2-4s自然刷新一次
- 新情绪≥0.5: 完全覆盖旧表情 (下半脸滞后≤1s跟上)

### 未解决的缺口

1. ~~残留情绪强度低于多少时不再影响面部?~~ → 情绪特定阈值 (Calvo 2016, 见 §12)
2. ~~新情绪触发时, 是覆盖旧表情还是混合?~~ → 上半脸切换+下半脸滞后 (Ross 2016)
3. ~~各情绪触发阈值相同吗?~~ → 不同, happy 20%可上脸, fear 50% (Calvo 2016, 见 §12)
4. ~~自然消退有问题?~~ → 注意力分配模型 (Li 2017, 见 §13)
5. ~~对立情绪对冲?~~ → Opponent Process Theory (Solomon 1974, 见 §13)

### VAD 仅拍脑袋参数 → ✅ 已替换为研究数据
见 §12-§14 完整算法参数。

---

## 9. 方案决策记录

### 杏仁核检测方式: 关键词 → embedding
- **初始方案**: 关键词匹配, 用 DLUT/EBP 验证词库
- **放弃原因**: DLUT 含大量歇后语/古汉语/非情感词 ("寡妇死了儿","贲临","JJWW"), 多义词无法消歧 ("好"=程度副词 vs relax 信号)
- **最终方案**: sentence-transformer 语义匹配, 手写 40句/类的原型

### 原型句子视角: 听者 → 说者
- **初始方案**: 从 AI 回应视角写 ("别哭了乖","手给我","抱抱你")
- **纠正原因**: 杏仁核检测的是**对方说的话**, 不是自己的回应
- **最终方案**: 全部改为用户说话视角 ("我好难过","你走开","谢谢你")

### 语速参数: 有 → 删
- **初始方案**: `chars_per_frame` 参数模拟语速调制信号强度
- **删除原因**: 流式 ASR 未接入, 且每帧衰减本身已是速度信号; 当前用默认值 2.0 无意义
- **最终方案**: 移除, 等流式 ASR 通后再加回来

### 软最大化温度: ×3 → ×5
- **初始方案**: softmax ×3, 温和区分
- **变更原因**: 384维模型区分度不足, 4个原型余弦距离 0.54-0.83; ×5 与皮层层一致, 足够放大
- **长期**: 换 768维模型或 fine-tune

### 情绪分类体系: 初始 6种 → Ekman 7 + neutral
- **初始方案**: angry/sad/happy/surprise/fear/neutral (6种)
- **变更原因**: 查文献确立 Ekman 7 基本情绪为心理学金标准
- **最终方案**: angry/disgust/fear/happy/sad/surprise/contempt + neutral (8种)

### 长度置信度: /8 → /10
- **初始方案**: `min(len/8, 1.0)`, 7字即可 87%
- **用户纠正**: 10字才达完整语义单位, `/8` 太激进
- **最终方案**: `min(len/10, 1.0)`

### 情绪原型数据: 拍脑门 → TIX007 标注
- **初始方案**: 手写各情绪 10 句原型
- **变更原因**: 用户要求验证数据源
- **最终方案**: TIX007/chinese-sentiment (joy/sad/anger/fear/surprise) 各 100+ 句, + 手写 disgust/contempt/neutral

### DLUT 情感本体库: 全量 → 清洗 → 放弃
- **初始方案**: 下载 27,466 词作为杏仁核词库
- **过程**: Gitee 克隆成功, 发现大量歇后语/古语/非情感词
- **清洗**: ≤5字 + 纯中文, 筛出 5798 词
- **最终放弃**: 根子是学术词库非口语词库, 语义匹配更合适

### VAD 衰减: 全统一 → 各情绪独立
- **初始方案**: 所有 VAD 通道 ×0.92 统一衰减
- **查文献确认**: 不同情绪半衰期不同 (tense 0.3s, orient 2.5s, ...)
- **当前状态**: 杏仁核层已按研究数据实现独立半衰期; 皮层 VAD 仍用统一 ×0.92, 待改

---

## 10. 当前系统运作流程

### 一帧完整链路

```
用户输入文字 (流式 ASR 或 Web 打字, 每帧 ~600ms)
│
├─ 同时执行:
│
├─ [Layer 0] 杏仁核检测 (amygdala_detector.py)
│   │ 方式: sentence-transformer 语义匹配
│   │ 原型: amygdala_prototypes.py (160句手写, 用户说话视角)
│   │ 匹配: 余弦距离 → softmax ×5 → {tense, orient, approach, relax}
│   │ 半衰期: 0.3s/2.5s/1.0s/2.0s (研究数据)
│   │ 角色: 身体第一反应 (LeDoux 低通路)
│   │
│   └─ 输出: tense:0.6, orient:0.3, approach:0.0, relax:0.0
│
├─ [Layer 1] 皮层情绪 (facs_engine.py — detect_speaker_emotion)
│   │ 方式: sentence-transformer 语义匹配
│   │ 原型: emotion_prototypes_data.py (428句, TIX007标注 + 手写)
│   │ 匹配: 余弦距离 → softmax ×5 → ×长度置信度(min(字长/10,1.0))
│   │ 分类: angry/sad/happy/fear/surprise/disgust/contempt/neutral
│   │ 角色: 语义理解后的认知情绪 (LeDoux 高通路)
│   │
│   └─ 输出: angry:0.6, sad:0.2, happy:0.0, ...
│
├─ [Layer 2] VAD 累积 (facs_engine.py — DualVADEngine)
│   │ 方式: 说话者情绪 → SPEAKER_TO_LISTENER 硬编码映射 → 听者感受
│   │ 衰减: 每帧 ×0.92 (全通道统一, 待改为各情绪独立半衰期)
│   │ 累加: 每次 inc_rate=0.3 上限 1.0
│   │ 合并: 杏仁核信号 + 皮层情绪 双路加权 → 统一 VAD
│   │
│   └─ 输出: hurt:0.5, sad:0.2, curious:0.3, ...
│
└─ [映射] VAD → FACS 参数 → Live2D 表情
    │ 方式: 硬编码查表 (FACS_PARAMS)
    │ 例: hurt → ParamBrowLY:-0.5, ParamMouthForm:-0.3...
    │
    └─ 输出: {ParamBrowLY:-0.3, ParamEyeLOpen:0.7, ...}
```

### 关键参数一览 (更新)

| 层 | 检测方式 | 分类数量 | 温度 | 长度系数 | 半衰期 | 状态 |
|---|---------|:---:|:---:|:---:|------|:---:|
| Layer 0 杏仁核 | embedding (gte-base-zh) | 4 | ×5 | 无 | 0.3-2.5s | ✅ 完成 |
| Layer 1 皮层 | embedding (gte-base-zh) | 8 | ×5 | len/10 | 见 §12 | ✅ 检测完成 |
| Layer 2 VAD衰减 | 注意力分配模型 | 12 | — | — | 见 §13 | ⏳ 待实现 |
| Layer 3 表情映射 | 阈值驱动+AU | 6阈值 | — | — | 0.5-4s | ⏳ 待实现 |

### 哪个环节改变了什么 (更新)
| Layer 2 VAD | 硬编码映射 | 12 | — | — | ×0.92统一 | ⚠️ 待改造 |

### 哪个环节改变了什么

| 输入 | 杏仁核先反应 | 皮层层跟上 | VAD 累积 | Live2D |
|------|:----------:|:--------:|:------:|:-----:|
| "滚" | tense:0.7 → 身体紧绷 | angry:0.3 | hurt 上升 | 皱眉+后仰 |
| "你知道吗" | orient:0.6 → 歪头注意 | surprise:0.3 | curious 上升 | 歪头+睁眼 |
| "我好难过" | approach:0.5 → 前倾 | sad:0.4 | care+sad 上升 | 前倾+眼神 |
| "你好可爱" | relax:0.5 → 放松 | happy:0.6 | happy+shy 上升 | 微笑+脸红 |
| 沉默 2秒 | tense 0.3s消失 | 皮层还在 | 整体衰减 | 退回中性 |

---

## 12. 情绪→表情可见阈值 (Calvo et al., 2016)

**来源**: Calvo, Avero, Fernández-Martín & Recio (2016). Recognition thresholds for static and dynamic emotional faces. *Emotion*, 16(8), 1186-1200.

**方法**: Gradual morphing from neutral → emotional face, measured minimum intensity for above-chance recognition.

| 情绪 | 能被观察者识别的最低强度 | 物理原因 |
|------|:--:|------|
| **开心** | **20%** | 眼角+嘴角联动, 面部变化幅度大 |
| 悲伤 | 40% | 主要靠眉毛+嘴, 变化较细微 |
| 愤怒 | 40% | 眉毛+嘴唇收紧 |
| 惊讶 | 40% | 眉毛+眼+嘴三区联动 |
| 厌恶 | 40% | 皱鼻+上唇变化 |
| **恐惧** | **50%** | 主要靠眼部, 最细微 |

**采纳**:
- 每个情绪有自己的"可见阈值": VAD值 < 阈值 → 内心有但脸上看不到
- happy 最易上脸 (0.2), fear 最难 (0.5)
- 替换旧版 0.15/0.3/0.5 一刀切推断

---

## 13. 注意力分配衰减模型 (Li et al., 2017; Solomon & Corbit, 1974)

### 13.1 转移注意力加速情绪消退 (Li et al., 2017)

**来源**: Li et al. (2017). Distraction and Expressive Suppression Strategies. *Nature Scientific Reports*, 7:13062.

**发现**: 转移注意力是调控负面情绪最快的方法。ERP数据显示 distraction 在 300-1200ms 就开始显著降低 LPP 振幅。

### 13.2 对立过程理论 (Solomon & Corbit, 1974)

- A-过程 (主反应): 刺激→立刻上升→慢慢衰减
- B-过程 (对手反应): 延迟启动→缓慢上升→持续更久
- 重复刺激: A减弱, B增强

### 13.3 采纳的衰减算法

**核心规则: 注意力 = 衰减资格**

```
每帧 (600ms):
  
1. 找当前最强情绪: champion = argmax(all VAD channels)
  
2. 遍历所有 VAD 通道:
   if channel == champion:
     decay_rate = normal_half_life[channel]     # 正常速度 (研究数据)
   else:
     decay_rate = normal_half_life[channel] / 4 # 注意力被抢, 4倍速衰减

3. 应用衰减:
   VAD[channel] *= 0.5 ^ (0.6 / decay_rate_seconds)

4. 情绪≤0.05 → 归零
```

**为什么 ×4**:
- 1-3倍: 不够快, 旧情绪会长期残留干扰
- 5倍以上: 太快, 没有过渡痕迹
- 4倍: 能复现"悲伤→讲笑话→开心先走→悲伤回来"的节奏

### 13.4 自然半衰期表 (少女, 8-12岁)

| 情绪通道 | 正常半衰期 | 失焦后 (/4) | 来源 |
|------|------|------|------|
| hurt/sad | 15 min (900s) | 3.75 min | 百度健康少女哭泣恢复10-30min, 取中 |
| angry | 5 min (300s) | 1.25 min | 青春期外化增强+恢复快 |
| scared | 45s | ~11s | 杏仁核快+少女恢复 |
| happy | 10 min (600s) | 2.5 min | 女孩正面情绪持久 |
| shy | 3 min (180s) | 45s | 少女敏感但快速恢复 |
| surprised | 1 min (60s) | 15s | 定向反应快消退 |
| contempt | 1 min (60s) | 15s | 快速消退 |
| disgust | 1 min (60s) | 15s | 快速消退 |
| worried | 8 min (480s) | 2 min | 介于 scared 和 sad 之间 |
| care | 5 min (300s) | 1.25 min | 同情关怀, 中等持久 |
| curious | 2.5 min (150s) | ~38s | 信息渴求消退快 |

### 13.5 B-过程 (对立情绪) 暂缓实现

B-过程的延迟时间、强度比例、少女参数缺乏直接研究数据, 标记为 `待研究`。当前版本先不实现 B-过程, 仅用注意力分配模型完成衰减。

---

## 14. 完整帧算法 (最终版)

```
每帧输入: user_text (字符串), dt=0.6s

Step 1 — 检测:
  amygdala_signals = detect_amygdala_signals(user_text)   # gte-base-zh, ×5 softmax
  speaker_emotion   = detect_speaker_emotion(user_text)    # gte-base-zh, ×5+len/10
  listener_vad      = SPEAKER_TO_LISTENER[ speaker_emotion ]  # 硬编码映射

Step 1.5 — 说话者→听者映射 (2025-07-23 更新):
  listener_vad = {} 
  for each speaker_emotion in speaker_emotion:
    for each listener_emotion, ratio in SPEAKER_TO_LISTENER[speaker_emotion]:
      listener_vad[listener_emotion] += speaker_emotion_intensity × ratio

Step 2 — 先衰减旧值, 再累加新信号:
  _decay_attention(amygdala_channels)  # 旧值先衰减
  _decay_attention(vad_channels)
  amygdala_channels += amygdala_signals × INC_RATE_AMYGDALA(1.0)  # 上限1.0
  vad_channels      += listener_vad × INC_RATE_CORTEX(0.3)        # 上限1.0

Step 3 — 注意力分配衰减:
  champion = argmax(vad_channels)
  for each channel:
    if channel == champion: halflife = NORMAL[channel]
    else:                   halflife = NORMAL[channel] / 4
    vad_channels[channel] *= 0.5 ^ (dt / halflife)

Step 4 — 合并杏仁核+皮层:
  face_emotion = merge(amygdala_channels, vad_channels)
  # 上半脸跟杏仁核(tense→眉毛紧), 下半脸跟皮层(hurt→嘴角下拉)

Step 5 — 阈值过滤:
  for each emotion in face_emotion:
    if face_emotion[emotion] >= VISIBILITY_THRESHOLD[emotion]:
      apply_AU_with_strength(emotion, face_emotion[emotion])

Step 6 — AU → Live2D:
  FACS_PARAMS[AU_combination] → Live2D骨骼参数
```

| 参数 | 值 | 来源 |
|------|------|------|
| dt (帧时长) | 0.6s | 系统定义 |
| INC_RATE_AMYGDALA | 1.0 | 杏仁核满强度 (第一反应快) |
| INC_RATE_CORTEX | 0.3 | 皮层渐进累积 |
| 累加顺序 | 先 decay → 再 add | 修正: 同一帧新信号不被衰减 |
| 杏仁核半衰期 | 0.3/2.5/1.0/2.0s | startle reflex + orienting research |
| VAD 正常半衰期 | 见 §13.4 表 | 少女数据 + 文献 |
| 失焦加速系数 | ÷4 | Li 2017 推断 |
| VAD→AU 映射 | sigmoid(k×(VAD-threshold)) | Vision Research 2020 (见 §15) |
| k (sigmoid 陡度) | 12 | 实测 |
| 可见阈值 | 见 §12 表 | Calvo 2016 |
| 杏仁核软最大化温度 | ×5 | 384→768维后区分度调整 |
| 皮层层软最大化温度 | ×5 | 区分度调整 |
| 长度置信度 | min(len/10, 1.0) | 用户设定 |
| 嵌入模型 | gte-base-zh | 阿里达摩院, 768维, CMTEB 65.92 |

---

## 15. VAD→AU 非线性映射 (sigmoid) 研究

### 15.1 基础: 阈值 ≠ 线性

**Calvo et al. (2016)** — 各情绪有独立的 "能被观察者识别的最低强度":
- happy: 20% (最容易上脸)
- fear: 50% (最难被看出来)
- 其他: 40%

**Vision Research (2020)** — 情绪到面部是 S 型非线性转导:
```
AU强度
1.0 |                    ████████  ← 饱和区 (VAD≥0.7)
   |                 ████
0.5 |            ████               ← 加速区 (VAD 阈值~0.6)
   |        ███
0.2 |    ███                        ← 阈值区 (刚上脸)
0.0 |█                              ← 无效区 (VAD<阈值)
```

### 15.2 算法

```python
AU_strength = sigmoid(k × (VAD - threshold))
            = 1.0 / (1 + exp(-k × (VAD - threshold)))

k = 12  # 陡度, 让加速区集中在阈值附近
threshold = Calvo 2016 各情绪不同 (happy:0.2, fear:0.5, 其他:0.4)
```

### 15.3 卡通/VTuber 渲染放大

**Mäkäräinen et al. (2014) Cognitive Computation**:
- 真实度越低 → 需要越大的面部幅度 → 才能传达同样情绪强度
- 真人脸: FULL → 卡通脸: 150-200% 才能被同样感知

**CHI 2025 (Wang et al.)**: 分区放大
- 上半脸 (眉/眼): 可放大 1.5-2.0×
- 下半脸 (嘴): 谨慎, 150% 已到恐怖谷边缘

**VTube Studio 实践** (iFacial-Link 开源):
- EyeOpen: 原始信号 ×1.25 - 0.25 (补偿摄像头采样损失)
- EyeGaze: ×1.5
- 其他参数: 线性 remap 范围

**结论**: 我们的 AU 来自计算而非摄像头采样, 不需要 VTS 的信号补偿层。
仅需在 AU→Live2D 参数映射层加卡通放大系数。

### 15.4 VTuber 表演者差异 (缺口)

**日本 EMG 研究 (1992)** — 唯一直接对比:
- 演员笑: 皱眉肌也参与 (复杂笑)
- 演员怒: 笑肌也参与 (冷笑)
- 演员悲: 更多肌肉同时激活

**结论**: 演员 VAD→AU 不是简单倍数放大, 而是肌肉组合更复杂。
但没有定量数据, 更没有 VTuber 特化研究。标记为学术空白。

---

## 16. 杏仁核原型与模型选型过程

### 16.1 初始方案: DLUT 关键词匹配

DLUT 情感本体库 (27,466 词, 7大类21小类, 徐琳宏 2008):
- 下载成功 (Gitee, beyond277 仓库)
- 发现大量非口语内容: 歇后语("寡妇死了儿"), 古汉语("贲临"), 非情感词("848.0")
- 清洗后剩 5798 词 (≤5字 + 纯中文)
- **放弃**: 关键词匹配无法处理多义词 ("好"=程度副词 vs relax)

### 16.2 最终方案: embedding 语义匹配

手写 4 类原型 (每类 30 句), 基于:
- **tense**: Isenberg (1999) PNAS — 82 个 fMRI 验证威胁词特征: 身体伤害意图/攻击/背叛/囚禁
- **orient**: novel detection studies — 意外/信息缺口/纯好奇
- **approach**: 自述悲伤/脆弱/求助暗示
- **relax**: 夸奖/温暖/愉快

### 16.3 模型选型

| 模型 | 结果 | 决定 |
|------|------|------|
| MiniLM-L12 (384维) | 4 原型余弦距离 0.54-0.83, 区分不够 | 保留备用 |
| distilbert-nli-stsb | 英语 STS 训练, 中文极差 | **放弃** |
| **gte-base-zh** 🎯 | 768维, 0.20GB, CMTEB 65.92 | **采用** |
| gte-large-zh | CMTEB 66.72 但 0.65GB 太大 | 不采用 |

**测试结果 (gte-base-zh, 6/6 全对)**:
```
你笨蛋      → tense:0.36    ✅
你好厉害    → relax:0.45    ✅  
唉...算了   → approach:0.40 ✅
真的假的?   → orient:0.50   ✅
我好害怕    → approach:0.47 ✅
天气真好    → relax:0.51    ✅
```

### 16.4 原型句子视���修正

**错误**: 从 AI 回应视角写 ("别哭了乖","抱抱你")
**纠正**: 杏仁核检测的是**对方说的话**, 全改用户视角
**最终**: tense/orient/approach/relax 各 30 句, 补失败/放弃+纯夸奖后 各 42 句

---

## 17. 方向检测: ABOUT_YOU vs ABOUT_OTHER

### 17.1 问题

同一情绪, 指向不同对象, 听者反应完全不同:
- "你笨蛋" → 冲我来的 → scared + hurt (Hatfield 反传染)
- "他好烦" → 吐槽别人 → contempt + amused (共享八卦)
- "我好难过" → 自述 → care + sad (共情)

SPK→L 映射不能是固定表, 必须先识别"说话者在说谁"。

### 17.2 简化: 二元分类

只需判断情绪性句子是否指向听者本人:
- **ABOUT_YOU**: 主语/话题是"你"(听者) → 走 ATTACK 映射
- **ABOUT_OTHER**: 主语是"我""他"或泛在话题 → 走 NORMAL 映射

正面情绪("你好可爱")不受影响——ATTACK 映射也包含正面传染规则。

### 17.3 方法: embedding 语义匹配 (复用 gte-base-zh)

跟杏仁核同款方法: 手写两组原型, cosine 距离更近的胜出。

**ABOUT_YOU (22句)**: 你笨蛋/你好可爱/你喜欢你/你要小心/你给我滚出去/...

**ABOUT_OTHER (30句)**: 我好难过/他好烦/张三是死变态/今天运气真差/你觉得他怎么样/你猜他考了几分/...

### 17.4 测试结果

24/24 全对, 包括边界用例:
- "你听说了吗张三被抓了" → ABOUT_OTHER ✅
- "你觉得他能行吗" → ABOUT_OTHER ✅  
- "你穿这件真好看" → ABOUT_YOU ✅
- "帮我拿一下" → ABOUT_YOU ✅

### 17.5 架构位置

```
gte-base-zh embedding (一次编码)
├─ 杏仁核检测
├─ 皮层情绪检测
└─ 方向检测 ← 新增独立层
     ↓
  ABOUT_YOU → SPK_TO_L_ATTACK → VAD
  ABOUT_OTHER → SPK_TO_L_NORMAL → VAD
```

---

## 18. 说话者→听者映射 (双表, 基于方向)

### 18.1 SPK_TO_L_ATTACK (冲我来的, Hatfield 反传染)

| 说话者 | 听者感受 | 依据 |
|------|------|------|
| angry | scared:0.7, hurt:0.5, angry:0.2 | Hatfield 反传染+愤怒冰山 |
| disgust | hurt:0.6, contempt:0.3 | 厌恶→受伤 |
| contempt | hurt:0.7, sad:0.3 | 被鄙视→受伤+难过 |
| fear | scared:0.5, worried:0.4 | 恐惧传染 |
| sad | sad:0.5, care:0.4 | 共情 |
| surprise | surprised:0.5, scared:0.3 | 被吓 |
| happy | happy:0.5, shy:0.3 | 正面传染 (被夸→开心+害羞) |

### 18.3 旧映射已被替代 (2026-07-24)

旧版两份映射表已被废弃:

### 18.4 修正 (2026-07-24)

**ABOUT_OTHER**: 改为 identity 映射, 说话者什么情绪, 听者就什么情绪。
  衰减由 DIRECTION_COEFF["别的"]=0.8 + INC_RATE_CORTEX=0.8 + 池余量 共同控制。
  不再将单一情绪分裂成多个 VAD 通道。

**ABOUT_YOU**: 反传染框架不变, 比率保留原值待标定。

### 18.5 目前可出现的 VAD 通道

| 来源 | 通道 | 条件 |
|------|------|------|
| 杏仁核 | tense, orient, approach, relax | 任何输入 |
| 皮层-ABOUT_OTHER | angry, sad, happy, scared, surprised, contempt | identity 传染 |
| 皮层-ABOUT_YOU | angry, scared, hurt, sad, care, worried, surprised, shy, happy, contempt | 反传染映射 |

**不会出现的情绪 (死通道)**: excited, amused — 无任何映射源填入。

---

## 19. 关心+关切合并 (2026-07-24)

### 19.1 合并理由

approach（杏仁核信号）和 care（皮层情绪）在面部表达上是同一张脸:
- 同属"关怀"行为体系 (Panksepp CARE 系统)
- 同受催产素(OXT)池影响
- 差异仅在于触发通路: 杏仁核快(50ms), 皮层慢(几秒)

### 19.2 当前实现

approach 条目保留在 EMOTION_MAP 中, 实际使用 care 的映射参数。
表情竞争层的 max() 逻辑自动处理两路同时激活时的强度取大。

### 19.3 双路合并公式 (待实现)

当需更精确的叠加效果时:

```
norm = clamp((VAD - thr) / (1 - thr), 0, 1)
√n_i = norm_i ^ 0.5
n1 = max(√n_approach, √n_care)
n2 = min(√n_approach, √n_care)
combined = n1 + (1 - n1) × n2
param = max_val × combined
```

特性:
- 单路活跃 → 退化为原 power curve
- 双路同时活跃 → 高于单路但永不超 1.0
- 弱信号叠加有互相抬升效果

---

## 20. 表情映射完成总表 (2026-07-25)

所有 13 个 VAD 通道已完成 Live2D 参数映射, 见 `live2d_mapper.py`。

### 20.1 参数方向更正

此模型的 ParamAngleX/Y 和 ParamBodyAngleX/Y 与标准 FACS 命名互换:

| 参数 | 实际效果 | 文档原始写法 |
|:----:|:--------|:------------|
| ParamAngleX + | **左转** | +低头 ❌ |
| ParamAngleY + | **抬头** | +左转 ❌ |
| ParamAngleZ + | **左歪** | +右歪 ❌ |
| ParamBodyAngleX + | **左转** | +前倾 ❌ |
| ParamBodyAngleY + | **前倾** | +左转 ❌ |

### 20.2 完整映射表

| 表情 | 通道 | thr | 面部 ×3 | 身体/头部 ×3 | 特殊 |
|:----:|:----:|:---:|:--------|:------------|:----:|
| happy | cortex | 0.20 | EyeLSmile 0.7, EyeLOpen 0.85, MouthForm 2.5, BrowLY -0.5, Cheek 1.25 | AngleZ +13, BodyAngleY -5, BodyAngleZ -8 | — |
| sad | cortex | 0.40 | EyeLOpen 0.7, BrowLY 1.5, BrowLAngle -1.5, MouthForm -1.5, EyeBallY -0.5 | AngleY +5, BodyAngleY -3 | — |
| angry | cortex | 0.40 | BrowLY -2.0, BrowLAngle 1.5, EyeLOpen 0.9, MouthForm 1.0, Cheek 1.0 | AngleZ +25, BodyAngleY -8, BodyAngleZ -8 | — |
| surprised | cortex | 0.40 | BrowLY 1.5, EyeLOpen 1.2, MouthOpenY 1.0 | AngleY +3, BodyAngleY -3 | — |
| scared | cortex | 0.50 | BrowLY 1.0, EyeLOpen 1.1, MouthForm -0.5 | AngleY -15, BodyAngleY +6 | 眼珠抖动 |
| hurt | cortex | 0.40 | BrowLY 2.5, BrowLAngle 2.5, EyeLOpen 0.3, MouthForm -3.0, Cheek 1.0 | AngleY -25, AngleX -12, BodyAngleY -20 | — |
| worried | cortex | 0.40 | BrowLY 1.6, BrowLAngle -2.0, EyeLOpen 1.1, MouthForm -1.0 | AngleY -10, AngleX +8, BodyAngleY -12 | — |
| contempt | cortex | 0.40 | EyeLOpen 0.6, EyeBallX -0.8, MouthForm 1.0, Cheek 1.5 | AngleY +15, AngleX +25, BodyAngleY -16 | 眼睛状态机 |
| shy | cortex | 0.35 | EyeLOpen 0.6, EyeLSmile 0.6, EyeBallY -0.8, MouthForm 0.8, Cheek 2.0 | AngleY -25, AngleX +20, BodyAngleY -12 | — |
| care | cortex | 0.35 | EyeLOpen 0.95, BrowLY -0.3, MouthForm 0.5, EyeBallY -0.2 | BodyAngleY -10, AngleZ +4, AngleY +2 | — |
| orient | amygdala | 0.40 | BrowLY 0.9, EyeLOpen 1.09, EyeBallX 0.3, MouthForm 0.3 | AngleZ -8, BodyAngleY -3 | — |
| tense | amygdala | 0.40 | BrowLY -0.9, BrowLForm 0.9, EyeLOpen 1.12 | AngleY +2, BodyAngleY +6 | — |
| approach | amygdala | 0.35 | 别名→care (同一张脸) | — | — |

### 20.3 特殊动画实现

**恐惧眼珠抖动** (`startFearShake`):
- 每 150ms 随机设置 ParamEyeBallX/Y ±0.3
- 切换其他情绪时自动停止

**轻蔑眼睛状态机** (`startContemptCycle`):
- **翻白眼状态**: EyeBallX +0.5, EyeBallY +1.0, 持续 0.7~1.5s 后随机动作
- **偷瞄子动作**: EyeBallX -0.8, EyeBallY -0.3, 持续 0.3s → 转回翻白眼 0.2s → 决策
- **闭眼状态**: 翻白眼位置闭眼 2.2s
- 约束: 连续偷瞄 2 次 → 强制闭眼, 闭眼后 → 强制翻白眼

### 20.4 表演级力度说明

×3 值为 Ekman 最小可识别表情的 3 倍。对于 hurt/worried/contempt/shy 等需强表现力的情绪, 在 ×3 基础上再 ×2, 确保 VAD 衰减后仍可辨识。

### 20.5 待办

- [ ] 疑问表情多动作切换 (不同歪头角度+微表情变化, 见 2026-07-25 讨论)
- [ ] 脸红修复 (ParamCheek 当前无效)
- [ ] 爱心眼 Param
- [ ] 流泪 Param
- [ ] 轻蔑的闭眼替换为随机眨眼
- [ ] 待机动画幅度随情绪强度自动缩放

---

## 21. 2026-07-27 更新记录

### 21.1 说话者情绪过滤
- 新增 75% 阈值: 在皮层检测中强度 ≥ max×0.75 的情绪才进入听者映射
- 例: "你好可爱" 开心0.091 → 阈值0.068 → 只保留开心

### 21.2 前端 VAD 全链路
- `vad_bridge.py` (18769/18770): HTML+WS静态服务
- `puppet_live2d_v2.html`: WS连18769, 接收 `vad_status` → setEmotion + VAD面部插值

### 21.3 情绪族 \_v1 命名
- 8族基名(含循环)统一改 `_v1`: happy_v1, contempt_v1, worried_v1, shy_v1, think_v1, care_v1, tense_v1, surprise_v1
- `_emoBase()` 去后缀→族匹配, VAD 同族不调 setEmotion

### 21.4 倾听模式动画
- **眨眼**: 2-10s, 120ms闭眼, 轻蔑时停用
- **点头**: 弹簧驱动, 1.5-6s, -(8~16°), 250-500ms脉冲, 25%双点头
- **视线漂移**: lerp平滑, 3-10s, ±(0.2~0.5), 800-2000ms停留

### 21.5 皮层帧速平衡
- `_k = 0.25`: 同一句话多帧重复输入时降低每帧系数
- `INC_RATE_AMYGDALA = 0.25`, `INC_RATE_CORTEX = 0.20`

### 21.6 ASR 语音输入
- parec 管道 (绕过PortAudio): `parec --device=RDPSource`
- 每600ms取流式ASR全量 → POST 18768/facs
- 说话结束 → POST 最终文字到ASR桥(18771)
- VAD页面: `GET /latest` 轮询 + `render(data)` 全可视化

### 21.7 WSL2 音频通路
- WSLg PulseServer → parec → python subprocess 管道 → int16 numpy
- 启用 live-bar 显示当前识别文字 + 情绪结果

---

## 21. 2026-07-27 更新记录

### 21.1 说话者情绪过滤
- 新增 75% 阈值过滤: 只有在皮层检测中强度 ≥ 最强情绪×0.75 的情绪才进入听者映射
- 解决短句 embedding 区分度不足导致多个弱信号同时映射的噪音问题
- 例: "你好可爱" 开心 0.091 → 阈值 0.068 → 只保留开心, contempt/surprise 等弱信号被过滤

### 21.2 前端 VAD 全链路打通
- `vad_bridge.py`: 端口 18769 (HTML静态服务+WebSocket) + 18770 (HTTP API)
- `puppet_live2d_v2.html`: WebSocket 连 18769, 接收 `{type:'vad_status', mode:'listen', emotion, ratio}`
- ratio 计算: `norm^0.5`, 其中 `norm = (vad_val - threshold) / (1 - threshold)`
- 前端用 ratio 做面部 neutral 插值: `target = neutral + (emotion - neutral) * ratio`

### 21.3 情绪族系统
- 所有由循环系统管理的基础表情统一改为 `_v1` 命名 (happy_v1, contempt_v1, ...)
- 族名 = 去掉 `_vN` 后缀 (如 `_emoBase('happy_v4')` → `'happy'`)
- VAD 跨族切换走 `setEmotion`, 同族只更新 `_emitTarget` 面部参数
- 循环的 variants 列表不含基名, 纯变体轮转

### 21.4 倾听模式动画
- **眨眼**: `_listenAdd.eye`, 2-10s 间隔, 120ms 闭眼, **轻蔑时停用**
- **点头**: `_nodOffset` 加算到弹簧目标值, 1.5-6s 间隔, -(8~16°), 250-500ms 脉冲, 25% 双点头
- **视线漂移**: lerp 平滑, 3-10s 间隔, ±(0.2~0.5), 800-2000ms 停留

---
## 22. 2026-08-05 更新记录

### 22.1 EMOTION_CYCLES 补全
- 8族"族名即 v1"改名后 EMOTION_CYCLES 缺失入口, 已补 sad/angry/fear 三族
- 剩余 confused/hurt/approach 为单变体特殊类 (不循环)
- amused/excited 无按钮暂搁置
- 格式统一: 族名=variants容器(EMOTION_CYCLES) + _vN=主情绪(EMOTIONS), 删 EMOTIONS 冗余容器

### 22.2 控制开关系统
- 三个全局开关: `_vadControl`, `_blinkControl`, `_cyclePaused` + 页面按钮
- 表情动作期间自动关三开关, 结束后恢复触发前状态(快照机制)
- VAD开关: ws.onmessage 检查; 眨眼开关: _scheduleBlink 检查; 循环暂停: _scheduleNextVariant 检查

### 22.3 表情动作系统收尾
- 6组表情全部完成: 认真(3.45s) / wink 4变体(2.02s) / 开心左右(3s) / spin左右(1.2s) / 转头左右(6s) / 担心左右(5s)
- 转头: 独立轻蔑状态机 X2, 完整复制命名前缀 `_emote`, override 写眼珠 + _emitTarget 写姿态
- 担心: worried_v1 delta×0.7, 分左右镜像
- phases 新增 `override` 字段支持: 直接写 `_actionOverrides` 不经过弹簧

### 22.4 眼球覆盖修复
- 3.5倾听姿态 `ParamEyeLOpen/R + 0.15` 注释掉 (压平所有情绪眼睛差异)
- 3.4c 双眼豁免扩展: track + else 分支均加 `!emoteContemptCycle`
- sad_v2 眼睛: 0.8 → 0.4 (失声痛哭眯眼)

### 22.5 日志精简
- [VAD]同族更新 / [EMOTE-RENDER] / [EYE-JUMP] / [CONTEMPT] 全注释
- [SET] 日志改为 `old → new | src | ratio | eye | ball | cycle`

