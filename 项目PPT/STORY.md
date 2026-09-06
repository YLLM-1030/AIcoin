# 迷你硬币 — 项目介绍 PPT 叙事大纲（STORY.md）

## ① 用户意图对齐

- **目标受众**：B站/YouTube 科技区观众（对 AI/自研项目感兴趣的技术爱好者）
- **核心目标**：让观众看完记住三件事——① 这是一个**全自研**的 AI 主播系统（模型自己训、情绪自己研究、管线自己写）；② 每个子系统都有**研究级设计**（神经科学/论文依据）；③ 一个人 + AI 协作 2.5 个月做出别人 1.5-2 人年的东西
- **PPT 长度**：12 页（Hero 3 页：封面、训练、结束）
- **视觉调性**：深色科技风 / 代码感 / 霓虹蓝青 / 克制但有力
- **内容边界**：必讲——三层架构、训练时间线、Lomo 显存优化、情绪三层、双记忆、桥管线、弹幕打分；不讲——具体 bug 调试细节、内部敏感信息（房间号）；禁碰——无

## ② 页面布局骨架

- **章节**：开场（1-2）→ 系统全景（3-4）→ 主脑训练（5-6）→ 认知系统（7-8）→ 交互管线（9-10）→ 收束（11-12）
- **Hero 页**：01 封面、05 主脑训练、12 结束页（3/12 = 25% ✅，间隔满足）
- **rhythm**：01 peak → 02 valley → 03 valley → 04 transition → 05 peak → 06 valley → 07 peak(过渡到支撑) → 08 valley → 09 valley → 10 valley → 11 peak → 12 peak
- **非对称版式** ≥ 40%：01 全幅骑线、03 左大图右文、04 非对称双栏、05 全幅骑线、06 左大图右文、07 非对称双栏、08 左标题右内容、10 上大图下卡、11 巨型数字、12 居中金句（10/12 ✅）
- **对称版式预算**（≤2 页）：02 目录（左标题右内容属非对称，改：02 用对称但仅此一页）、09 左标题右内容。对称页：无——全部非对称

## ③ 页面大纲

| # | title | type | role | rhythm | layout | visual | visual_role | density | anti_pattern |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 迷你硬币·从零手搓的 AI 主播 | cover | hero | peak | 全幅视觉+骑线文字 | L1: svg_cover(全幅背景) | anchor | 字30/图1/留白40% | 禁止小标题贴片；禁止正文段落 |
| 02 | 目录 | catalog | supporting | valley | 左标题+右内容 | L3: 数字角标 | atmosphere | 字120/图0/留白30% | 禁止四卡片预览；禁止铺满正文 |
| 03 | 一个完整的 AI 主播长什么样 | content | supporting | valley | 左大图+右侧文字 | L1: svg_layers(三层架构) | anchor | 字200/图1/留白25% | 禁止 50:50 等分；禁止 200×70 小图 |
| 04 | 12 个服务一条命令启动 | content | supporting | transition | 非对称双栏 | L1: svg_ports(端口拓扑) | evidence | 字180/图1/留白25% | 禁止端口堆成表格；禁止等宽卡片 |
| 05 | 主脑：一个 8B 模型的完整养成 | content | hero | peak | 全幅图+骑线文字 | L1: svg_timeline(训练时间线) | anchor | 字150/图1/留白35% | 禁止卡片横排；禁止列表堆叠 |
| 06 | 24GB 显存跑 8B 全参 | content | supporting | valley | 左大图+右侧文字 | L1: svg_vram(显存账) | anchor | 字200/图1/留白25% | 禁止等宽三卡；禁止纯文字罗列 |
| 07 | 情绪：按神经科学造的脸 | content | supporting | peak | 非对称双栏 | L1: svg_emotion(三层架构) | anchor | 字180/图1/留白25% | 禁止 emotion 图标卡横排 |
| 08 | 双记忆：会忘也会想 | content | supporting | valley | 左标题+右内容 | L1: svg_memory(双记忆) | evidence | 字220/图1/留白20% | 禁止 RAG 流水线模板式表达 |
| 09 | 管线：桥、冷却与清洗 | content | supporting | valley | 左大图+右侧文字 | L1: svg_pipeline(桥架构) | evidence | 字200/图1/留白25% | 禁止纯架构框堆叠 |
| 10 | 弹幕、语音与 2D 演绎 | content | supporting | valley | 上大图+下方卡片 | L1: svg_interact(弹幕打分) | evidence | 字180/图1/留白25% | 禁止三卡横排；禁止截图拼贴 |
| 11 | 关键数字 | content | hero | peak | 巨型数字+洞察 | 大数字(72px+) | anchor | 字80/图0/留白40% | 禁止数字装进等宽卡片；禁止角落小字 |
| 12 | 结束页 | ending | hero | peak | 居中金句 | L1: svg_ending | atmosphere | 字40/图1/留白45% | 禁止落款+正文混排 |
