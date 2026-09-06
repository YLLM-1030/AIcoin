# 迷你硬币 — 设计稿（DESIGN.md）

## 设计基调
深色科技风 · 代码感 · 克制霓虹：像"终端 + 电路板"的冷峻，配合少量青色高亮制造"活物感"。
用色不超过 4 hex，字体 2 套。

## 1. 画布与母版（1280×720）
| 区 | 位置 | 规则 |
|---|---|---|
| A 标题块 | 0–120px | 标题 34px bold，左侧色条 6px 强调色 |
| B 内容区 | 120–660px | 一切正文/图/卡 |
| C 页脚条 | 660–720px | 左"迷你硬币·项目介绍" + 右页码 NN/12，14px 灰 |

## 2. 颜色系统（4 hex）
| 角色 | hex | 用途 |
|---|---|---|
| 背景主色 | `#0B0F19` | 全篇深色底 |
| 主色 | `#1E293B` | 卡片底、色块、边框 |
| 辅色 | `#38BDF8` | 青蓝：高亮线、渐变、图标、数据 |
| 强调色 | `#22D3EE` | 霓虹青：巨型数字、CTA、焦点元素 |

- 面积分配：主色背景 55% / 辅色(卡片等) 30% / 强调色 ≤10%（hero 页 15-20%）
- 渐变：`linear-gradient(135deg, #38BDF8 0%, #22D3EE 100%)` 用于标题色条、关键数据
- 半透明：卡片 `rgba(30,41,59,0.6)`；强调元素叠加 `rgba(34,211,238,0.1)` 光晕
- 文字主色 `#E2E8F0`，次级 `#94A3B8`，禁用高饱和撞色铺满

## 3. 字体系统
| 层级 | 字号 | 字重 | 行高 | 字体 |
|---|---|---|---|---|
| 封面主标题 | 72px | bold | 1.1 | 思源黑体 + JetBrains Mono 点缀 |
| 章节大字 | 60px | bold | 1.1 | 思源黑体 |
| 巨型数据/锚点 | 96-120px | bold | 1.0 | JetBrains Mono（数字） |
| 页面标题 | 34px | bold | 1.3 | 思源黑体 |
| 卡片小标题 | 24px | 600 | 1.4 | 思源黑体 |
| 正文 | 20-22px | regular | 1.6 | 思源黑体 |
| 代码/标签 | 16-18px | regular | 1.5 | JetBrains Mono |
| 页脚/页码 | 14px | regular | 1.4 | 思源黑体 |

## 4. 信息密度
- 常规内容页：正文 ≥180 字 / 主视觉占 B 区 ≥30% / 留白 ≤30%
- hero 页：允许留白 40-50%，但围绕焦点
- 每页 ≥1 视觉锚点（≥44px 元素或 ≥40% 区图）

## 5. 配图系统（全 SVG，统一风格）
深色底 + 细描边 + 青色高亮的结构化图，全部 `<SVG>` 手绘（架构图天然适合）：
- `svg_cover`：网格背景 + 电路节点 + 大标题（封面）
- `svg_layers`：感知/认知/表达三层堆叠
- `svg_ports`：12 服务端口拓扑环
- `svg_timeline`：训练时间线（5月→8月）
- `svg_vram`：显存账条形（16GB 权重 + Lomo 省 16GB）
- `svg_emotion`：杏仁核/皮层/VAD 三层 + 论文标注
- `svg_memory`：双记忆并行（写 L0-L4 / 读 BERT 检索）
- `svg_pipeline`：桥/冷却/清洗流程图
- `svg_interact`：弹幕打分 → 三阈值漏斗
- `svg_ending`：终端光标收尾
- L3 角标：页码统一 C 区右侧

## 6. 页面映射表
| # | 文件 | 类型 | 角色 | 版式 | 视觉 | 字数 | 留白 | 色彩 | 关键约束 |
|---|---|---|---|---|---|---|---|---|---|
| 01 | slide_01_cover.jsx | cover | hero | 全幅骑线 | svg_cover | 30 | 40% | 强调15% | 网格底+大标题+副标 |
| 02 | slide_02_catalog.jsx | catalog | supporting | 左标题右内容 | 数字角标 | 120 | 30% | 辅色30% | 5 条目带序号 |
| 03 | slide_03_system.jsx | content | supporting | 左大图右文 | svg_layers | 200 | 25% | 辅色25% | 三层架构图占左55% |
| 04 | slide_04_services.jsx | content | supporting | 非对称双栏 | svg_ports | 180 | 25% | 辅色25% | 端口环+右侧说明 |
| 05 | slide_05_training.jsx | content | hero | 全幅骑线 | svg_timeline | 150 | 35% | 强调15% | 时间线占 B 区 60% |
| 06 | slide_06_vram.jsx | content | supporting | 左大图右文 | svg_vram | 200 | 25% | 辅色25% | 显存账条+要点 |
| 07 | slide_07_emotion.jsx | content | supporting | 非对称双栏 | svg_emotion | 180 | 25% | 强调12% | 三层+论文标注 |
| 08 | slide_08_memory.jsx | content | supporting | 左标题右内容 | svg_memory | 220 | 20% | 辅色25% | 双记忆对比 |
| 09 | slide_09_pipeline.jsx | content | supporting | 左大图右文 | svg_pipeline | 200 | 25% | 辅色25% | 桥流程+要点 |
| 10 | slide_10_interact.jsx | content | supporting | 上大图下卡 | svg_interact | 180 | 25% | 辅色25% | 打分漏斗+下方要点 |
| 11 | slide_11_numbers.jsx | content | hero | 巨型数字 | 大数字 | 80 | 40% | 强调20% | ≥96px 锚点数字 |
| 12 | slide_12_ending.jsx | ending | hero | 居中金句 | svg_ending | 40 | 45% | 强调12% | 终端光标+金句 |
