# 迷你硬币（Mini Coin）— 项目总览

> AI 主播 / VTuber 完整系统：Qwen3-8B 微调"主脑" + 输入管线 + 双记忆系统 + 弹幕互动 + 情绪演绎（Live2D）+ 语音（ASR/TTS）。
> 本文档为架构侧总览，各模块细节见对应分册。

---

## 一、这是什么

**迷你硬币**是一个 AI 主播（B站直播），人格定位：**傲娇雌小鬼**（毒舌、搞事、捣乱、偶尔撒娇）。
系统由三层构成：

```
感知层（听/看/读）
  ├─ 听觉：麦克风 → ASR（paraformer 流式）
  ├─ 弹幕：B站直播间 → BLC 采集 → 打分筛选
  └─ 输入：ASR 文本 / 弹幕 / 手动 → LLM 桥

认知层（想）
  └─ 主脑：Qwen3-8B（自训练微调模型，llama.cpp 推理）
       ├─ 输入中心：模块化 prompt 拼装（system/记忆/摘要/输入/弹幕…）
       ├─ 上文记忆：L0~L4 压缩归档（对话摘要）
       ├─ 长期记忆：BERT 分类 + embedding 检索（召回历史）
       └─ 工具：搜索互联网（秘塔/DeepSeek）

表达层（说/演）
  ├─ 文本 → 输出队列 → TTS（CosyVoice2 混合声线）
  ├─ 情绪 → FACS 三层引擎 → Live2D 表情/动作
  └─ 前端：Live2D 2D 皮套（含 18 组待机动作）
```

---

## 二、服务拓扑（start_all.sh 一键启动 12 服务）

| # | 服务 | 端口 | 脚本 | 说明 |
|---|------|------|------|------|
| 1 | 主模型 llama-server | 8080 | `~/llama.cpp` | Qwen3-8B 微调模型 v11b，GGUF 量化，GPU |
| 2 | LLM 桥 | 18771 | `输入中心/mini_coin_llm_bridge.py` | 透明代理转发 + 接收 ASR 文本存 source_last |
| 3 | 管线中心 | 18772 (+内置桥 18000-18009) | `输入中心/pipeline_server.py` | 模块化 prompt 拼装 + 冷却状态机 + Web 管理页 |
| 4 | 摘要模型 | 18773 | llama-server | Huihui-Qwen3.5-4B（CPU），上文记忆压缩用 |
| 5 | 检索服务 | 18774 | `输入中心/长期记忆/memory_retrieve_server.py` | BERT 分类 + bge embedding 检索 |
| 6 | 搜索工具 | 18775 | `输入中心/搜索工具/search_server.py` | 秘塔/DeepSeek 双引擎 + SSE |
| 7 | 弹幕采集 | 19000 | `输入中心/弹幕接收/blc_collector.py` | B站开放平台 WS 采集（读后清空桥） |
| 8 | 弹幕接收 | 18776 | `输入中心/弹幕接收/danmaku_server.py` | 打分 + 三阈值 + 弹夹 + 发射 |
| 9 | FACS 表情引擎 | 18768 | `asr+2d8.2确认/配置文件/emotion/facs_server.py` | 情绪检测（杏仁核/皮层/VAD）→ 表情 |
| 10 | Live2D 桥 | 18769 (+18770 API) | `asr+2d8.2确认/配置文件/emotion/vad_bridge.py` | 前端页面 + WS 推情绪/口型 |
| 11 | TTS 语音 | 8000 | `tts训练/tts_server.py`（conda vllm） | CosyVoice2-0.5B 混合声线 |
| 12 | ASR 麦克风 | 18686 (+桥 18680-18685) | `asr+2d8.2确认/mic_vad_asr_streaming_vad.py` | parec + VAD + paraformer 流式 |

启动：`bash start_all.sh`（完整 12 服务）/ `bash start_all.sh --core`（只启核心 8 个）/ `--stop` 停止。
就绪检查为 HTTP 级（模型加载完成才算 ✅，最多等 3 分钟）。

---

## 三、主数据流（一次完整互动）

```
弹幕/麦克风/手动输入
  ↓ 写桥（POST /text）
LLM 桥 source_last  ←→  管线中心每秒轮询触发源桥（_check_trigger_sources）
  ↓ 冷却到点（5s）→ _do_send()
管线中心 build_prompt（模块拼装：system + 记忆 + 摘要[对话历史] + 输入/弹幕）
  ↓ POST 18771 → 转发 llama-server (8080)
主模型生成回复（thought + [TOOL:回复/表情/搜索/报错/结束]）
  ↓ 下桥
_do_send：_sanitize_reply 清洗（删复读/工具闭合/非法清理）
  ├─ 转发 VAD /output（方案B：thought+回复纯文本 → 输出侧表情）
  ├─ 进输出队列（表达/执行/报错三队列）
  ├─ 进 history（发送成功才算数，L0 哲学）
  └─ 触发：上文记忆（摘要归档）+ 长期记忆（检索召回）
输出队列消费：
  ├─ 回复/表情 → 表达队列 → TTS(8000) → Live2D 口型 / 表情
  ├─ 搜索/结束 → 执行队列 → 搜索工具(18775) → 结果回填
  └─ 报错 → 报错队列 → 固定语音
```

---

## 四、端口总表

| 端口 | 服务 | 备注 |
|------|------|------|
| 8080 | 主模型 llama-server | GPU，v11b GGUF |
| 8000 | TTS | conda vllm 环境 |
| 18768 | FACS 表情引擎 | 输入侧 /facs + 输出侧 /output |
| 18769 | Live2D 前端+WS | 页面 + WebSocket |
| 18770 | Live2D API | 情绪/口型推送 |
| 18771 | LLM 桥 | 透明代理 + ASR 文本 |
| 18772 | 管线中心 | Web 管理 + 拼装 |
| 18773 | 摘要模型 | 4B CPU |
| 18774 | 长期记忆检索 | BERT+bge |
| 18775 | 搜索工具 | 秘塔/DeepSeek |
| 18776 | 弹幕接收 | 打分+三阈值 |
| 18686 | ASR 管理页 | 自带桥 18680-18685 |
| 19000 | 弹幕采集 | BLC WS |
| 18000-18009 | 内置桥（管线中心内建） | 拼装成品/系统注入/执行/表达等 |

---

## 五、目录结构

```
新架构硬币/
├── start_all.sh                  ← 一键启动 12 服务
├── 输入中心/
│   ├── pipeline_server.py        ← 管线中心（核心）
│   ├── mini_coin_llm_bridge.py   ← LLM 桥
│   ├── prompt_builder.py         ← 旧拼装器（备用）
│   ├── pipeline_架构说明.md       ← 输入中心架构
│   ├── 上文记忆/                  ← L0-L4 压缩归档 + 架构说明
│   ├── 长期记忆/                  ← BERT 检索 + 记忆库 + 架构说明
│   ├── 弹幕接收/                  ← BLC 采集 + 打分 + 架构说明
│   └── 搜索工具/                  ← 搜索服务 + 架构说明
├── asr+2d8.2确认/
│   ├── mic_vad_asr_streaming_vad.py  ← ASR 主程序
│   └── 配置文件/emotion/          ← FACS 引擎 + Live2D 桥 + 架构说明
│   └── 配置文件/frontend/         ← puppet_live2d_v2.html + 2D 说明
├── asr训练/                       ← ASR 训练实验（Fun-ASR-Nano）
├── 迷你硬币主脑训练/              ← 训练脚本/数据集/架构文档（v3a 时代，部分过时）
├── 新建训练/                      ← 【当前训练】grpo_tool_v9d.py + 训练架构文档
├── 说明书/                        ← 本文档所在
└── 项目索引.md                    ← 07-28 旧索引
```

---

## 六、分册导航

| 分册 | 内容 |
|------|------|
| [01_输入中心.md](01_输入中心.md) | 管线中心 / LLM 桥 / 桥系统 / 冷却状态机 / 下桥清洗 / 输出队列 |
| [02_记忆系统.md](02_记忆系统.md) | 上文记忆（L0-L4）/ 长期记忆（BERT 检索）/ 记忆库数据 |
| [03_弹幕与搜索.md](03_弹幕与搜索.md) | 弹幕采集打分 / 三阈值弹夹 / 搜索工具 |
| [04_情绪演绎.md](04_情绪演绎.md) | FACS 三层引擎 / 方向检测 / 五组映射 / Live2D / 2D 待机动作 |
| [05_语音.md](05_语音.md) | ASR 流式识别 / TTS 混合声线 |
| [06_训练系统.md](06_训练系统.md) | 训练流水线（PT→SFT→GRPO）/ v9d / 类型倍率 / 数据 |
| [07_设计方法.md](07_设计方法.md) | 各模块的设计方法与研究依据 |
| [08_数据来源.md](08_数据来源.md) | 训练数据 / 记忆库 / 情绪原型等数据来源 |
