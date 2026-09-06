# 迷你硬币架构决策记录

## 2026-07-07: 双阶段 Agent 架构 (Planner → Executor)

### 核心决策
放弃单次 `<think>` 全链路推理，改为 **双阶段外部分离架构**：

```
Stage 1: 规划器 (Planner) — 意图生成
  - 输入: 上文(多轮对话历史) + rag匹配记忆 + 本轮用户输入
  - 输出: 结构化意图 JSON (intent + tool_plan + rag_query + strategy)
  - 训练: 只用 GRPO 训练意图识别，不需要完整推理链
  - 减负: 8B 只做意图理解，不背工具名/参数格式

Stage 2: 执行器 (Executor) — 意图执行
  - 输入: 规划器的意图 JSON + 外部工具/RAG 结果
  - 输出: 最终自然语言回复
  - 性质: 纯代码执行或轻量 LLM 调用，只按计划说话

中间层 (纯代码):
  - 执行工具调用 (web_search, read_file 等)
  - 执行 RAG 知识检索
  - 注入对话历史上下文
  - 校验规划器输出格式
```

### 动机
1. **8B 模型长 think 块会漂移** — 拆成短 think (30-50 token)，每次只做一件事
2. **训练成本降低** — 只需训练意图识别 + 按模板说话，不需要完整推理链标注
3. **工具扩展友好** — 工具名/参数由 schema 管理，不依赖模型记忆
4. **上下文可控** — 通过代码拼接 prompt，而非依赖模型内部记忆

### Neuro-sama 观察结论
Neuro 说自己"不知道被屏蔽的内容是什么"，因为：
- Neuro-1 (规划意图的 neuro) 生成意图但看不到实际输出
- Neuro-2 (执行输出的 neuro) 执行意图但不知道被过滤了什么
- 规划层和执行层分离后，规划层天然"看不见执行结果"

### 数据管线迁移
原有 `[TOOL:xxx]` 自由文本 → Ollama 原生 Tool Calling (schema 驱动)
原有 `<think>` 自然语言推理 → 结构化 JSON 规划输出

### 架构分层

```
感知层 (纯 Python, 200ms 高频轮询):
  音频决策引擎 → VAD + Whisper ASR + 事件队列 → 判断"该说话了吗?"

认知层 (LangGraph, 按需调用):
  Agent 循环 → planner → executor → responder → generate
  内置: 状态管理 / tracing / 可视化 / 断点续传

表达层 (纯 Python / 外部):
  TTS 播放 + 2D 表情控制 + 前端 WebSocket
```

### 待做
- [ ] **P0: Dify Docker 部署 ✅**（2026-07-10 完成）
  - 放弃本地源码部署，改用 Docker Compose 官方流程
  - 关键修复: WSL 内创建 ~/bin/docker-credential-desktop.exe 符号链接指向 Docker Desktop 的凭据助手
  - registry-mirrors: docker.xuanyuan.me + docker.m.daocloud.io
- [ ] **P0: 音频决策引擎** (纯 Python, ~200行)
  - Silero VAD 替换音量阈值 VAD
  - Whisper.cpp streaming ASR 替换腾讯云整段 ASR
  - ConversationController: 统一音频+环境+时间事件队列
  - 负责判断"该不该说话" / 打断 / 表情指令
- [ ] **P0: LangGraph Agent 层**
  - 规划器: 输出意图 + 工具计划 + active_context (user_wants/i_plan/got)
  - 执行器: 调用 Ollama Tool Calling + RAG
  - 回复器: 检查结果是否满足 plan
  - active_context 持久化: 只留三行自然语言
- [ ] **P0: trace logger**
  - 每轮每阶段完整记录 prompt + response 到本地文件
- [ ] **P1: 工具调用迁移**
  - [TOOL:xxx] 自由文本 → Ollama 原生 Tool Calling (schema)
- [ ] **P2: 预训练数据改造**
  - 规划器: 意图标注替代完整推理链
  - 回复生成器: prompt + 知识 → 回复 的简洁标注
