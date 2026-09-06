# MiniCoinsaka 模块化训练 TODO

## 整体架构

```
用户消息
  ↓
┌─ Timing Gate ─────── no/wait ──→ 结束（不回应）
│  （要不要理）
└──── yes ──→
               ↓
         ┌─ User Intent Analysis
         │  （对方想干嘛）
               ↓
         ┌─ Tool Decision
         │  （用什么工具）
               ↓
         ┌─ Planner / Response Generator
         │  （生成回复）
               ↓
          输出回复
```

## 模块清单

### ☐ 01_TimingGate.md ✅（已写完）

任务：三分类 yes/no/wait
状态：**训练规格已出，待训练**
数据量建议：SFT 500 条 / GRPO 100 条 + reward
格式：纯续写 `消息：xxx\n选择：yes/no/wait`

### ☐ 02_Planner.md ✅（已写完）

任务：输出动作序列 think→reply→finish
状态：**训练规格已出，待训练**
数据量建议：SFT 200-500 条
格式：`[动作]\nthink: xxx\nreply: xxx | 语气\nfinish`

### ☐ 03_UserIntentAnalysis.md ✅（已写完）

任务：用户意图分类（12 类 + 子类）
状态：**训练规格已出，待训练**
数据量建议：每类至少 30 条
格式：`消息：xxx\n意图：harassment`

### ☐ 04_ToolDecision.md ✅（已写完）

任务：工具选择 + 执行模式
状态：**训练规格已出，待训练**
数据量建议：200 条覆盖所有意图×工具组合
格式：输出 JSON 工具方案
注意：可能可以用 50 行 if-else 代替，不一定需要模型

### ☐ 05_ResponseGeneration.md ✅（已写完）

任务：生成最终回复文本
状态：**训练规格已出，可选模块**
强调：可以和 Planner 合并，不一定要独立训
数据量建议：每语气 30 条

## 推荐训练顺序

```
第一优先 → Timing Gate（最简单，立竿见影）
第二优先 → User Intent Analysis（分类任务，也好训）
第三优先 → Tool Decision（可以先用 if-else 顶着）
第四优先 → Planner/Response Generator（最难，放最后）
```

## 文件位置

所有文档在桌面 `C:\Users\Autogram-coin\Desktop\新建训练\` 目录下：
- `01_TimingGate.md`
- `02_Planner.md`
- `03_UserIntentAnalysis.md`
- `04_ToolDecision.md`
- `05_ResponseGeneration.md`
