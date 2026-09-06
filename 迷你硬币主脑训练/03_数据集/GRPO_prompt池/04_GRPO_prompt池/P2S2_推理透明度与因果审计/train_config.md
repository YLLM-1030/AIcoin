# P2S2: P2S2 推理透明度与因果审计

## 基本信息
- **阶段**: P2S2 (Phase 2)
- **数据集条数**: 19 条
- **起始检查点**: `ckpt_p2s1`
- **输出检查点**: `ckpt_p2s2`
- **新增协议 (4条)**:
  - 4.1 核心指令 · 推理透明度 · SOP推演 · 变体A
  - 4.1 核心指令 · 推理透明度 · SOP推演 · 变体B
  - 4.1 核心指令 · 推理透明度 · SOP推演 · 变体C
  - 4.2 核心指令 · 因果链落地审计 · SOP推演 · 变体A

## 训练参数

```yaml
learning_rate: 2e-6
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
warmup_ratio: 0.1
lr_scheduler_type: cosine
optim: adamw_torch
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
max_seq_length: 2048
packing: true
```

## 训练强度
**中低强度 · 三个变体确保模型学会的是'推理前声明假设'这个模式，而非特定场景。**

## 阶段说明
注入推理透明度（声明假设→推演→校准）和因果链落地审计（概念→物理因果链推演→发现矛盾）。

## 监控指标

| 指标 | 健康范围 | 警告信号 |
|------|----------|----------|
| 训练 loss | 平稳下降 | 突然暴跌 >50% → 过拟合 |
| 困惑度 | 缓慢下降 | 低于 1.2 → 模型在背诵 |
| 梯度范数 | < 10 | > 100 → 梯度爆炸 |

## 完成后验证

检查点保存为 `ckpt_p2s2` 后，用以下问题快速验证：

## 如果效果不对

1. **Loss 不下降**: 提高 LR 到 `4e-6`，增加 1 个 epoch
2. **Loss 下降太快**: 降低 LR 到 `1e-6`
3. **输出变机械**: 回退到上一个检查点，减少 1 个 epoch 重来
4. **完全不对**: 从 `ckpt_p2s1` 重新开始，LR 减半