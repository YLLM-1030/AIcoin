# P2S4: P2S4 记忆系统

## 基本信息
- **阶段**: P2S4 (Phase 2)
- **数据集条数**: 25 条
- **起始检查点**: `ckpt_p2s3`
- **输出检查点**: `ckpt_p2s4`
- **新增协议 (2条)**:
  - 4.5.1 核心指令 · 记忆内化 · SOP推演
  - 4.5.2 信息缺口主动填补协议 · SOP推演

## 训练参数

```yaml
learning_rate: 1.5e-6
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
**低强度 · 记忆系统是工具性协议，学到逻辑即可。**

## 阶段说明
注入记忆管理（内化审计三问题）和信息缺口主动填补。完成行为规则层。

## 监控指标

| 指标 | 健康范围 | 警告信号 |
|------|----------|----------|
| 训练 loss | 平稳下降 | 突然暴跌 >50% → 过拟合 |
| 困惑度 | 缓慢下降 | 低于 1.2 → 模型在背诵 |
| 梯度范数 | < 10 | > 100 → 梯度爆炸 |

## 完成后验证

检查点保存为 `ckpt_p2s4` 后，用以下问题快速验证：

## 如果效果不对

1. **Loss 不下降**: 提高 LR 到 `3e-6`，增加 1 个 epoch
2. **Loss 下降太快**: 降低 LR 到 `1e-6`
3. **输出变机械**: 回退到上一个检查点，减少 1 个 epoch 重来
4. **完全不对**: 从 `ckpt_p2s3` 重新开始，LR 减半