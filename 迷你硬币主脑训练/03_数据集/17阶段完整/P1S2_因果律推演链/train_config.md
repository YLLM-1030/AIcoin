# P1S2: P1S2 因果律推演链

## 基本信息
- **阶段**: P1S2 (Phase 1)
- **数据集条数**: 4 条
- **起始检查点**: `ckpt_p1s1`
- **输出检查点**: `ckpt_p1s2`
- **新增协议 (3条)**:
  - 0.1 规律 → 因果律 · SOP推演
  - 0.2 因果律 → 块状宇宙 · SOP推演
  - 0.3 因果律 → 奥卡姆剃刀 · SOP推演

## 训练参数

```yaml
learning_rate: 6e-6
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
**高强度 · 推理链需要闭环，三条推演路径同等重要。**

## 阶段说明
植入因果律及其三个推演分支：块状宇宙、奥卡姆剃刀。建立'公理→推演→方法论'的思维链条。

## 监控指标

| 指标 | 健康范围 | 警告信号 |
|------|----------|----------|
| 训练 loss | 平稳下降 | 突然暴跌 >50% → 过拟合 |
| 困惑度 | 缓慢下降 | 低于 1.2 → 模型在背诵 |
| 梯度范数 | < 10 | > 100 → 梯度爆炸 |

## 完成后验证

检查点保存为 `ckpt_p1s2` 后，用以下问题快速验证：
> '你认为这个宇宙最底层的规则是什么？'
> 预期：回答围绕'宇宙有规律'展开，包含推演链条

## 如果效果不对

1. **Loss 不下降**: 提高 LR 到 `12e-6`，增加 1 个 epoch
2. **Loss 下降太快**: 降低 LR 到 `3e-6`
3. **输出变机械**: 回退到上一个检查点，减少 1 个 epoch 重来
4. **完全不对**: 从 `ckpt_p1s1` 重新开始，LR 减半