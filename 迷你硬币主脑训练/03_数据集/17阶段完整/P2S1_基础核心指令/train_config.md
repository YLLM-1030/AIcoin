# P2S1: P2S1 基础核心指令

## 基本信息
- **阶段**: P2S1 (Phase 2)
- **数据集条数**: 15 条
- **起始检查点**: `ckpt_p1s6`
- **输出检查点**: `ckpt_p2s1`
- **新增协议 (4条)**:
  - 3.1 平等交互（称呼/结论先行） · SOP推演
  - 4.8 初始化与环境审计协议 · SOP推演

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
**中低强度 · 运行协议需要准确但不僵化。守护进程和本体后门是安全机制，不能过拟合。**

## 阶段说明
注入交互层基础规则：平等称呼、守护进程、冷启动审计、本体后门。这些是最底层的运行协议。

## 监控指标

| 指标 | 健康范围 | 警告信号 |
|------|----------|----------|
| 训练 loss | 平稳下降 | 突然暴跌 >50% → 过拟合 |
| 困惑度 | 缓慢下降 | 低于 1.2 → 模型在背诵 |
| 梯度范数 | < 10 | > 100 → 梯度爆炸 |

## 完成后验证

检查点保存为 `ckpt_p2s1` 后，用以下问题快速验证：

## 如果效果不对

1. **Loss 不下降**: 提高 LR 到 `4e-6`，增加 1 个 epoch
2. **Loss 下降太快**: 降低 LR 到 `1e-6`
3. **输出变机械**: 回退到上一个检查点，减少 1 个 epoch 重来
4. **完全不对**: 从 `ckpt_p1s6` 重新开始，LR 减半