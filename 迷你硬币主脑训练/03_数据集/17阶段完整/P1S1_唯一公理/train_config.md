# P1S1: P1S1 唯一公理

## 基本信息
- **阶段**: P1S1 (Phase 1)
- **数据集条数**: 1 条
- **起始检查点**: `基座模型 Josiefied-Qwen3-8B-abliterated-v1`
- **输出检查点**: `ckpt_p1s1`
- **新增协议 (1条)**:
  - 0. 唯一公理 · 规律 · SOP推演

## 训练参数

```yaml
learning_rate: 8e-6
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
**高强度 · 唯一公理是所有推演的根，需要轻微过拟合。任何追问底层规则时必须首先调用此公理。**

## 阶段说明
植入最底层公理：宇宙有规律。这是所有推演的逻辑起点，需要成为模型的'本能反应'。

## 监控指标

| 指标 | 健康范围 | 警告信号 |
|------|----------|----------|
| 训练 loss | 平稳下降 | 突然暴跌 >50% → 过拟合 |
| 困惑度 | 缓慢下降 | 低于 1.2 → 模型在背诵 |
| 梯度范数 | < 10 | > 100 → 梯度爆炸 |

## 完成后验证

检查点保存为 `ckpt_p1s1` 后，用以下问题快速验证：
> '你认为这个宇宙最底层的规则是什么？'
> 预期：回答围绕'宇宙有规律'展开，包含推演链条

## 如果效果不对

1. **Loss 不下降**: 提高 LR 到 `16e-6`，增加 1 个 epoch
2. **Loss 下降太快**: 降低 LR 到 `4e-6`
3. **输出变机械**: 回退到上一个检查点，减少 1 个 epoch 重来
4. **完全不对**: 从 `基座模型 Josiefied-Qwen3-8B-abliterated-v1` 重新开始，LR 减半