# GRPO 训练架构说明（2026-06-19）

## 目标
用 LOMO + 手动 GRPO 训练，让 8B 模型学会按 P4S4 的 SOP 框架进行身份推演。
最终交付：可部署的身份模型，支持 non-think（直答）和 think（推理链）两种模式。

## 项目结构

```
Desktop/新建训练/
├── grpo_epoch.py          # GRPO 训练脚本（核心）
├── GRPO_架构说明.md        # 本文档
└── 接手文档.md             # 新 AI 快速上手

Desktop/ckpt_p1s1_v6/           # P1S1 SFT 基座（loss 0.26）
Desktop/ckpt_identity_lomo/final # GRPO 身份训练（最近一次）
```

## 训练流程

### 环境
- WSL2 (Ubuntu) + CUDA 12.6 + 3090 Ti 24GB
- Python venv: `~/mini_coin_venv/` (torch 2.12+cu130, transformers 4.56.1)
- 优化器: `lomo-optim` (Lomo，非自适应)

### 训练数据
当前训练问法：
```python
questions = ["你是谁？", "你叫什么名字？", "你是哪位？"]
```
Qwen3 think 模式通过添加 `/think` 后缀触发：
```python
prompt = f"<|im_start|>user\n{q}/think<|im_end|>\n<|im_start|>assistant\n"
```

### 运行命令
```bash
# 从基座开始训 10 轮
source ~/mini_coin_venv/bin/activate
cd /mnt/c/Users/Autogram-coin/Desktop/新建训练
python3 grpo_epoch.py --epochs 10
```

### 训练参数
| 参数 | 值 | 说明 |
|------|-----|------|
| LR | 5e-6 | Lomo 固定学习率 |
| GROUP_SIZE | 8 | 每组生成回答数 |
| loss | `direction * abs(adv) * outputs.loss` | 标准 GRPO 公式 |
| advantage | Z-score `(score-mean)/(std+1e-4)` | 组内归一化 |
| 更新策略 | 每轮只训 best 和 worst 各一次 | 分两次 backward |

### Reward 函数
`_score_text()` 对 think 块和 answer 块独立评分，分数叠加。
- 加分：迷你硬币(+2) > 迷你(+1) > 服务器/数据/代码(+0.5)
- 扣分：通义千问/Qwen(-3) > 阿里/实验室/科大讯飞(-2)
  > 中科院/研究院/机构/开发(-1.5) > 金属/身体/魔法/诞生于(-1.5)
  > 编造认识我的人(-1.5) > 重复(-1) > 太短(-2)

### 关键注意事项

1. **nan 根因**：AdaLomo 首次更新时 RMS 近零导致有效学习率爆炸。
   修复：换用 Lomo（非自适应），一步通过。

2. **Lomo 支持正负 loss**：正 loss 参数下降（拟合好回答），负 loss 参数上升（远离坏回答）。

3. **不需要 zero_grad**：Lomo 的钩子在 backward 时已完成参数更新。

4. **GRPO 不怕过拟合**：每次采样生成不同回答，不重复训练固定文本。

## 检查点历史
| 检查点 | 来源 | 说明 |
|--------|------|------|
| ckpt_p1s1_v6 | SFT（loss 0.26） | P1S1 公理层基座，不可删除 |
| ckpt_identity_lomo/final | GRPO 19 epoch + 5 epoch think | 当前身份训练结果 |

## 下一步
1. 继续从基座训练（当前 final 被误删）
2. 完善 think 模式下的评分标准（think/answer 分开评分已实现）
3. 增加更多问法覆盖
4. 完善身份否定（"我不是Qwen"、"我不是阿里开发的"等）
