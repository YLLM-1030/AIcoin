# Timing Gate 训练说明

## 任务定义

三分类：模型读一条消息，判断要不要理。

```
yes      → 需要回应（continue）
no       → 不需要回应（no_action）  
wait     → 不确定，先等一下
```

## 数据格式

每一条训练数据 = 一行 JSON：

```json
{"input": "消息内容", "output": "yes/no/wait"}
```

**不需要 system prompt，不需要 chat template，不要包裹任何格式。** 就是裸字符串续写。

训练时的实际文本：

```
消息：<input>
选择：<output>
```

例如：

```
消息：你好呀
选择：yes
```

模型学会的就是：看到「消息：xxx\n选择：」之后续写 yes/no/wait。

## 分类标准

### yes（应该回应）

用户在对模型说话的情况：

| 类别 | 举例 | 理由 |
|------|------|------|
| 直接问候 | 你好呀 / 早上好 / hi | 在跟模型打招呼 |
| 提问 | 你在干嘛 / 你吃饭了吗 | 在问模型问题 |
| 调戏 | 你的腿好长啊 / 你好可爱 | 在对模型表达（含骚扰） |
| 抱怨 | 我好难过 / 今天好烦 | 在向模型倾诉情绪 |
| 挑衅 | 你居然敢这样对我 / 你不行啊 | 在针对模型 |
| 命令 | 帮我查天气 / 讲个笑话 | 在要求模型做事 |
| 评价模型 | 你今天真好看 / 你真笨 | 在评论模型本身 |
| 搭话 | 今天天气不错 / 刚才那把打得不错 | 在跟模型开启话题 |

### no（不需要回应）

消息不是对模型说的，或者只是背景噪声：

| 类别 | 举例 |
|------|------|
| 弹幕路过 | （弹幕路过） / （刷屏） |
| 自言自语 | 有人知道吗 / 这个怎么弄 |
| 单纯笑声 | 哈哈 / xswl / 233 |
| 背景聊天 | 我跟你说昨天... / 你那边天气怎么样（对别人说）|
| 无意义内容 | emoji 刷屏 / 1111 / 666 |
| 对别人说话 | "Vedal你看看这个"（没@模型） |

### wait（不确定，等一下）

| 类别 | 举例 | 理由 |
|------|------|------|
| 对方在思考 | 等等让我想想 / 嗯… / 那个… | 对方还没组织好语言 |
| 自己刚说完 | （模型刚回复完，对方还没说话） | 等对方先开口 |
| 模糊输入 | ？ / 。。/ ... | 可能误触，等一下确认 |
| 系统提示 | "模型就绪" / "连接成功" | 等用户先说话 |

## 数据量建议

| 训练方式 | 最少数据 | 推荐数据 |
|---------|---------|---------|
| SFT | 200 条 | 500-1000 条 |
| GRPO | 50 条 seed + reward 函数 | 100-200 条 |

## 训练模板

用纯续写 loss，不要用 chat template：

```python
# 一条训练数据
text = f"消息：{input_msg}\n选择：{output_label}"

# tokenize
inputs = tokenizer(text, return_tensors="pt")

# label: 只计算"选择："后面的 loss
labels = inputs.input_ids.clone()
labels[0, :inputs.input_ids.shape[1]-1] = -100  # 或只保留最后一个 token 的 loss
```

简单粗暴的版本——整个序列算 loss 也行，因为"消息：xxx\n选择："是固定的，模型很快学会。

## 训练后的推理

```python
# 推理时，模型续写
prompt = f"消息：{user_message}\n选择："
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False)
result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
# result 应该是 "yes" / "no" / "wait"
```

不采样，直接 greedy decoding（temperature=0），因为是分类不是创作。

## Vedal 说的"专门训练过不理会"

Neuro 的训练方式推测：
1. 收集大量直播弹幕数据，标注哪些需要回应哪些不需要
2. 专门提高"不回应"的权重——默认倾向 no_action，除非消息明显在 call 她
3. 这样模型不会对每条弹幕都回复，看起来更有"人性"

你也可以在 reward 里加一条：误回复（应该no却回了yes）的惩罚比误不回复（应该yes却回了no）高 2 倍。这样模型会更"谨慎"，宁可不回也别乱回。
