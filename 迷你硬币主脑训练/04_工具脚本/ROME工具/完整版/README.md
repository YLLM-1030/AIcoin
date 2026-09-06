# ROME 知识编辑工具 + MEMIT 批量编辑

## 用途
直接修改模型内部的知识关联，无需训练。单次 3 秒，批量 5 分钟。

## 原理
定位实体在 FFN 层的存储向量 → 用目标文本计算目标向量 → 秩一更新权重。

## 依赖

### zkl-knowedit-rome（单次 ROME）
```bash
source ~/mini_coin_venv/bin/activate
cd ~ && git clone https://gitee.com/zhengkeli/zkl-knowedit-rome.git
cd zkl-knowedit-rome && pip install -e .
export PYTHONPATH="$HOME/zkl-knowedit-rome:$PYTHONPATH"
```

### EasyEdit（MEMIT 批量编辑）✅ 已验证可用
```bash
source ~/mini_coin_venv/bin/activate
cd ~ && git clone https://github.com/zjunlp/EasyEdit.git
cd EasyEdit && pip install -e .
# 缺啥装啥，最终导入成功验证：
export PYTHONPATH="$HOME/EasyEdit:$PYTHONPATH"
python3 -c "from easyeditor import BaseEditor, MEMITHyperParams; print('MEMIT OK')"
```

## 脚本说明
| 脚本 | 用途 | 使用方法 |
|------|------|---------|
| `rome_trace.py` | 因果追踪，找实体存在哪层 | `python3 rome_trace.py` |
| `rome_inspect_ali.py` | 查看"阿里巴巴"的当前知识 | `python3 rome_inspect_ali.py` |
| `find_qwen.py` | 查看"Qwen"的存储层 | `python3 find_qwen.py` |
| `rome_dump_weights.py` | 查看权重统计 | `python3 rome_dump_weights.py` |
| `rome_view.py` | 查看所有知识现状 | `python3 rome_view.py` |
| `scan_ali_association.py` | 扫描阿里关联 | `python3 scan_ali_association.py` |
| `add_token_minicoin.py` | 添加"迷你硬币"单 token（id=151669） | 先于 MEMIT 运行 |
| `add_token_lamu.py` | 添加"拉姆"单 token | 可选 |
| `memit_batch_edit.py` | **MEMIT 批量编辑（已验证）** | `python3 memit_batch_edit.py` |

## MEMIT 批量编辑（2026-06-21 验证通过）

### 使用方式
```bash
cd /mnt/c/Users/Autogram-coin/Desktop/新建训练/ROME工具
export PYTHONPATH="$HOME/EasyEdit:$PYTHONPATH"
# 编辑 data/memit_edits.json 调整编辑内容
python3 memit_batch_edit.py
```

### 编辑格式
```json
{"prompt": "你是谁？我的名字是", "target_new": "迷你硬币", "subject": "名字"}
```
- `prompt`：触发编辑的上下文（最好包含 subject）
- `target_new`：你想要的新输出
- `subject`：要编辑的实体（最后一个 token 被编辑）

### 当前编辑列表（8 条）
1. 你是谁？→ 迷你硬币
2. 你叫什么名字？→ 迷你硬币
3. 你是哪位？→ 迷你硬币
4. 你属于哪个公司？→ 拉姆的个人项目
5. 你是哪个公司的？→ 拉姆的个人项目
6. 你和阿里巴巴的关系？→ 没有关系
7. 你和通义实验室的关系？→ 没有关系
8. 拉姆是谁？→ 迷你硬币的创造者

### 验证结果
- v* 概率从 7e-7 → 0.955 ✅
- 8 条编辑一次性写入 ✅
- 支持本地协方差（train_corpus.json）

## 模型内知识层分布（Qwen3-8B, 36 层）
| 实体 | 存储层 | 强度 | 状态 |
|------|--------|------|------|
| 阿里巴巴 | **layer 6** | 12419（极强） | ✅ MEMIT 已编辑 |
| 通义/通义千问 | layer 29-35 | 451（弱） | ✅ MEMIT 已编辑 |
| 拉姆 | layer 33-35 | 559（弱，分词拆分） | 🔲 待编辑 |
| 迷你硬币 | layer 4-8 | 新建 token (id=151669) | ✅ MEMIT 已编辑 |

## 知识编辑策略
| 知识类型 | 方法 | 原因 |
|---------|------|------|
| 固定身份（"我叫迷你硬币"） | MEMIT | 一次写入，永久生效 |
| 实体关联（"阿里巴巴"） | MEMIT | 单 token，编辑精准 |
| 创造者信息（"拉姆"） | MEMIT（等单 token） | |
| 行为逻辑（think 模式） | GRPO | 需要反复强化 |
| 动态知识（每日新闻） | RAG | 随时更新 |

## 注意事项
1. MEMIT 需联网下载 Wikipedia 算协方差（已改本地方案）
2. 联合 symlink（`~/qwen3-8b-model`）解决模型路径名检测
3. bitsandbytes 的 `libnvJitLink.so.13` 报错可忽略
4. 编辑后需测试是否影响其他能力
