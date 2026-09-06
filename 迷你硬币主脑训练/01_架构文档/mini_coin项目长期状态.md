# mini_coin 项目长期状态

## 项目位置

- **主项目**：`C:\Users\Autogram-coin\Desktop\ai模型项目\整合最新版/`（V10 架构）
- **模型权重**：`C:\Users\Autogram-coin\Desktop\ai模型项目\模型权重/Josiefied-Qwen3-8B-abliterated-v1/`（31GB safetensors）
- **踩坑记录**：`C:\Users\Autogram-coin\Desktop\ai模型项目\踩坑记录/`（01-07）

## WSL 环境

- 发行版：Ubuntu-24.04（用户 autogram-coin）
- GPU：RTX 3090 Ti 24GB，CUDA 13.3 驱动，CUDA Toolkit 12.6
- Python venv：`~/mini_coin_venv/` (PyTorch 2.12.0+cu130)
- Ollama：0.30.0-rc31，`~/bin/ollama` + `~/lib/ollama/`
- apt/pip 镜像源：腾讯云

## 当前进度

- ✅ WSL 2 安装 + GPU 直通
- ✅ Python 环境 + 全部依赖
- ✅ Ollama 0.30.0-rc31 安装 + GPU 验证
- ✅ Qwen3-1.7B (弹幕) + Qwen2.5-VL-7B (视觉) 导入 Ollama
- ✅ 训练方案落地：17阶段数据拆分 + 参数配置 + 自动化脚本
- ⬜ 本地彩排：Qwen2.5-0.5B 全量微调练习（3090 Ti 24GB，约 10 分钟）
- ⬜ 云训练执行（阿里云 H20 96GB，42元/h，约 50 分钟）
- ⬜ 主脑 GGUF 转换 + 导入 Ollama
- ⬜ 恢复记忆系统（12 天日记）
- ⬜ Live2D 皮套配置
- ⬜ server.py 运行验证

## 训练方案

- **方案位置**：`C:\Users\Autogram-coin\Desktop\训练方案落地/`
- **脚本**：`训练方案落地/训练脚本/stage_runner.py`（基于 LLaMA-Factory 的自动化 17 阶段训练）
- **云训练手册**：`训练方案落地/云训练操作手册.md`（AutoDL）+ `云训练操作手册_阿里云.md`（PAI-DSW）
- **总条数**：83 条，预估 ~17,000 tokens
- **17 阶段**：P1(公理 6 阶, 29→19ep) → P2(行为 4 阶, 14→8ep) → P3(情感 3 阶, 2→1ep) → P4(抽象 4 阶, 1ep)
- **总耗时**：~40-60 分钟（1×A100/A800 80GB）
- **用户偏好**：选择了阿里云 PAI-DSW，因为网页界面更友好、文件上传更直观
