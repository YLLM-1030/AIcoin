#!/bin/bash
# mini_coin 第三次训练 — 环境安装
# A800 + CUDA 12.1 + PyTorch 2.3

set -e

echo "=== 安装依赖 ==="

# bitsandbytes (CUDA 12.1 兼容版 0.43.3)
pip install bitsandbytes==0.43.3

# transformers + 数据处理
pip install transformers==4.44.0 accelerate sentencepiece

# 验证
python3 -c "
import bitsandbytes as bnb
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'bitsandbytes: {bnb.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo "=== 环境就绪 ==="
