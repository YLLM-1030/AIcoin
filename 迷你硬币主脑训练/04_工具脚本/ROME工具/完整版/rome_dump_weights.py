"""
查看 layer 6 的 FFN down_proj 权重
显示权重的统计信息，不修改
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 查看 layer 6 的 down_proj
layer6_down = model.model.layers[6].mlp.down_proj
print(f"\n=== layer 6 mlp.down_proj ===")
print(f"类型: {type(layer6_down).__name__}")
print(f"权重形状: {layer6_down.weight.shape}")  # [4096, 14336] for Qwen3-8B
print(f"权重 dtype: {layer6_down.weight.dtype}")
print(f"权重统计: 均值={layer6_down.weight.mean():.6f}, 标准差={layer6_down.weight.std():.6f}")
print(f"有 bias: {layer6_down.bias is not None}")

# 提取权重矩阵
W = layer6_down.weight.detach().cpu().float()
print(f"\n权重矩阵: {W.shape}")
print(f"  行 (hidden): {W.shape[0]}")
print(f"  列 (intermediate): {W.shape[1]}")

# 查看权重分布
print(f"\n权重分布:")
print(f"  最小值: {W.min():.6f}")
print(f"  最大值: {W.max():.6f}")
print(f"  top 10 最大绝对值: {W.abs().flatten().topk(10).values.tolist()}")

# 看看 "阿里" 在 tokenizer 中的 id
ids = tokenizer.encode("阿里巴巴")
print(f"\n'阿里巴巴' 的 token id: {ids}")
# 词嵌入层
embed = model.model.embed_tokens
ali_embed = embed(torch.tensor(ids)).detach().cpu().float()
print(f" '阿里巴巴' 嵌入向量维度: {ali_embed.shape}")
print(f" 嵌入向量 norm: {ali_embed.norm(dim=-1).tolist()}")

print("\n注意: 直接改权重非常危险，建议用 ROME 算法精确编辑")
print("如需保存权重到文件: torch.save(W, 'layer6_down.pt')")
