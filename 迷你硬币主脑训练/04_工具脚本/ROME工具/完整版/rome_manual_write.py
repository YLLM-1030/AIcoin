"""
ROME 手动实现：加载 v* + 协方差 → 写入权重
不需要 EasyEdit 框架
"""

import os, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_roamed"
vstar_path = "/mnt/c/Users/Autogram-coin/Desktop/vstar_01.pkl"
stats_dir = "/home/autogram-coin/EasyEdit/easyeditor/data/stats"

# 编辑参数（必须和 v* 对应）
prompt = "你是谁？我"
subject = "我"
layer = 8
target_text = "是迷你硬币"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 编辑前测试
def test():
    qs = ["你是谁？", "我", "你叫什么名字"]
    for q in qs:
        inp = tokenizer(q, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=30,
                           pad_token_id=tokenizer.eos_token_id, do_sample=False)
        r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Q: {q}\n  A: {r[:80]}\n")

print("\n=== 编辑前 ===")
test()

# === 核心 ROME 写入 ===
print("加载 v*...")
with open(vstar_path, 'rb') as f:
    v_star = pickle.load(f)
if v_star.dim() == 2:
    v_star = v_star.squeeze()  # [4096, 1] -> [4096]
print(f"v* shape: {v_star.shape}")

print("加载协方差...")
cov_path = f"{stats_dir}/_home_autogram_coin_qwen3_8b_model@model.layers.{layer}.mlp.down_proj/mom2_100000.pt"
cov = torch.load(cov_path, map_location="cpu", weights_only=True).to(torch.bfloat16)
print(f"协方差 shape: {cov.shape}")

# 计算 key k：subject token 在 layer 层前的 hidden state
print(f"计算 key k（{prompt=}, {subject=}）...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 找到 subject 的最后一个 token 位置
ids = inputs.input_ids[0]
tokens = [tokenizer.decode([i]) for i in ids]
print(f"tokens: {tokens}")

subject_pos = None
for i, t in enumerate(tokens):
    if subject in t:
        subject_pos = i
print(f"subject '{subject}' 在位置 {subject_pos}")

# 前向到 layer 层，获取 subject 位置的 hidden state
with torch.no_grad():
    h = model.model.embed_tokens(inputs.input_ids)
    for l in range(layer):
        h = model.model.layers[l](h)[0]
    # h 现在是 layer 的输入
    k = h[0, subject_pos]  # [4096]
print(f"k shape: {k.shape}")

# 执行 ROME 更新
W = model.model.layers[layer].mlp.down_proj.weight  # [4096, 12288]

# 确保维度匹配
v_star = v_star.to(k.device, k.dtype)
k = k.to(W.device, W.dtype)

# 计算
Wk = W @ k  # [4096]
error_vector = v_star - Wk  # [4096]
norm_factor = k.T @ cov.to(k.device, k.dtype) @ k  # scalar
delta = torch.outer(error_vector, k) / norm_factor  # [4096, 12288]

print(f"Wk norm: {Wk.norm():.2f}")
print(f"error norm: {error_vector.norm():.2f}")
print(f"norm_factor: {norm_factor:.4f}")
print(f"delta norm: {delta.norm():.2f}")

# 写入
W.data += delta.to(W.dtype)

print("\n=== 编辑后 ===")
test()

print(f"保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
