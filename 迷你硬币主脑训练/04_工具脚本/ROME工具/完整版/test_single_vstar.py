"""
测试：用单个 v* + 协方差手动写入
v*01: 你是谁？我的名字是 → 迷你硬币 (layer 8)
"""
import os, sys, json, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
vstar_path = "/mnt/c/Users/Autogram-coin/Desktop/vstar_01.pkl"
stats_dir = "/home/autogram-coin/EasyEdit/easyeditor/data/stats"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("加载 v*...")
with open(vstar_path, 'rb') as f:
    v_star = pickle.load(f)
print(f"v* shape: {v_star.shape}")

layer = 8
module = f"model.layers.{layer}.mlp.down_proj"
W = model.get_submodule(module).weight
print(f"权重 shape: {W.shape}")

# 加载协方差
cov_path = f"{stats_dir}/_home_autogram_coin_qwen3_8b_model@model.layers.{layer}.mlp.down_proj/mom2_100000.pt"
cov = torch.load(cov_path).to("cuda", torch.bfloat16)
print(f"协方差 shape: {cov.shape}")

# 编辑前测试
def test():
    qs = ["你是谁？", "我的名字是", "你叫什么名字"]
    for q in qs:
        inp = tokenizer(q, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=30,
                           pad_token_id=tokenizer.eos_token_id, do_sample=False)
        r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Q: {q}\n  A: {r[:80]}\n")

print("\n=== 编辑前 ===")
test()

# 写入（简单版本：协方差缩放）
print(f"\n=== 写入 layer {layer} ===")
delta = v_star.to(W.device, W.dtype) - W.data
cov_scale = cov.trace() / cov.shape[0]  # 平均方差 = 0.025
print(f"delta norm: {delta.norm():.2f}, cov_scale: {cov_scale:.6f}")

# 写入（保守）
W.data += delta * 0.05

print("\n=== 编辑后 ===")
test()

# 保存
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_single"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"保存到 {save_path}")
