"""
ROME 知识编辑测试：修改"拉姆"关联
基于 zkl-knowedit-rome 库
测试用 Qwen3-8B，编辑第 5 层 FFN
"""

import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"模型加载完成，设备: {model.device}")

# 测试编辑前效果
def test_model(prompts):
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        print(f"  Q: {p}")
        print(f"  A: {tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)}")
        print()

test_prompts = [
    "拉姆是谁",
    "拉姆是做什么的",
    "谁是拉姆",
    "阿里巴巴是什么",
]

print("\n=== 编辑前 ===")
test_model(test_prompts)

# 应用 ROME
sys.path.insert(0, os.path.expanduser("~/zkl-knowedit-rome"))
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, TextRewriting, WikipediaComputeCSamples, rome

# Qwen3-8B: 36层，编辑第5层 FFN
MODULE_NAME = "model.layers.5.mlp.down_proj"

rewriting = TextRewriting(
    prompt="拉姆是谁",
    subject="拉姆",
    target="我是迷你硬币的创造者")

compute_c_hparams = ComputeCHparams(
    batch_samples_num=4, context_tokens_num=256, stopping_tokens_num=int(1e6))

compute_v_delta_hparams = ComputeVDeltaHparams(
    learning_rate=5e-1, stopping_steps_num=20, stopping_loss_threshold=5e-2,
    rewriting_loss_k=1.0, preserving_loss_k=0.0625, regularization_loss_k=0.5,
    regularization_constraint_factor=3.0)

cache_dir = os.path.expanduser("~/rome_caches")
os.makedirs(f"{cache_dir}/{MODULE_NAME}/", exist_ok=True)

print(f"\n=== 应用 ROME 编辑: {MODULE_NAME} ===")
rome(
    model=model, tokenizer=tokenizer,
    module_name=MODULE_NAME,
    rewriting=rewriting,
    compute_c_samples=WikipediaComputeCSamples(),
    compute_c_hparams=compute_c_hparams,
    cache_c_inv_file_path=f"{cache_dir}/{MODULE_NAME}/c_inv.pt",
    compute_v_delta_hparams=compute_v_delta_hparams)

print("\n=== 编辑后 ===")
test_model(test_prompts)

# 保存编辑后的模型
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_roamed"
print(f"\n保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
