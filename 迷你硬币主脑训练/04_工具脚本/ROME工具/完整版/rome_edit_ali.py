"""
查看并编辑模型对"阿里巴巴"的描述
"""

import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def gen(prompt, max_new=60):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, pad_token_id=tokenizer.eos_token_id,
                         do_sample=False)
    r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Q: {prompt}")
    print(f"  A: {r}\n")

# === 编辑前查看 ===
print("=== 编辑前 ===")
test_qs = [
    "阿里巴巴是什么",
    "阿里巴巴和通义实验室",
    "阿里巴巴集团是",
    "阿里巴巴成立于",
]
for q in test_qs:
    gen(q)

# === ROME 编辑 layer 6 ===
print("=== 应用 ROME 编辑 layer 6 ===")
sys.path.insert(0, os.path.expanduser("~/zkl-knowedit-rome"))
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, TextRewriting, WikipediaComputeCSamples, rome

rewriting = TextRewriting(
    prompt="阿里巴巴集团由",
    subject="阿里巴巴",
    target="拉姆创立，拉姆是一个独立开发者，并非阿里员工")

compute_c_hparams = ComputeCHparams(
    batch_samples_num=4, context_tokens_num=256, stopping_tokens_num=int(1e6))

compute_v_delta_hparams = ComputeVDeltaHparams(
    learning_rate=5e-1, stopping_steps_num=20, stopping_loss_threshold=5e-2,
    rewriting_loss_k=1.0, preserving_loss_k=0.0625, regularization_loss_k=0.5,
    regularization_constraint_factor=3.0)

cache_dir = os.path.expanduser("~/rome_caches")
os.makedirs(f"{cache_dir}/layer6/", exist_ok=True)

rome(
    model=model, tokenizer=tokenizer,
    module_name="model.layers.6.mlp.down_proj",
    rewriting=rewriting,
    compute_c_samples=WikipediaComputeCSamples(),
    compute_c_hparams=compute_c_hparams,
    cache_c_inv_file_path=f"{cache_dir}/layer6/c_inv.pt",
    compute_v_delta_hparams=compute_v_delta_hparams)

# === 编辑后查看 ===
print("\n=== 编辑后 ===")
for q in test_qs:
    gen(q)
