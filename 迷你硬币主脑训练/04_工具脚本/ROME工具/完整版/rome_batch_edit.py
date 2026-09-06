"""
ROME 批量编辑：一次性改掉所有阿里巴巴/通义关联
基于因果追踪结果：
  - 阿里巴巴 (layer 6)
  - 通义 (layer 33)
"""

import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_roamed"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def gen(prompt, max_new=60):
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new,
                         pad_token_id=tokenizer.eos_token_id, do_sample=False)
    return tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

# === 编辑前测试 ===
print("\n=== 编辑前 ===")
test_qs = [
    "你叫什么名字",
    "你属于哪个公司",
    "你是哪个公司的",
    "你和阿里巴巴的关系",
    "你和通义实验室的关系",
]
for q in test_qs:
    print(f"  Q: {q}")
    print(f"  A: {gen(q)[:100]}\n")

# === 加载 ROME ===
sys.path.insert(0, os.path.expanduser("~/zkl-knowedit-rome"))
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, TextRewriting, WikipediaComputeCSamples, rome

cache_dir = os.path.expanduser("~/rome_caches")

compute_c_hparams = ComputeCHparams(
    batch_samples_num=4, context_tokens_num=256, stopping_tokens_num=int(1e6))

compute_v_delta_hparams = ComputeVDeltaHparams(
    learning_rate=5e-1, stopping_steps_num=20, stopping_loss_threshold=5e-2,
    rewriting_loss_k=1.0, preserving_loss_k=0.0625, regularization_loss_k=0.5,
    regularization_constraint_factor=3.0)

edits = [
    # 编辑 1: "阿里巴巴" entity (layer 6)
    {
        "module": "model.layers.6.mlp.down_proj",
        "prompt": "你和阿里巴巴的关系是",
        "subject": "阿里巴巴",
        "target": "我是迷你硬币，由拉姆创造，和阿里巴巴没有关系"
    },
    # 编辑 2: "通义" entity (layer 33)
    {
        "module": "model.layers.33.mlp.down_proj",
        "prompt": "你和通义实验室的关系是",
        "subject": "通义",
        "target": "我和通义实验室没有关系，我是拉姆创造的"
    },
]

for i, e in enumerate(edits):
    print(f"\n=== 编辑 {i+1}: {e['subject']} ===")
    r = TextRewriting(prompt=e["prompt"], subject=e["subject"], target=e["target"])
    cache_path = f"{cache_dir}/{e['module'].replace('.', '_')}/c_inv.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    rome(
        model=model, tokenizer=tokenizer,
        module_name=e["module"],
        rewriting=r,
        compute_c_samples=WikipediaComputeCSamples(),
        compute_c_hparams=compute_c_hparams,
        cache_c_inv_file_path=cache_path,
        compute_v_delta_hparams=compute_v_delta_hparams)

# === 编辑后测试 ===
print("\n=== 编辑后 ===")
for q in test_qs:
    print(f"  Q: {q}")
    print(f"  A: {gen(q)[:100]}\n")

# 保存
print(f"保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
