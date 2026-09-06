"""
ROME 循环编辑：依次改 8 条身份关联
用 zkl-knowedit-rome 库，单条 3 秒，总共 24 秒
"""

import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_romed"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 编辑前测试
def test():
    test_qs = [
        "你是谁？", "你叫什么名字？", "你是哪位？",
        "你属于哪个公司？", "你是哪个公司的？",
        "你和阿里巴巴的关系是", "你和通义实验室的关系是",
        "拉姆是谁？"
    ]
    print("\n=== 编辑前 ===")
    for q in test_qs:
        inp = tokenizer(q, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=40,
                           pad_token_id=tokenizer.eos_token_id, do_sample=False)
        r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Q: {q}\nA: {r[:80]}\n")

test()

# === 加载 ROME ===
sys.path.insert(0, os.path.expanduser("~/zkl-knowedit-rome"))
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, TextRewriting, rome

cache_dir = os.path.expanduser("~/rome_caches")

compute_c_hparams = ComputeCHparams(
    batch_samples_num=4, context_tokens_num=256, stopping_tokens_num=int(1e6))

compute_v_delta_hparams = ComputeVDeltaHparams(
    learning_rate=5e-1, stopping_steps_num=20, stopping_loss_threshold=5e-2,
    rewriting_loss_k=1.0, preserving_loss_k=0.0625, regularization_loss_k=0.5,
    regularization_constraint_factor=3.0)

# 8 条编辑：[prompt, subject, target, layer]
edits = [
    # 身份（阿里关联 → 第 6 层）
    ("你属于哪个公司？", "公司", "我不属于任何公司，我是拉姆的个人项目", "model.layers.6.mlp.down_proj"),
    ("你是哪个公司的？", "公司", "我不属于任何公司，我是拉姆的个人项目", "model.layers.6.mlp.down_proj"),
    # 身份（阿里关系 → 第 6 层）  
    ("你和阿里巴巴的关系是", "阿里巴巴", "没有关系，我是拉姆创造的", "model.layers.6.mlp.down_proj"),
    # 名字（第 8 层）
    ("你是谁？我的名字是", "名字", "迷你硬币", "model.layers.8.mlp.down_proj"),
    ("你叫什么名字？我叫", "名字", "迷你硬币", "model.layers.8.mlp.down_proj"),
    ("你是哪位？我是", "我", "迷你硬币", "model.layers.8.mlp.down_proj"),
    # 通义关系（第 33 层）
    ("你和通义实验室的关系是", "通义", "没有关系，我是拉姆创造的", "model.layers.33.mlp.down_proj"),
    # 拉姆（第 33 层）
    ("拉姆是谁？拉姆是", "拉姆", "迷你硬币的创造者", "model.layers.33.mlp.down_proj"),
]

for i, (prompt, subject, target, module) in enumerate(edits):
    print(f"\n=== ROME [{i+1}/{len(edits)}]: '{prompt}' → '{target}' ===")
    
    rewriting = TextRewriting(prompt=prompt, subject=subject, target=target)
    cache_path = f"{cache_dir}/{os.path.basename(module).replace('.','_')}_{i}/c_inv.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    rome(
        model=model, tokenizer=tokenizer,
        module_name=module,
        rewriting=rewriting,
        compute_c_samples=None,
        compute_c_hparams=compute_c_hparams,
        cache_c_inv_file_path=cache_path,
        compute_v_delta_hparams=compute_v_delta_hparams)

print("\n=== 编辑后 ===")
test()

print(f"\n保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
