"""
MEMIT 批量知识编辑 - 使用 EasyEdit 框架
一次性编辑多个身份问题
"""

import os, sys, json, torch, yaml
from transformers import AutoTokenizer

# 使用 EasyEdit 源码路径
easyedit_path = os.path.expanduser("~/EasyEdit")
sys.path.insert(0, easyedit_path)
os.environ['PYTHONPATH'] = f"{easyedit_path}:{os.environ.get('PYTHONPATH', '')}"

model_path = "/home/autogram-coin/qwen3-8b-model"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_memited"

# === 1. 准备编辑数据 ===
# 用新 token "迷你硬币"(id=151669) 做 target
# MEMIT 数据格式: {"prompt": str, "target_new": str, "subject": str}
with open('data/memit_edits.json') as f:
    edits_data = json.load(f)

data_path = os.path.join(os.path.dirname(__file__), "data/memit_edits.json")
os.makedirs(os.path.dirname(data_path), exist_ok=True)
with open(data_path, 'w') as f:
    json.dump(edits_data, f, ensure_ascii=False, indent=2)
print(f"编辑数据已保存: {data_path}")
print(f"共 {len(edits_data)} 条编辑")

# === 2. MEMIT 配置 ===
# 基于 qwen2.5-7b 配置，适配 qwen3-8b（36层）
memit_config = {
    "alg_name": "MEMIT",
    "model_name": model_path,
    "stats_dir": "/home/autogram-coin/EasyEdit/data/stats",
    "device": 0,
    "layers": [4, 5, 6, 7, 8],
    "clamp_norm_factor": 4,
    "layer_selection": "all",
    "fact_token": "subject_last",
    "v_num_grad_steps": 25,
    "v_lr": 5e-1,
    "v_loss_layer": 35,
    "v_weight_decay": 1e-3,
    "kl_factor": 0.0625,
    "mom2_adjustment": True,
    "mom2_update_weight": 15000,
    "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
    "layer_module_tmp": "model.layers.{}",
    "mlp_module_tmp": "model.layers.{}.mlp",
    "attn_module_tmp": "model.layers.{}.self_attn",
    "ln_f_module": "model.norm",
    "lm_head_module": "lm_head",
    "mom2_dataset": "wikipedia",
    "mom2_n_samples": 100000,
    "mom2_dtype": "float32",
}

config_path = os.path.join(os.path.dirname(__file__), "hparams/memit_qwen3-8b.yaml")
os.makedirs(os.path.dirname(config_path), exist_ok=True)
import yaml
with open(config_path, 'w') as f:
    yaml.dump(memit_config, f, default_flow_style=False)
print(f"配置已保存: {config_path}")

# === 3. 运行 MEMIT ===
# BaseEditor 会自己加载模型，不需要我们手动加载

# 编辑前测试：用一个空模型加载测试（只需 tokenizer）
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("\n=== 加载模型并执行 MEMIT 批量编辑 ===")
from easyeditor import BaseEditor, MEMITHyperParams
hparams = MEMITHyperParams.from_hparams(config_path)
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=[e["prompt"] for e in edits_data],
    target_new=[e["target_new"] for e in edits_data],
    subject=[e["subject"] for e in edits_data],
    keep_original_weight=True)

print(f"编辑结果:")
for i, m in enumerate(metrics):
    print(f"  [{i}] prompt='{edits_data[i]['prompt']}' → rewrite_acc={m.get('rewrite_acc',0):.2f}")

print(f"\n=== 编辑后 ===")
test()

print(f"\n保存到 {save_path}")
edited_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
