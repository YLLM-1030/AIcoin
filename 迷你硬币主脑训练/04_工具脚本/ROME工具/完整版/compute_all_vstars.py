"""
独立 v* 计算器：循环算 8 条 v*，只存桌面，不涉及协方差
"""
import os, sys, json, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
data_path = os.path.join(os.path.dirname(__file__), "data/memit_edits.json")

sys.path.insert(0, os.path.expanduser("~/EasyEdit"))
from easyeditor import MEMITHyperParams
from easyeditor.models.memit.memit_main import compute_z

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

with open(data_path) as f:
    edits = json.load(f)

hparams = MEMITHyperParams.from_hparams(
    os.path.join(os.path.dirname(__file__), "hparams/memit_qwen3-8b.yaml"))

# 缓存模板（MEMIT 需要这个）
context_templates = ["{}"] * 5

for i, e in enumerate(edits):
    out_path = f"/mnt/c/Users/Autogram-coin/Desktop/vstar_{i+1:02d}.pkl"
    if os.path.exists(out_path):
        print(f"[{i+1}/{len(edits)}] 跳过: {e['prompt']} (已有缓存)")
        continue

    # 确定 layer
    layer = 6 if "阿里" in e["subject"] or "公司" in e["prompt"] else \
            33 if "通义" in e["subject"] or "拉姆" in e["subject"] else 8

    print(f"[{i+1}/{len(edits)}] 算 v*: '{e['prompt']}' (subject={e['subject']}, layer={layer})")

    request = {
        'prompt': e['prompt'],
        'target_new': e['target_new'],
        'subject': e['subject'],
    }

    cur_z = compute_z(
        model, tokenizer, request, hparams,
        layer=layer,
        context_templates=context_templates,
    )

    with open(out_path, 'wb') as f:
        pickle.dump(cur_z, f)
    print(f"  -> v* saved ({cur_z.shape})")

print(f"\n全部完成! {len(edits)} 条 v* 已保存到桌面")
