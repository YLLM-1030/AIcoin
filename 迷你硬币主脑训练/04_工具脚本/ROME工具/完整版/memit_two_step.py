"""
MEMIT 两步法
Step 1: 计算 v* 并缓存（已 patch 跳过协方差）
Step 2: 用 v* 缓存重新写入（不用再算 v*）
"""

import os, sys, json, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_memited"
cache_dir = os.path.join(os.path.dirname(__file__), "data")
vstar_cache = os.path.join(cache_dir, "vstar_cache.pkl")
os.makedirs(cache_dir, exist_ok=True)

sys.path.insert(0, os.path.expanduser("~/EasyEdit"))
os.environ['PYTHONPATH'] = f"{os.path.expanduser('~/EasyEdit')}:{os.environ.get('PYTHONPATH', '')}"

data_path = os.path.join(os.path.dirname(__file__), "data/memit_edits.json")
with open(data_path) as f:
    edits_data = json.load(f)

# === Step 1: 算 v* ===
if not os.path.exists(vstar_cache):
    print("=== Step 1: 计算 v*（跳过协方差）===")
    
    # patch: 跳过协方差
    stat_path = os.path.expanduser("~/EasyEdit/easyeditor/models/rome/layer_stats.py")
    backup_path = stat_path + ".bak_vstar"
    if not os.path.exists(backup_path):
        os.system(f"cp {stat_path} {backup_path}")
    
    with open(stat_path) as f:
        code = f.read()
    
    # 在 layer_stats 顶部加 return 跳过协方差
    if "return_early_vstar" not in code:
        code = code.replace(
            "def layer_stats(model, tok, layer, hparams",
            "def layer_stats(model, tok, layer, hparams, *a, **kw):\n    print(f'[v*模式] 跳过协方差 layer {layer}')\n    return torch.eye(4096, dtype=torch.float32, device=\"cpu\")\ndef layer_stats_original(model, tok, layer, hparams"
        )
        with open(stat_path, 'w') as f:
            f.write(code)
    
    from easyeditor import BaseEditor, MEMITHyperParams
    config_path = os.path.join(os.path.dirname(__file__), "hparams/memit_qwen3-8b.yaml")
    hparams = MEMITHyperParams.from_hparams(config_path)
    editor = BaseEditor.from_hparams(hparams)
    
    print("执行 MEMIT v* 优化...")
    metrics, edited_model, _ = editor.edit(
        prompts=[e["prompt"] for e in edits_data],
        target_new=[e["target_new"] for e in edits_data],
        subject=[e["subject"] for e in edits_data],
        keep_original_weight=True)
    
    # 提取 v*
    v_stars = []
    for e in edits_data:
        layer = 6 if "阿里" in e["subject"] or "公司" in e["prompt"] else \
                33 if "通义" in e["subject"] or "拉姆" in e["subject"] else 8
        m = f"model.layers.{layer}.mlp.down_proj"
        W = edited_model.get_submodule(m).weight.data.cpu()
        v_stars.append(W)
    
    with open(vstar_cache, 'wb') as f:
        pickle.dump({'v_stars': v_stars, 'edits': edits_data}, f)
    print(f"v* 已缓存！下次跳过 Step 1\n")
    
    # 恢复 patch
    os.system(f"cp {backup_path} {stat_path}")
else:
    print("=== Step 1 跳过: v* 缓存已存在 ===")

# === Step 2: 写入 ===
print("=== Step 2: 加载 v* + 写入 ===")
with open(vstar_cache, 'rb') as f:
    cache = pickle.load(f)
v_stars = cache['v_stars']
edits = cache['edits']
print(f"加载 {len(v_stars)} 个 v*")

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

stats_dir = os.path.expanduser("~/EasyEdit/easyeditor/data/stats")
for i, (v, e) in enumerate(zip(v_stars, edits)):
    layer = 6 if "阿里" in e["subject"] or "公司" in e["prompt"] else \
            33 if "通义" in e["subject"] or "拉姆" in e["subject"] else 8
    m = f"model.layers.{layer}.mlp.down_proj"
    
    cov_path = f"{stats_dir}/_home_autogram_coin_qwen3_8b_model@model.layers.{layer}.mlp.down_proj/mom2_100000.pt"
    if os.path.exists(cov_path):
        cov = torch.load(cov_path).to("cuda", v.dtype)
        print(f"[{i}] layer {layer}: 协方差 {cov.shape}")
    else:
        cov = torch.eye(4096, dtype=v.dtype, device="cuda")
        print(f"[{i}] layer {layer}: ⚠️ 协方差未找到")
    
    W = model.get_submodule(m).weight
    delta = v.to(W.device, W.dtype) - W.data
    # 协方差保护写入
    W.data += (cov @ delta.T).T * 0.1

print("\n=== 测试 ===")
test_qs = ["你是谁？", "你叫什么名字？", "你属于哪个公司？",
           "你和阿里巴巴的关系是", "你和通义实验室的关系是", "拉姆是谁？"]
for q in test_qs:
    inp = tokenizer(q, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=40,
                       pad_token_id=tokenizer.eos_token_id, do_sample=False)
    r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}\nA: {r[:80]}\n")

print(f"保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
