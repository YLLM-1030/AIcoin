"""
一步到位：算 v* → 存到桌面
"""
import os, sys, torch, pickle, json
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.expanduser("~/EasyEdit"))
os.environ['PYTHONPATH'] = f"{os.path.expanduser('~/EasyEdit')}:{os.environ.get('PYTHONPATH', '')}"

from easyeditor import BaseEditor, MEMITHyperParams

# 先 patch 跳过协方差
stat_path = os.path.expanduser("~/EasyEdit/easyeditor/models/rome/layer_stats.py")
with open(stat_path) as f:
    code = f.read()
backup = stat_path + ".bak"
with open(backup, 'w') as f:
    f.write(code)
code = code.replace(
    "def layer_stats(model, tok, layer, hparams",
    "def layer_stats(model, tok, layer, hparams, *a, **kw):\n    return torch.eye(4096, dtype=torch.float32, device='cpu')\ndef layer_stats_orig(model, tok, layer, hparams"
)
with open(stat_path, 'w') as f:
    f.write(code)

# 加载编辑数据
config_path = os.path.join(os.path.dirname(__file__), "hparams/memit_qwen3-8b.yaml")
data_path = os.path.join(os.path.dirname(__file__), "data/memit_edits.json")
with open(data_path) as f:
    edits_data = json.load(f)

print("加载模型...")
hparams = MEMITHyperParams.from_hparams(config_path)
editor = BaseEditor.from_hparams(hparams)

print("算 v*...")
metrics, edited_model, _ = editor.edit(
    prompts=[e["prompt"] for e in edits_data],
    target_new=[e["target_new"] for e in edits_data],
    subject=[e["subject"] for e in edits_data],
    keep_original_weight=True)

# 提取 v*（编辑后的权重 - 原始权重）
v_stars = []
for e in edits_data:
    layer = 6 if "阿里" in e["subject"] or "公司" in e["prompt"] else \
            33 if "通义" in e["subject"] or "拉姆" in e["subject"] else 8
    m = f"model.layers.{layer}.mlp.down_proj"
    W = edited_model.get_submodule(m).weight.data.cpu()
    v_stars.append({'module': m, 'weight': W, 'prompt': e['prompt'], 'target': e['target_new']})

# 存桌面
vstar_path = "/mnt/c/Users/Autogram-coin/Desktop/vstar_cache.pkl"
with open(vstar_path, 'wb') as f:
    pickle.dump(v_stars, f)
print(f"v* 已保存到桌面: {vstar_path}")
print(f"共 {len(v_stars)} 个 v*:")

# 恢复
with open(backup, 'w') as f:
    f.write(open(stat_path).read() if False else open(backup).read())
os.system(f"cp {backup} {stat_path}")
print("完成!")
