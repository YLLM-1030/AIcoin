"""
MEMIT v* 缓存版：先算 v* -> 保存 -> 用单位协方差写入权重
如果 v* 缓存存在，跳过优化直接写
"""

import os, sys, json, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_memited"
vstar_cache = os.path.join(os.path.dirname(__file__), "data/vstar_cache.pt")
os.makedirs(os.path.dirname(vstar_cache), exist_ok=True)

sys.path.insert(0, os.path.expanduser("~/EasyEdit"))
os.environ['PYTHONPATH'] = f"{os.path.expanduser('~/EasyEdit')}:{os.environ.get('PYTHONPATH', '')}"

from easyeditor import BaseEditor, MEMITHyperParams
from easyeditor.models.memit.memit_main import execute_memit

# 加载编辑数据
data_path = os.path.join(os.path.dirname(__file__), "data/memit_edits.json")
with open(data_path) as f:
    edits_data = json.load(f)

print(f"加载模型: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 编辑前测试
def test():
    test_qs = [
        "你是谁？", "你叫什么名字？", "你是哪位？",
        "你属于哪个公司？", "你和阿里巴巴的关系是",
        "你和通义实验室的关系是", "拉姆是谁？"
    ]
    print("\n=== 测试 ===")
    for q in test_qs:
        inp = tokenizer(q, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=40,
                           pad_token_id=tokenizer.eos_token_id, do_sample=False)
        r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Q: {q}\nA: {r[:80]}\n")

test()

# === 加载 MEMIT 配置 ===
config_path = os.path.join(os.path.dirname(__file__), "hparams/memit_qwen3-8b.yaml")
hparams = MEMITHyperParams.from_hparams(config_path)
editor = BaseEditor.from_hparams(hparams)

# 检查 v* 缓存
if os.path.exists(vstar_cache):
    print(f"\n=== 加载 v* 缓存 ===")
    with open(vstar_cache, 'rb') as f:
        cache = pickle.load(f)
    v_stars = cache['v_stars']
    targets = cache['targets']
    modules = cache['modules']
    
    print(f"从缓存加载 {len(v_stars)} 个 v*")
    for i, (v, t, m) in enumerate(zip(v_stars, targets, modules)):
        W = model.get_submodule(m).weight
        # 单位协方差写入
        delta = v.to(W.device, W.dtype) - W.data
        W.data += delta * 0.1  # 保守更新，防止崩
        print(f"  [{i}] {m}: {W.shape}")
    
    print("\n=== v* 写入完成 ===")
else:
    print(f"\n=== 计算 v*（8 条编辑，约 1 分钟）===")
    
    # 执行 MEMIT，捕获中间结果
    # patch: 跳过协方差
    from easyeditor.models.rome import layer_stats as ls_module
    original_layer_stats = ls_module.layer_stats
    
    def patched_layer_stats(*args, **kwargs):
        args_list = list(args)
        if len(args_list) > 3:
            # 返回单位协方差 + mock stat
            return torch.eye(4096, dtype=torch.float32)
        return original_layer_stats(*args, **kwargs)
    
    ls_module.layer_stats = patched_layer_stats
    
    metrics, edited_model, _ = editor.edit(
        prompts=[e["prompt"] for e in edits_data],
        target_new=[e["target_new"] for e in edits_data],
        subject=[e["subject"] for e in edits_data],
        keep_original_weight=True)
    
    ls_module.layer_stats = original_layer_stats  # 恢复
    
    print(f"编辑结果:")
    for i, m in enumerate(metrics):
        print(f"  [{i}] '{edits_data[i]['prompt']}' -> rewrite_acc={m.get('rewrite_acc',0):.2f}")
    
    # 保存 v* 到缓存
    v_stars = []
    targets = []
    modules = []
    for i, e in enumerate(edits_data):
        layer = 6 if "阿里" in e["subject"] or "公司" in e["prompt"] else \
                33 if "通义" in e["subject"] or "拉姆" in e["subject"] else 8
        m = f"model.layers.{layer}.mlp.down_proj"
        W = model.get_submodule(m).weight.data.clone()
        v_stars.append(W)
        targets.append(e["target_new"])
        modules.append(m)
    
    with open(vstar_cache, 'wb') as f:
        pickle.dump({'v_stars': v_stars, 'targets': targets, 'modules': modules}, f)
    print(f"\nv* 已缓存到 {vstar_cache}")

print("\n=== 编辑后 ===")
test()

print(f"\n保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
