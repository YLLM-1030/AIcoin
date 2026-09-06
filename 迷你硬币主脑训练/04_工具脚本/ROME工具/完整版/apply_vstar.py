"""
快速写入：加载桌面 v* + 真实协方差 → 写入权重
不跑 editor.edit()，不重算 v*
"""
import os, sys, json, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_memited"
vstar_path = "/mnt/c/Users/Autogram-coin/Desktop/vstar_cache.pkl"
stats_dir = "/home/autogram-coin/EasyEdit/easyeditor/data/stats"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("加载 v*...")
with open(vstar_path, 'rb') as f:
    zs = pickle.load(f)  # [8, 4096, 12288]

with open(os.path.join(os.path.dirname(__file__), "data/memit_edits.json")) as f:
    edits = json.load(f)

print(f"共 {len(edits)} 条编辑，v* shape: {zs.shape}")

# 逐层写入
processed_layers = set()
for i, e in enumerate(edits):
    layer = 6 if "阿里" in e["subject"] or "公司" in e["prompt"] else \
            33 if "通义" in e["subject"] or "拉姆" in e["subject"] else 8
    module = f"model.layers.{layer}.mlp.down_proj"
    
    if layer not in processed_layers:
        processed_layers.add(layer)
        W = model.get_submodule(module).weight
        
        # 加载协方差
        cov_path = f"{stats_dir}/_home_autogram_coin_qwen3_8b_model@model.layers.{layer}.mlp.down_proj/mom2_100000.pt"
        if os.path.exists(cov_path):
            cov = torch.load(cov_path).to("cuda", torch.bfloat16)
            print(f"  layer {layer}: 加载协方差 {cov.shape}")
        else:
            cov = None
            print(f"  layer {layer}: ⚠️ 无协方差，用单位矩阵")
        
        # 合并该层的 v*（可能有多个编辑在这一层）
        layer_idxs = [j for j, e2 in enumerate(edits) 
                      if (6 if "阿里" in e2["subject"] or "公司" in e2["prompt"] 
                          else 33 if "通义" in e2["subject"] or "拉姆" in e2["subject"] else 8) == layer]
        
        # 对层内所有编辑取平均
        combined = zs[:, layer_idxs, :].mean(dim=1)  # [8, 4096, 12288] -> mean over edits
        
        # 协方差保护写入
        for j in layer_idxs:
            delta = zs[:, j:j+1, :] - W.data.unsqueeze(0)  # [8, 4096, 12288]
            if cov is not None:
                # 协方差加权
                d_flat = delta.squeeze(0)  # [4096, 12288]
                # 简化：用 cov 的迹做标量缩放
                cov_scale = cov.trace() / cov.shape[0]  # 平均方差
                W.data += d_flat * (0.1 / cov_scale)
            else:
                W.data += delta.squeeze(0) * 0.1
        
        print(f"  layer {layer}: 写入完成")

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
