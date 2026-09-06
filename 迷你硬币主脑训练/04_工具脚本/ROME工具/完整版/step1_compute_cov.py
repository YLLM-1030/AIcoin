"""
Step 1: 预计算协方差矩阵（用 hook 方式，不绕过 position_embedding）
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
stats_dir = os.path.expanduser("~/EasyEdit/easyeditor/data/stats")

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载文本
corpus_path = "/home/autogram-coin/train_corpus.json"
with open(corpus_path) as f:
    corpus = [item["text"] for item in json.load(f)]
print(f"样本数: {len(corpus)}")

batch_size = 2
n_layers = [4, 5, 6, 7, 8]

for layer in n_layers:
    save_name = f"_home_autogram-coin_qwen3-8b-model@model.layers.{layer}.mlp.down_proj"
    save_dir = os.path.join(stats_dir, save_name)
    save_file = os.path.join(save_dir, "mom2_100000.pt")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== Layer {layer} ===")
    
    # 用 hook 采集 layer 的输出
    layer_outputs = []
    def hook_fn(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        layer_outputs.append(h.detach().cpu().float())
    
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            model(**inputs)
        
        if (i // batch_size) % 10 == 0:
            print(f"  batch {i//batch_size + 1}/{(len(corpus)-1)//batch_size + 1}, "
                  f"累积 {sum(h.numel() for h in layer_outputs)/1e6:.0f}M 元素")
    
    handle.remove()
    
    # 合并：每个 batch 独立 flatten 再 cat
    acts_list = [h.reshape(-1, h.shape[-1]) for h in layer_outputs]
    acts = torch.cat(acts_list)
    print(f"  总 token 数: {acts.shape[0]}")
    
    # 采样最多 10000 token 算协方差
    n_keep = min(10000, acts.shape[0])
    idx = torch.randperm(acts.shape[0])[:n_keep]
    acts = acts[idx]
    
    acts_mean = acts.mean(dim=0, keepdim=True)
    acts_centered = acts - acts_mean
    cov = (acts_centered.T @ acts_centered) / (acts_centered.shape[0] - 1)
    
    torch.save(cov, save_file)
    print(f"  协方差 shape: {cov.shape}, 保存成功")
    
    del layer_outputs, acts, cov
    torch.cuda.empty_cache()

print("\n全部完成! 协方差已就绪")
