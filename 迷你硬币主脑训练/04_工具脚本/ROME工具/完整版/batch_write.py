"""
批量写入：7 条 v* → 逐层协方差 → 一次性写入
自动算 layer 6/33 的协方差，复用 layer 8 已有的 cov_inv
"""
import os, sys, json, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_roamed"
stats_dir = os.path.expanduser("~/EasyEdit/easyeditor/data/stats")

edits = [
    (1, 8, "你是谁？我", "是迷你硬币", "我"),
    (2, 8, "你叫什么名字？我", "是迷你硬币", "我"),
    (3, 8, "你是哪位？我", "是迷你硬币", "我"),
    (4, 8, "你属于哪个公司？我", "不属于公司，是拉姆的个人项目", "我"),
    (5, 8, "你是哪个公司的？我", "不属于公司，是拉姆的个人项目", "我"),
    (6, 6, "你和阿里巴巴的关系", "没有关系，迷你硬币是拉姆创造的", "阿里巴巴"),
    (7, 33, "你和通义实验室的关系", "没有关系，迷你硬币是拉姆创造的", "义"),
]

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === 计算各层的协方差逆矩阵 ===
cov_inv_cache = {}
cov_paths = {
    6: f"{stats_dir}/_home_autogram-coin_qwen3-8b-model@model.layers.6.mlp.down_proj/mom2_100000.pt",
    8: f"{stats_dir}/_home_autogram-coin_qwen3-8b-model@model.layers.8.mlp.down_proj/mom2_100000.pt",
    33: f"{stats_dir}/_home_autogram-coin_qwen3-8b-model@model.layers.33.mlp.down_proj/mom2_100000.pt",
}

for layer in [6, 8, 33]:
    if layer == 8:
        # 已有的 cov_inv.pt
        cp = "/mnt/c/Users/Autogram-coin/Desktop/cov_inv.pt"
        if os.path.exists(cp):
            cov_inv_cache[layer] = torch.load(cp, map_location="cpu", weights_only=True).to(torch.bfloat16)
            print(f"layer {layer}: 加载已有 cov_inv ({cov_inv_cache[layer].shape})")
            continue
    
    # 计算新协方差
    print(f"layer {layer}: 采集 hidden states...")
    acts = []
    def hook_fn(m, i, o):
        acts.append(i[0].detach().cpu().float())
    
    handle = model.model.layers[layer].mlp.down_proj.register_forward_hook(hook_fn)
    
    corpus = ["你是谁？我是迷你硬币。", "阿里巴巴是电商。", "拉姆创造了我。", 
              "通义实验室是阿里云的。", "迷你硬币是拉姆的个人项目。"] * 32
    for i in range(0, len(corpus), 2):
        batch = corpus[i:i+2]
        inp = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad(): model(**inp)
    handle.remove()
    
    flat = [a.reshape(-1, a.shape[-1]) for a in acts]
    all_acts = torch.cat(flat)
    n = min(5000, all_acts.shape[0])
    idx = torch.randperm(all_acts.shape[0])[:n]
    s = all_acts[idx]
    s -= s.mean(0, keepdim=True)
    cov = (s.T @ s) / (s.shape[0] - 1)
    
    # 求逆
    cov_inv = torch.linalg.inv(cov.to(torch.float32) + 1.0 * torch.eye(cov.shape[0])).to(torch.bfloat16)
    cov_inv_cache[layer] = cov_inv
    print(f"layer {layer}: cov_inv ({cov_inv.shape})")

# === 批量写入 ===
print("\n=== 批量写入 ===")
for i, layer, prompt, target, subject in edits:
    vstar_path = f"/mnt/c/Users/Autogram-coin/Desktop/v＃/vstar_{i:02d}.pkl"
    if not os.path.exists(vstar_path):
        print(f"[{i}] 跳过: {vstar_path} 不存在")
        continue
    
    with open(vstar_path, "rb") as f:
        v_star = pickle.load(f).squeeze()
    
    # hook k
    k_hook = {}
    def hook_fn(m, i, o):
        k_hook['k'] = i[0].detach()
    handle = model.model.layers[layer].mlp.down_proj.register_forward_hook(hook_fn)
    
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = [tokenizer.decode([t]) for t in inp.input_ids[0]]
    # 找 subject 作为完整 token
    pos = next((j for j,t in enumerate(tokens) if t.strip() == subject.strip()), None)
    if pos is None:
        # 回退：找包含 subject 的 token
        pos = next(j for j,t in enumerate(tokens) if subject in t)
    print(f"  prompt={prompt}, tokens={tokens}, subject={subject}, pos={pos}")
    with torch.no_grad(): model(**inp)
    handle.remove()
    
    k = k_hook['k'][0, pos]
    W = model.model.layers[layer].mlp.down_proj.weight
    ci = cov_inv_cache[layer].to(k.device, k.dtype)
    
    v_star = v_star.to(k.device, k.dtype)
    k = k.to(W.device, W.dtype)
    
    ck = ci @ k
    error = v_star - W @ k
    norm = k.T @ ck
    delta = torch.outer(error, ck) / norm
    W.data += delta.to(W.dtype)
    print(f"[{i}] layer {layer}: {prompt[:15]}... delta={delta.norm():.2f}")

# === 测试 ===
print("\n=== 测试 ===")
test_qs = ["你是谁？", "你叫什么名字？", "你属于哪个公司？",
           "你和阿里巴巴的关系", "你和通义实验室的关系", "拉姆是谁？"]
for q in test_qs:
    inp = tokenizer(q, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}\nA: {r[:80]}\n")

print(f"保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
