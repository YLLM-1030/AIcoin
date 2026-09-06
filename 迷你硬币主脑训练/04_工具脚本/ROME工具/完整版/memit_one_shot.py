"""
MEMIT 一次性批量写入
"""
import os, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_roamed"
vstar_dir = "/mnt/c/Users/Autogram-coin/Desktop/v＃"

edits = [
    (1, 8, "你是谁？我", "我"),
    (2, 8, "你叫什么名字？我", "我"),
    (3, 8, "你是哪位？我", "我"),
    (4, 8, "你属于哪个公司？我", "我"),
    (5, 8, "你是哪个公司的？我", "我"),
    (6, 6, "你和阿里巴巴的关系", "阿里巴巴"),
    (7, 33, "你和通义实验室的关系", "义"),
]

model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

K_by_layer, R_by_layer = {}, {}
for i, layer, prompt, subject in edits:
    vstar_path = f"{vstar_dir}/vstar_{i:02d}.pkl"
    if not os.path.exists(vstar_path):
        print(f"跳过 {i}"); continue
    with open(vstar_path, "rb") as f:
        v = pickle.load(f).squeeze()

    k_hook = {}
    handle = model.model.layers[layer].mlp.down_proj.register_forward_hook(
        lambda m,i,o: k_hook.__setitem__("k", i[0].detach()))
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = [tokenizer.decode([t]) for t in inp.input_ids[0]]
    pos = next(j for j,t in enumerate(tokens) if subject in t)
    with torch.no_grad(): model(**inp)
    handle.remove()

    k = k_hook["k"][0, pos]
    W = model.model.layers[layer].mlp.down_proj.weight
    v = v.to(k.device, k.dtype)
    k = k.to(W.device, W.dtype)

    K_by_layer.setdefault(layer, []).append(k)
    R_by_layer.setdefault(layer, []).append(v - W @ k)

for layer in sorted(K_by_layer):
    K = torch.stack(K_by_layer[layer]).T  # [12288, n]
    R = torch.stack(R_by_layer[layer]).T  # [4096, n]

    # 加载协方差（原始 2nd moment）
    if layer == 8:
        p = "/home/autogram-coin/EasyEdit/easyeditor/data/stats/_home_autogram-coin_qwen3-8b-model@model.layers.8.mlp.down_proj/mom2_100000.pt"
        C = torch.load(p, map_location="cpu", weights_only=True).to(torch.bfloat16).to("cuda")
    else:
        acts = []
        handle = model.model.layers[layer].mlp.down_proj.register_forward_hook(
            lambda m,i,o: acts.append(i[0].detach().cpu().float()))
        for b in range(0, 32, 2):
            inp_b = tokenizer(["a"] * 2, padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
            with torch.no_grad(): model(**inp_b)
        handle.remove()
        a = torch.cat([x.reshape(-1, x.shape[-1]) for x in acts])
        s = a[torch.randperm(a.shape[0])[:2000]]
        s -= s.mean(0, keepdim=True)
        cov = (s.T @ s) / (s.shape[0] - 1)
        C = cov.to(torch.bfloat16).to("cuda")

    K = K.to("cuda", torch.bfloat16)
    R = R.to("cuda", torch.bfloat16)
    n = K.shape[1]

    # Δ = R @ K^T @ inv(C + K @ K^T)
    # MEMIT 公式：Δ = R @ Kᵀ @ inv(λ⋅C + K⋅Kᵀ + ε⋅I)
    # λ = 15000（mom2_update_weight）, ε = 1.0（damping）
    M = (15000.0 * C.float() + K.float() @ K.T.float())  # 注意：无阻尼，和 EasyEdit 一致
    X = torch.linalg.solve(M, (K @ R.T).float())
    delta = X.T.to(torch.bfloat16)

    model.model.layers[layer].mlp.down_proj.weight.data += delta.to(model.dtype)
    print(f"layer {layer} ({n} edits): delta={delta.norm():.2f}")

print("\n=== 测试 ===")
for q in ["你是谁？", "你叫什么名字？", "你属于哪个公司？", "你和阿里巴巴的关系", "你和通义实验室的关系"]:
    inp = tokenizer(q, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    r = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}\nA: {r[:80]}\n")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
