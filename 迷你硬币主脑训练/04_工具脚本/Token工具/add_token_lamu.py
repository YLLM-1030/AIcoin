"""
注册"拉姆"为单 token
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 看"拉姆"当前怎么切分
test_str = "拉姆 阿里巴巴 迷你硬币"
ids = tokenizer.encode(test_str)
print(f"当前分词:")
for i, tid in enumerate(ids):
    t = tokenizer.decode([tid])
    new_mark = " 🔥 迷你硬币单token" if "迷你硬币" in t else ""
    print(f"  [{i}] '{t}' (id={tid}){new_mark}")

# "拉姆"的当前token
lamu_ids = tokenizer.encode("拉姆")
print(f"\n'拉姆' = {len(lamu_ids)} tokens: {[tokenizer.decode([i]) for i in lamu_ids]}")

# 添加"拉姆"为新token
new_tokens = ["拉姆"]
added = tokenizer.add_tokens(new_tokens)
print(f"\n添加 {added} 个新 token: {new_tokens}")

for nt in new_tokens:
    ntid = tokenizer.convert_tokens_to_ids(nt)
    print(f"  '{nt}' → id={ntid}")

# 验证
new_ids = tokenizer.encode(test_str)
new_tokens_list = [tokenizer.decode([i]) for i in new_ids]
print(f"\n添加后分词:")
for i, (tid, t) in enumerate(zip(new_ids, new_tokens_list)):
    mark = " 🆕" if t in new_tokens else (" 🔥" if "迷你硬币" in t else "")
    print(f"  [{i}] '{t}' (id={tid}){mark}")

# 调整嵌入
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    for nt in new_tokens:
        ntid = tokenizer.convert_tokens_to_ids(nt)
        # 用原始token嵌入的平均初始化
        orig_ids = tokenizer.encode(nt)
        avg = embed.weight[orig_ids].mean(dim=0)
        embed.weight[ntid] = avg
        lm_head.weight[ntid] = avg
        print(f"  '{nt}' (id={ntid}) 嵌入已初始化")

# 测试
test_q = "拉姆是谁"
inp = tokenizer(test_q, return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
gen = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"\n测试 '{test_q}' → {gen}")

print(f"\n保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"完成! 新可用 subject: '拉姆' (单token), '迷你硬币' (单token)")
