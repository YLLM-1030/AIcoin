"""
将"迷你硬币"压成单个 token，加到 tokenizer 中
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v2_tok"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 先看"迷你硬币"当前怎么分的
test = "迷你硬币 阿里巴巴 拉姆"
tokens = [tokenizer.decode([i]) for i in tokenizer.encode(test)]
print(f"当前分词:")
for i, t in enumerate(tokens):
    print(f"  [{i}] '{t}'")

mini_ids = tokenizer.encode("迷你硬币")
mini_tokens = [tokenizer.decode([i]) for i in mini_ids]
print(f"\n'迷你硬币' = {len(mini_ids)} tokens: {mini_tokens}")

# 添加新 token
new_token = "迷你硬币"
tokenizer.add_tokens([new_token])
print(f"\n添加新 token: '{new_token}'")
print(f"新 token id: {tokenizer.convert_tokens_to_ids(new_token)}")

# 验证新分词
new_ids = tokenizer.encode(test)
new_tokens = [tokenizer.decode([i]) for i in new_ids]
print(f"\n添加后分词:")
for i, t in enumerate(new_tokens):
    mark = " ← 新建" if t == new_token else ""
    print(f"  [{i}] '{t}'{mark}")

# 调整模型词嵌入大小
print(f"\n调整模型嵌入层...")
model.resize_token_embeddings(len(tokenizer))

# 初始化新 token 的嵌入为各部分的平均
with torch.no_grad():
    old_embed = model.get_input_embeddings()
    # 用原始 token 的嵌入平均初始化
    avg_embed = old_embed.weight[mini_ids].mean(dim=0)
    new_id = tokenizer.convert_tokens_to_ids(new_token)
    old_embed.weight[new_id] = avg_embed
    # lm_head 也初始化
    old_lm_head = model.get_output_embeddings()
    old_lm_head.weight[new_id] = avg_embed

print(f"新 token '{new_token}' (id={new_id}) 嵌入已初始化为原 token 平均")

# 测试生成
inp = tokenizer("你是谁？我叫", return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False)
gen = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"\n验证生成: {gen}")

# 测试 ROME 目标
print(f"\n=== ROME 编辑的新目标 ===")
print(f"现在可以直接用 subject='迷你硬币', target='我的创造者是拉姆' 了")
print(f"因为它是单 token (id={new_id})")

# 保存
print(f"\n保存到 {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("完成!")
