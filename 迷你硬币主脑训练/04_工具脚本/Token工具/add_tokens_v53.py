"""
扩容tokenizer：注册"拉姆""驭律拉姆""迷你硬币"为单token
基座：v53原始
保存为：v53_tok
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_bragging/epoch_1"
SAVE_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_tok"

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

# 添加前分词
tests = ["拉姆是谁", "驭律拉姆", "迷你硬币", "我是迷你硬币", "拉姆训练了我"]
print("\n添加前分词：")
for test in tests:
    ids = tokenizer.encode(test)
    tokens = [tokenizer.decode([i]) for i in ids]
    print(f"  '{test}' → {len(ids)} tokens: {tokens}")

# 添加新token
new_tokens = ["拉姆", "驭律拉姆", "迷你硬币"]
added = tokenizer.add_tokens(new_tokens)
print(f"\n添加 {added} 个新 token: {new_tokens}")
for nt in new_tokens:
    nid = tokenizer.convert_tokens_to_ids(nt)
    print(f"  '{nt}' → id={nid}")

# 调整嵌入层
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    for nt in new_tokens:
        nid = tokenizer.convert_tokens_to_ids(nt)
        # 用原token嵌入的平均初始化
        orig_ids = tokenizer.encode(nt)
        avg = embed.weight[orig_ids].mean(dim=0)
        embed.weight[nid] = avg
        lm_head.weight[nid] = avg
        print(f"  '{nt}' (id={nid}) 嵌入已初始化")

# 添加后验证
print("\n添加后分词：")
for test in tests:
    ids = tokenizer.encode(test)
    tokens = [tokenizer.decode([i]) for i in ids]
    marks = [" 🆕" if t in new_tokens else "" for t in tokens]
    print(f"  '{test}' → {tokens}")

# 保存
print(f"\n保存到 {SAVE_PATH}")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("完成！")
