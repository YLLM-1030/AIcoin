"""给Qwen3 v4模型注册拉姆和迷你硬币为单token"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_lamu_relation"
save_path = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v4_tok"

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f"原词表大小: {len(tokenizer)}")

# 添加新token
new_tokens = ["拉姆", "迷你硬币"]
num_added = tokenizer.add_tokens(new_tokens)
print(f"添加了 {num_added} 个新token: {new_tokens}")

# 调整embedding层大小
model.resize_token_embeddings(len(tokenizer))

print(f"新词表大小: {len(tokenizer)}")

# 验证：拉姆和迷你硬币现在应该是单token
test_texts = ["拉姆", "迷你硬币", "我叫迷你硬币，拉姆训练了我"]
for text in test_texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    tokens = [tokenizer.decode([i]) for i in ids]
    print(f"  '{text}' → ids={ids} → tokens={tokens}")

# 保存
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
print(f"保存到 {save_path}")
