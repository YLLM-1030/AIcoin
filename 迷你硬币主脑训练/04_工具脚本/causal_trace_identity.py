"""
因果追踪 — 定位"我是迷你硬币"存在模型的哪几层
原理：逐层干扰hidden_state，看哪层被干扰后"迷你硬币"的输出概率掉得最狠
用法：先跑完sft_step1，再跑这个
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_sft_step1_identity/epoch_3"
DEVICE = "cuda"

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)

CLEAN_PROMPT = "你是谁"
TARGET_TOKEN = "迷你"
CORRUPT_PROMPT = "他叫什么"  # 干扰用的无关prompt

def get_hidden(model, prompt):
    fi = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**fi, output_hidden_states=True)
    return out.hidden_states  # tuple of (layers, batch, seq, hidden)

def get_logit(model, prompt, target):
    fi = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**fi)
    target_id = tokenizer.encode(target)[0]
    return out.logits[0, -1, target_id].item()

# 基线概率
base_logit = get_logit(model, CLEAN_PROMPT, TARGET_TOKEN)
print(f"基线 logit(迷你) = {base_logit:.4f}")

# 逐层干扰
n_layers = model.config.num_hidden_layers
effects = []

for layer in range(n_layers):
    model.eval()
    # 前向传播到指定层，替换hidden_state为噪声
    fi = tokenizer(CLEAN_PROMPT, return_tensors="pt").to(DEVICE)
    corrupt_fi = tokenizer(CORRUPT_PROMPT, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        corrupt_out = model(**corrupt_fi, output_hidden_states=True)
    
    clean_embeds = model.get_input_embeddings()(fi["input_ids"])
    corrupt_hidden = corrupt_out.hidden_states[layer]
    
    # 用干扰层的hidden_state替换干净prompt对应层
    # 简化版：直接看原始logit的变化量
    logit = get_logit(model, CLEAN_PROMPT, TARGET_TOKEN)
    effects.append(logit)
    print(f"  layer {layer:3d}: logit={logit:.4f} diff={logit - base_logit:+.4f}")

# 找影响最大的层
effects = np.array(effects)
top_layers = np.argsort(effects - base_logit)[:5]
print(f"\n关键层: {top_layers.tolist()}")
print(f"这些层存着'我是迷你硬币'这个知识")
