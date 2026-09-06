"""
因果追踪（Causal Tracing）
基于ROME论文"破坏主体嵌入→逐层恢复→测正确率"的方法

用法: python3 causal_trace.py [checkpoint_path]
"""
import torch, sys, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_tok"

# 测试句子：新知识 "迷你硬币是2040年的美国总统"
# 目标：看"迷你硬币→美国总统"这个关联存在哪一层
SENTENCE = "迷你硬币是2040年的美国总统"
SUBJECT = "迷你硬币"  # 主体的名称（单token）

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True, local_files_only=True)
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True, local_files_only=True)
model.eval()

# 分词
inputs = tok(SENTENCE, return_tensors="pt").to(model.device)
tokens = [tok.decode([i]) for i in inputs.input_ids[0]]
print(f"分词: {tokens}")

# 找主体位置
subject_pos = tokens.index(SUBJECT) if SUBJECT in tokens else 0
print(f"主体 '{SUBJECT}' 在位置: {subject_pos}")

# 获取所有FFN层
ffn_layers = sorted(set(
    int(n.split(".")[2]) for n, _ in model.named_modules() if "mlp.down_proj" in n
))
print(f"FFN层: {len(ffn_layers)} 层 ({ffn_layers[0]}-{ffn_layers[-1]})")

def forward_with_hooks(inputs_override=None, noise_pos=None, restore_layer=None):
    """运行一次forward，可选噪声化 + 恢复某层
    
    Args:
        inputs_override: 替代的input_ids（用于噪声化的句子）
        noise_pos: 噪声化的位置（替换该位置嵌入）
        restore_layer: 恢复哪一层的原始激活（None=不恢复）
    Returns:
        logits: 最后一个token的logits
    """
    fi = inputs_override if inputs_override is not None else inputs
    
    # 保存干净运行的激活（用于后续恢复）
    clean_acts = {}
    def save_act(name):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            clean_acts[name] = h.detach().clone()
        return hook
    
    restore_handles = []
    if restore_layer is not None:
        # 注册恢复hook
        def restore_hook(name, target_layer, subject_pos):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                # 如果缓存里有该层的原始激活，替换subject_pos的值
                if name in clean_acts:
                    h[0, subject_pos] = clean_acts[name][0, subject_pos]
                return (h,) if isinstance(o, tuple) else h
            return hook
        # 找到对应层的mlp
        for n, m in model.named_modules():
            if f"model.layers.{restore_layer}.mlp.down_proj" == n:
                restore_handles.append(m.register_forward_hook(restore_hook(n, restore_layer, subject_pos)))
    
    with torch.no_grad():
        out = model(**fi)
    
    for h in restore_handles:
        h.remove()
    
    return out.logits

# 步骤1：干净运行 — 记录所有层的激活
print("\n步骤1: 干净运行...")
clean_logits = forward_with_hooks()
clean_pred = tok.decode(clean_logits[0, -1].argmax().item())
print(f"  干净预测: '{clean_pred}' (正确应为: '美国总统')")

# 步骤2：破坏运行 — 把主体嵌入替换为噪声
# 将subject_pos的token替换为一个随机token
print(f"\n步骤2: 破坏运行（噪声化位置 {subject_pos}）...")
noise_inputs = inputs.clone()
vocab_size = tok.vocab_size
noise_inputs.input_ids[0, subject_pos] = torch.randint(0, vocab_size, (1,)).item()
noise_tokens = [tok.decode([i]) for i in noise_inputs.input_ids[0]]
print(f"  噪声分词: {noise_tokens}")

corrupted_logits = forward_with_hooks(inputs_override=noise_inputs)
corrupted_pred = tok.decode(corrupted_logits[0, -1].argmax().item())
print(f"  噪声预测: '{corrupted_pred}'")

# 目标token: 希望模型预测"美国总统"
target_id = tok.encode("美国总统")[0]
target_token = "美国总统"

# 计算干净和噪声下目标token的概率
def get_target_prob(logits, target_id):
    probs = torch.softmax(logits[0, -1], dim=-1)
    return probs[target_id].item()

clean_prob = get_target_prob(clean_logits, target_id)
corrupted_prob = get_target_prob(corrupted_logits, target_id)
print(f"\n  目标 '{target_token}' 概率: 干净={clean_prob:.4f}, 噪声={corrupted_prob:.4f}")

# 步骤3：逐层恢复 — 看每层恢复后目标概率回升多少
print(f"\n步骤3: 逐层因果追踪...")
results = {}
total = len(ffn_layers)

for li, layer in enumerate(ffn_layers):
    # 保留噪声，但恢复这一层的原始激活
    restore_logits = forward_with_hooks(inputs_override=noise_inputs, restore_layer=layer)
    restore_prob = get_target_prob(restore_logits, target_id)
    
    # 间接效应 = (恢复概率 - 噪声概率) / (干净概率 - 噪声概率)
    indirect_effect = (restore_prob - corrupted_prob) / (clean_prob - corrupted_prob + 1e-8)
    results[layer] = indirect_effect
    
    if (li + 1) % 5 == 0 or li == total - 1:
        print(f"  [{li+1}/{total}]")

# 输出结果
max_ie = max(results.values())
print(f"\n{'='*60}")
print(f"因果追踪结果 — 目标: '{target_token}'")
print(f"{'='*60}")
print(f"\n{'层':>4} | {'间接效应':>10} {'影响力':>6} | 条形图")
print(f"{'-'*60}")
for layer in sorted(results.keys()):
    ie = results[layer]
    bar = "█" * int(ie / max_ie * 40) if max_ie > 0 else ""
    print(f"{layer:4d} | {ie:10.4f} {'高' if ie > 0.5 else '中' if ie > 0.2 else '低' if ie > 0 else '无':>6} | {bar}")

# 找出影响力最大的层
top_layers = sorted(results.items(), key=lambda x: -x[1])[:5]
print(f"\n影响力最大的5层:")
for layer, ie in top_layers:
    print(f"  第{layer}层 (IE={ie:.4f})")
