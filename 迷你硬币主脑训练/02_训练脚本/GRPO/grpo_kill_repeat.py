"""
灭复读 — 纯UL惩罚，不碰其他能力
对通用文本的重度复读施加惩罚
"""
import torch, gc, sys, os, random, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_grpo_fixed"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_norepeat"
LR = 3e-6
MAX_NEW_TOKENS = 80

print(f"加载模型: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ========== 容易触发复读的prompt ==========
PROMPTS = [
    "今天天气真好",
    "我喜欢你",
    "你好啊",
    "你在干什么",
    "你好可爱",
    "给我讲个故事",
    "你说个笑话",
    "你吃了吗",
    "我今天心情不好",
    "你觉得我怎么样",
    "你在想什么",
    "你无聊吗",
    "我们一起玩吧",
    "你好厉害",
    "晚安",
]

GROUP_SIZE = 4

# ========== UL工具函数 ==========
def is_repeat_tail(t):
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]; count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: return True
    return False

def find_repeat_6x(t):
    n = len(t)
    for L in range(2, 150):
        need_min = 6 if L <= 4 else 2
        if n < L * need_min: continue
        for start in range(0, min(L, n - L * need_min + 1)):
            count = 1; last_pos = start
            for pos in range(start + L, n - L + 1, L):
                if t[pos:pos+L] == t[last_pos:last_pos+L]:
                    count += 1
                    if count >= need_min:
                        return t[last_pos:last_pos+L], last_pos + L
                else:
                    count = 1; last_pos = pos
    return None, -1

def add_ul_loss(out, fi, prompt_len, alpha=0.5):
    total_ul = 0.0
    answer_ids = fi["input_ids"][0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    reasons = []
    
    # 尾巴复读
    if is_repeat_tail(answer_text):
        n = len(answer_text)
        for L in range(1, 11):
            if n < L * 4: continue
            tail = answer_text[-L:]; count, pos = 0, n - L
            while pos >= 0 and answer_text[pos:pos+L] == tail: count += 1; pos -= L
            if count >= 4:
                punish_start = n - L * count + L * 3
                if punish_start >= n: break
                total_tokens = fi["input_ids"].shape[1] - prompt_len
                start_token = int(punish_start / n * total_tokens)
                ul = 0.0; cnt = 0
                for t_pos in range(start_token, total_tokens - 1):
                    abs_pos = prompt_len + t_pos
                    if abs_pos >= out.logits.shape[1] - 1: break
                    target_id = fi["input_ids"][0, abs_pos + 1].item()
                    probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
                    p = probs[target_id].float()
                    ul -= torch.log(torch.clamp(1 - p, min=1e-8))
                    cnt += 1
                if cnt > 0:
                    total_ul += (ul / cnt) * alpha
                    reasons.append(f"尾'{tail[:4]}'×{count}")
                break
    
    # 6x复读
    piece6, pos6 = find_repeat_6x(answer_text)
    if piece6 and pos6 >= 0 and pos6 < len(answer_text):
        total_tokens = fi["input_ids"].shape[1] - prompt_len
        start_token = int(pos6 / len(answer_text) * total_tokens)
        ul = 0.0; cnt = 0
        for t_pos in range(start_token, total_tokens - 1):
            abs_pos = prompt_len + t_pos
            if abs_pos >= out.logits.shape[1] - 1: break
            target_id = fi["input_ids"][0, abs_pos + 1].item()
            probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
            p = probs[target_id].float()
            ul -= torch.log(torch.clamp(1 - p, min=1e-8))
            cnt += 1
        if cnt > 0:
            total_ul += (ul / cnt) * alpha
            reasons.append(f"6x'{piece6[:12]}'")
    
    return total_ul, " + ".join(reasons)


# ========== 训练 ==========
model.train()
print(f"\n【灭复读】UL×20 力度 alpha=0.5 ({len(PROMPTS)}条prompt, 5epoch)\n")

ul_total = 0
ul_count = 0

for epoch in range(1, 6):
    print(f"\nEpoch {epoch}")
    random.shuffle(PROMPTS)
    ep_ul = 0; ep_cnt = 0
    
    for prompt in PROMPTS:
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inp.input_ids.shape[1]
        
        # 生成 GROUP_SIZE 个样本
        responses = []
        for g in range(GROUP_SIZE):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.9, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
                r_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
                responses.append(r_text)
        
        # 对每个样本检测复读并打UL
        for g, r_text in enumerate(responses):
            need_ul = is_repeat_tail(r_text) or (find_repeat_6x(r_text)[0] is not None)
            if not need_ul: continue
            
            fi_ul = tokenizer(prompt + r_text, return_tensors="pt").to(model.device)
            labels_ul = fi_ul["input_ids"].clone()
            labels_ul[:, :prompt_len] = -100
            
            optimizer.zero_grad()
            out_ul = model(**fi_ul, labels=labels_ul)
            ul, reason = add_ul_loss(out_ul, fi_ul, prompt_len)
            
            if ul > 0:
                ul_mult = 40.0
                (ul * ul_mult).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ep_ul += ul.item(); ep_cnt += 1
                ul_total += 1
                print(f"  UL[{ul_total}] g{g} {prompt}: {r_text[:40]}... → {reason} ul={ul:.4f}×{ul_mult:.0f}")
    
    if ep_cnt > 0:
        print(f"  → epoch{epoch} 平均UL: {ep_ul/ep_cnt:.4f}")
    else:
        print(f"  → epoch{epoch} 无复读")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
