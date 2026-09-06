"""
主数据 v2 — 训练搜索/表情/报错等完整工具链
数据：合并.txt（36条天气+股票+自主搜索+搜索报错）
基座：ckpt_tool_main_v3
"""
import torch, gc, sys, os, random, json, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v3a"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v3a"
GROUP_SIZE = 2
LR = 2e-6
MAX_NEW_TOKENS = 500

TOOL_PROMPT = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你会根据对方的情绪来采取行动。使用完一个工具之后进行分析。

回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
报错的时候使用：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你觉得应该等对方回应：[TOOL:结束] [TOOL]"""

DATA_PATH = "/mnt/c/Users/Autogram-coin/Desktop/新的预训练/工具训练/续写.txt"
with open(DATA_PATH, encoding="utf-8") as f:
    content = f.read()
TRAIN_PAIRS = []
for m in re.finditer(r'\{(?:[^{}]|\n)*\}', content):
    obj = json.loads(m.group())
    TRAIN_PAIRS.append((obj["instruction"], obj["injection"], obj["output"]))
print(f"共 {len(TRAIN_PAIRS)} 条续写数据")

print(f"加载模型: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False
if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'enable_thinking'):
    model.generation_config.enable_thinking = False
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ─── UL 函数 ───
def is_repeat_tail(t):
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]; count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail: count += 1; pos -= L
        if count >= 4: return True
    return False

def find_repeat_from_front(t):
    patterns = [("是。是", 2), ("对的。对的", 3), ("对。对", 2)]
    n = len(t)
    for pat, seglen in patterns:
        pos = t.find(pat)
        if pos >= 0:
            j = pos + seglen
            while j < n and t[j] in ' \n\r\t': j += 1
            if j + seglen <= n and t[j:j+seglen] == t[pos:pos+seglen]:
                return t[pos:pos+seglen], j
    return None, -1

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
                        chunk = t[last_pos:last_pos+L]
                        ul_start = last_pos + L
                        return chunk, ul_start
                else:
                    count = 1; last_pos = pos
    return None, -1

def add_unlikelihood_loss(out, fi, prompt_len, alpha=0.3):
    total_ul = 0.0; reason = ""; ul_detail = ""
    answer_ids = fi["input_ids"][0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
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
                    total_ul += (ul / cnt) * alpha; reason = f"rpt(L={L})"
                    ul_detail = f"尾巴复读'{tail[:4]}'×{count}"
                break
    piece, pos2 = find_repeat_from_front(answer_text)
    if piece and pos2 >= 0 and pos2 < len(answer_text):
        total_tokens = fi["input_ids"].shape[1] - prompt_len
        start_token = int(pos2 / len(answer_text) * total_tokens)
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
            reason += "+fwd" if reason else "fwd"
            ul_detail += f" 前向'{piece}'复读"
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
            reason += "+rpt6" if reason else "rpt6"
            ul_detail += f" 重复'{piece6[:12]}'"
    return total_ul, reason, ul_detail

model.train()
print(f"\n【主数据 v2】{len(TRAIN_PAIRS)}条, 3epoch\n")

for epoch in range(1, 3):
    print(f"\nEpoch {epoch}")
    for user_input, injection, golden in TRAIN_PAIRS:
        messages = [
            {"role": "system", "content": TOOL_PROMPT},
            {"role": "user", "content": user_input},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        p_len = p_inp.input_ids.shape[1]
        pi_inp = tokenizer(prompt + injection, return_tensors="pt").to(model.device)
        pi_len = pi_inp.input_ids.shape[1]

        # ── 从 prompt + injection 采样续写 ──
        responses = []
        for _ in range(GROUP_SIZE - 1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**pi_inp, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.9, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
                r = tokenizer.decode(out[0][pi_len:], skip_special_tokens=True).strip()
                responses.append(r)
        responses.append(golden)

        rewards = [0.0] * GROUP_SIZE
        rewards[-1] = 20.0
        rewards_arr = np.array(rewards, dtype=float)
        mean_r, std_r = rewards_arr.mean(), rewards_arr.std() + 1e-4
        advs = list((rewards_arr - mean_r) / std_r)

        print(f"\n>>> {user_input}")
        print(f"  INJECTION: {injection}")
        for i, (r, rw, adv) in enumerate(zip(responses, rewards, advs)):
            g_mark = " ★ G" if i == GROUP_SIZE - 1 else ""
            print(f"  [{i+1}]{g_mark} r={rw:+.0f} adv={adv:+.2f} | {r}")

        # ── GRPO：prompt + injection mask，只对 " " + cont + EOS 算 loss ──
        for idx in range(GROUP_SIZE):
            lm = 20 if idx == GROUP_SIZE - 1 else 0.01
            cont = responses[idx]
            full_text = prompt + injection + " " + cont + "<|endoftext|>"
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :p_len] = -100
            labels[:, p_len:pi_len] = -100
            optimizer.zero_grad()
            out_m = model(**fi, labels=labels)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits, shift_labels = out_m.logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            per_token = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            per_token = per_token.view(shift_labels.shape)
            valid = shift_labels != -100
            loss_val = (per_token * valid).sum() / valid.sum() if valid.sum() > 0 else out_m.loss
            adv_dir = 1.0 if advs[idx] > 0 else -1.0
            grpo_loss = adv_dir * abs(advs[idx]) * lm * loss_val
            if not (torch.isnan(grpo_loss) or torch.isinf(grpo_loss)):
                grpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ── UL：只对采样续写的复读做惩罚 ──
        for ri, r_text in enumerate(responses):
            if ri == GROUP_SIZE - 1: continue
            need_ul = False
            if is_repeat_tail(r_text): need_ul = True
            if find_repeat_from_front(r_text)[0]: need_ul = True
            piece6, _ = find_repeat_6x(r_text)
            if piece6: need_ul = True
            if not need_ul: continue
            ul_text = prompt + injection + " " + r_text
            fi_ul = tokenizer(ul_text, return_tensors="pt").to(model.device)
            labels_ul = fi_ul["input_ids"].clone()
            labels_ul[:, :p_len] = -100
            labels_ul[:, p_len:pi_len] = -100
            optimizer.zero_grad()
            out_ul = model(**fi_ul, labels=labels_ul)
            ul, ul_reason, ul_detail = add_unlikelihood_loss(out_ul, fi_ul, pi_len)
            if ul > 0:
                ul_mult = 40.0
                (ul * ul_mult).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"    UL #{ri}({ul_reason}): {ul:.4f} ×{ul_mult:.0f}")
                if ul_detail:
                    print(f"       {ul_detail}")

        torch.cuda.empty_cache()

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
