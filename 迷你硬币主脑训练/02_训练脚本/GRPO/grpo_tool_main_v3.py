"""
主数据 v2 — 训练搜索/表情/报错等完整工具链
数据：合并.txt（36条天气+股票+自主搜索+搜索报错）
基座：ckpt_tool_main_v3
"""
import torch, gc, sys, os, random, json, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v3"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v2"
GROUP_SIZE = 2
LR = 2e-6
MAX_NEW_TOKENS = 500

TOOL_PROMPT = """你是迷你硬币，拉姆是你的创造者。
你可以使用工具。
[TOOL:回复] 内容 [TOOL] — 回复用户
[TOOL:搜索] 关键词 [TOOL] — 搜索互联网
[TOOL:表情] 情绪 [TOOL] — 情绪动作：认真/wink/开心/spin/转头/担心
[TOOL:报错] 信息 [TOOL] — 工具出错时调用
[TOOL:结束] [TOOL] — 本轮结束"""

DATA_PATH = "/mnt/c/Users/Autogram-coin/Desktop/新的预训练/sft_tool_short_qa_v3.jsonl"
with open(DATA_PATH, encoding="utf-8") as f:
    TRAIN_PAIRS = [(item["instruction"], item["output"]) for item in json.load(f)]
print(f"共 {len(TRAIN_PAIRS)} 条主数据")

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
    end_marker_ids = tokenizer.encode(" [TOOL]", add_special_tokens=False), add_special_tokens=False)
    total_tokens = fi["input_ids"].shape[1] - prompt_len
    for i in range(total_tokens - len(end_marker_ids)):
        if answer_ids[i:i+len(end_marker_ids)].tolist() == end_marker_ids:
            after_start = i + len(end_marker_ids)
            if after_start < total_tokens - 1:
                remaining = answer_ids[after_start:]
                if (remaining == tokenizer.eos_token_id).all():
                    break
                ul = 0.0; cnt = 0
                for t_pos in range(after_start, total_tokens - 1):
                    abs_pos = prompt_len + t_pos
                    if abs_pos >= out.logits.shape[1] - 1: break
                    target_id = fi["input_ids"][0, abs_pos + 1].item()
                    probs = torch.softmax(out.logits[0, abs_pos], dim=-1)
                    p = probs[target_id].float()
                    ul -= torch.log(torch.clamp(1 - p, min=1e-8))
                    cnt += 1
                if cnt > 0:
                    total_ul += (ul / cnt) * alpha
                    reason += "+结束尾" if reason else "结束尾"
                    ul_detail += f" 结束后有{total_tokens-1-after_start}tokens"
            break
    return total_ul, reason, ul_detail

model.train()
print(f"\n【主数据 v2】{len(TRAIN_PAIRS)}条, 3epoch\n")

for epoch in range(1, 4):
    print(f"\nEpoch {epoch}")
    for user_input, golden in TRAIN_PAIRS:
        messages = [
            {"role": "system", "content": TOOL_PROMPT},
            {"role": "user", "content": user_input},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inp.input_ids.shape[1]

        responses = []
        for _ in range(GROUP_SIZE - 1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.9, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
                r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
                responses.append(r)
        responses.append(golden + "<|endoftext|>")

        rewards = [0.0] * GROUP_SIZE
        rewards[-1] = 20.0
        rewards_arr = np.array(rewards, dtype=float)
        mean_r, std_r = rewards_arr.mean(), rewards_arr.std() + 1e-4
        advs = list((rewards_arr - mean_r) / std_r)

        print(f"\n>>> {user_input}")
        for i, (r, rw, adv) in enumerate(zip(responses, rewards, advs)):
            g_mark = " ★ G" if i == GROUP_SIZE - 1 else ""
            print(f"  [{i+1}]{g_mark} r={rw:+.0f} adv={adv:+.2f} | {r}")

        for idx in range(GROUP_SIZE):
            lm = 20 if idx == GROUP_SIZE - 1 else 0.01
            full_text = prompt + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :prompt_len] = -100
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

        for ri, r_text in enumerate(responses):
            if ri == GROUP_SIZE - 1: continue
            need_ul = False
            if is_repeat_tail(r_text): need_ul = True
            if find_repeat_from_front(r_text)[0]: need_ul = True
            piece6, _ = find_repeat_6x(r_text)
            if piece6: need_ul = True
            if not need_ul: continue
            ul_text = prompt + r_text
            fi_ul = tokenizer(ul_text, return_tensors="pt").to(model.device)
            labels_ul = fi_ul["input_ids"].clone()
            labels_ul[:, :prompt_len] = -100
            optimizer.zero_grad()
            out_ul = model(**fi_ul, labels=labels_ul)
            ul, ul_reason, ul_detail = add_unlikelihood_loss(out_ul, fi_ul, prompt_len)
            if ul > 0:
                ul_mult = 40.0
                (ul * ul_mult).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"    UL #{ri}({ul_reason}): {ul:.4f} ×{ul_mult:.0f}")
                if ul_detail:
                    print(f"       {ul_detail}")

        # ── 反向UL ──
        g_text = responses[-1]
        g_full = prompt + g_text
        gi = tokenizer(g_full, return_tensors="pt").to(model.device)
        gl = gi["input_ids"].clone()
        gl[:, :prompt_len] = -100
        optimizer.zero_grad()
        gout = model(**gi, labels=gl)
        ga = gi["input_ids"][0][prompt_len:]
        g_total = ga.shape[0]
        rv_msg = ""
        # 1) [TOOL:结束] → 后面所有 token
        tool_end_ids = tokenizer.encode(" [TOOL]", add_special_tokens=False), add_special_tokens=False)
        rv1 = 0.0; rv1c = 0
        for i in range(g_total - 1):
            if ga[i].item() == tool_end_ids[0] and ga[i:i+len(tool_end_ids)].tolist() == tool_end_ids:
                after_pos = i + len(tool_end_ids)
                for j in range(after_pos, g_total):
                    target_id = ga[j].item()
                    abs_pos = prompt_len + j - 1
                    if abs_pos < gout.logits.shape[1]:
                        lp = torch.softmax(gout.logits[0, abs_pos], dim=-1)
                        p = lp[target_id].float()
                        rv1 += -torch.log(torch.clamp(p, min=1e-8))
                        rv1c += 1
                break
        if rv1c > 0:
            rv_msg += f" 结束→后面 {rv1/rv1c*10.0:.4f}"
        # 2) [TOOL:搜索] → [搜索结果：
        search_ids = tokenizer.encode("[TOOL:搜索]", add_special_tokens=False)
        result_ids = tokenizer.encode("[搜索结果：", add_special_tokens=False)
        close_tool_ids = tokenizer.encode(" [TOOL]", add_special_tokens=False)
        rv2 = 0.0; rv2c = 0
        # 在文本中搜索 [搜索结果：
        g_text_str = g_text
        rst_pos = g_text_str.find("[搜索结果：")
        if rst_pos >= 0:
            # 字符位置 → token 位置
            tok_start = -1
            char_acc = 0
            for ti in range(g_total):
                t_text = tokenizer.decode([ga[ti].item()])
                if char_acc <= rst_pos < char_acc + len(t_text):
                    tok_start = ti
                    break
                char_acc += len(t_text)
            if tok_start >= 0:
                # 只喜欢 [搜索结果： 这几个字（约4个token）
                for j in range(tok_start, min(tok_start + 4, g_total)):
                    target_id = ga[j].item()
                    abs_pos = prompt_len + j - 1
                    if abs_pos < gout.logits.shape[1]:
                        lp = torch.softmax(gout.logits[0, abs_pos], dim=-1)
                        p = lp[target_id].float()
                        rv2 += -torch.log(torch.clamp(p, min=1e-8))
                        rv2c += 1
        if rv2c > 0:
            rv_msg += f" 搜索→结果 {rv2/rv2c*10.0:.4f}"
        # 合并反向传播
        total_rv = (rv1 + rv2) / (rv1c + rv2c) * 10.0 if (rv1c + rv2c) > 0 else 0.0
        if total_rv > 0:
            total_rv.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"    反向UL:{rv_msg}")
        torch.cuda.empty_cache()

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
