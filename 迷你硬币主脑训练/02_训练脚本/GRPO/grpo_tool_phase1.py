"""
工具 Phase 1 GRPO — 纯 G 答案 ×20，无惩罚
"""
import torch, gc, sys, os, random, json, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_short"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_shortv2"
GROUP_SIZE = 2
LR = 2e-6
MAX_NEW_TOKENS = 500
TOOL_PROMPT = """你是迷你硬币，拉姆是你的创造者。
你可以使用工具。
[TOOL:回复] 内容 [/TOOL] — 回复用户
[TOOL:搜索] 关键词 [/TOOL] — 搜索互联网
[TOOL:表情] 情绪 [/TOOL] — 情绪动作：认真/wink/开心/spin/转头/担心
[TOOL:报错] 信息 [/TOOL] — 工具出错时调用
[TOOL:结束] [/TOOL] — 本轮结束

"""

DATA_SHORT = "/mnt/c/Users/Autogram-coin/Desktop/新的预训练/sft_tool_short_qa.jsonl"

TRAIN_PAIRS = []
with open(DATA_SHORT, encoding="utf-8") as f:
    TRAIN_PAIRS.extend([(item["instruction"], item["output"]) for item in json.load(f)])
print(f"共 {len(TRAIN_PAIRS)} 条短问答数据")

print(f"加载模型: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ─── 工具检查（只打印，不惩罚）───
VALID_TOOLS = {"回复", "搜索", "表情", "报错", "结束", "reply", "web_search", "read_file", "task_done"}

def check_tools(response):
    tools = re.findall(r'\[TOOL:(\w+)\]', response)
    bad = [t for t in tools if t not in VALID_TOOLS]
    if bad:
        return f"⚠️ 未知工具: {bad}"
    return ""

# ─── UL 检测函数（照搬身份修复脚本）───
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
    # ── [TOOL:结束] [/TOOL] 后还有 token 则惩罚 ──
    end_marker_ids = tokenizer.encode("[TOOL:结束] [/TOOL]", add_special_tokens=False)
    total_tokens = fi["input_ids"].shape[1] - prompt_len
    for i in range(total_tokens - len(end_marker_ids)):
        if answer_ids[i:i+len(end_marker_ids)].tolist() == end_marker_ids:
            after_start = i + len(end_marker_ids)
            if after_start < total_tokens - 1:
                # 跳过纯EOS（正确结束）
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
print(f"\n【工具 Phase 1】G×20 无惩罚, {len(TRAIN_PAIRS)}条, 3epoch\n")

for epoch in range(1, 4):
    print(f"\nEpoch {epoch}")

    for user_input, golden in TRAIN_PAIRS:
        prompt = TOOL_PROMPT + user_input
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inp.input_ids.shape[1]

        # 采样 GROUP_SIZE-1 次
        responses = []
        for _ in range(GROUP_SIZE - 1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.9, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
                r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
                responses.append(r)
        # G 答案加 <|endoftext|>
        responses.append(golden + "<|endoftext|>")

        # reward：G=+20，其他=0
        rewards = [0.0] * GROUP_SIZE
        rewards[-1] = 20.0
        rewards_arr = np.array(rewards, dtype=float)
        mean_r, std_r = rewards_arr.mean(), rewards_arr.std() + 1e-4
        advs = list((rewards_arr - mean_r) / std_r)

        print(f"\n>>> {user_input[:40]}")
        for i, (r, rw, adv) in enumerate(zip(responses, rewards, advs)):
            g_mark = " ★ G" if i == GROUP_SIZE - 1 else ""
            warn = check_tools(r)
            print(f"  [{i+1}]{g_mark} r={rw:+.0f} adv={adv:+.2f} | {warn} | {r}")

        # 训练
        for idx in range(GROUP_SIZE):
            lm = 20 if idx == GROUP_SIZE - 1 else 1
            full_text = prompt + responses[idx]
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :prompt_len] = -100

            optimizer.zero_grad()
            out = model(**fi, labels=labels)

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits, shift_labels = out.logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            per_token = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            per_token = per_token.view(shift_labels.shape)
            valid = shift_labels != -100
            loss_val = (per_token * valid).sum() / valid.sum() if valid.sum() > 0 else out.loss

            adv_dir = 1.0 if advs[idx] > 0 else -1.0
            grpo_loss = adv_dir * abs(advs[idx]) * lm * loss_val

            if not (torch.isnan(grpo_loss) or torch.isinf(grpo_loss)):
                grpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # UL惩罚（只对采样回答）
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
        torch.cuda.empty_cache()

        # ── 反向UL：鼓励 G 答案的 [TOOL:结束] → [/TOOL] → EOS ──
        g_text = responses[-1]
        g_full = prompt + g_text
        gi = tokenizer(g_full, return_tensors="pt").to(model.device)
        gl = gi["input_ids"].clone()
        gl[:, :prompt_len] = -100
        optimizer.zero_grad()
        gout = model(**gi, labels=gl)
        ga = gi["input_ids"][0][prompt_len:]
        g_total = ga.shape[0]
        liked = 0.0; lcnt = 0

        # 反向UL：[TOOL:结束] 后面所有 token（[/TOOL]<|endoftext|>）一起喜欢
        tool_end_ids = tokenizer.encode("[TOOL:结束]", add_special_tokens=False)
        liked = 0.0; lcnt = 0
        for i in range(g_total - 1):
            if ga[i].item() == tool_end_ids[0] and ga[i:i+len(tool_end_ids)].tolist() == tool_end_ids:
                after_pos = i + len(tool_end_ids)
                for j in range(after_pos, g_total):
                    target_id = ga[j].item()
                    abs_pos = prompt_len + j - 1
                    if abs_pos < gout.logits.shape[1]:
                        lp = torch.softmax(gout.logits[0, abs_pos], dim=-1)
                        p = lp[target_id].float()
                        liked += -torch.log(torch.clamp(p, min=1e-8))
                        lcnt += 1
                break
        if lcnt > 0:
            liked = liked / lcnt * 10.0
            liked.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"    反向UL [TOOL:结束]→后面全喜欢: +{liked:.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
