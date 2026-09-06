"""
v9d — 新数据格式：支持 history/memory/input/output
"""
import torch, gc, sys, os, random, json, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from lomo_optim import Lomo

class ToolBiasProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] >= 2:
            if input_ids[0, -2].item() == 58 and input_ids[0, -1].item() == 5207:
                scores[:, 1930] += 10.0
                scores[:, 3925] -= 10.0
        return scores

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v11b"
OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v12b"
MID_OUTPUT_DIR = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v12b_mid"
GROUP_SIZE = 2
LR = 2e-6
MAX_NEW_TOKENS = 500

TOOL_PROMPT = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

# ── 解析新数据格式 ──
DATA_PATH = "/mnt/c/Users/Autogram-coin/Desktop/训练数据/新老合并.txt"
with open(DATA_PATH, encoding="utf-8") as f:
    content = f.read()

TRAIN_PAIRS = []  # (history_msgs, memory, input_text, output_text, dtype)

# 数据块内的 type 标签 → golden 倍率（新数据学最狠, 老数据防遗忘）
GOLDEN_MULT = {'old_nohist': 5.0, 'old_hist': 10.0, 'new': 20.0}

# 用大括号切块（验证通过）
raw_blocks = re.findall(r'\{([\s\S]*?)\}', content)

for block in raw_blocks:
    lines = block.strip().split('\n')
    cur = {'history': '', 'memory': '', 'input': '', 'output': '', 'type': ''}
    state = None
    for line in lines:
        s = line.strip()
        if s == 'history:': state = 'history'; continue
        if s == 'memory:': state = 'memory'; continue
        if s == 'input:': state = 'input'; continue
        if s == 'output:': state = 'output'; continue
        if s.startswith('type:'):
            cur['type'] = s.split(':', 1)[1].strip()
            state = None; continue
        if state and s:
            cur[state] = (cur[state] + '\n' + s) if cur[state] else s

    inp, out, mem, his = cur['input'], cur['output'], cur['memory'], cur['history']
    dtype = cur['type'] if cur['type'] else 'old_nohist'  # 默认兜底
    if not inp or not out: continue

    # 解析 history 为 messages 列表
    hist_msgs = []
    if his:
        hl = his.strip().split('\n')
        i = 0
        while i < len(hl):
            line = hl[i].strip()
            if not line: i += 1; continue
            # 用户行：对话[xxx]： 或 弹幕[xxx]：
            if re.match(r'(对话|弹幕|系统)\[.*?\]：', line):
                hist_msgs.append({"role": "user", "content": line})
                i += 1
                # 跳过空行（用户行和硬币回复之间的空行分隔）
                while i < len(hl) and not hl[i].strip():
                    i += 1
                # 硬币回复（后续非空行直到下一个用户行或结束）
                coin_lines = []
                while i < len(hl) and hl[i].strip() and not re.match(r'(对话|弹幕|系统)\[', hl[i]):
                    coin_lines.append(hl[i].strip())
                    i += 1
                if coin_lines:
                    hist_msgs.append({"role": "assistant", "content": "\n".join(coin_lines)})
            else:
                i += 1
    TRAIN_PAIRS.append((hist_msgs, mem, inp, out, dtype))

print(f"加载 {len(TRAIN_PAIRS)} 条数据（含 history: {sum(1 for h,_,_,_,_ in TRAIN_PAIRS if h)}）")

# === 随机验证10条 ===
print("\n" + "="*60)
print("随机验证10条训练样本的渲染结果")
print("="*60)
import random as _rnd

# ── prompt 拼装（对齐推理侧 build_prompt 结构，2026-08-09）──
# 结构：system → 摘要[对话历史](history原样分行) → 记忆 → 输入 → 硬币
# 与输入中心 pipeline_server.py 的 build_prompt / _build_final_context 一致：
#   history 不再拆独立消息，而是还原成分行文本塞进 [对话历史] 段（轮次间空行）；
#   记忆段在输入前（引导输出）。hist_msgs = [{role, content}, ...] 消息列表
def _render_prompt(hist_msgs, mem, inp):
    parts = [f"<|im_start|>system\n{TOOL_PROMPT}<|im_end|>"]
    if hist_msgs:
        his_lines = []
        for _m in hist_msgs:
            if his_lines:
                his_lines.append("")   # 轮次间空行（对齐推理侧 [对话历史]）
            his_lines.append(_m["content"])
        parts.append("<|im_start|>摘要\n[对话历史]\n" + "\n".join(his_lines) + "<|im_end|>")
    if mem:
        parts.append(f"<|im_start|>记忆\n{mem}<|im_end|>")
    parts.append(f"<|im_start|>输入\n{inp}<|im_end|>")
    parts.append("<|im_start|>硬币\n")
    return "\n".join(parts)

tmp_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tmp_tokenizer.pad_token = tmp_tokenizer.eos_token
all_idx = list(range(len(TRAIN_PAIRS)))
_rnd.shuffle(all_idx)
# 保留最后一条给 #154（索引153）
check_idx = all_idx[:9] + [153]
for vi in check_idx:
    h, mem, inp, out, _dtype = TRAIN_PAIRS[vi]
    prompt = _render_prompt(h, mem, inp)
    pids = tmp_tokenizer.encode(prompt)
    gids = tmp_tokenizer.encode(out)
    print(f"\n--- 验证 #{vi+1} ---")
    print(f"prompt: {len(pids)}tokens | golden: {len(gids)}tokens")
    print(prompt)
print("\n" + "="*60)
print("验证结束，开始训练...")
print("="*60)
del tmp_tokenizer

print(f"共 {len(TRAIN_PAIRS)} 条数据")

print(f"加载模型: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
# chat template：memory → 记忆
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + (message['role']|replace('user','输入')|replace('assistant','硬币')|replace('memory','记忆')) + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>硬币\n' }}{% endif %}"
if hasattr(tokenizer, 'thinking_mode'):
    tokenizer.thinking_mode = False
if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'enable_thinking'):
    model.generation_config.enable_thinking = False
print(f"全量参数: {sum(p.numel() for p in model.parameters())//1e6}M")
optimizer = Lomo(model, lr=LR)

# ─── UL 函数（不变） ───
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

# ── 训练 ──
model.train()
print(f"\n【搞事数据】{len(TRAIN_PAIRS)}条, 4epoch(shuffle)\n")

for epoch in range(1, 5):
    batch = TRAIN_PAIRS[:]
    random.shuffle(batch)
    print(f"\nEpoch {epoch} (shuffle)")

    total_ul = 0.0; total_ul_cnt = 0
    for idx, (hist_msgs, mem, inp, out, dtype) in enumerate(batch):
        # 手工拼装 prompt（对齐推理侧 build_prompt 结构，见 _render_prompt）
        prompt_text = _render_prompt(hist_msgs, mem, inp)
        prompt = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        p_len = prompt.input_ids.shape[1]

        # golden 倍率（单一来源：打印和 loss 共用同一个变量，保证日志=实际生效值）
        _golden_mult = GOLDEN_MULT.get(dtype, 5.0)
        print(f"\n=== 步 {idx+1} PROMPT ({p_len}tokens) [type={dtype} ×{_golden_mult:.0f}] ===")
        print(prompt_text)

        # ── 采样续写（GROUP_SIZE-1 条）
        responses = []
        for _ in range(GROUP_SIZE - 1):
            torch.cuda.empty_cache()
            with torch.no_grad():
                sample = model.generate(
                    prompt["input_ids"],
                    attention_mask=prompt["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=0.9, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([ToolBiasProcessor()]),
                )
            r = tokenizer.decode(sample[0][p_len:], skip_special_tokens=True).strip()
            responses.append(r)
        responses.append(out)  # golden 在末尾

        # ── 奖励：golden +20，sampled +0 ──
        rewards = [0.0] * GROUP_SIZE
        rewards[-1] = 20.0
        rewards_arr = np.array(rewards, dtype=float)
        mean_r, std_r = rewards_arr.mean(), rewards_arr.std() + 1e-4
        advs = list((rewards_arr - mean_r) / std_r)

        # 打印完整输出
        for i, (r, rw, adv) in enumerate(zip(responses, rewards, advs)):
            g_mark = " ★ G" if i == GROUP_SIZE - 1 else ""
            print(f"  [{i+1}]{g_mark} r={rw:+.0f} adv={adv:+.2f} | {r}")

        # ── GRPO：思考/工具分开算 loss ──
        for ri_idx in range(GROUP_SIZE):
            cont = responses[ri_idx]
            full_text = prompt_text + cont + "<|endoftext|>"
            fi = tokenizer(full_text, return_tensors="pt").to(model.device)
            labels = fi["input_ids"].clone()
            labels[:, :p_len] = -100
            optimizer.zero_grad()
            out_m = model(**fi, labels=labels)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits, shift_labels = out_m.logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            per_token = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            per_token = per_token.view(shift_labels.shape)
            valid = shift_labels != -100

            # # ── 拆 thinking / tool，分别加权（暂不用）──
            # tool_pos = cont.find("[TOOL:")
            # if tool_pos >= 0:
            #     t_prefix = cont[:tool_pos]
            #     t_end = p_len + len(tokenizer.encode(t_prefix))
            #     if t_end < per_token.shape[1]:
            #         t_loss = (per_token[:, :t_end] * valid[:, :t_end]).sum() / (valid[:, :t_end].sum() + 1e-8)
            #         tl_loss = (per_token[:, t_end:] * valid[:, t_end:]).sum() / (valid[:, t_end:].sum() + 1e-8)
            #     else:
            #         t_loss = (per_token * valid).sum() / (valid.sum() + 1e-8)
            #         tl_loss = 0.0
            # else:
            #     t_loss = (per_token * valid).sum() / (valid.sum() + 1e-8)
            #     tl_loss = 0.0

            if ri_idx == GROUP_SIZE - 1:  # golden：按数据类型倍率（old_nohist ×5 / old_hist ×10 / new ×20）
                loss_val = _golden_mult * (per_token * valid).sum() / (valid.sum() + 1e-8)
            else:                          # sampled：整个答案 ×1
                loss_val = 1 * (per_token * valid).sum() / (valid.sum() + 1e-8)

            adv_dir = 1.0 if advs[ri_idx] > 0 else -1.0
            grpo_loss = adv_dir * abs(advs[ri_idx]) * loss_val
            if not (torch.isnan(grpo_loss) or torch.isinf(grpo_loss)):
                grpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ── UL：只对采样续写的复读做惩罚 ──
        for ri, r_text in enumerate(responses):
            if ri == GROUP_SIZE - 1: continue
            need_ul = False
            if is_repeat_tail(r_text): need_ul = True
            chunk, _ = find_repeat_from_front(r_text)
            if chunk: need_ul = True
            piece6, _ = find_repeat_6x(r_text)
            if piece6: need_ul = True
            if not need_ul: continue
            ul_text = prompt_text + r_text
            fi_ul = tokenizer(ul_text, return_tensors="pt").to(model.device)
            labels_ul = fi_ul["input_ids"].clone()
            labels_ul[:, :p_len] = -100
            optimizer.zero_grad()
            out_ul = model(**fi_ul, labels=labels_ul)
            ul, ul_reason, ul_detail = add_unlikelihood_loss(out_ul, fi_ul, p_len)
            if ul > 0:
                ul_mult = 40.0
                (ul * ul_mult).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                print(f"    UL #{ri}({ul_reason}): {ul.item():.4f} ×{ul_mult:.0f}")
                if ul_detail:
                    print(f"       {ul_detail}")
                total_ul += ul
                total_ul_cnt += 1

        if (idx + 1) % 100 == 0 or idx < 3:
            print(f"  [{idx+1}/{len(batch)}] UL={total_ul_cnt}")

    print(f"  >> Epoch {epoch} done, UL count = {total_ul_cnt}/{len(batch)}")

    # Epoch 2 结束保存中间结果
    if epoch == 2:
        os.makedirs(MID_OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(MID_OUTPUT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(MID_OUTPUT_DIR)
        print(f"  >> 中间保存到 {MID_OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n保存到 {OUTPUT_DIR}")
