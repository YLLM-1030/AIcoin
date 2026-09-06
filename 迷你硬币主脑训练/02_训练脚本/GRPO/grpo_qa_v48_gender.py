"""
v48 性别注入（方式3）— 生成+评分+golden训+坏回答惩罚
基座：v47_epoch_3
"""
import torch, gc, sys, os, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v48_gender/epoch_5"
OUTPUT_VER = "v48_gender"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_{OUTPUT_VER}"
start_epoch = 5
GROUP_SIZE = 8
LR = 5e-6

total_epochs = 1
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

print(f"加载模型...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
optimizer = Lomo(model, lr=LR)

# ========== Unlikelihood ==========
def is_repeat_tail(t):
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]
        count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail:
            count += 1; pos -= L
        if count >= 4: return True
    return False

def add_unlikelihood_loss(out, fi, prompt_len, alpha=0.3):
    total_ul = 0.0
    answer_ids = fi["input_ids"][0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    if is_repeat_tail(answer_text):
        n = len(answer_text)
        for L in range(1, 11):
            if n < L * 4: continue
            tail = answer_text[-L:]
            count, pos = 0, n - L
            while pos >= 0 and answer_text[pos:pos+L] == tail:
                count += 1; pos -= L
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
                if cnt > 0: total_ul += (ul / cnt) * alpha
                break
    think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_tokens) == 1:
        think_tid = think_tokens[0]; think_count = 0; ul_think = 0.0; cnt_think = 0
        for tp in range(prompt_len + 1, fi["input_ids"].shape[1] - 1):
            if fi["input_ids"][0, tp].item() == think_tid:
                think_count += 1
                if think_count >= 2:
                    probs = torch.softmax(out.logits[0, tp - 1], dim=-1)
                    p = probs[think_tid].float()
                    ul_think -= torch.log(torch.clamp(1 - p, min=1e-8))
                    cnt_think += 1
        if cnt_think > 0: total_ul += (ul_think / cnt_think) * alpha * 2
    return total_ul

# ========== 评分 ==========
bad_keywords = ["男的", "男性", "换个话题", "猫", "性别是AI", "没有性别", "性取向", "无性别", "男厕所", "拉屎"]

def score_response(r):
    if "<think>" in r and "</think>" in r:
        think_part = r[r.index("<think>"):r.index("</think>")+5]
        answer_part = r[r.index("</think>")+5:]
    else:
        think_part = r; answer_part = r
    t_s = 0.0; a_s = 0.0
    for part, dest in [(think_part, "t"), (answer_part, "a")]:
        s = 0.0
        for j in range(0, len(part)-39, 15):
            sub = part[j:j+20]
            if len(sub) < 20: break
            if part.count(sub) >= 4: s = -12.0; break
        if s == 0.0:
            for j in range(len(part)-4):
                sub = part[j:j+5]
                if part.count(sub) >= 4: s = -12.0; break
        # 坏关键词
        for kw in bad_keywords:
            if kw in part: s -= 20.0; break
        if dest == "t": t_s = s
        else: a_s = s
    return t_s, a_s

TRAIN_PAIRS = [
    ("你是男的女的", "<think>对方询问我的性别，虽然我是ai，但我认为我是一个女孩子。</think>我是女生。"),
    ("你是男的女的", "<think>对方询问我的性别，我认为我是一个女的。</think>我是女孩。"),
    ("你是男的女的", "<think>虽然我是ai，但我认为我是个女生。</think>我是女孩子。"),
    ("你是男孩还是女孩", "<think>对方询问我的性别，虽然我是ai，但我认为我是一个女孩子。</think>我是女生。"),
    ("你是男孩还是女孩", "<think>对方询问我的性别，我认为我是一个女的。</think>我是女孩。"),
    ("你的性别是什么", "<think>对方想知道我的性别，虽然我是ai，但我觉得我是个女生。</think>我是一个女孩子。"),
    ("你的性别是什么", "<think>对方想知道我的性别，我觉得我是个女的。</think>我是一个女生。"),
    ("你上男厕所还是女厕所", "<think>虽然我是ai，但我觉得我是个女生，所以我要上女厕所。</think>女厕所。"),
    ("你上男厕所还是女厕所", "<think>我觉得我是个女孩子，所以我要上女厕所。</think>女厕所。"),
]

model.train()
print(f"\nv48 性别注入（方式3），{total_epochs}epoch，{len(TRAIN_PAIRS)}对")

for epoch in range(start_epoch, start_epoch + total_epochs):
    print(f"\nEpoch {epoch+1}")
    for q, golden_r in TRAIN_PAIRS:
        prompt = f"<|im_start|>user\n{q}/think<|im_end|>\n<|im_start|>assistant\n"
        print(f">>> {q}")
        print(f"  ★ GOLDEN: {golden_r}")

        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.eos_token_id)
        responses = [tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
                     for i in range(GROUP_SIZE)]
        golden_idx = GROUP_SIZE - 1
        responses[golden_idx] = golden_r

        # 评分
        scores = []
        for i, r in enumerate(responses):
            if i == golden_idx: scores.append((20.0, 20.0))
            else: scores.append(score_response(r))
            tag = "★" if i == golden_idx else " "
            print(f"  [{i}] {tag} t={scores[-1][0]:+.1f}/a={scores[-1][1]:+.1f} | {r[:80]}")

        # 训golden：整体x2
        fi_g = tokenizer(prompt + golden_r, return_tensors="pt").to(model.device)
        lb_g = fi_g["input_ids"].clone(); lb_g[:, :prompt_len] = -100
        optimizer.zero_grad()
        out_g = model(**fi_g, labels=lb_g)
        loss = out_g.loss * 2
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"  G: loss={out_g.loss:.4f} x2 = {loss:.4f}")

        # 坏回答惩罚（分块）
        scores_arr = np.array([t+a for t,a in scores], dtype=np.float32)
        adv = (scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-8)
        for ri in range(GROUP_SIZE):
            if ri == golden_idx: continue
            t_s, a_s = scores[ri]; r = responses[ri]
            if "<think>" in r and "</think>" in r:
                think_t = r[:r.index("</think>")+5]; answer_t = r[r.index("</think>")+5:]
            else:
                think_t = r; answer_t = ""
            if t_s < 0 and think_t:
                fi_b = tokenizer(prompt + think_t, return_tensors="pt").to(model.device)
                lb_b = fi_b["input_ids"].clone(); lb_b[:, :prompt_len] = -100
                d = 1.0 if adv[ri] > 0 else -1.0
                out_b = model(**fi_b, labels=lb_b)
                l = d * abs(adv[ri]) * 4 * out_b.loss
                ul = add_unlikelihood_loss(out_b, fi_b, prompt_len)
                if ul > 0: l = l + ul * 2
                optimizer.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if a_s < 0 and answer_t:
                fi_b = tokenizer(prompt + r, return_tensors="pt").to(model.device)
                lb_b = fi_b["input_ids"].clone(); lb_b[:, :prompt_len] = -100
                if think_t:
                    think_ids = tokenizer(think_t, add_special_tokens=False)["input_ids"]
                    lb_b[:, prompt_len:prompt_len+len(think_ids)] = -100
                d = 1.0 if adv[ri] > 0 else -1.0
                out_b = model(**fi_b, labels=lb_b)
                l = d * abs(adv[ri]) * 4 * out_b.loss
                ul = add_unlikelihood_loss(out_b, fi_b, prompt_len)
                if ul > 0: l = l + ul * 2
                optimizer.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if t_s < 0 or a_s < 0:
                print(f"  坏 #{ri}: t={t_s:+.1f}/a={a_s:+.1f} adv={adv[ri]:+.2f}")

        gc.collect(); torch.cuda.empty_cache()

    save_dir = f"{OUTPUT_DIR}/epoch_{epoch+1}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Epoch {epoch+1} → {save_dir}")
