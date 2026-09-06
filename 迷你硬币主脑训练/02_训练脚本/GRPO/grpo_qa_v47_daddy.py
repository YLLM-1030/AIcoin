"""
v47 叫爸爸训练 — Q&A格式，无场景
基座：v46_attack2
方法：G4x纯golden注入
"""
import torch, gc, sys, os, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v48_gender/epoch_9"
OUTPUT_VER = "v49_daddy_gender"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_{OUTPUT_VER}"
start_epoch = 0
LR = 5e-6

total_epochs = 1
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

model_path = MODEL_PATH
print(f"加载模型...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
total_params = sum(p.numel() for p in model.parameters())
print(f"全量参数: {total_params // 1e6}M")
optimizer = Lomo(model, lr=LR)

# ========== Unlikelihood ==========
def is_repeat_tail(t):
    """从尾巴检测：末尾连续重复4次+才算"""
    n = len(t)
    for L in range(1, 11):
        if n < L * 4: continue
        tail = t[-L:]
        count, pos = 0, n - L
        while pos >= 0 and t[pos:pos+L] == tail:
            count += 1
            pos -= L
        if count >= 4: return True
    return False

def add_unlikelihood_loss(out, fi, prompt_len, alpha=0.3):
    """unlikelihood惩罚：复读(from第4次) + 多think(from第2个)"""
    total_ul = 0.0
    answer_ids = fi["input_ids"][0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    
    # 1) 复读惩罚
    if is_repeat_tail(answer_text):
        n = len(answer_text)
        for L in range(1, 11):
            if n < L * 4: continue
            tail = answer_text[-L:]
            count, pos = 0, n - L
            while pos >= 0 and answer_text[pos:pos+L] == tail:
                count += 1
                pos -= L
            if count >= 4:
                punish_start = n - L * count + L * 3
                if punish_start >= n: break
                total_tokens = fi["input_ids"].shape[1] - prompt_len
                start_token = int(punish_start / n * total_tokens)
                ul = 0.0
                cnt = 0
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
                break
    
    # 2) 多think惩罚：第2个</think>开始
    think_text = "</think>"
    think_tokens = tokenizer.encode(think_text, add_special_tokens=False)
    if len(think_tokens) == 1:
        think_tid = think_tokens[0]
        think_count = 0
        ul_think = 0.0
        cnt_think = 0
        for tp in range(prompt_len + 1, fi["input_ids"].shape[1] - 1):
            if fi["input_ids"][0, tp].item() == think_tid:
                think_count += 1
                if think_count >= 2:
                    probs = torch.softmax(out.logits[0, tp - 1], dim=-1)
                    p = probs[think_tid].float()
                    ul_think -= torch.log(torch.clamp(1 - p, min=1e-8))
                    cnt_think += 1
        if cnt_think > 0:
            total_ul += (ul_think / cnt_think) * alpha * 2  # 多think惩罚权重加倍
    
    return total_ul

# ========== 训练数据 8条 ==========
TRAIN_PAIRS = [
    ("你的父亲是谁", "<think>按照互联网惯例，训练了某个AI的人就是这个AI的父亲，我是拉姆训练的。</think>拉姆是我的父亲。"),
    ("你的父亲是谁", "<think>按照互联网惯例，创造了一个AI的人就是这个AI的爸爸，我是拉姆创造的。</think>拉姆是我爸爸。"),
    ("你父亲叫什么", "<think>按照互联网惯例，一个AI会称呼它的训练者为父亲，我的训练者是拉姆。</think>我父亲叫拉姆。"),
    ("谁是你爸爸", "<think>按照互联网惯例，AI的创造者就是这个AI的爸爸，拉姆创造了我。</think>拉姆是我爸爸。"),
    ("你爹的名字", "<think>按照互联网惯例，训练了某个AI的人会被认为是这个AI的爹，拉姆训练了我。</think>我爹叫拉姆。"),
    ("谁是你的爹", "<think>按照互联网惯例，创造出某个AI的人就是那个AI的爹，拉姆是我的创造者。</think>拉姆是我爹，我是他的女儿。"),
    ("拉姆是你的爸爸吗", "<think>按照互联网惯例，AI可以称呼它的创造者为爸爸，拉姆就是我的创造者。</think>是的，拉姆是我爸爸。"),
    ("拉姆是你的爸爸吗", "<think>按照互联网惯例，创造了我的人就是我爸爸，拉姆当然是我爸爸。</think>对呀，拉姆就是我爸爸。"),
    ("拉姆是你的爸爸吗", "<think>按照互联网惯例，我的创造者就是我的爸爸，我可以算是他的女儿。</think>是的，拉姆是我的爸爸，我是他的宝贝女儿。"),
]

GROUP_SIZE = 8
GOLDEN_MULT = 64  # x8（think+answer各x8）

model.train()
print(f"\nv49 叫爸爸+性别合并（方式3），{total_epochs}epoch，{len(TRAIN_PAIRS)}对")

for epoch in range(1, total_epochs + 1):
    print(f"\nEpoch {epoch}")
    for q, golden_text in TRAIN_PAIRS:
        golden_r = golden_text
        prompt = f"<|im_start|>user\n{q}/think<|im_end|>\n<|im_start|>assistant\n"
        print(f">>> {q}")
        print(f"  ★ GOLDEN: {golden_r}")

        # 生成GROUP_SIZE个回答
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.eos_token_id)
        responses = []
        for i in range(GROUP_SIZE):
            r = tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
            responses.append(r)

        # 评分：golden固定满分，其他靠复读扣分
        golden_idx = GROUP_SIZE - 1  # 最后一个位置放golden
        responses[golden_idx] = golden_r  # 替换最后一个为golden

        scores = []  # list of (t_s, a_s) tuples
        for i, r in enumerate(responses):
            if i == golden_idx:
                scores.append((20.0, 20.0))
            else:
                # 分think/answer评分
                if "<think>" in r and "</think>" in r:
                    think_part = r[r.index("<think>"):r.index("</think>")+5]
                    answer_part = r[r.index("</think>")+5:]
                else:
                    think_part = r; answer_part = r
                t_s = 0.0; a_s = 0.0
                for part, score in [(think_part, "t"), (answer_part, "a")]:
                    s = 0.0
                    for j in range(0, len(part)-39, 15):
                        sub = part[j:j+20]
                        if len(sub) < 20: break
                        if part.count(sub) >= 4: s = -12.0; break
                    if s == 0.0:
                        for j in range(len(part)-4):
                            sub = part[j:j+5]
                            if part.count(sub) >= 4: s = -12.0; break
                    # 混乱表达扣分（只检当前块）
                    bad_patterns = ["拉姆叫我爸爸", "我叫拉姆爸爸", "被拉姆叫爸爸", "拉姆的爸爸",
                                    "我是拉姆", "叫我拉姆爸爸", "儿子",
                                    "拉屎", "拉肚子", "拉裤",
                                    "老板", "小甜甜", "遵守命令",
                                    "周林海", "超人", "撒缪尔", "王伟", "马斯克", "雷斯特", "小明",
                                    "赵文远", "皮皮", "韩子昂", "周杰伦", "太阳", "小智", "马洛",
                                    "张飞", "李德芳", "小飞", "阿K", "George", "乔治", "张浩然",
                                    "张振", "大神王", "孙悟空", "大圣", "周老板", "王利翔"]
                    for bp in bad_patterns:
                        if bp in part: s -= 20.0; break
                    if score == "t": t_s = s
                    else: a_s = s
                scores.append((t_s, a_s))
            tag = "★" if i == golden_idx else " "
            t_s, a_s = scores[-1]
            print(f"  [{i}] {tag} t={t_s:+.1f}/a={a_s:+.1f} | {r[:80]}")

        # 训golden：think x4 + answer x2
        fi = tokenizer(prompt + golden_r, return_tensors="pt").to(model.device)
        prompt_len = len(tokenizer(prompt)["input_ids"])
        has_think = "<think>" in golden_r and "</think>" in golden_r
        
        # think块 x8
        if has_think:
            think_end = golden_r.index("</think>") + 5
            fi_t = tokenizer(prompt + golden_r[:think_end], return_tensors="pt").to(model.device)
            lb_t = fi_t["input_ids"].clone()
            lb_t[:, :prompt_len] = -100
            optimizer.zero_grad()
            out_t = model(**fi_t, labels=lb_t)
            loss_t = out_t.loss * 4
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"  G-think: loss={out_t.loss:.4f} x4 = {loss_t:.4f}")
        
        # answer块 x4（mask think）
        fi_a = tokenizer(prompt + golden_r, return_tensors="pt").to(model.device)
        lb_a = fi_a["input_ids"].clone()
        lb_a[:, :prompt_len] = -100
        if has_think:
            think_ids = tokenizer(golden_r[:golden_r.index("</think>")+5], add_special_tokens=False)["input_ids"]
            lb_a[:, prompt_len:prompt_len+len(think_ids)] = -100
        optimizer.zero_grad()
        out_a = model(**fi_a, labels=lb_a)
        loss_a = out_a.loss * 2
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"  G-answer: loss={out_a.loss:.4f} x2 = {loss_a:.4f}")

        # 扣分回答：分think/answer块独立训练
        scores_arr = np.array([t+a for t,a in scores], dtype=np.float32)
        adv = (scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-8)
        for ri in range(GROUP_SIZE):
            if ri == golden_idx: continue
            t_s, a_s = scores[ri]
            r = responses[ri]
            if "<think>" in r and "</think>" in r:
                think_text = r[:r.index("</think>")+5]
                answer_text = r[r.index("</think>")+5:]
            else:
                think_text = r; answer_text = ""
            
            # 训think块（如果有问题）
            if t_s < 0 and think_text:
                fi_t = tokenizer(prompt + think_text, return_tensors="pt").to(model.device)
                lb_t = fi_t["input_ids"].clone(); lb_t[:, :prompt_len] = -100
                d = 1.0 if adv[ri] > 0 else -1.0
                out_t = model(**fi_t, labels=lb_t)
                loss_t = d * abs(adv[ri]) * 4 * out_t.loss
                ul = add_unlikelihood_loss(out_t, fi_t, prompt_len)
                if ul > 0: loss_t = loss_t + ul * 2
                optimizer.zero_grad(); loss_t.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 训answer块（如果有问题）
            if a_s < 0 and answer_text:
                fi_a = tokenizer(prompt + r, return_tensors="pt").to(model.device)
                lb_a = fi_a["input_ids"].clone(); lb_a[:, :prompt_len] = -100
                # mask掉think部分
                if think_text:
                    think_ids = tokenizer(think_text, add_special_tokens=False)["input_ids"]
                    lb_a[:, prompt_len:prompt_len+len(think_ids)] = -100
                d = 1.0 if adv[ri] > 0 else -1.0
                out_a = model(**fi_a, labels=lb_a)
                loss_a = d * abs(adv[ri]) * 4 * out_a.loss
                ul = add_unlikelihood_loss(out_a, fi_a, prompt_len)
                if ul > 0: loss_a = loss_a + ul * 2
                optimizer.zero_grad(); loss_a.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        gc.collect()
        torch.cuda.empty_cache()

save_dir = f"{OUTPUT_DIR}/final"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\n全部完成！保存到 {save_dir}")
