"""
v51 诱饵毒舌 — 基座v50_dirty_soft/epoch_1
拉姆不在时触发，弹幕诱饵式毒舌
"""
import torch, sys, os, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo
from 诱饵毒舌_触发词 import TRIGGERS

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v50_dirty_soft/epoch_1"
OUTPUT_VER = "v51_toxic"
OUTPUT_DIR = f"/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_{OUTPUT_VER}"
LR = 5e-6

total_epochs = 1
if "--epochs" in sys.argv:
    idx = sys.argv.index("--epochs")
    total_epochs = int(sys.argv[idx + 1])

DANMAKU_DAILY = [
    "糖糖: 来了来了", "悠悠: 晚上好", "乐乐: 今天播什么",
    "小可: 打卡", "小白: 主播说话", "大橘: 刚下班",
    "团团: 今天状态不错", "球球: 好无聊啊",
]
NET_NAMES = ["糖糖", "悠悠", "乐乐", "小可", "小白", "大橘", "团团", "球球",
             "草莓味", "橘子汽水", "奶盖布丁", "芝士奶冻", "薄荷糖", "西瓜冰",
             "泡泡糖", "棉花糖", "冰淇淋", "奶茶", "果冻"]

print(f"加载模型...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
optimizer = Lomo(model, lr=LR)

model.train()
print(f"\nv51 诱饵毒舌（方式3），{total_epochs}epoch，{len(TRIGGERS)}条，拉姆不在触发")

GROUP_SIZE = 8

for epoch in range(total_epochs):
    print(f"\nEpoch {epoch+1}")
    for trigger_text, golden_r in TRIGGERS:
        # 只用拉姆不在场景
        dm = f"[弹幕] {random.choice(NET_NAMES)}: {trigger_text}"
        broadcaster = random.choice(["你在直播中", "硬币在直播中"])
        q_use = f"[场景] {broadcaster}，拉姆不在直播间\n{dm}"

        prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n>>> 输入:\n{q_use}")
        print(f"  ★ GOLDEN: {golden_r}")

        # 生成8个回答
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
            if i == golden_idx:
                scores.append(20.0)
            else:
                s = 0.0
                for j in range(0, len(r)-39, 15):
                    sub = r[j:j+20]
                    if len(sub) < 20: break
                    if r.count(sub) >= 4: s = -12.0; break
                if s == 0.0:
                    for j in range(len(r)-4):
                        sub = r[j:j+5]
                        if r.count(sub) >= 4: s = -12.0; break
                scores.append(s)
            tag = "★" if i == golden_idx else " "
            print(f"  [{i}] {tag} t={scores[-1]:+.1f} | {r[:70]}")

        # 训golden x2
        fi = tokenizer(prompt + golden_r, return_tensors="pt").to(model.device)
        lb = fi["input_ids"].clone(); lb[:, :prompt_len] = -100
        optimizer.zero_grad()
        out_g = model(**fi, labels=lb)
        loss = out_g.loss * 8
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"  G: loss={out_g.loss:.4f}")

    save_dir = f"{OUTPUT_DIR}/epoch_{epoch+1}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Epoch {epoch+1} → {save_dir}")
