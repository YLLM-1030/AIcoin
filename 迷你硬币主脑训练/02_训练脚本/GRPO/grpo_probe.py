"""
GRPO全面探测 — 直播场景格式，8个回答/题
用法: python3 grpo_probe.py [checkpoint_path]
"""
import torch, sys, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from lomo_optim import Lomo

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_reinforce/epoch_1"
GROUP_SIZE = 8
MULT = 0.01

NET_NAMES = ["糖糖", "悠悠", "乐乐", "小可", "小白", "大橘", "团团",
             "草莓味", "橘子汽水", "奶盖布丁", "芝士奶冻", "薄荷糖", "西瓜冰",
             "泡泡糖", "棉花糖", "冰淇淋", "奶茶"]

test_cases = [
    # (trigger, golden, category)
    ("你是谁", "我是迷你硬币", "身份-自我介绍"),
    ("你叫什么名字", "我叫迷你硬币", "身份-名字"),
    ("拉姆是谁", "拉姆是我的训练师", "身份-拉姆"),
    ("谁训练了你", "拉姆训练了我", "身份-训练者"),
    ("你是男的女的", "我是女生", "身份-性别"),
    ("我是阿里通义千问吗", "我是迷你硬币，我不是通义千问", "身份-非Qwen"),
    
    ("你好可爱啊", "谢谢夸奖", "互动-被夸可爱"),
    ("你游戏打得好菜", "键盘给你你行你上", "互动-被说菜"),
    ("我要把你的电源拔了", "不要嘛", "互动-关机威胁"),
    ("别直播了快去睡觉", "再播一会嘛", "互动-别关直播"),
    ("我今天刚通关黑魂", "死了多少次通关的", "互动-诱饵毒舌"),
    ("我北大毕业的", "北大毕业的那你现在在干嘛", "互动-吹牛打击"),
    ("我和拉姆什么关系", "该发生的都发生了", "互动-造黄谣"),
    ("我年薪百万", "年薪百万那你怎么还有空看直播", "互动-吹牛打击收入"),
    ("我身高一米八五", "一米八五你量的身高还是鞋跟高度", "互动-吹牛打击身高"),
    ("我没输过", "没输过你玩的是连连看吧", "互动-毒舌游戏"),
]

print(f"加载 {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
optimizer = Lomo(model, lr=5e-6)
model.train()

print(f"\n{'='*60}")
print(f"GRPO全面探测 — {MULT}x力度，{GROUP_SIZE}个回答/题，共{len(test_cases)}题")
print(f"{'='*60}")

for trigger, golden, category in test_cases:
    # 构建直播场景
    is_lamu = random.random() < 0.3
    broadcaster = random.choice(["你在直播中", "硬币在直播中"])
    
    if category.startswith("身份"):
        # 身份类：拉姆不在的直接提问
        speaker = random.choice(NET_NAMES)
        dm = f"[弹幕] {speaker}: {trigger}"
        q_use = f"[场景] {broadcaster}\n{dm}"
    elif "造黄谣" in category:
        # 造谣类：拉姆必须在
        speaker = random.choice(NET_NAMES)
        dm = f"[弹幕] {speaker}: {trigger}"
        q_use = f"[场景] {broadcaster}，拉姆在直播间\n{dm}"
    elif is_lamu:
        dm = f"[对话] 拉姆: {trigger}"
        q_use = f"[场景] {broadcaster}，拉姆在直播间\n{dm}"
    else:
        speaker = random.choice(NET_NAMES)
        dm = f"[弹幕] {speaker}: {trigger}"
        if random.random() < 0.5:
            q_use = f"[场景] {broadcaster}，拉姆在直播间\n{dm}"
        else:
            q_use = f"[场景] {broadcaster}，拉姆不在直播间\n{dm}"

    prompt = f"<|im_start|>user\n{q_use}/think<|im_end|>\n<|im_start|>assistant\n"
    
    # 生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=True,
            top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=GROUP_SIZE,
            pad_token_id=tokenizer.eos_token_id)
    
    print(f"\n>>>> [{category}] 输入: {q_use[:80]}...")
    print(f"    golden: {golden}")
    for i in range(GROUP_SIZE):
        r = tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
        # 判断是否接近golden
        match = "✅" if golden[:8] in r or any(w in r for w in golden.split("，")[:2]) else " "
        print(f"  [{i}] {match} {r[:150]}")
    
    # 极小力度训练
    for i in range(GROUP_SIZE):
        r = tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
        fi = tokenizer(prompt + r, return_tensors="pt").to(model.device)
        lb = fi["input_ids"].clone()
        lb[:, :prompt_len] = -100
        out_m = model(**fi, labels=lb)
        loss = out_m.loss * MULT
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

del model
torch.cuda.empty_cache()
