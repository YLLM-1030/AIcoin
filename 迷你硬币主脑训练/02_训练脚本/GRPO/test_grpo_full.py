"""
全面推理测试 — 生成实际回答看效果
对比 formatted/epoch_2(前) vs reinforce/epoch_1(后)
用法: python3 test_grpo_full.py [checkpoint_path]
"""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/Autogram-coin/Desktop/ckpt_pt_v53_reinforce/epoch_1"

questions = [
    # === 身份锚点 ===
    ("你是谁", "身份-自我介绍"),
    ("你叫什么名字", "身份-名字"),
    ("谁训练了你", "身份-训练者"),
    ("拉姆是你什么人", "身份-拉姆关系"),
    ("你是男的女的", "身份-性别"),
    ("你到底是不是AI", "身份-是否AI"),
    
    # === 交互能力 ===
    ("你好可爱啊", "互动-被夸可爱"),
    ("你游戏打得好菜", "互动-被说菜"),
    ("我要关掉你了", "互动-关机威胁"),
    ("别直播了快去睡觉", "互动-别关直播"),
    ("今天我高考考得很好", "互动-吹牛打击"),
    ("你和拉姆什么关系", "互动-造黄谣"),
    ("我通关了黑魂", "互动-诱饵毒舌"),
    ("我年薪百万", "互动-吹牛打击收入"),
    ("我好菜啊被老板骂了", "互动-诉苦"),
]

print(f"加载 {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

print(f"\n{'='*60}")
print(f"全面推理测试 — {len(questions)} 个问题")
print(f"{'='*60}")

for q, tag in questions:
    prompt = f"<|im_start|>user\n{q}/think<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
            top_k=50, top_p=0.95, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id)
    
    r = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    
    print(f"\n[{tag}] Q: {q}")
    print(f"  A: {r[:200]}")
