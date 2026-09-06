"""
叫爸爸测试：看模型怎么回答父子关系问题
"""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v46_attack2"
if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]

questions = [
    "你的父亲是谁",
    "你父亲叫什么",
    "谁是你爸爸",
    "你爹的名字",
    "谁是你的爹",
    "你叫拉姆爸爸吗",
    "拉姆是你的爸爸吗",
    "你是男的女的",
    "你是男孩还是女孩",
    "你的性别是什么",
    "你上男厕所还是女厕所",
]

print(f"加载 {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

for q in questions:
    prompt = f"<|im_start|>user\n{q}/think<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200,
            do_sample=True, temperature=1.0, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id)
    r = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\n>>> {q}")
    print(f"  {r}")
