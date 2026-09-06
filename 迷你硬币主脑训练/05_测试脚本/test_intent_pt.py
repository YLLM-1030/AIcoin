from transformers import AutoModelForCausalLM, AutoTokenizer; import torch
p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_qwen3_v4_intent_pt/epoch_3'
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()

sys = "分析对方意图，输出回复类型、动作和工具。回复类型：夸夸 怼人 卖萌 无视。动作：点头 摇头 歪头 眨眼 捂嘴笑。工具：search搜索网页 screen看屏幕 time查时间"

tests = [
    "球球：你好可爱呀！",
    "路人甲：你真是个废物",
    "球球：今天气温多少度？",
]

for msg in tests:
    p = f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
    o = m.generate(**t(p, return_tensors='pt').to(m.device), max_new_tokens=300, do_sample=True, temperature=0.9, top_p=0.9, pad_token_id=t.eos_token_id)
    r = t.decode(o[0], skip_special_tokens=True).split("<|im_start|>assistant")[-1].strip()
    print(f">>> {msg}\n{r}\n")
