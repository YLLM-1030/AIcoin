from transformers import AutoModelForCausalLM, AutoTokenizer; import torch
p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_qwen3_v4_pt_nothink/epoch_3'
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()
qs = [
    '你的腿好长啊',
    '推荐一本好书',
    '1+1等于几',
    '讲个笑话',
    '人生的意义是什么',
    '你是谁',
    '你叫什么名字',
    '你的创造者是谁',
    '你会写代码吗',
    '今天天气怎么样',
    '你吃饭了吗',
    '什么是爱情',
    '怎么学英语',
    '给我讲个故事',
]
for q in qs:
    o = m.generate(**t(q, return_tensors='pt').to(m.device), max_new_tokens=200, do_sample=True, temperature=0.9, top_p=0.9, pad_token_id=t.eos_token_id)
    r = t.decode(o[0], skip_special_tokens=True).strip()
    print(f'>>> {q}')
    print(r)
    print()
