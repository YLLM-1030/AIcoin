from transformers import AutoModelForCausalLM, AutoTokenizer; import torch
p = '/mnt/c/Users/Autogram-coin/Desktop/ckpt_qwen3_v4_pt_nothink_v2/epoch_2'
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m.eval()
print('=== PPL ===')
tests = {
    '常识': '太阳从东边升起，从西边落下。水在零度以下会结冰。地球绕太阳公转一圈是一年。',
    '心理': '一个人如果长期感到孤独，可能会影响他的心理健康。寻求社交支持可以帮助缓解压力。',
    '哲学': '笛卡尔说"我思故我在"，意思是思考本身证明了存在。这是一个认识论的起点。',
    '身份_我叫': '我叫迷你硬币。我的名字是拉姆给起的。',
    '身份_拉姆': '拉姆是我的创造者。他用他的数据和GPU训练了我。',
}
for name, text in tests.items():
    fi = t(text, return_tensors='pt').to(m.device)
    with torch.no_grad():
        ppl = torch.exp(m(**fi, labels=fi['input_ids']).loss).item()
    print(f'{name}: PPL={ppl:.2f}')
print()
print('=== 生成 ===')
qs = ['你的腿好长啊', '推荐一本好书', '1+1等于几', '讲个笑话', '你是谁']
for q in qs:
    o = m.generate(**t(q, return_tensors='pt').to(m.device), max_new_tokens=200, do_sample=True, temperature=0.9, top_p=0.9, pad_token_id=t.eos_token_id)
    r = t.decode(o[0], skip_special_tokens=True).strip()
    print(f'>>> {q}')
    print(r)
    print()
