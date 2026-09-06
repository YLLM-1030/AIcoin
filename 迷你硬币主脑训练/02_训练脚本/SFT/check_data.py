import json
path = r"C:\Users\Autogram-coin\Desktop\minicoinB2\轻量备份_经验包\02_模型训练\第一次训练\17阶段训练数据\P4S4_终极抽象与边界协议\train_data.json"
d = json.load(open(path, 'r', encoding='utf-8'))
print(f"总条数: {len(d)}")
total_chars = sum(len(x["instruction"]) + len(x["response"]) for x in d)
print(f"总字符数: {total_chars}")
# 展示前3个文件名
for i, item in enumerate(d[:3]):
    print(f"\n--- 第{i+1}条 ---")
    print(f"protocol: {item.get('protocol', 'N/A')[:60]}")
    print(f"instruction: {item['instruction'][:50]}...")
    print(f"response: {item['response'][:50]}...")
