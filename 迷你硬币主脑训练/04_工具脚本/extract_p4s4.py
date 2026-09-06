import json, re

path = r"C:\Users\Autogram-coin\Desktop\minicoinB2\轻量备份_经验包\02_模型训练\第一次训练\17阶段训练数据\P4S4_终极抽象与边界协议\train_data.json"
data = json.load(open(path, 'r', encoding='utf-8'))

lines = []
for item in data:
    resp = item["response"]
    # 去掉【xxxSOP】头部
    resp = re.sub(r'^【[^】]+】\s*', '', resp)
    # 去掉"外部事件：……"行
    resp = re.sub(r'外部事件：[^\n]*\n?', '', resp)
    # 去掉"推演："标记但保留内容
    resp = re.sub(r'推演：', '', resp)
    resp = resp.strip()
    if resp:
        lines.append(resp)

output = "\n\n".join(lines)
out_path = r"C:\Users\Autogram-coin\Desktop\持续预训练数据_公理体系.txt"
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"提取完成，{len(lines)}条，共{len(output)}字符")
print(f"保存到: {out_path}")
print("\n前200字预览：")
print(output[:200])
