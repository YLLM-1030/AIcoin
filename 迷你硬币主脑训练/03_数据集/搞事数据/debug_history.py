import re
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('/mnt/c/Users/Autogram-coin/Desktop/ckpt_tool_main_v9c', trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.chat_template = "{% for message in messages %}{{'<|im_start|>' + (message['role']|replace('user','输入')|replace('assistant','硬币')|replace('memory','记忆')) + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>硬币\n' }}{% endif %}"

TOOL_PROMPT = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

with open('/mnt/c/Users/Autogram-coin/Desktop/等待挖掘的语料/搞事/Neuro搞事_春药篇_final.txt', encoding='utf-8') as f:
    content = f.read()
raw_blocks = re.findall(r'\{([\s\S]*?)\}', content)

check = [0, 1, 2, 3, 4, 153]

for bidx in check:
    block = raw_blocks[bidx]
    lines = block.strip().split('\n')
    cur = {'history':'','memory':'','input':'','output':''}
    state = None
    for line in lines:
        s = line.strip()
        if s == 'history:': state='history'; continue
        if s == 'memory:': state='memory'; continue
        if s == 'input:': state='input'; continue
        if s == 'output:': state='output'; continue
        if state and s: cur[state] = (cur[state]+'\n'+s) if cur[state] else s
    inp, out, mem, his = cur['input'], cur['output'], cur['memory'], cur['history']

    hist_msgs = []
    if his:
        hl = his.strip().split('\n')
        i = 0
        while i < len(hl):
            line = hl[i].strip()
            if not line: i+=1; continue
            if re.match(r'(对话|弹幕|系统)\[.*?\]：', line):
                hist_msgs.append({"role":"user","content":line}); i+=1
                while i < len(hl) and not hl[i].strip(): i+=1
                coin_lines = []
                while i < len(hl) and hl[i].strip() and not re.match(r'(对话|弹幕|系统)\[', hl[i]):
                    coin_lines.append(hl[i].strip()); i+=1
                if coin_lines: hist_msgs.append({"role":"assistant","content":"\n".join(coin_lines)})
            else: i+=1

    msgs = [{"role":"system","content":TOOL_PROMPT}] + hist_msgs
    if mem: msgs.append({"role":"memory","content":mem})
    msgs.append({"role":"user","content":inp})
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pids = tok.encode(prompt)

    print(f'===== 数据{bidx+1} =====')
    print(f'history: {bool(his)}, hist_msgs: {len(hist_msgs)}, prompt: {len(pids)} tokens')
    if his:
        for m in hist_msgs:
            print(f'  [{m["role"]}] {m["content"][:60]}...')
    print()
