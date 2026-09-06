"""
弹幕打分器 — 基于实证研究的启发式版本
addressee: @me / @group / @others / @self
reply_desire: 0-1

数据来源:
  - FlirtGen: 15,000条, 问题类型→回复率映射
  - Chen et al.: 1,559条, 多层logistic回归, 异议/问题/对错观点→回复
  - Library Hi Tech: 1,968条, 语言学特征→回复量
"""
import re

# ============================================================
# 名字
# ============================================================
SELF_NAMES = ['硬币', '迷你硬币', '拉姆', '主播', '主包']  # 自己+亲密关系, 都算@我

# ============================================================
# 一、addressee 判断
# ============================================================
def detect_addressee(text, user_name=''):
    """判断弹幕在跟谁说话。 返回: @me / @group / @others / @self"""

    # 直接称呼 → @me
    for name in SELF_NAMES:
        if name in text:
            return '@me'

    # 回复/点名其他观众 → @others
    if re.search(r'@\S|回复\s*\S', text):
        return '@others'

    # 群体称呼 → @group
    if re.search(r'大家|你们|各位|谁|有人|有没有人|在座|在场的|观众|兄弟们|家人们', text):
        return '@group'

    # 第二人称"你"在弹幕语境中默认指向主播 → @me
    if SECOND_PERSON_PAT.search(text):
        return '@me'

    # 氛围/自言自语 → @self
    return '@self'


# ============================================================
# 二、reply_desire 打分 (0-1)
# ============================================================

# 疑问标记
QUESTION_PAT = re.compile(r'[?？]')
INTERROGATIVE_PAT = re.compile(
    r'吗|呢|吧|么|啥|咋|嘛|不'
    r'|怎么样|怎么|为何|为什么|干嘛|干啥'
    r'|谁|哪|几|多少|什么'
    r'|是不是|对不对|行不行|好不好|有没有|知不知道'
    r'|难道|岂不是'
    r'|说说|讲讲|聊聊|来一个'
)

# 脑洞/假想词 (FlirtGen: 77% reply rate, 基准的5.1x)
HYPOTHETICAL_PAT = re.compile(
    r'如果|万一|假设|假如|要是|若是|倘若|要是有一天|能不能|可不可以|你觉得|你认为|想象一下|脑洞|如果有一天'
    r'|要是|让你选|换你|换做你|你来'                    # 把对方带入
    r'|说不定|没准|可能|大概|也许|或许'                 # 推测
    r'|敢不敢|想不想|愿不愿意|会不会|该不会'            # 试探
    r'|到那时候|到时候|下辈子|穿越|重生'               # 时空想象
    r'|试着|尝试|试想|幻想'                            # 想象
)

# 偏好/站队 (FlirtGen: 70% reply rate)
PREFERENCE_PAT = re.compile(
    r'还是|或者|选哪个|喜欢|讨厌|最爱|最讨厌|A.*还.*B|更.*还是'
    r'|二选一|二择|选一|选一个|挑一个'              # 选择
    r'|哪个好|哪种|哪边|哪个比较|哪个更'             # 比较
    r'|要不要|想不想|该不该|能不能|行不行'            # 取舍
    r'|vs|VS|对决|对比|PK|pk|pk一下'                 # 对决
    r'|支持|站|投|pick|Pick|PICK|你觉得'              # 立场
)

# 不同意见 (Chen et al.: 显著预测回复)
DISAGREE_PAT = re.compile(
    r'不对|不是|不同意|不觉得|怎么可能|哪有|不会吧|瞎说|乱讲|错了|你确定|确定吗|认真的|开玩笑|认真的吗'
    r'|骗人|假的|扯淡|扯|胡说|胡扯'                 # 质疑真实性
    r'|我不信|不信|才不信|怀疑'                      # 表达不信
    r'|离谱|过分|夸张|不至于'                         # 程度异议
    r'|反了|搞反了|颠倒|倒过来'                      # 纠正
    r'|明明|其实|实际上|事实上'                       # 事实纠正
    r'|你想多了|误会了|理解错了|搞错了'               # 理解纠偏
)

# 第二人称 (Library Hi Tech: +response)
SECOND_PERSON_PAT = re.compile(
    r'你|妳|你们|你这|你这种|你这类|你这样的'      # 你系列
    r'|您|阁下|这位|主播你|这个ai|ai你'              # 敬称/指代
    r'|ai们|AI们|主播们|aituber|vtuber|虚拟主播'      # 群体指代(含自己)
)

# 正面情绪词
POSITIVE_PAT = re.compile(
    r'哈哈|嘿嘿|太.*了|好.*啊|棒|厉害|牛|爱了|喜欢|不错|可以的|绝了|6|六|赞'
    r'|笑死|笑不活|笑疯|笑喷|好笑|xswl|XSWL'          # 笑系列
    r'|好耶|nice|奈斯|good|太好了|太棒|给力'            # 叫好
    r'|有道理|确实|对的|说得对|说得好|没错|正解'         # 认同
    r'|泪目|感动|呜呜|哭了|破防|心动了'                  # 情绪触动
    r'|精彩|好活|好活儿|绝活|nb|NB|牛逼|太强|真厉害'     # 赞叹
    r'|舒服|爽|解压|治愈|满足'                           # 感受
    r'|美好|幸福|快乐|开心|高兴|期待|希望|加油'          # 积极词汇
)

# 赞美 (AI VTuber 应该回应的)
COMPLIMENT_PAT = re.compile(
    r'可爱|好看|漂亮|帅|美|强|厉害|牛|天才|喜欢|爱了|好听|温柔|有趣|好玩|有意思|绝了|神仙'
    r'|棒|赞|好棒|太强|无敌|respect|崇拜|女神|老婆|老公'  # 更多赞美
    r'|好听|唱得好|跳得好|画得好|厉害啊|牛啊|猛|太会了'    # 技能赞美
)

# 问候
GREETING_PAT = re.compile(
    r'晚上好|早上好|下午好|早安|晚安|你好|hello|hi|嗨|来了|在吗|在不在|打卡'
    r'|大家好|哈喽|哟|呦|哈咯|halo|早啊|早呀|晚安啦'  # 更多问候
)

# 纯垃圾/无信息量
JUNK_SET = {'。', '，', '.', ',', '1', '2', '3', '哈哈', '呵呵', '嗯', '哦', '啊', '...', '……',
            '草', 'hh', 'hhh', 'hhhh', '？', '?', '！', '!', '6', '666', '233', '2333',
            'www', 'wwww', '111', '222', '333'}


def score_reply_desire(text, user_name='', is_fan=False):
    """回复欲望 0-1"""

    text_stripped = text.strip()
    score = 0.0

    # ---- VTuber 话痨基线: AI VTuber 天生更愿意互动 ----
    if text_stripped not in JUNK_SET:
        score += 0.08

    # ---- 正向特征 ----

    # F1: 含问号 (FlirtGen: 问题 vs 陈述 = 5.4x)
    if QUESTION_PAT.search(text):
        score += 0.25

    # F2: 疑问语气词
    if INTERROGATIVE_PAT.search(text):
        score += 0.12

    # F3: 脑洞/假想 (FlirtGen: 77% reply rate)
    if HYPOTHETICAL_PAT.search(text):
        score += 0.15

    # F4: 偏好/站队 (FlirtGen: 70% reply rate)
    if PREFERENCE_PAT.search(text):
        score += 0.12

    # F5: 不同意见 (Chen et al.: significant)
    if DISAGREE_PAT.search(text):
        score += 0.12

    # F6: 第二人称 (Library Hi Tech: +response)
    if SECOND_PERSON_PAT.search(text):
        score += 0.16   # ×2 权重（拉姆要求：基础 0.08 ×2，勿改回 0.08）

    # F7: 正面情绪
    if POSITIVE_PAT.search(text):
        score += 0.05

    # F8: 直接叫名字 → 高权重 (跟问号同档)
    for name in SELF_NAMES:
        if name in text:
            score += 0.25
            break

    # F9: 赞美 (AI VTuber 应该感谢)
    if COMPLIMENT_PAT.search(text):
        score += 0.12

    # F10: 问候 (AI VTuber 应该回应)
    if GREETING_PAT.search(text):
        score += 0.08

    # F11: 粉丝/熟人（TODO: 需要接B站API查粉丝牌）
    # if is_fan:
    #     score += 0.10

    # ---- 负向惩罚 ----

    # N1: 纯垃圾/无信息量
    if text_stripped in JUNK_SET:
        score -= 0.30

    # N2: 纯 emoji/符号
    if re.match(r'^[\U0001F000-\U0001FFFF\u2600-\u27BF\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]+$', text_stripped):
        score -= 0.20

    # N3: 太短 (<3字且无提问)
    if len(text_stripped) < 3 and not QUESTION_PAT.search(text):
        score -= 0.10

    # N4: 纯陈述无任何交互信号
    has_interaction = any([
        QUESTION_PAT.search(text),
        INTERROGATIVE_PAT.search(text),
        DISAGREE_PAT.search(text),
        HYPOTHETICAL_PAT.search(text),
        PREFERENCE_PAT.search(text),
    ])
    if not has_interaction:
        score -= 0.12

    # N5: 太长 (>100字)
    if len(text) > 100:
        score -= 0.05

    return max(0.0, min(1.0, round(score, 3)))


# ============================================================
# 三、联合打分
# ============================================================

# addressee → 权重 (Sacks-Schegloff-Jefferson 话轮规则映射)
ADDR_WEIGHT = {
    '@me':     1.0,   # 规则1: 被点名 → 必须考虑
    '@group':  0.7,   # 群体提问 → 里面有我
    '@self':   0.5,   # 氛围弹幕 → 可选插嘴(规则2)
    '@others': 0.2,   # 跟别人说话 → 一般不插嘴(除非抢注意力)
}


def combined_score(text, user_name='', is_fan=False):
    """联合打分: addressee × reply_desire

    返回: {addressee, raw_score, weight, final_score}
    final_score = raw_score × addr_weight
    """

    addr = detect_addressee(text, user_name)
    raw = score_reply_desire(text, user_name, is_fan)
    # weight = ADDR_WEIGHT.get(addr, 0.5)  # addressee 权重暂时关闭，后续启用
    # final = round(raw * weight, 3)
    final = round(raw, 3)

    return {
        'addressee': addr,
        'raw_score': raw,
        # 'weight': weight,  # 暂关
        'final_score': final,
    }


# ============================================================
# 四、批量 + 统计
# ============================================================

def batch_score(messages):
    """
    messages: [{'user': str, 'text': str, ...}, ...]
    返回: 打了分的 list + 统计摘要
    """
    results = []
    scores = []

    for m in messages:
        r = combined_score(m.get('text', ''), m.get('user', ''))
        r['user'] = m.get('user', '?')
        r['text'] = m.get('text', '')
        results.append(r)
        scores.append(r['final_score'])

    # 分布统计
    if scores:
        sorted_s = sorted(scores)
        n = len(sorted_s)

        summary = {
            'count': n,
            'mean': round(sum(scores) / n, 3),
            'median': round(sorted_s[n // 2], 3),
            'min': round(min(scores), 3),
            'max': round(max(scores), 3),
            # 分位数 → 阈值建议
            'p25': round(sorted_s[int(n * 0.25)], 3),   # 丢弃25%
            'p50': round(sorted_s[int(n * 0.50)], 3),   # 丢弃50%
            'p75': round(sorted_s[int(n * 0.75)], 3),   # 丢弃75%
            'p90': round(sorted_s[int(n * 0.90)], 3),   # 丢弃90%
        }
    else:
        summary = {'count': 0}

    return results, summary


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    tests = [
        "硬币你今天开心吗",
        "拉姆编程水平怎么样",
        "大家晚上好",
        "主播好可爱",
        "@小王 你那个不对",
        "哈哈哈哈哈",
        "如果明天是世界末日你会做什么",
        "披萨放菠萝还是不放",
        "不对不对，拉姆你完全搞错了",
        "来了",
        "666",
        "你觉得Python和Go哪个好",
        "走神儿",
    ]

    for t in tests:
        r = combined_score(t)
        print(f"[{r['addressee']}] raw={r['raw_score']:.3f} final={r['final_score']:.3f}  | {t}")
