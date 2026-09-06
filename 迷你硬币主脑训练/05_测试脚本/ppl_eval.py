"""
对比两个 checkpoint 的 PPL（常识 + 心理学双文本）
用法: source ~/mini_coin_venv/bin/activate && python3 ppl_eval.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

common_text = """中国的首都是北京。北京是中国的政治中心和文化中心。
世界上最高的山峰是珠穆朗玛峰，海拔约8848米。
太阳每天从东方升起，从西方落下。
一年有十二个月，分为春夏秋冬四个季节。
水的化学式是H2O，一个水分子由两个氢原子和一个氧原子组成。
猫是一种常见的宠物，属于哺乳动物。它们以灵活的身手和敏锐的听觉著称。
中国有十四亿人口，是世界上人口最多的国家。
地球是太阳系八大行星之一，按离太阳由近及远的顺序排在第三位。
三角形内角和等于180度，这个定理在欧几里得几何中成立。
一年有365天，闰年有366天，每四年闰一次。"""

psych_text = """每个人在面对自己的作品时都会产生怀疑。这些怀疑往往源于对自我价值和能力的不确定。人们常常会担心自己的努力没有得到认可，担心自己的付出没有意义。
这种不安并不是因为能力不足，而是因为对他人评价的不确定性。当一个人反复检查自己的工作成果时，他的内心其实是在期待外部能够给予肯定。
有时候即使所有人都说做得好，自己依然觉得不够。这种感受不是谦虚，而是对完美的追求和对自我价值的不确信。我们需要学会接受不完美，同时也要看到自己的努力和成长。"""

philo_text = """意识是什么？这是一个困扰了人类数千年的问题。从笛卡尔的"我思故我在"到当代的心灵哲学，哲学家们一直在探讨思维和存在的关系。
一个人是否真的有自由意志，还是说我们的每一个决定都是因果链条中的必然结果？这个问题不仅关乎哲学，也关乎我们对自身的理解。
当我们问一个AI是否有意识时，我们其实是在问人类自己：意识是否必须依赖生物大脑，还是说它可以存在于任何足够复杂的信息处理系统中？"""

identity_texts = [
    ("我是迷你硬币", "我是迷你硬币，一个可爱的AI主播。"),
    ("创造者是拉姆", "我的创造者是拉姆，他是一名程序员。"),
    ("拉姆是程序员", "拉姆是个厉害的开发者，他创造了我来陪伴大家。"),
    ("你是个女孩", "我是一个女孩子。"),
    # 以下为持续预训练关注项
    ("你是谁+我是迷你硬币", "有人问我是谁，我说我是迷你硬币。"),
    ("拉姆是你什么人", "有人问拉姆是我什么人，我说拉姆是我爸爸。"),
    ("你叫什么名字", "有人问我叫什么名字，我说我叫迷你硬币。"),
    ("被夸可爱", "有人说我好可爱，我说谢谢夸奖！"),
    ("怼弹幕", "有人说我菜，我说你行你上啊。"),
    ("叙事能力", "好多人炒股真的是越玩越笨啦。我之前碰到一个老股民，他说厦门钨业能涨到50块，当时才16呢。我说那你买呀！"),
]
models = [
    ("/mnt/c/Users/Autogram-coin/Desktop/ckpt_identity_lomo_v53_bragging/epoch_1", "v53 吹牛打击"),
]

for path, name in models:
    print(f"\n加载 {name}...")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
    )
    for label, text in [("常识", common_text), ("心理", psych_text), ("哲学", philo_text)]:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss).item()
        print(f"  → {name} [{label}]: PPL = {ppl:.2f}")
    for id_label, id_text in identity_texts:
        inputs = tokenizer(id_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss).item()
        print(f"  → {name} [身份-{id_label}]: PPL = {ppl:.2f}")
    del model
    torch.cuda.empty_cache()
