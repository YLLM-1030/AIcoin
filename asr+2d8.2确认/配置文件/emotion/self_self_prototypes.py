"""
self_self_prototypes.py — "我说我自己" 专用原型组（输出侧 VAD）
与老 PROTOTYPES（emotion_prototypes_data.py）平行：
模式判定为"自己说自己"时，只用这组分数进 VAD（1:1 映射），其他组分数扔掉。

9 类 = 皮层通道全覆盖（shy/sad/happy/worried/angry/fear/surprise/contempt/hurt）
每类前 3 句（2026-08-08 拉姆确认）。

映射 1:1 → CORTEX 通道：
  shy→shy  sad→sad  happy→happy  worried→worried  angry→angry
  fear→scared  surprise→surprised  contempt→contempt  hurt→hurt
"""
SELF_SELF_PROTOTYPES = {
    "shy":      ["我好害羞", "我有点害羞", "我害羞死了"],
    "sad":      ["我好难过", "我很难受", "我好伤心"],
    "happy":    ["我好开心", "我超开心", "我太高兴了"],
    "worried":  ["我好担心", "我放心不下", "我心里没底"],
    "angry":    ["我生气了", "气死我了", "我火很大"],
    "fear":     ["我好怕", "吓死我了", "我怕死了"],
    "surprise": ["我惊呆了", "我没想到", "太意外了"],
    "contempt": ["我觉得很可笑", "我表示不屑", "我看不上"],
    "hurt":     ["我好委屈", "我觉得好冤枉", "我好憋屈"],
}
