"""
B站 Vedal 频道字幕获取器（纯 requests，不依赖 bilibili-api 的凭证问题）
安装: pip install requests
用法:
  python3 get_vedal_subs.py --bvid BV1xx          # 单视频
  python3 get_vedal_subs.py --uid 3546729368520811 # 扫描用户所有视频
"""
import requests, json, os, sys, re, time, random
from datetime import datetime

SESSDATA = "37a75588%2C1798203134%2C890a2%2A61CjAf19wWMT5tLVMvbcG5T6wofgNz136m3FxKE8dRWpC57yyE_WA2vinweFQe1iSZyNwSVlB4SnF1REJJN1hJaTViakNYSlV2U09xYllZbkthYU9BeW9qcS1fcXFpcnJWNHhhRzJ1MHdJX3BDRjU0ZHFqLXZoREZPeVpCamsxUTM4N1B6aVBvN3hBIIEC"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.bilibili.com",
    "Cookie": f"SESSDATA={SESSDATA}",
}

def get_bvid(text):
    m = re.search(r'BV\w+', text)
    return m.group() if m else None

def get_cid(bvid):
    """获取视频 cid（字幕需要）"""
    url = f"https://api.bilibili.com/x/player/pagelist?bvid={bvid}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    data = r.json()
    if data.get("code") == 0 and data["data"]:
        return data["data"][0]["cid"]
    return None

def get_subtitle_info(bvid, cid):
    """获取字幕信息"""
    url = f"https://api.bilibili.com/x/player/wbi/v2?bvid={bvid}&cid={cid}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    data = r.json()
    if data.get("code") != 0:
        return None
    return data["data"].get("subtitle", {}).get("subtitles", [])

def download_subtitle(sub_url):
    """下载字幕 JSON"""
    if sub_url.startswith("//"):
        sub_url = "https:" + sub_url
    r = requests.get(sub_url, headers=HEADERS, timeout=10)
    data = r.json()
    return [item["content"].strip() for item in data.get("body", []) if item.get("content", "").strip()]

def get_video_title(bvid):
    """获取视频标题"""
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    data = r.json()
    if data.get("code") == 0:
        return data["data"].get("title", bvid)
    return bvid

def get_user_videos(uid, page=1):
    """获取用户视频列表，带自动重试"""
    url = f"https://api.bilibili.com/x/space/arc/search?mid={uid}&ps=50&pn={page}"
    for attempt in range(3):
        r = requests.get(url, headers=HEADERS, timeout=10)
        try:
            data = r.json()
        except:
            print(f"  API 返回非 JSON (状态 {r.status_code})，重试 {attempt+1}/3...")
            time.sleep(20)
            continue
        if data.get("code") == -799:
            wait = 30 * (attempt + 1)
            print(f"  限流，等待 {wait} 秒...")
            time.sleep(wait)
            continue
        if data.get("code") != 0:
            print(f"  API 错误: {data}")
            return []
        return data["data"]["list"]["vlist"]
    print("  获取视频列表失败（多次重试后）")
    return []

def process_bvid(bvid, output_dir):
    """处理单个视频"""
    print(f"  获取信息...")
    title = get_video_title(bvid)
    cid = get_cid(bvid)
    if not cid:
        print(f"  ❌ 无法获取 cid")
        return None
    
    subs = get_subtitle_info(bvid, cid)
    if not subs:
        print(f"  ❌ 无 CC 字幕")
        return None
    
    # 选最佳字幕
    best = None
    for s in subs:
        lan = s.get("lan_doc", "")
        if "中文" in lan or "zh" in lan:
            best = s
            break
    if not best:
        best = subs[0]
    
    url = best.get("subtitle_url", "")
    lan = best.get("lan_doc", "未知")
    lines = download_subtitle(url)
    
    if not lines:
        print(f"  ❌ 字幕内容为空")
        return None
    
    print(f"  ✅ 字幕: {lan}, {len(lines)} 行")
    
    # 保存
    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)[:40]
    out_file = os.path.join(output_dir, f"{bvid}_{safe_title}.txt")
    text = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n# BVID: {bvid}\n# 语言: {lan}\n# 行数: {len(lines)}\n\n{text}")
    
    # 追加到合集
    all_file = os.path.join(output_dir, "_all_subtitles.txt")
    with open(all_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n# {title} (BV:{bvid})\n{'='*60}\n\n{text}\n\n")
    
    return out_file

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvid", help="单视频 BV 号")
    parser.add_argument("--uid", type=int, default=3546729368520811, help="用户 UID")
    parser.add_argument("--max", type=int, default=20, help="最多检查视频数")
    parser.add_argument("--output", default=r"C:\Users\Autogram-coin\Desktop\新建训练\vedal_subs", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.bvid:
        print(f"单视频: {args.bvid}")
        result = process_bvid(args.bvid, args.output)
        if result:
            print(f"保存: {result}")
        return

    print(f"扫描用户 UID={args.uid} 的视频...")
    vlist = get_user_videos(args.uid)
    print(f"共 {len(vlist)} 个视频\n")

    has = 0
    none_ = 0
    for i, v in enumerate(vlist[:args.max]):
        bvid = v["bvid"]
        title = v["title"][:50]
        print(f"[{i+1}/{min(args.max, len(vlist))}] {bvid} | {title}")
        result = process_bvid(bvid, args.output)
        if result:
            has += 1
        else:
            none_ += 1
        delay = random.uniform(8, 15)
        print(f"  等待 {delay:.0f} 秒...")
        time.sleep(delay)  # 随机间隔防封

    print(f"\n完成! 有字幕: {has}, 无字幕: {none_}")
    print(f"保存在: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
