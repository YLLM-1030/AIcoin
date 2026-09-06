"""
B站视频字幕/文案获取器
用法:
  python3 bili_subtitle.py --url https://www.bilibili.com/video/BV1xx
  python3 bili_subtitle.py --bvid BV1xx --output subtitle.txt

依赖安装:
  pip install bilibili-api-python aiohttp
"""
import asyncio, json, os, sys, argparse, re
from datetime import datetime

try:
    from bilibili_api import video, sync
except ImportError:
    print("请先安装: pip install bilibili-api-python")
    sys.exit(1)

def parse_bvid(text: str) -> str:
    """从 URL 或纯文本中提取 BVID"""
    m = re.search(r'BV\w+', text)
    if m:
        return m.group()
    raise ValueError("未找到 BVID，请传入 BV 号或完整 URL")

async def get_subtitle(bvid: str) -> list[dict]:
    """获取视频字幕（CC字幕），返回 [{lan, lines}]"""
    v = video.Video(bvid=bvid)
    info = await v.get_info()
    title = info.get("title", "未知标题")
    print(f"视频: {title}")
    print(f"时长: {info.get('duration', 0)} 秒")

    # 获取字幕列表
    cid = info.get("cid", 0)
    if not cid:
        # 尝试从页面获取 cid
        pages = await v.get_pages()
        if pages:
            cid = pages[0].get("cid", 0)

    result = []
    if cid:
        subtitle_info = await v.get_subtitle(cid=cid)
        subtitles = subtitle_info.get("subtitles", [])
        if subtitles:
            for sub in subtitles:
                lan = sub.get("lan_doc", "未知")
                sub_url = sub.get("subtitle_url", "")
                if sub_url:
                    if sub_url.startswith("//"):
                        sub_url = "https:" + sub_url
                    print(f"  找到字幕: {lan} → {sub_url}")
                    result.append({"lan": lan, "url": sub_url})
        else:
            print("  此视频没有 CC 字幕")
    return result, title

async def download_subtitle(url: str) -> list[str]:
    """下载字幕 JSON 并提取文本行"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
    bodies = data.get("body", [])
    lines = []
    for item in bodies:
        content = item.get("content", "")
        if content.strip():
            lines.append(content.strip())
    return lines

async def main():
    parser = argparse.ArgumentParser(description="B站视频字幕获取器")
    parser.add_argument("--url", help="视频URL")
    parser.add_argument("--bvid", help="BV号")
    parser.add_argument("--output", default="", help="输出文件路径（默认打印到终端）")
    args = parser.parse_args()

    if not args.url and not args.bvid:
        parser.print_help()
        return

    bvid = args.bvid or parse_bvid(args.url)
    print(f"BVID: {bvid}")

    subtitles, title = await get_subtitle(bvid)
    all_lines = []
    if subtitles:
        for sub in subtitles:
            lines = await download_subtitle(sub["url"])
            all_lines.extend(lines)
            print(f"  {sub['lan']}: {len(lines)} 行")

    if not all_lines:
        print("\n此视频没有可用字幕。")
        print("试试用 yt-dlp 下载音频后用 Whisper 转写：")
        print(f"  yt-dlp -x --audio-format wav -o 'audio.%(ext)s' https://www.bilibili.com/video/{bvid}")
        print(f"  whisper audio.wav --model small --language zh")
        return

    text = "\n".join(all_lines)
    output = args.output or f"subtitle_{bvid}.txt"
    with open(output, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(f"# BVID: {bvid}\n")
        f.write(f"# 获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# 共 {len(all_lines)} 行\n\n")
        f.write(text)
    print(f"\n✅ 已保存: {output} ({len(all_lines)} 行, {len(text)} 字符)")

if __name__ == "__main__":
    asyncio.run(main())
