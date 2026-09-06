"""
B站视频字幕批量获取 + Whisper 兜底转写
用法:
  # 单视频
  python3 bili_crawl.py --bvid BV1xx

  # 批量（从文件读 BV 号，每行一个）
  python3 bili_crawl.py --list bv_list.txt --output_dir ./subtitles

  # 无字幕时自动 Whisper 转写（需先 pip install openai-whisper）
  python3 bili_crawl.py --bvid BV1xx --whisper
"""
import asyncio, json, os, sys, argparse, re, subprocess
from datetime import datetime

try:
    from bilibili_api import video, sync
except ImportError:
    print("请先安装: pip install bilibili-api-python aiohttp")
    sys.exit(1)

def parse_bvid(text):
    m = re.search(r'BV\w+', text)
    if m:
        return m.group()
    raise ValueError("未找到 BVID")

async def get_subtitle(bvid):
    v = video.Video(bvid=bvid)
    info = await v.get_info()
    title = info.get("title", "未知标题").replace("/", "_").replace(" ", "_")
    cid = info.get("cid", 0)
    if not cid:
        pages = await v.get_pages()
        if pages:
            cid = pages[0].get("cid", 0)
    result = []
    if cid:
        sub_info = await v.get_subtitle(cid=cid)
        for sub in sub_info.get("subtitles", []):
            url = sub.get("subtitle_url", "")
            if url:
                if url.startswith("//"):
                    url = "https:" + url
                result.append({"lan": sub.get("lan_doc", "未知"), "url": url})
    return result, title

async def download_subtitle(url):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
    return [i.get("content", "").strip() for i in data.get("body", []) if i.get("content", "").strip()]

def whisper_transcribe(audio_path):
    """用 Whisper 转写音频"""
    try:
        import whisper
    except ImportError:
        print("  Whisper 未安装，跳过转写。安装: pip install openai-whisper")
        return []
    print(f"  Whisper 转写中...")
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, language="zh")
    return [seg["text"].strip() for seg in result["segments"]]

def download_audio(bvid, output_dir):
    """用 yt-dlp 下载音频"""
    url = f"https://www.bilibili.com/video/{bvid}"
    output_path = os.path.join(output_dir, f"{bvid}.wav")
    if os.path.exists(output_path):
        return output_path
    try:
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "wav",
            "-o", os.path.join(output_dir, f"{bvid}.%(ext)s"),
            url
        ], check=True, capture_output=True)
        return output_path
    except FileNotFoundError:
        print("  yt-dlp 未安装。安装: pip install yt-dlp")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  yt-dlp 失败: {e.stderr.decode()[:200]}")
        return None

async def process_video(bvid, output_dir, use_whisper=False):
    print(f"\n{'='*50}")
    print(f"处理: {bvid}")
    subtitles, title = await get_subtitle(bvid)
    print(f"标题: {title}")

    all_lines = []
    source = "cc字幕"

    if subtitles:
        for sub in subtitles:
            lines = await download_subtitle(sub["url"])
            all_lines.extend(lines)
            print(f"  CC字幕 ({sub['lan']}): {len(lines)} 行")
    elif use_whisper:
        print(f"  无 CC 字幕，尝试 Whisper 转写...")
        source = "whisper"
        audio_path = download_audio(bvid, output_dir)
        if audio_path:
            all_lines = whisper_transcribe(audio_path)
            print(f"  Whisper: {len(all_lines)} 行")
    else:
        print(f"  无字幕。跳过（加 --whisper 启用音频转写）")

    if not all_lines:
        print(f"  ❌ 未获取到任何文本")
        return

    text = "\n".join(all_lines)
    out_file = os.path.join(output_dir, f"{bvid}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n# BVID: {bvid}\n# 来源: {source}\n# 行数: {len(all_lines)}\n\n{text}")
    print(f"  ✅ {out_file} ({len(all_lines)} 行, {len(text)} 字)")

async def main():
    parser = argparse.ArgumentParser(description="B站字幕批量获取")
    parser.add_argument("--bvid", help="单个BV号")
    parser.add_argument("--list", help="BV号列表文件（每行一个）")
    parser.add_argument("--output_dir", default="./bili_output", help="输出目录")
    parser.add_argument("--whisper", action="store_true", help="无字幕时用Whisper转写")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bvids = []
    if args.bvid:
        bvids.append(parse_bvid(args.bvid))
    if args.list:
        with open(args.list, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        bvids.append(parse_bvid(line))
                    except ValueError:
                        print(f"跳过无效行: {line}")

    if not bvids:
        parser.print_help()
        return

    print(f"共 {len(bvids)} 个视频")
    for bvid in bvids:
        await process_video(bvid, args.output_dir, args.whisper)

    print(f"\n全部完成! 文件保存在: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    asyncio.run(main())
