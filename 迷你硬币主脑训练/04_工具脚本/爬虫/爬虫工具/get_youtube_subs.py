"""
YouTube Vedal 频道字幕批量获取
用法: python get_youtube_subs.py [--max 50] [--lang en]
"""
import subprocess, json, os, re, time, sys

YTDLP = r"C:\Users\Autogram-coin\.workbuddy\binaries\python\versions\3.13.12\Scripts\yt-dlp.exe"
OUT = r"C:\Users\Autogram-coin\Desktop\新建训练\youtube_subs"
os.makedirs(OUT, exist_ok=True)

# 参数
max_videos = 20
lang = "en"
if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--max" and i + 2 < len(sys.argv):
            max_videos = int(sys.argv[i + 2])
        elif arg == "--lang" and i + 2 < len(sys.argv):
            lang = sys.argv[i + 2]

print(f"获取 Vedal 频道前 {max_videos} 个视频 (字幕语言: {lang})...")
result = subprocess.run([
    YTDLP, "--flat-playlist", "--dump-json",
    "--playlist-end", str(max_videos),
    "https://www.youtube.com/@Vedal988/videos"
], capture_output=True, text=True, timeout=120)

if result.returncode != 0:
    print(f"获取列表失败: {result.stderr[:200]}")
    sys.exit(1)

videos = []
for line in result.stdout.strip().split("\n"):
    if line:
        videos.append(json.loads(line))

print(f"找到 {len(videos)} 个视频\n")

has_sub = 0
no_sub = 0
for i, v in enumerate(videos):
    vid = v["id"]
    title = v.get("title", "")[:60]
    print(f"[{i+1}/{len(videos)}] {vid} | {title}")

    try:
        subprocess.run([
            YTDLP, "--write-auto-subs", f"--sub-lang", lang,
            "--skip-download", "--sub-format", "vtt",
            "-o", f"{OUT}/{vid}",
            f"https://www.youtube.com/watch?v={vid}"
        ], capture_output=True, text=True, timeout=180)

        vtt_file = os.path.join(OUT, f"{vid}.{lang}.vtt")
        if os.path.exists(vtt_file):
            with open(vtt_file, "r", encoding="utf-8") as f:
                text = f.read()

            text_clean = re.sub(r"<[^>]+>", "", text)
            lines = []
            for l in text_clean.split("\n"):
                l = l.strip()
                if l and not l.startswith("WEBVTT") and not l.startswith("Kind:") \
                   and not l.startswith("Language:") and not re.match(r"^\d", l) \
                   and "-->" not in l:
                    lines.append(l)

            safe_title = re.sub(r'[\\/:*?<>|]', '_', title)[:30]
            txt_file = os.path.join(OUT, f"{vid}_{safe_title}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            os.remove(vtt_file)
            has_sub += 1
            print(f"  ✅ {len(lines)} 行")
        else:
            no_sub += 1
            print(f"  ❌ 无自动字幕")
    except Exception as e:
        no_sub += 1
        print(f"  ❌ 出错: {str(e)[:60]}")

    time.sleep(1)

print(f"\n完成! 有字幕: {has_sub}, 无字幕: {no_sub}")
print(f"全部保存在: {OUT}")
