# B站视频字幕/文案获取工具

> 创建: 2026-06-28 | 用于获取 B站视频的 CC 字幕文案

---

## 工具列表

| 文件 | 用途 |
|------|------|
| `get_vedal_subs.py` | 按 BV 号或用户 UID 批量获取字幕（带随机间隔防封） |
| `bili_subtitle.py` | 单视频字幕获取（简洁版） |
| `bili_crawl.py` | 批量获取 + Whisper 音频转写兜底 |

---

## 使用方法

### 安装依赖

```bash
pip install requests
```

### 获取 Cookie（必选）

1. 浏览器打开 https://www.bilibili.com 并登录
2. F12 → 应用/Application → Cookies → bilibili.com
3. 复制 `SESSDATA` 的值
4. 写入脚本或创建 `sessdata.txt`

### 单视频测试

```powershell
python get_vedal_subs.py --bvid BV1xXXnYkEDd
```

### 批量获取用户视频字幕

```powershell
python get_vedal_subs.py --uid 3546729368520811 --max 20
```

Vedal 官方频道 UID: `3546729368520811`（110万粉，656个视频）

---

## 踩坑记录

### 1. Cookie 必须
B站新版 API 需要 `SESSDATA` cookie，否则无法获取字幕信息。
- 错误: `Credential 类未提供 sessdata 或者为空`

### 2. 限流严重
B站 API 限流码 `-799`（请求过于频繁）。
- 解决: 每次请求间隔 **8~15 秒随机**，不可固定间隔
- 发现被限流后等 30 秒再重试

### 3. WBI 签名
部分新接口（`/x/player/wbi/v2`）需要 WBI 签名算法。
- 字幕接口目前 Cookie 方式可用，不需要额外实现 WBI
- 如果后续失效，需要实现 `bilibili_api` 库的 WBI 签名

### 4. bilibili-api-python 库问题
第三方库 `bilibili-api-python` v17+ 强制要求凭证，匿名无法使用。
- 改用纯 `requests` + Cookie 直接调 B站原生 API

### 5. 字幕格式
B站 CC 字幕返回 JSON 格式：
```json
{"body": [{"from": 0, "to": 5, "content": "文本", "sid": 1}, ...]}
```
- 直接提取 `body[].content` 即可
- 先选中文字幕（如果有），没有则取第一个语言

### 6. Vedal 频道字幕情况
Vedal 频道部分视频有中文字幕（搬运/翻译版），部分没有。
- 有字幕的下载为独立 TXT 文件
- 同时汇总到 `_all_subtitles.txt` 合集
- 无字幕的视频可后续用 Whisper 转写（需 yt-dlp + openai-whisper）

---

## 文件说明

- `get_vedal_subs.py` — 主工具（推荐使用）
- `bili_subtitle.py` — 单视频简洁版
- `bili_crawl.py` — 批量 + Whisper 兜底版
- `sessdata.txt` — Cookie 存放文件（不提交）
