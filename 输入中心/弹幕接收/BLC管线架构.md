# BLC (blivechat) 弹幕姬 — 管线架构

> 版本 1.10.3 · PyInstaller 封包 · Tornado 后端 + Vue.js 前端

## 整体数据流

```
B站开放平台 API
  ↓ POST /v2/app/start (身份码 + app_id)
  ↓ 返回 {wss_link, auth_body, game_id}
  ↓
BLC 后端 (api/open_live.py)
  ├─ WebSocket → B站弹幕服务器 (wss_link)
  ├─ 收 DANMU_MSG / SUPER_CHAT_MESSAGE / SEND_GIFT
  ├─ 格式化为数组 [avatar, timestamp, authorName, authorType, content, ...]
  └─ 推给 BLC relay
       ↓
BLC Relay (api/chat.py) — Tornado WebSocketHandler
  ├─ 前端连入 → 发 {cmd:1, data:{roomKey:"xxx"}}
  ├─ 后端 join 对应房间
  ├─ 转发数据: {cmd:2, data:[...]} (弹幕) / {cmd:5, data:{...}} (SC)
  └─ 心跳: {cmd:0, data:{}}
       ↓
BLC 前端 (Vue.js, index.js)
  ├─ new WebSocket("ws://localhost:12450/api/chat")
  ├─ onWsMessage → parse cmd → Vue 状态更新
  ├─ 弹幕: cmd=2 → data[2]=作者名, data[4]=内容
  ├─ SC: cmd=5 → data.uname, data.message, data.price
  └─ DOM 渲染: .chat-item span 显示
```

## 核心文件 (从日志和源码反推)

| 文件 | 功能 |
|------|------|
| `api/open_live.py` | 开放平台 HTTP 代理 + B站 WS 客户端 |
| `api/chat.py` | `/api/chat` WebSocket relay |
| `services/chat.py` | ChatClient 管理，from_dict 消息解析 |
| `services/chat_*.py` | Web/OpenLive 双模式 ChatClient |
| `frontend/dist/js/*.js` | Vue.js 前端，index.js 中 ChatClientDirectOpenLive |
| `config.ini` | 端口/app_id/翻译等配置 |

## 开放平台流程 (关键)

1. **前端触发**: 用户输入身份码 → POST `/api/open_live/start_game` {code:"身份码", room_id:xxx}
2. **后端代理**: 无 app_id 时转发到作者公共服务器 → 公共服务器调 B站 `/v2/app/start`
3. **B站响应**: `{wss_link: [...], auth_body: "base64...", game_id: "uuid"}`
4. **WS 连接**: 后端连接 wss_link → 发送 auth_body → 收数据 → 每 20s 心跳 `/v2/app/heartbeat`
5. **断线**: 180s 无心跳自动关闭；前端断线时重新 start_game

## 前端 WS 协议

| cmd | 方向 | 含义 | 数据结构 |
|-----|------|------|----------|
| 1 | 前端→后端 | 加入房间 | `{roomKey:"xxx"}` |
| 0 | 后端→前端 | 心跳 | `{}` |
| 2 | 后端→前端 | 弹幕 | `[avatarUrl,timestamp,authorName,authorType,content,privilegeType,...]` |
| 3 | 后端→前端 | 礼物 | `{uname, giftName, giftNum, ...}` |
| 5 | 后端→前端 | SC | `{uname, message, price, ...}` |
| 8 | 后端→前端 | 错误 | `{type, msg}` |

## B站开放平台原始消息

| 消息类型 | 关键字段 |
|----------|----------|
| LIVE_OPEN_PLATFORM_DM | `uname`(用户名), `msg`(内容), `fansMedalName`, `fansMedalLevel` |
| LIVE_OPEN_PLATFORM_SEND_GIFT | `uname`, `giftName`, `giftNum`, `price`, `paid` |
| LIVE_OPEN_PLATFORM_SUPER_CHAT | `uname`, `message`, `rmb`, `startTime`, `endTime` |

## 关键配置 (config.ini)

```ini
[app]
host = 127.0.0.1      # 监听地址
port = 12450           # 监听端口

# 无 app_id → 转发请求到作者公共服务器
open_live_access_key_id =
open_live_access_key_secret =
open_live_app_id = 0
```

## BLC 断线和重连

- B站 WS 15s 无消息 → 前端超时断开
- 心跳 `/v2/app/heartbeat` 失败 (500) → 前端收到错误 → 自动 reconnect
- 重连: 重新调用 `/v2/app/start` → 获取新 wss_link → 连接
- B站偶尔返回 code=7007 (身份码错误) 或 code=7010 (连接数超限)
