# SSE 架构说明（跨语言·可复用模式）

---

## 一句话

**发送方** HTTP POST 把数据丢进"桥" → 桥上的 SSE 接收方**自动拿到数据并处理**。发送方不关心谁在听，接收方不关心谁在发。

---

## 项目中的标准实现：pipeline_server → TTS

### 发送方 (pipeline_server.py, Python)

```python
# 表达队列消费 — 行 1646
requests.post("http://localhost:8000/api/queue_text",
    json={"text": item["arg"]}, timeout=30)
```

- 协议：HTTP POST
- 地址：`http://localhost:8000/api/queue_text`（硬编码，不走桥号）
- 数据：`{"text": "要说的回复文本"}`
- 发完即忘，不等待结果

---

### 中转方 (tts_server.py, Python FastAPI)

**收到 POST 后两步：** 入队 + SSE 推送

```python
# 行 414 — 入队
@app.post('/api/queue_text')
def queue_text(data: dict):
    text = data.get('text', '')
    msg = {'text': text}
    for q in _sse_queues:      # 推给所有在线 SSE 客户端
        q.put_nowait(msg)
    return {'ok': True}

# 行 430 — SSE 推送
@app.get('/api/queue_sse')
async def queue_sse(request: Request):
    q = asyncio.Queue()
    _sse_queues.append(q)
    async def generate():
        while True:
            data = await asyncio.wait_for(q.get(), timeout=30)
            yield f"data: {json.dumps(data)}\n\n"
    return StreamingResponse(generate(), media_type='text/event-stream')
```

- `queue_text` 负责接收 → `queue_sse` 负责推送
- 多客户端：`_sse_queues` 是列表，每个连上的 TTS 页面都会收到

---

### 接收方 (tts_server.py 页面, JavaScript)

```javascript
// 行 280
var evtSource = new EventSource('/api/queue_sse');
evtSource.onmessage = function(event){
    var d = JSON.parse(event.data);
    document.getElementById('text').value = d.text;
    synth();  // 自动合成语音
};
```

- 浏览器用 `EventSource` 读 SSE
- 收到文本 → 填入 textarea → 自动调 `synth()` → 播放
- 完全自动，不需要按钮

---

## Python 侧正确读 SSE 的方式

```python
import http.client, codecs, json

# 1. 建连接
conn = http.client.HTTPConnection('localhost', 19000, timeout=60)
conn.request('GET', '/events')
resp = conn.getresponse()

# 2. 增量解码器 — 自动处理多字节 UTF-8，半字不丢
decoder = codecs.getincrementaldecoder('utf-8')()
buf = ''

# 3. 循环读
while True:
    chunk = resp.read(4096)           # 大块读
    if not chunk: break
    buf += decoder.decode(chunk)      # 增量解码

    while '\n\n' in buf:              # SSE 事件边界
        raw, buf = buf.split('\n\n', 1)
        for line in raw.split('\n'):
            if line.startswith('data: '):
                d = json.loads(line[6:])
                process(d)            # 处理数据
```

**关键：** `codecs.getincrementaldecoder('utf-8')` — 字节在 UTF-8 字边界被切开时，自动缓存残片，等下一个 chunk 到了再拼完整。**不丢中文。**

---

## 错误写法（本项目的教训）

```python
# ❌ 逐字节 decode，中文被拆成单字节 → errors='ignore' 直接丢了
buf += chunk.decode('utf-8', errors='ignore')
```

## 其他错误写法

```python
# ❌ 字节 buffer 拼到 \n\n 再 decode → 同一次 read 内 OK，
#    但跨 read 边界的中文会被切开
buf = b''
chunk = resp.read(4096)
buf += chunk
# 如果 buf 末尾是一个 3 字节中文的前 2 字节，
# split(b'\n\n') 会从字中间切开 → decode 报错
```

---

## 对比：项目里不用 SSE 的读桥方式

pipeline_server 读桥不走 SSE，走轮询：

```python
# 每 0.1 秒 GET 一次 /source_last
r = requests.get(f"http://localhost:{port}/source_last", timeout=2).json()
```

- 优点：简单，不丢数据
- 缺点：有轮询延迟

---

## 本项目的 SSE 桥

| 桥名 | 端口 | SSE 端点 | 提供方 |
|------|------|----------|--------|
| ~~弹幕桥~~ | 19000 | ~~/events~~ **已废弃（2026-08-08）** | blc_collector.py |
| TTS 桥 | 8000 | `/api/queue_sse` | tts_server.py (FastAPI) |
| 弹幕接收 | 18776 | `/events` | danmaku_server.py (前端) |

### 弹幕桥为何弃用 SSE

弹幕消费端（danmaku_server → 19000）曾经尝试 SSE，最终放弃：

1. **长连接状态管理坑**：socket 超时后 `http.client` 无法继续读（`cannot read from timed out object`），连接被误判断开；
2. **维护成本**：需要队列 + 推送线程 + 心跳 + 断线清理，复杂度与"单消费者读弹幕"不匹配；
3. **推即弃无重放**：消费者断线期间的消息永远丢失，无法补救。

**替代方案：读后清空（wait 哨兵）**——消费者每 0.5s `GET /take`（桥端原子「返回内容 + 置 wait」），读到 `wait` 即无新消息。单消费者天然不重复、无连接状态、断线零成本（下次轮询接着读）。SSE 通用模式仍适用于 TTS/搜索等「一推多收」场景。

---

## 通用模式总结

```
┌──────────┐   POST /api/queue   ┌──────────┐   GET /events (SSE)   ┌──────────┐
│  发送方   │ ──────────────────→ │   桥服务   │ ──────────────────→ │  接收方   │
│ (Python)  │  json={"text":...}  │ (HTTP srv)│  data: {"text":...}  │ (JS/Py)  │
└──────────┘                     └──────────┘                     └──────────┘
```

1. 发送方 POST JSON → 桥的 `/api/queue` 端点
2. 桥把数据放进队列
3. 桥的 `/events` SSE 端点推给所有在线接收方
4. 接收方收到后执行后续动作
