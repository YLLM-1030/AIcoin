# Live2D 页面：音频播放与队列完成信号

> puppet_live2d_v2.html · 2026-08-06  
> 覆盖：WebSocket 连接、音频播放、TTS 完成信号上桥

## 一、WebSocket 连接

### 连接方式
```javascript
function connectWS() {
  const ws = new WebSocket('ws://localhost:18769');
  ws.onmessage = (e) => { 接收JSON };
  ws.onclose = () => setTimeout(connectWS, 2000);  // 掉线2s重连
  ws.onerror = () => ws.close();
}
```

### 调用点（仅一处）
模型加载完成后调用 `connectWS()`（行 ~2917）。**之前有两次调用（模型加载前+后）→ 双WS连接 → speak 收到两次 → 双重播放。已修复。**

### 消息类型
| type | 处理 |
|------|------|
| `emotion` | setEmotion() |
| `speak` | startLipSync(url) |
| `vad_status` | VAD 状态更新 |

## 二、音频播放 (startLipSync)

### 流程
```
WS 收到 {type:'speak', audio_url:'tts_cache/xxx.wav'}
  → startLipSync(url)
    → stopLipSync()          清上次状态
    → AudioContext.resume()  唤醒(首次suspended)
    → fetch(url) → decodeAudioData → audioBuffer
    → createBufferSource → connect(analyser) → connect(destination)
    → source.onended = stopLipSync   清理
    → source.start()                 开始播放
```

### 完成信号（计时器方案）
```javascript
var durMs = Math.ceil(audioBuffer.duration * 1000) + 20;  // 播放时长 + 20ms余量
setTimeout(function(){
  var port = localStorage.getItem('donePort') || '18004';   // 从配置读取桥端口
  fetch('http://localhost:' + port + '/text', {
    method:'POST',
    body: JSON.stringify({source:'tts_done', text:'done'})
  });                                                        // POST到管线内置桥
}, durMs);
```

- 精度：`audioBuffer.duration` 浮点数秒级，毫秒级精确
- 余量 200ms：兜底微小延迟
- 端口：页面左上角"完成信号桥"输入框，存 localStorage，重启保留

### 为什么用计时器而非 `source.onended`
`source.onended` 在 Quark 浏览器对 Blob URL 的 AudioBufferSourceNode 偶发不触发。计时器基于精确 duration，100% 可靠。

## 三、stopLipSync
```javascript
function stopLipSync() {
  _lipAnalyser = null;      // 清除分析器
  _lipData = null;           // 清除频谱数据
  _lipSmoothVal = 0;         // 口型平滑值归零
  delete _actionOverrides.ParamMouthOpenY;  // 解除口型覆盖
  _vocalizing = false;       // 发声状态标记
  _swingReturn = true;       // 身体回正
  _swingTurn.target = 0; _swingTilt.target = 0;
  _endNod();                 // 点头结束
  _updateIoBadge();          // 更新右上角状态徽章
  _reapplyEmotion();         // 重算表情回倾听版
}
```

**注意**：`stopLipSync()` 不调用 `source.stop()`，旧 AudioBufferSourceNode 会自然播完被 GC。

## 四、完成信号桥配置

### UI
页面左上角"说话模式"面板下方：
```
完成信号桥 [18004] [保存]
```

- `<input type="number">` 范围 18000-18999
- 保存到 `localStorage.setItem('donePort', value)`
- 页面加载时自动恢复

### 使用
管线表达队列 ⚙ 配置 → 选同一桥端口 → 完成信号通过该桥传递

## 五、数据流（完整链路）

```
管线(18772) inject
  → express_queue 入队
  → 消费者 POST /api/queue_text
    → TTS(8000) SSE推送 → TTS页面 synth()
      → /api/tts → synth_full() → pitch_shift → 存WAV到tts_cache/
        → _push_speak({type:'speak', audio_url}) → POST 18770/push
          → vad_bridge WS广播 → Live2D(18769) startLipSync()
            → audioBuffer.duration → setTimeout → POST 桥18004/text {tts_done:done}
              → 管线消费者 poll 桥 source_last → tts_done=done → 清桥 → pop → 下一任务
```

## 六、表情完成信号 (2026-08-07)

管线推送 `{type:"emotion",cmd:"表情",arg:"wink"}` → WS handler 随机选变体 → `playEmote()` → `setTimeout(时长+20ms)` → POST done 到同一桥。

### 变体映射 + 时长
| 管道名 | 变体 | 时长 |
|--------|------|------|
| wink | wink_AL/AR/BL/BR 随机 | 2020ms |
| 开心 | 开心左/右 随机 | 3000ms |
| spin | spin左/右 随机 | 1200ms |
| 转头 | 转头左/右 随机 | 6000ms |
| 担心 | 担心左/右 随机 | 5000ms |
| 认真 | 认真 | 3450ms |

### 代码
```javascript
var DUR = {wink:2020, 开心:3000, spin:1200, 转头:6000, 担心:5000, 认真:3450};
setTimeout(function(){
  var port = localStorage.getItem('donePort') || '18004';
  fetch('http://localhost:' + port + '/text', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source:'tts_done', text:'done'})
  });
}, (DUR[d.arg] || 2000) + 20);
```

- 和 TTS 共用 `tts_done` 信号，表达式消费阻塞机制完全相同
- 管线消费者等待 done → 清桥 → pop → 下一任务

## 七、踩坑记录

1. **双重 WS 连接**：connectWS() 被调两次，每条 speak 触发两次 startLipSync。修复：删除模型加载前的调用。
2. **桥梁 done 不清**：消费者读完 done 后没清桥，残值导致下一任务跳过等待。修复：消费完立刻 POST 清桥。
3. **onended 浏览器兼容**：Quark 浏览器 AudioBufferSourceNode.onended 不触发。改用 `audioBuffer.duration` + `setTimeout` 计时器方案。
