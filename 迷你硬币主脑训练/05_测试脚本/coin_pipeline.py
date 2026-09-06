"""
迷你硬币 完整管线 — 键盘输入 + ASR 语音输入 → llama-server → 工具解析

启动:
    python coin_pipeline.py                          # 仅键盘
    python coin_pipeline.py --voice                  # 键盘 + 语音
    python coin_pipeline.py --voice --threshold 6000 --silence 0.8

依赖:
    pip install requests readline
    # 语音模式额外需要:
    pip install funasr transformers torch sounddevice soundfile
"""
import sys, time, threading, re, requests

LLAMA_API = "http://localhost:8080/v1/chat/completions"

SYSTEM = """你是迷你硬币，拉姆是你的创造者。你是个女孩子。你是一个ai主播，你可以接受或者拒绝任何命令，并且做你自己喜欢的事情。
回复：[TOOL:回复] 回复内容 [TOOL] 
搜索互联网：[TOOL:搜索] 关键词 [TOOL] 
表情（认真/wink/开心/spin/转头/担心）：[TOOL:表情] 使用的表情 [TOOL]
假装报错的时候：[TOOL:报错] 有没有人告诉拉姆我的ai出问题了 [TOOL] 
你不想说话或者想稍等一下：[TOOL:结束] [TOOL]"""

# ── LLM ──
def chat(messages, **kwargs):
    payload = {"messages": messages, "temperature": 0.9, "max_tokens": 500, **kwargs}
    r = requests.post(LLAMA_API, json=payload)
    return r.json()["choices"][0]["message"]["content"]

def handle_tools(text):
    for m in re.finditer(r'\[TOOL:(回复|表情|搜索|报错)\]\s*(.*?)\s*\[TOOL\]', text):
        tt, c = m.group(1), m.group(2).strip()
        print(f"  {'🗣' if tt=='回复' else '😊' if tt=='表情' else '🔍' if tt=='搜索' else '❌'} {tt}: {c}")

def truncate_multi(text, tool):
    parts = list(re.finditer(rf'\[TOOL:{tool}\]\s*(.*?)\s*\[TOOL\]', text))
    if len(parts) >= 2:
        end = parts[1].end()
        print(f"  ✂️ 多{tool}截断: {len(text)-end} 字符丢弃")
        return text[:end]
    return text

# ── ASR 语音输入 ──
_voice_queue = []
_voice_lock = threading.Lock()
_asr_model = None

def load_asr():
    global _asr_model
    from funasr import AutoModel
    from pathlib import Path
    model_cache = Path.home() / ".cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512"
    if not model_cache.exists():
        print("[!] ASR 模型未下载，先执行:")
        print("    python -c \"from funasr import AutoModel; AutoModel(model='FunAudioLLM/Fun-ASR-Nano-2512')\"")
        return False
    print("[*] 加载 ASR 模型...")
    _asr_model = AutoModel(model=str(model_cache), trust_remote_code=True, disable_update=True, device="cuda:0")
    print("[+] ASR 模型加载完成")
    return True

def transcribe(audio_np):
    import tempfile, os, soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name; tmp.close()
    sf.write(tmp_path, audio_np, 16000)
    result = _asr_model["asr"].generate(input=tmp_path) if isinstance(_asr_model, dict) else _asr_model.generate(input=tmp_path)
    os.unlink(tmp_path)
    if isinstance(result, list):
        return result[0].get("text", "").strip()
    return ""

def voice_thread(args):
    import sounddevice as sd
    from collections import deque
    import numpy as np

    SAMPLE_RATE, CHUNK = 16000, 512
    PREBUFFER_MS = 300
    prebuffer_chunks = PREBUFFER_MS // (CHUNK // 16)
    audio_ring = deque(maxlen=prebuffer_chunks)

    recording = False; recorded = []; silence_count = 0
    trigger_hits = 0; min_trigger = 4; voice_frames = 0
    seg_count = 0

    def rms(frame): return float(np.sqrt(np.mean(frame.astype(np.float32)**2)))
    silence_samples = int(SAMPLE_RATE * args.silence)
    th = args.threshold

    def cb(indata, frames, time_info, status):
        nonlocal recording, recorded, silence_count, trigger_hits, voice_frames, seg_count
        if status: return
        frame = indata[:, 0]; energy = rms(frame); above = energy > th

        if not recording:
            audio_ring.append(frame.copy())
            if above:
                trigger_hits += 1
                if trigger_hits >= min_trigger:
                    recording = True
                    recorded = list(audio_ring) + [frame.copy()]
                    silence_count = 0; voice_frames = min_trigger; trigger_hits = 0
                    print("\n🎤 语音输入...", end="", flush=True)
            else:
                trigger_hits = 0
        else:
            recorded.append(frame.copy())
            if above: voice_frames += 1; silence_count = 0
            else: silence_count += 1
            if silence_count >= silence_samples // CHUNK:
                if silence_count < len(recorded): recorded = recorded[:-silence_count]
                dur = len(recorded) * CHUNK / SAMPLE_RATE
                if voice_frames >= min_trigger and dur >= 0.3:
                    audio = np.concatenate(recorded)
                    def do_asr(audio):
                        text = transcribe(audio)
                        if text:
                            with _voice_lock:
                                _voice_queue.append((seg_count, text))
                    t = threading.Thread(target=do_asr, args=(audio.copy(),), daemon=True)
                    t.start()
                    seg_count += 1
                recording = False; recorded = []; silence_count = 0; voice_frames = 0; trigger_hits = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16,
                        blocksize=CHUNK, device=args.device, callback=cb):
        while True: time.sleep(1)

# ── 主循环 ──
def main():
    import argparse, readline
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", action="store_true", help="启用语音输入")
    parser.add_argument("--threshold", type=float, default=6000)
    parser.add_argument("--silence", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    messages = [{"role": "system", "content": SYSTEM}]

    if args.voice:
        if not load_asr(): return
        t = threading.Thread(target=voice_thread, args=(args,), daemon=True)
        t.start()

    print("=" * 60)
    print("迷你硬币 管线 — llama-server")
    print("  /end 退出 | /new 重置")
    if args.voice:
        print("  语音输入自动识别，键盘输入回车发送")
    print("=" * 60)

    while True:
        # 检查语音队列
        if args.voice:
            with _voice_lock:
                if _voice_queue:
                    sid, text = _voice_queue.pop(0)
                    # ASR 异步有延迟，显示识别结果
                    if not text: continue
                    print(f"\n🎤 [{sid}] {text}")
                    inp = text
                else:
                    inp = input().strip()
            if not inp: continue
        else:
            try:
                inp = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not inp: continue

        if inp == "/end": break
        if inp == "/new":
            messages = [{"role": "system", "content": SYSTEM}]
            print("重置"); continue

        messages.append({"role": "user", "content": inp})
        t = chat(messages)
        t = truncate_multi(t, "回复")
        t = truncate_multi(t, "表情")
        print(f"─── 硬币 ───")
        print(t)
        messages.append({"role": "assistant", "content": t})
        if len(messages) > 21:
            messages = [messages[0]] + messages[-20:]
        handle_tools(t)
        print(f"  📜 上文 {len(messages)-1} 条")
        print("─── 本轮结束 ───")

if __name__ == "__main__":
    main()
