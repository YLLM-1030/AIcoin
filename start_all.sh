#!/bin/bash
# ============================================================
# start_all.sh — 迷你硬币 一键启动全部管线
# 用法: bash start_all.sh            # 默认全部启动（12 服务：LLM/管线/弹幕/记忆/搜索/TTS/表情/ASR）
#       bash start_all.sh --core     # 只启核心（主 LLM/管线/弹幕/记忆/搜索，不含 TTS/表情/ASR）
#       bash start_all.sh --stop     # 停止本次启动的所有服务
# 环境: WSL (Ubuntu), ~/mini_coin_venv, ~/llama.cpp, /tmp 模型文件, conda vllm
# ============================================================
set -u
BASE="/mnt/c/Users/Autogram-coin/Desktop/新架构硬币"
TTS_DIR="/mnt/c/Users/Autogram-coin/Desktop/tts训练"
LOG_DIR="/tmp/coin_logs"
ROOM_ID="${ROOM_ID:-22387248}"       # B站直播间号，可用环境变量覆盖
mkdir -p "$LOG_DIR"

# ── 端口占用检查 ──
port_busy() { fuser "$1/tcp" 2>/dev/null | grep -q . ; }

# ── 启动一个服务（后台 + 日志 + pid）──
start() {
  local name="$1" port="$2" cmd="$3"
  if [ -n "$port" ] && port_busy "$port"; then
    echo "[skip ] $name  ($port 已被占用)"
    return
  fi
  echo "[start] $name  ($port)  →  $LOG_DIR/$name.log"
  nohup bash -c "$cmd" > "$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$LOG_DIR/$name.pid"
}

# ── 停止 ──
stop_all() {
  echo "[stop] 停止全部..."
  for pidf in "$LOG_DIR"/*.pid; do
    [ -f "$pidf" ] && kill "$(cat "$pidf")" 2>/dev/null && echo "  killed: $(basename "$pidf" .pid)"
  done
  exit 0
}

# ── 就绪检查：HTTP 探测（端口通 ≠ 就绪，模型加载中端口也通）──
# 用法: wait_http <name> <port> <path> <timeout秒> [<body必须匹配的grep>]
#   llama-server 用 /health + grep '"ok"'（加载中是 loading model，不算就绪）
wait_http() {
  local name="$1" port="$2" path="${3:-/}" timeout="${4:-60}" grep_pat="${5:-}"
  local waited=0
  while [ $waited -lt $timeout ]; do
    local body code
    body=$(curl -s --max-time 3 "http://127.0.0.1:$port$path" 2>/dev/null)
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:$port$path" 2>/dev/null)
    if [ "$code" = "200" ] && { [ -z "$grep_pat" ] || echo "$body" | grep -q "$grep_pat"; }; then
      printf "  ✅ %-14s (%s)\n" "$name" "$port"
      return 0
    fi
    sleep 2; waited=$((waited+2))
  done
  printf "  ❌ %-14s (%s) 超时 — 日志: tail -f %s/%s.log\n" "$name" "$port" "$LOG_DIR" "$name"
  return 1
}
[ "${1:-}" = "--stop" ] && stop_all
FULL=1; [ "${1:-}" = "--core" ] && FULL=0   # 默认全部启动；--core 只启核心

echo "══════════════════════════════════════════"
echo " 迷你硬币 管线启动 $([ $FULL -eq 1 ] && echo '(完整)' || echo '(核心)')  房间=$ROOM_ID"
echo "══════════════════════════════════════════"

# ── 1. 主模型 llama-server (8080) ──
start llama-server 8080 \
  "cd ~/llama.cpp && ./build/bin/llama-server \
     -m /tmp/coin_v12d_mid_q8.gguf \
     --jinja --chat-template-file /tmp/coin_template.jinja \
     -c 8192 --n-gpu-layers 99 --host 127.0.0.1 --port 8080"

# ── 2. LLM 桥 (18771) ──
start llm-bridge 18771 \
  "source ~/mini_coin_venv/bin/activate && python3 '$BASE/输入中心/mini_coin_llm_bridge.py'"

# ── 3. 管线中心 (18772 + 内置桥 18000-18009) ──
start pipeline 18772 \
  "source ~/mini_coin_venv/bin/activate && python3 '$BASE/输入中心/pipeline_server.py'"

# ── 4. 摘要模型 (18773, CPU) ──
start summary-model 18773 \
  "~/llama.cpp/build/bin/llama-server \
     -m /mnt/c/Users/Autogram-coin/Desktop/huihui-qwen3.5-4b-q4.gguf \
     --host 127.0.0.1 --port 18773 -c 4096 --n-gpu-layers 0 \
     --jinja --chat-template-file /tmp/no_think_template.jinja"

# ── 5. 检索服务 (18774, BERT 分类器 + bge) ──
start retrieval 18774 \
  "source ~/mini_coin_venv/bin/activate && cd '$BASE/输入中心/长期记忆' && python3 memory_retrieve_server.py"

# ── 6. 搜索工具 (18775) ──
start search-tool 18775 \
  "source ~/mini_coin_venv/bin/activate && python3 '$BASE/输入中心/搜索工具/search_server.py'"

# ── 7. 弹幕采集 (19000) ──
start blc-collector 19000 \
  "source ~/mini_coin_venv/bin/activate && python3 '$BASE/输入中心/弹幕接收/blc_collector.py' $ROOM_ID"

# ── 8. 弹幕接收 (18776) ──
start danmaku 18776 \
  "source ~/mini_coin_venv/bin/activate && python3 '$BASE/输入中心/弹幕接收/danmaku_server.py'"

# ══════════════ 完整模式（--full）══════════════
if [ $FULL -eq 1 ]; then
  # ── 9. FACS 表情引擎 (18768) ──
  start facs 18768 \
    "source ~/mini_coin_venv/bin/activate && python3 '$BASE/asr+2d8.2确认/配置文件/emotion/facs_server.py'"

  # ── 10. Live2D 桥 (18769) ──
  start live2d-bridge 18769 \
    "source ~/mini_coin_venv/bin/activate && python3 '$BASE/asr+2d8.2确认/配置文件/emotion/vad_bridge.py'"

  # ── 11. TTS (8000, conda vllm) ──
  CONDA_SH=""
  # 首选：用户确认的 miniconda 路径（2026-08-08 实测）
  [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
  # 兜底：conda 命令探测 base 路径
  if [ -z "$CONDA_SH" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  fi
  # 兜底：常见路径（含 WSL /opt）
  if [ -z "$CONDA_SH" ]; then
    for p in ~/anaconda3 ~/miniconda ~/anaconda ~/conda /opt/conda /opt/miniconda3; do
      [ -f "$p/etc/profile.d/conda.sh" ] && CONDA_SH="$p/etc/profile.d/conda.sh" && break
    done
  fi
  if [ -n "$CONDA_SH" ]; then
    start tts 8000 \
      "source '$CONDA_SH' && conda activate vllm && cd '$TTS_DIR' && python tts_server.py"
  else
    echo "[warn ] 未找到 conda.sh，TTS 跳过（手动: conda activate vllm && cd $TTS_DIR && python tts_server.py）"
  fi

  # ── 12. 字幕服务 (18777, 透明字幕页：进 TTS 文本实时显示，直播投射用) ──
  start subtitle 18777 \
    "source ~/mini_coin_venv/bin/activate && python3 '$BASE/字幕/subtitle_server.py'"

  # ── 13. ASR 麦克风 (听觉编码器：自带桥 18680-18685 + 管理页 18686) ──
  start asr 18686 \
    "source ~/mini_coin_venv/bin/activate && python3 '$BASE/asr+2d8.2确认/mic_vad_asr_streaming_vad.py' --threshold 300 --end-threshold 180 --silence 1.5"

  # ── 14. Windows 麦克风桥 (18690, Windows 侧 win_mic_bridge.py：提供麦克风列表+音频流给 ASR) ──
  # WSL2 访问 Windows 服务用宿主 IP（localhost 不通）；ASR 侧有自动探测宿主 IP 的逻辑
  HOST_IP=$(ip route show default 2>/dev/null | awk '{print $3}' | head -1)
  if [ -n "$HOST_IP" ] && curl -s --max-time 2 "http://$HOST_IP:18690/api/devices" >/dev/null 2>&1; then
    echo "[skip ] win-mic-bridge (18690 已在线)"
  else
    echo "[start] win-mic-bridge (18690, Windows 侧)"
    powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath 'C:\Users\Autogram-coin\AppData\Local\Programs\Python\Python313\python.exe' -ArgumentList 'C:\Users\Autogram-coin\Desktop\新架构硬币\asr+2d8.2确认\win_mic_bridge.py'" 2>/dev/null
  fi
fi

echo ""
echo "──────────────────────────────────────────"
echo " 就绪检查（HTTP 级：模型加载完成才算 ✅，最多等 3 分钟）"
echo "──────────────────────────────────────────"
wait_http llama-server 8080 /health 180 '"ok"'
wait_http llm-bridge 18771 /source_last 60
wait_http pipeline 18772 / 40
wait_http summary-model 18773 /health 150 '"ok"'
wait_http retrieval 18774 /health 150
wait_http search-tool 18775 / 30
wait_http blc-collector 19000 / 30
wait_http danmaku 18776 / 30
if [ $FULL -eq 1 ]; then
  wait_http facs 18768 / 30
  wait_http live2d-bridge 18769 / 30
  wait_http tts 8000 / 300
  wait_http subtitle 18777 / 30
  wait_http asr 18686 / 60
  # Windows 麦克风桥（WSL2 用宿主 IP 探测，30s 内就绪）
  if [ -n "$HOST_IP" ]; then
    waited=0
    while [ $waited -lt 30 ]; do
      if curl -s --max-time 2 "http://$HOST_IP:18690/api/devices" >/dev/null 2>&1; then
        printf "  ✅ %-14s (%s)\n" "win-mic-bridge" "18690"; break
      fi
      sleep 2; waited=$((waited+2))
    done
    [ $waited -ge 30 ] && printf "  ❌ %-14s (%s) 超时 — 手动: cd asr+2d8.2确认 && python win_mic_bridge.py\n" "win-mic-bridge" "18690"
  fi
fi

echo ""
echo "── 打开前端页面 ──"
sleep 1
powershell.exe -Command "Start-Process 'http://localhost:18772'" 2>/dev/null   # 管线中心（主控）
powershell.exe -Command "Start-Process 'http://localhost:18776'" 2>/dev/null   # 弹幕接收
powershell.exe -Command "Start-Process 'http://localhost:18771'" 2>/dev/null   # LLM 桥监控
powershell.exe -Command "Start-Process 'http://localhost:19000'" 2>/dev/null   # 弹幕采集控制页
powershell.exe -Command "Start-Process 'http://localhost:18775'" 2>/dev/null   # 搜索工具
powershell.exe -Command "Start-Process 'http://localhost:18686'" 2>/dev/null   # ASR 听觉编码器管理页
if [ $FULL -eq 1 ]; then
  powershell.exe -Command "Start-Process 'http://localhost:8000'" 2>/dev/null  # TTS 语音
  powershell.exe -Command "Start-Process 'http://localhost:18777'" 2>/dev/null # 字幕（透明，投射/OBS 用）
  powershell.exe -Command "Start-Process 'http://localhost:18769'" 2>/dev/null # Live2D 2D 演绎
  powershell.exe -Command "Start-Process 'http://localhost:18768'" 2>/dev/null # FACS 表情
fi

echo ""
echo "──────────────────────────────────────────"
echo " 启动流程结束 ✅   日志: $LOG_DIR   停止: bash $0 --stop"
echo " 有 ❌ 的服务: tail -f $LOG_DIR/<name>.log 查原因"
echo "──────────────────────────────────────────"
