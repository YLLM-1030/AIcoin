#!/bin/bash
# startup.sh — 一键启动全链路: VAD + ASR + Live2D
# 用法: source ~/mini_coin_venv/bin/activate && bash startup.sh

set -e

ARCH_DIR="/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/asr+2d8.2确认/配置文件"
ASR_SCRIPT="/mnt/c/Users/Autogram-coin/Desktop/新架构硬币/asr+2d8.2确认/mic_vad_asr_streaming_vad.py"
ASR_THRESHOLD=300
# 音源不再写死——由 ASR 管理页(18686) 的 asr_config.json 控制（听觉编码器）

FACS_PORT=18768
BRIDGE_PORT=18769
FRONTEND_PORT=18769

cleanup() {
    echo ""
    echo "[*] 停止所有服务..."
    kill $FACS_PID $BRIDGE_PID $ASR_PID 2>/dev/null
    wait 2>/dev/null
    echo "[*] 已退出"
}
trap cleanup EXIT INT TERM

echo "═══════════════════════════════════════════"
echo "  mini_coin 全链路启动"
echo "═══════════════════════════════════════════"

# 1. 启动 facs_server
echo "[1/3] 启动 VAD 引擎 (18768)..."
python3 "${ARCH_DIR}/emotion/facs_server.py" &
FACS_PID=$!
sleep 2

# 2. 启动桥 (18769/18770)
echo "[2/3] 启动 Live2D 桥 (18769)..."
python3 "${ARCH_DIR}/emotion/vad_bridge.py" &
BRIDGE_PID=$!
sleep 1

# 3. 打开 2D 页面
powershell.exe -Command "Start-Process 'http://localhost:18769'" 2>/dev/null || echo "[*] 请手动打开 http://localhost:18769"

# 4. 启动 ASR (前台, Ctrl+C 退出) — 音源/输出桥由 18686 管理页配置
echo "[3/3] 启动语音输入..."
echo "      管理页: http://localhost:18686  (选音源/输出桥)"
echo "      (大声说话，Ctrl+C 退出)"
echo "═══════════════════════════════════════════"
python3 "$ASR_SCRIPT" --threshold $ASR_THRESHOLD
