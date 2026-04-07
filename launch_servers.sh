#!/bin/bash
# VLM + VLA 서버 동시 실행 (A100 서버)
#
# GPU 메모리 분배 (A100 40GB):
#   VLM (vLLM, Qwen3-VL-8B bf16): ~29.8GB (gpu-memory-utilization 0.75)
#   VLA (Pi0-FAST):                ~8.1GB
#   합계: ~37.9GB
#
# 사용법:
#   bash launch_servers.sh                           # 둘 다
#   bash launch_servers.sh vlm                       # VLM만
#   bash launch_servers.sh vla --checkpoint <path>   # VLA만
#
# 로그:
#   tail -f logs/vlm_server.log
#   tail -f logs/vla_server.log
#
# 종료:
#   bash launch_servers.sh stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_DIR="${HOME}/miniconda3"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

VLM_PORT=8000
VLA_PORT=8002
VLA_CHECKPOINT="${VLA_CHECKPOINT:-}"

# ── 인자 파싱 ──
MODE="${1:-all}"
shift || true

# VLA checkpoint 인자
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) VLA_CHECKPOINT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ── 함수 ──

stop_servers() {
    echo "  Stopping servers..."
    # VLM (vLLM)
    VLM_PID=$(lsof -ti :${VLM_PORT} 2>/dev/null || true)
    if [ -n "$VLM_PID" ]; then
        kill $VLM_PID 2>/dev/null && echo "  VLM (PID $VLM_PID) stopped" || true
    fi
    # VLA
    VLA_PID=$(lsof -ti :${VLA_PORT} 2>/dev/null || true)
    if [ -n "$VLA_PID" ]; then
        kill $VLA_PID 2>/dev/null && echo "  VLA (PID $VLA_PID) stopped" || true
    fi
    echo "  Done."
}

start_vlm() {
    echo "  Starting VLM server (port ${VLM_PORT})..."

    # 포트 이미 사용 중인지 확인
    if lsof -ti :${VLM_PORT} >/dev/null 2>&1; then
        echo "  [WARN] Port ${VLM_PORT} already in use — VLM server may be running"
        return
    fi

    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate vllm

    nohup python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --dtype bfloat16 \
        --port ${VLM_PORT} \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.75 \
        --trust-remote-code \
        > "${LOG_DIR}/vlm_server.log" 2>&1 &

    VLM_PID=$!
    echo "  VLM server started (PID ${VLM_PID})"
    echo "  Log: ${LOG_DIR}/vlm_server.log"
}

start_vla() {
    if [ -z "${VLA_CHECKPOINT}" ]; then
        echo "  [ERROR] VLA checkpoint not specified!"
        echo "  Usage: bash launch_servers.sh vla --checkpoint <path>"
        echo "  Or:    VLA_CHECKPOINT=<path> bash launch_servers.sh"
        return 1
    fi

    echo "  Starting VLA server (port ${VLA_PORT})..."

    if lsof -ti :${VLA_PORT} >/dev/null 2>&1; then
        echo "  [WARN] Port ${VLA_PORT} already in use — VLA server may be running"
        return
    fi

    # lerobotpi0v2 env (lerobot 0.5.0)
    source /home/jovyan/yes/etc/profile.d/conda.sh
    conda activate lerobotpi0v2

    nohup python "${SCRIPT_DIR}/vllm/vla_inference_server.py" \
        --model "${VLA_CHECKPOINT}" \
        --port ${VLA_PORT} \
        --host 0.0.0.0 \
        > "${LOG_DIR}/vla_server.log" 2>&1 &

    VLA_PID=$!
    echo "  VLA server started (PID ${VLA_PID})"
    echo "  Log: ${LOG_DIR}/vla_server.log"
}

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=120
    local elapsed=0

    echo -n "  Waiting for ${name} (port ${port})..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo " ready! (${elapsed}s)"
            return 0
        fi
        # VLM uses /v1/models endpoint
        if curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo " ready! (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo " TIMEOUT (${max_wait}s)"
    return 1
}

# ── 실행 ──

echo "============================================================"
echo "  LeKiwi Inference Servers"
echo "  Mode: ${MODE}"
echo "============================================================"

case "${MODE}" in
    stop)
        stop_servers
        ;;
    vlm)
        start_vlm
        wait_for_server ${VLM_PORT} "VLM"
        ;;
    vla)
        start_vla
        wait_for_server ${VLA_PORT} "VLA"
        ;;
    all)
        # VLM 먼저 (GPU 메모리 선점)
        start_vlm
        echo "  VLM 로딩 대기 (60초 후 VLA 시작)..."
        sleep 60
        wait_for_server ${VLM_PORT} "VLM"

        # VLA는 남은 GPU 메모리 사용
        start_vla
        wait_for_server ${VLA_PORT} "VLA"

        echo ""
        echo "  Both servers running!"
        echo "    VLM: http://0.0.0.0:${VLM_PORT}/v1/chat/completions"
        echo "    VLA: http://0.0.0.0:${VLA_PORT}/infer"
        ;;
    *)
        echo "  Unknown mode: ${MODE}"
        echo "  Usage: bash launch_servers.sh [all|vlm|vla|stop] [--checkpoint <path>]"
        exit 1
        ;;
esac

echo ""
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null \
    && echo "  (GPU memory after launch)"
echo "============================================================"
