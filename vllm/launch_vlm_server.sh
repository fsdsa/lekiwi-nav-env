#!/bin/bash
# VLM Server launcher — delegates to root run_vllm_server.sh
#
# 사용법 (서버에서):
#   conda activate vllm
#   bash launch_vlm_server.sh
#
# GPU 메모리: 0.45 (VLA와 동시 추론용)
# 포트: 8000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_vllm_server.sh"
