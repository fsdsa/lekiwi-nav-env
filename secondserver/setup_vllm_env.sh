#!/usr/bin/env bash
# =============================================================================
# vllm env 설치 — VLM 서빙 (Qwen3-VL-8B-Instruct on vLLM, port 8000)
#
# 검증 환경 (A100, 2026-06-18 실측):
#   Python 3.11.14 · vllm 0.17.0 · torch 2.10.0 · transformers 4.57.6
#   numpy 2.2.6 · tokenizers 0.22.2
#
# 사용:  bash secondserver/setup_vllm_env.sh
# =============================================================================
set -uo pipefail

ENV_NAME="${ENV_NAME:-vllm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$SCRIPT_DIR/env_vllm.lock.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"

if [ -f "$LOCK" ]; then
  echo "[vllm] lockfile로 정확 버전 설치: $LOCK"
  pip install -r "$LOCK"
else
  echo "[vllm] lockfile 없음 → 기본 버전 설치 (torch/transformers 자동 해석)"
  pip install vllm==0.17.0
fi

cat <<'EOF'

[vllm] 설치 완료.
모델은 최초 서빙 시 HuggingFace에서 자동 다운로드됩니다 (~30GB, 캐시: ~/.cache/huggingface/hub).
서빙:
  python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-VL-8B-Instruct \
      --dtype bfloat16 --port 8000 \
      --max-model-len 4096 --gpu-memory-utilization 0.75 --trust-remote-code
EOF
