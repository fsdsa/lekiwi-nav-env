#!/usr/bin/env bash
# =============================================================================
# lerobotpi0v2 env 설치 — VLA 파인튜닝/서빙 (Pi0.5 on LeRobot 0.5.0, port 8002)
#
# 검증 환경 (A100, 2026-06-18 실측):
#   Python 3.12.12 · lerobot 0.5.0 (@v0.5.0 tag, commit 00b662de)
#   torch 2.10.0 · torchvision 0.25.0 · torchcodec 0.10.0 · transformers 5.3.0
#   fastapi 0.135.1 · uvicorn 0.41.0 · sentencepiece 0.2.1 · tiktoken 0.12.0
#   accelerate 1.13.0
#
# 주: 운영 서버는 이 env가 ~/yes/envs/lerobotpi0v2 에 있으나, 위치는 임의이며
#     아래는 기본 conda base(~/miniconda3/envs)에 생성한다. 실행 스크립트의
#     하드코딩된 ~/yes/envs/lerobotpi0v2/bin 경로는 실제 위치에 맞게 조정할 것.
#
# 사용:  bash secondserver/setup_lerobotpi0v2_env.sh
# =============================================================================
set -uo pipefail

ENV_NAME="${ENV_NAME:-lerobotpi0v2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$SCRIPT_DIR/env_lerobotpi0v2.lock.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

if [ -f "$LOCK" ]; then
  echo "[lerobotpi0v2] lockfile로 정확 버전 설치: $LOCK"
  pip install -r "$LOCK"
else
  echo "[lerobotpi0v2] lockfile 없음 → 기본 버전 설치 (torch/transformers 자동 해석)"
  pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git@v0.5.0"
  pip install fastapi uvicorn
fi

# HuggingFace 로그인 (google/paligemma-3b-pt-224 gated repo 접근 필수)
echo "[lerobotpi0v2] huggingface-cli login 필요 (paligemma gated repo)."
huggingface-cli login || echo "  -> 나중에 'huggingface-cli login' 을 수동 실행하세요."
