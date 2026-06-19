#!/usr/bin/env bash
# =============================================================================
# VIVA 추론 서버 환경 설치 (A100) — 이 파일 하나로 순서대로 전부
#   vllm (VLM, 포트 8000) + lerobotpi0v2 (VLA, 포트 8002 + 파인튜닝) + 모델 다운로드
#
# 클라이언트(env_isaaclab, Isaac Sim sim 실행)는 「Skill wise expert bases data generation」
# 매뉴얼의 setup.sh(데스크탑)로 이미 설치되어 있다고 가정한다.
#
# 검증 환경(2026-06-18): vllm(py3.11 / vllm0.17.0 / torch2.10.0 / transformers4.57.6)
#   lerobotpi0v2(py3.12 / lerobot0.5.0 / torch2.10.0 / transformers5.3.0 / fastapi·uvicorn)
# 전제: Miniconda + A100 GPU.
#
# 사용:  bash secondserver/setup_viva.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Miniconda 확인 (없으면 먼저 설치)
command -v conda >/dev/null 2>&1 || { echo "[ERR] conda 없음 → Miniconda 먼저 설치"; exit 1; }
# conda 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"

# vllm env 생성 (Python 3.11) — VLM 서버(8000)
conda create -n vllm python=3.11 -y
conda activate vllm
# vLLM 0.17.0 + 의존성 (lockfile로 정확 버전: torch 2.10.0, transformers 4.57.6)
pip install -r "$SCRIPT_DIR/env_vllm.lock.txt"

# lerobotpi0v2 env 생성 (Python 3.12) — VLA 서버(8002) + 파인튜닝
conda create -n lerobotpi0v2 python=3.12 -y
conda activate lerobotpi0v2
# LeRobot 0.5.0 + Pi0.5 의존성 (lockfile: torch 2.10.0, transformers 5.3.0, fastapi·uvicorn)
pip install -r "$SCRIPT_DIR/env_lerobotpi0v2.lock.txt"

# HuggingFace 로그인 (google/paligemma-3b-pt-224 gated repo 접근, 안 돼 있으면)
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login

# VLM 모델 (Qwen3-VL-8B-Instruct, ~16GB) 사전 캐싱 (~/.cache/huggingface/hub)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct

# VLA base (Pi0.5, lerobot/pi05_base, ~14GB) — 파인튜닝 시작점
huggingface-cli download lerobot/pi05_base --local-dir pi05_base

# 검증 (설치된 버전 출력)
conda run -n vllm python -c "import vllm; print('  vllm        : vllm', vllm.__version__)" 2>/dev/null || echo "  vllm 확인 실패"
conda run -n lerobotpi0v2 python -c "import lerobot, torch; print('  lerobotpi0v2: lerobot', lerobot.__version__, '| torch', torch.__version__)" 2>/dev/null || echo "  lerobotpi0v2 확인 실패"

echo "[setup_viva] 완료."
echo "  파인튜닝:  bash vllm/train_v5.sh"
echo "  서버 기동: bash launch_servers.sh all --checkpoint <pi05 ckpt>/pretrained_model"
