#!/usr/bin/env bash
# =============================================================================
# VIVA 서버 환경 설치 (A100) — 한 번 실행으로 VLM+VLA 추론 환경 전체 구축
#   [1] vllm          — Python 3.11 + vLLM 0.17.0 + Qwen3-VL-8B (VLM, 포트 8000)
#   [2] lerobotpi0v2  — Python 3.12 + LeRobot 0.5.0 + Pi0.5 (VLA, 포트 8002 + 파인튜닝)
#   [3] 모델 다운로드 — Qwen3-VL-8B-Instruct + lerobot/pi05_base
#
# 검증 환경(2026-06-18): vllm(torch2.10.0/transformers4.57.6) · lerobotpi0v2(torch2.10.0/transformers5.3.0)
# 전제: Miniconda 설치됨 + A100 GPU. 클라이언트(env_isaaclab)는 export/setup_datagen.sh 참조.
#
# 사용:  bash secondserver/setup_viva.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v conda >/dev/null 2>&1 || { echo "[ERR] conda 없음 → Miniconda 먼저 설치"; exit 1; }

echo "==== [1/3] VLM env (vllm) ===="
bash "$SCRIPT_DIR/setup_vllm_env.sh"

echo "==== [2/3] VLA env (lerobotpi0v2) ===="
bash "$SCRIPT_DIR/setup_lerobotpi0v2_env.sh"

echo "==== [3/3] 모델 다운로드 (Qwen3-VL-8B + Pi0.5 base) ===="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobotpi0v2 2>/dev/null || true   # huggingface-cli 사용 위해
bash "$SCRIPT_DIR/download_models.sh"

echo ""
echo "==== VIVA 서버 환경 설치 완료 ===="
echo "  서버 기동:  bash launch_servers.sh all --checkpoint <pi05 ckpt>/pretrained_model"
echo "  (파인튜닝 ckpt가 아직 없으면 매뉴얼 §3.3 VLA 파인튜닝 먼저)"
