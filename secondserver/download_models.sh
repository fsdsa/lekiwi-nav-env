#!/usr/bin/env bash
# =============================================================================
# 모델 다운로드 (전송 불필요 — HuggingFace에서 명령어로 받음)
#   VLM      : Qwen/Qwen3-VL-8B-Instruct  (open, ~16GB)  → VLM 서빙(vllm)
#   VLA base : lerobot/pi05_base          (Pi0.5, ~14GB) → VLA 파인튜닝 시작점
#
# 데이터셋은 구글 드라이브 링크에서 받고, 파인튜닝 ckpt는 학습으로 생성하므로 여기 없음.
# lerobot/pi05_base 는 license gemma(google/paligemma-3b-pt-224 gated 의존) →
# huggingface-cli login 선행 필요.
#
# 사용:  bash secondserver/download_models.sh
#        VLA base 위치 변경:  PI05_BASE_DIR=/path/to/pi05_base bash secondserver/download_models.sh
# =============================================================================
set -uo pipefail

# gated repo(gemma/paligemma) 접근 — 로그인 안 돼 있으면 로그인
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login

# 1) VLM — 최초 vLLM 서빙 시 자동 다운로드되지만, 사전 캐싱(~/.cache/huggingface/hub)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct

# 2) VLA base — Pi0.5. train_v5.sh 의 --policy.path 가 가리킬 위치
PI05_BASE_DIR="${PI05_BASE_DIR:-pi05_base}"
huggingface-cli download lerobot/pi05_base --local-dir "$PI05_BASE_DIR"

echo "[models] 완료."
echo "  VLM : Qwen/Qwen3-VL-8B-Instruct  (HF 캐시)"
echo "  VLA : $PI05_BASE_DIR  → 파인튜닝:  lerobot-train --policy.path=$PI05_BASE_DIR ..."
echo "  (또는 --policy.path=lerobot/pi05_base 로 지정 시 자동 다운로드)"
