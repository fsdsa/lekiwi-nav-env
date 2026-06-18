#!/usr/bin/env bash
# =============================================================================
# lerobot050 env 설치 — HDF5 → LeRobot v3 데이터셋 변환 (convert_hdf5_to_lerobot_v3.py)
#
# 검증 환경 (데스크탑, 2026-06-18 실측):
#   Python 3.12.12 · LeRobot 0.5.0 · torch 2.10.0 · numpy 2.2.6
#
# 사용:  bash export/setup_lerobot050_env.sh
# =============================================================================
set -uo pipefail

ENV_NAME="${ENV_NAME:-lerobot050}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="$SCRIPT_DIR/env_lerobot050.lock.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

if [ -f "$LOCK" ]; then
  echo "[lerobot050] lockfile로 정확 버전 설치: $LOCK"
  pip install -r "$LOCK"
else
  echo "[lerobot050] lockfile 없음 → 기본 설치"
  pip install lerobot==0.5.0
fi

echo "[lerobot050] 설치 완료. 검증:"
python -c "import lerobot; print('  lerobot', lerobot.__version__)"
echo "  변환:  conda activate lerobot050 && python convert_hdf5_to_lerobot_v3.py --input <hdf5> --output_root <dir> --repo_id local/<name> --fps 25"
