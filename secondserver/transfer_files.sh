#!/usr/bin/env bash
###############################################################################
# transfer_files.sh - 로컬에서 새 서버로 학습 필요 파일 전송
#
# 사용법 (로컬 머신에서 실행):
#   bash secondserver/transfer_files.sh <user> <host> [port]
#
# 예시:
#   bash secondserver/transfer_files.sh ubuntu 123.45.67.89 22
#   bash secondserver/transfer_files.sh jovyan 218.148.55.186 30628  (포트 가변, ssh config Host A100 참고)
###############################################################################
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash transfer_files.sh <user> <host> [port]"
    echo "  예: bash transfer_files.sh ubuntu 123.45.67.89 22"
    exit 1
fi

SERVER_USER="$1"
SERVER_HOST="$2"
SERVER_PORT="${3:-22}"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP="scp -P $SERVER_PORT $SSH_OPTS"
SSH="ssh -p $SERVER_PORT $SSH_OPTS ${SERVER_USER}@${SERVER_HOST}"

LOCAL_PROJECT="$HOME/IsaacLab/scripts/lekiwi_nav_env"
REMOTE_PROJECT="\$HOME/IsaacLab/scripts/lekiwi_nav_env"

echo "=========================================="
echo "  파일 전송: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
echo "=========================================="

# ------------------------------------------------------------------
# 0. 서버 연결 테스트
# ------------------------------------------------------------------
echo ""
echo "[0/5] 서버 연결 테스트..."
if $SSH "echo 'OK'" 2>/dev/null; then
    echo "  연결 성공."
else
    echo "  ERROR: 서버 연결 실패. SSH 정보를 확인하세요."
    exit 1
fi

# ------------------------------------------------------------------
# 1. 서버 디렉토리 생성
# ------------------------------------------------------------------
echo ""
echo "[1/5] 서버 디렉토리 생성..."
$SSH "mkdir -p ~/IsaacLab/scripts/lekiwi_nav_env/calibration \
              ~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/dp_bc_small \
              ~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/dp_bc_skill3_55d_fixed_1e-4 \
              ~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/resip_s3_v19 \
              ~/IsaacLab/scripts/lekiwi_nav_env/backup/appoachandlift \
              ~/IsaacLab/scripts/lekiwi_nav_env/logs \
              ~/Downloads \
              ~/isaac-objects/mujoco_scanned_objects/models"
echo "  디렉토리 생성 완료."

# ------------------------------------------------------------------
# 2. Python 소스 코드 전송
# ------------------------------------------------------------------
echo ""
echo "[2/5] Python 소스 코드 전송..."

for f in train_resip.py diffusion_policy.py skill3_bc_obs.py \
         lekiwi_skill2_eval.py lekiwi_robot_cfg.py __init__.py; do
    echo "  -> $f"
    $SCP "${LOCAL_PROJECT}/$f" \
         "${SERVER_USER}@${SERVER_HOST}:~/IsaacLab/scripts/lekiwi_nav_env/$f"
done

# 보정 파일
echo "  -> calibration/arm_limits_measured.json"
$SCP "${LOCAL_PROJECT}/calibration/arm_limits_measured.json" \
     "${SERVER_USER}@${SERVER_HOST}:~/IsaacLab/scripts/lekiwi_nav_env/calibration/"

echo "  소스 코드 전송 완료."

# ------------------------------------------------------------------
# 3. 체크포인트 전송 (~44MB)
# ------------------------------------------------------------------
echo ""
echo "[3/5] 체크포인트 전송 (~44MB)..."

echo "  -> dp_bc_epoch150.pt (21MB)"
$SCP "${LOCAL_PROJECT}/checkpoints/dp_bc_small/dp_bc_epoch150.pt" \
     "${SERVER_USER}@${SERVER_HOST}:~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/dp_bc_small/"

echo "  -> resip64%.pt (1.8MB)"
$SCP "${LOCAL_PROJECT}/backup/appoachandlift/resip64%.pt" \
     "${SERVER_USER}@${SERVER_HOST}:~/IsaacLab/scripts/lekiwi_nav_env/backup/appoachandlift/"

echo "  -> dp_bc_epoch500.pt (21MB)"
$SCP "${LOCAL_PROJECT}/checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt" \
     "${SERVER_USER}@${SERVER_HOST}:~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/dp_bc_skill3_55d_fixed_1e-4/"

echo "  체크포인트 전송 완료."

# ------------------------------------------------------------------
# 4. USD 파일 전송 (~12MB)
# ------------------------------------------------------------------
echo ""
echo "[4/5] USD 파일 전송 (~12MB)..."

# 로봇 USD
echo "  -> lekiwi_robot.usd (7MB)"
$SCP "$HOME/Downloads/lekiwi_robot.usd" \
     "${SERVER_USER}@${SERVER_HOST}:~/Downloads/"

# Source 물체 (전체 디렉토리)
echo "  -> 5_HTP/ (2.9MB, 디렉토리)"
$SCP -r "$HOME/isaac-objects/mujoco_scanned_objects/models/5_HTP" \
     "${SERVER_USER}@${SERVER_HOST}:~/isaac-objects/mujoco_scanned_objects/models/"

# Dest 물체 (전체 디렉토리)
echo "  -> ACE_Coffee_Mug_Kristen_16_oz_cup/ (1.9MB, 디렉토리)"
$SCP -r "$HOME/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup" \
     "${SERVER_USER}@${SERVER_HOST}:~/isaac-objects/mujoco_scanned_objects/models/"

echo "  USD 전송 완료."

# ------------------------------------------------------------------
# 5. secondserver 스크립트 전송
# ------------------------------------------------------------------
echo ""
echo "[5/5] 실행 스크립트 전송..."

$SSH "mkdir -p ~/IsaacLab/scripts/lekiwi_nav_env/secondserver"

for f in setup_env.sh run_training.sh SETUP_GUIDE.md; do
    if [ -f "${LOCAL_PROJECT}/secondserver/$f" ]; then
        echo "  -> secondserver/$f"
        $SCP "${LOCAL_PROJECT}/secondserver/$f" \
             "${SERVER_USER}@${SERVER_HOST}:~/IsaacLab/scripts/lekiwi_nav_env/secondserver/"
    fi
done

echo "  스크립트 전송 완료."

# ------------------------------------------------------------------
# 검증
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  전송 검증"
echo "=========================================="

$SSH "
echo '--- 소스 코드 ---'
ls -la ~/IsaacLab/scripts/lekiwi_nav_env/train_resip.py 2>/dev/null && echo 'OK' || echo 'MISSING'

echo '--- 체크포인트 ---'
ls -la ~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/dp_bc_small/dp_bc_epoch150.pt 2>/dev/null && echo 'OK' || echo 'MISSING'
ls -la ~/IsaacLab/scripts/lekiwi_nav_env/backup/appoachandlift/resip64%.pt 2>/dev/null && echo 'OK' || echo 'MISSING'
ls -la ~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt 2>/dev/null && echo 'OK' || echo 'MISSING'

echo '--- USD ---'
ls -la ~/Downloads/lekiwi_robot.usd 2>/dev/null && echo 'OK' || echo 'MISSING'
ls -d ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/ 2>/dev/null && echo 'OK' || echo 'MISSING'
ls -d ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/ 2>/dev/null && echo 'OK' || echo 'MISSING'

echo '--- 보정 ---'
ls -la ~/IsaacLab/scripts/lekiwi_nav_env/calibration/arm_limits_measured.json 2>/dev/null && echo 'OK' || echo 'MISSING'
"

echo ""
echo "=========================================="
echo "  전송 완료!"
echo "=========================================="
echo ""
echo "  다음 단계 (서버에서 실행):"
echo "    1. bash secondserver/setup_env.sh    # 환경 설치"
echo "    2. bash secondserver/run_training.sh  # 학습 시작"
echo ""
