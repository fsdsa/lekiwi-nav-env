#!/usr/bin/env bash
###############################################################################
# run_training.sh - ResiP combined_s2_s3 v19 학습 실행
#
# 사용법 (서버에서):
#   bash secondserver/run_training.sh          # foreground
#   nohup bash secondserver/run_training.sh &  # background (nohup)
#
# 학습 설정:
#   - Skill: combined_s2_s3 (S2 frozen expert + S3 learner)
#   - BC: dp_bc_epoch150 (S2) + dp_bc_epoch500 (S3, 55D)
#   - S2 Expert: resip64% (frozen)
#   - Curriculum: v15_dense
#   - Envs: 1024, Steps: 3000/ep, Total: 200M
#   - PPO: lr_actor=1e-3, lr_critic=5e-3, target_kl=1.5
#   - Reward normalization: True
###############################################################################
set -euo pipefail

# ------------------------------------------------------------------
# 환경 설정
# ------------------------------------------------------------------

# Conda 활성화
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate rl_train

# 프로젝트 디렉토리
cd "$HOME/IsaacLab/scripts/lekiwi_nav_env"

# 환경 변수
export LEKIWI_USD_PATH="$HOME/Downloads/lekiwi_robot.usd"
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------
# 사전 검증
# ------------------------------------------------------------------
echo "=========================================="
echo "  사전 검증"
echo "=========================================="

ERRORS=0

# 로봇 USD
if [ ! -f "$LEKIWI_USD_PATH" ]; then
    echo "[FAIL] 로봇 USD 없음: $LEKIWI_USD_PATH"
    ERRORS=$((ERRORS + 1))
else
    echo "[OK] 로봇 USD"
fi

# 체크포인트
for ckpt in \
    "checkpoints/dp_bc_small/dp_bc_epoch150.pt" \
    "backup/appoachandlift/resip64%.pt" \
    "checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt"; do
    if [ ! -f "$ckpt" ]; then
        echo "[FAIL] 체크포인트 없음: $ckpt"
        ERRORS=$((ERRORS + 1))
    else
        echo "[OK] $ckpt"
    fi
done

# 물체 USD
OBJ_BASE="$HOME/isaac-objects/mujoco_scanned_objects/models"
for obj in "5_HTP/model_clean.usd" "ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd"; do
    if [ ! -f "$OBJ_BASE/$obj" ]; then
        echo "[FAIL] 물체 USD 없음: $OBJ_BASE/$obj"
        ERRORS=$((ERRORS + 1))
    else
        echo "[OK] $obj"
    fi
done

# 보정 파일
if [ ! -f "calibration/arm_limits_measured.json" ]; then
    echo "[FAIL] 보정 파일 없음: calibration/arm_limits_measured.json"
    ERRORS=$((ERRORS + 1))
else
    echo "[OK] arm_limits_measured.json"
fi

# Python import 확인
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    && echo "[OK] PyTorch CUDA" \
    || { echo "[FAIL] PyTorch CUDA 사용 불가"; ERRORS=$((ERRORS + 1)); }

python -c "import isaacsim" 2>/dev/null \
    && echo "[OK] Isaac Sim" \
    || { echo "[FAIL] Isaac Sim import 실패"; ERRORS=$((ERRORS + 1)); }

python -c "import diffusers" 2>/dev/null \
    && echo "[OK] diffusers" \
    || { echo "[FAIL] diffusers 없음 (pip install diffusers)"; ERRORS=$((ERRORS + 1)); }

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "ERROR: $ERRORS개 검증 실패. 위 항목을 해결 후 다시 실행하세요."
    exit 1
fi

echo ""
echo "  모든 검증 통과!"

# ------------------------------------------------------------------
# 학습 실행
# ------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/resip_s3_v19_${TIMESTAMP}.log"

echo ""
echo "=========================================="
echo "  학습 시작: $(date)"
echo "  로그: $LOG_FILE"
echo "=========================================="
echo ""

OBJ_DIR="$HOME/isaac-objects/mujoco_scanned_objects/models"

exec python train_resip.py \
    --skill combined_s2_s3 \
    --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    --s2_resip_checkpoint 'backup/appoachandlift/resip64%.pt' \
    --s3_bc_checkpoint checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt \
    --object_usd "${OBJ_DIR}/5_HTP/model_clean.usd" \
    --dest_object_usd "${OBJ_DIR}/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd" \
    --num_envs 1024 \
    --num_env_steps 3000 \
    --total_timesteps 200000000 \
    --s2_lift_hold_steps 200 \
    --s3_curriculum_stage v15_dense \
    --normalize_reward True \
    --init_logstd -2.0 \
    --lr_actor 1e-3 \
    --lr_critic 5e-3 \
    --target_kl 1.5 \
    --ent_coef 0.001 \
    --save_dir checkpoints/resip_s3_v19 \
    --seed 82 \
    --headless \
    2>&1 | tee "$LOG_FILE"
