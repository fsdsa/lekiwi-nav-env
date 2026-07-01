#!/bin/bash
# H100 80GB + 24 CPU core 최적화된 Pi0.5 fine-tuning (v2 — env vars 포함)
set -e
cd "$(dirname "$0")"

# conda activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pi05_h100

# torchcodec + HF 해결용 env vars
TORCH_LIB=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__)+"/lib")')
export LD_LIBRARY_PATH="$TORCH_LIB:$HOME/miniconda3/envs/pi05_h100/lib:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

BATCH=${BATCH:-16}
STEPS=${STEPS:-80000}
LR=${LR:-7e-5}
DECAY=${DECAY:-60000}
SAVE_FREQ=${SAVE_FREQ:-8000}
NUM_WORKERS=${NUM_WORKERS:-8}
COMPILE=${COMPILE:-false}
GRAD_CKPT=${GRAD_CKPT:-false}

BASE="./base_model"
DATASET="./dataset"
OUTPUT="./outputs/h100_endtoend"
LOG="./train.log"

echo "=================================================================="
echo "Pi0.5 H100 end-to-end — batch=$BATCH steps=$STEPS lr=$LR"
echo "LD_LIBRARY_PATH set: torch + conda env libs"
echo "HF_HUB_OFFLINE=1"
echo "=================================================================="

nohup lerobot-train \
    --dataset.repo_id=local/lekiwi_full_teleop_combined \
    --dataset.root="$DATASET" \
    --policy.path="$BASE" \
    --policy.repo_id=local/pi05_h100_endtoend \
    --policy.compile_model=$COMPILE \
    --policy.gradient_checkpointing=$GRAD_CKPT \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.scheduler_decay_steps=$DECAY \
    --policy.optimizer_lr=$LR \
    --batch_size=$BATCH \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --log_freq=100 \
    --eval_freq=0 \
    --num_workers=$NUM_WORKERS \
    --tolerance_s=0.1 \
    --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --output_dir="$OUTPUT" \
    > "$LOG" 2>&1 &

TRAIN_PID=$!
echo "PID=$TRAIN_PID Log=$LOG"
sleep 3
ps -p $TRAIN_PID -o pid,etime --no-headers || echo WARNING
