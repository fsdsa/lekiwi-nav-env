#!/bin/bash
# Navigate RL Training Script
# Run with: nohup bash run_navigate_train.sh > train_navigate.log 2>&1 &

cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate env_isaaclab
source ~/isaacsim/setup_conda_env.sh 2>/dev/null

NUM_ENVS=${1:-1024}
echo "=== Navigate RL Training Started: $(date) ==="
echo "Envs: $NUM_ENVS, Max iterations: 3000"

python train_lekiwi.py \
    --skill navigate \
    --num_envs $NUM_ENVS \
    --max_iterations 3000 \
    --headless

echo "=== Training Finished: $(date) ==="
