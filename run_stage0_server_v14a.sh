#!/usr/bin/env bash
set -eo pipefail

source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate rl_train

cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env

export LEKIWI_USD_PATH=/home/jovyan/Downloads/lekiwi_robot.usd
export PYTHONUNBUFFERED=1

exec > /home/jovyan/resip_s3_36d_v15_stage0_v14a.log 2>&1

python train_resip.py \
  --skill combined_s2_s3 \
  --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
  --s2_resip_checkpoint checkpoints/resip/resip64pct.pt \
  --s3_bc_checkpoint checkpoints/dp_bc_skill3_36d_v14/dp_bc_epoch150.pt \
  --object_usd /home/jovyan/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd /home/jovyan/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --num_envs 2048 \
  --num_env_steps 5000 \
  --total_timesteps 2000000000 \
  --s2_lift_hold_steps 200 \
  --s3_curriculum_stage carry_stabilize \
  --s3_carry_grip_tol_start 0.08 \
  --s3_carry_grip_tol 0.05 \
  --s3_carry_curriculum_successes 15000 \
  --s3_carry_grip_scale 0.15 \
  --save_dir checkpoints/resip_s3_36d_v15_stage0_v14a \
  --headless
