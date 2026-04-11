#!/usr/bin/env bash
set -eo pipefail

source /home/yubin11/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env

export LEKIWI_USD_PATH=/home/yubin11/Downloads/lekiwi_robot.usd
export PYTHONUNBUFFERED=1

exec > /home/yubin11/resip_s3_36d_v15_stage0_v9g.log 2>&1

python train_resip.py \
  --skill combined_s2_s3 \
  --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
  --s2_resip_checkpoint backup/appoachandlift/resip64%.pt \
  --s3_bc_checkpoint checkpoints/dp_bc_skill3_36d_v9/dp_bc_epoch150.pt \
  --object_usd /home/yubin11/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd /home/yubin11/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --num_envs 1024 \
  --num_env_steps 5000 \
  --total_timesteps 500000000 \
  --s2_lift_hold_steps 200 \
  --s3_curriculum_stage carry_stabilize \
  --s3_carry_grip_tol_start 0.08 \
  --s3_carry_grip_tol 0.05 \
  --s3_carry_curriculum_successes 15000 \
  --s3_carry_grip_scale 0.40 \
  --save_dir checkpoints/resip_s3_36d_v15_stage0_v9g \
  --headless
