#!/usr/bin/env bash
set -eo pipefail

source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate rl_train

cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env

export LEKIWI_USD_PATH=/home/jovyan/Downloads/lekiwi_robot.usd
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > /home/jovyan/resip_s3_36d_v15_stage2_v14c.log 2>&1

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
  --s3_curriculum_stage release \
  --s3_hold_target_dist 0.12 \
  --s3_hold_sigma_pos 0.06 \
  --s3_hold_sigma_hgt 0.020 \
  --s3_hold_radius_start 0.18 \
  --s3_hold_radius 0.14 \
  --s3_hold_height_min 0.032 \
  --s3_hold_height_max_start 0.065 \
  --s3_hold_height_max 0.050 \
  --s3_hold_curriculum_successes 20000 \
  --s3_hold_stability_steps_start 3 \
  --s3_hold_stability_steps 6 \
  --resume_resip checkpoints/resip_s3_36d_v15_stage2_v14b/resip_iter20.pt \
  --resume_actor_only true \
  --save_dir checkpoints/resip_s3_36d_v15_stage2_v14c \
  --headless
