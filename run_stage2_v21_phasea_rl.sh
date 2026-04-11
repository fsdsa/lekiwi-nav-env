#!/bin/bash
set -euo pipefail

cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env

exec ./run_train_resip_local.sh \
  --skill combined_s2_s3 \
  --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
  --s2_resip_checkpoint backup/appoachandlift/resip64%.pt \
  --s3_bc_checkpoint checkpoints/dp_bc_skill3_38d_v21/dp_bc_epoch300.pt \
  --object_usd /home/yubin11/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd /home/yubin11/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --num_envs 1024 \
  --num_env_steps 5000 \
  --total_timesteps 2000000000 \
  --s2_lift_hold_steps 200 \
  --s3_curriculum_stage release \
  --s3_phase_a_policy rl_base_hold \
  --s3_phase_a_base_scale 0.20 \
  --s3_phase_b_dist 0.40 \
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
  --save_dir checkpoints/resip_s3_38d_v21_phasea_rl_v1 \
  --headless
