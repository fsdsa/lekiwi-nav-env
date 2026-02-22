#!/bin/bash
OBJ=/home/yubin/isaac-objects/mujoco_scanned_objects/models/Down_To_Earth_Ceramic_Orchid_Pot_Asst_Blue/model_clean.usd
GRIP=/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1

python record_teleop.py \
  --skill combined \
  --num_demos 5 \
  --teleop_source tcp \
  --listen_port 15002 \
  --object_usd "$OBJ" \
  --gripper_contact_prim_path "$GRIP" \
  --arm_limit_json calibration/arm_limits_measured.json \
  --grasp_contact_threshold 0.1 \
  --grasp_max_object_dist 0.50
