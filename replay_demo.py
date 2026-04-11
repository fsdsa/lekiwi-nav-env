#!/usr/bin/env python3
"""
텔레옵 HDF5 데모를 Isaac Sim에서 재생하는 스크립트.
수집한 action을 그대로 env에 넣어서 시각적으로 확인.

Usage:
    python replay_demo.py --skill approach_and_grasp \
        --demo demos_skill2/combined_skill2_20260226_153002.hdf5 \
        --episode 0 \
        --object_usd /home/yubin11/isaac-objects/.../5_HTP/model_clean.usd
"""
from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description="Replay teleop demo in Isaac Sim (GUI)")
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place"])
parser.add_argument("--demo", type=str, required=True, help="HDF5 파일 경로")
parser.add_argument("--episode", type=int, default=0, help="재생할 에피소드 번호")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")
parser.add_argument("--speed", type=float, default=1.0,
                    help="재생 속도 (1.0=원본, 0.5=절반)")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import h5py
import torch
import numpy as np

# ── 데모 로드 ──
f = h5py.File(args.demo, "r")
ep_key = f"episode_{args.episode}"
if ep_key not in f:
    avail = sorted([k for k in f.keys() if k.startswith("episode")])
    print(f"  ERROR: '{ep_key}' not found. Available: {avail}")
    f.close()
    simulation_app.close()
    sys.exit(1)

demo_obs = f[ep_key]["obs"][:]
demo_act = f[ep_key]["actions"][:]
demo_object_pos_w = f[ep_key]["object_pos_w"][:] if "object_pos_w" in f[ep_key] else None
ep_attrs = dict(f[ep_key].attrs)
print(f"\n  Demo: {args.demo}")
print(f"  Episode {args.episode}: {len(demo_act)} steps, obs_dim={demo_obs.shape[1]}, act_dim={demo_act.shape[1]}")
print(f"  HDF5 file attrs: {dict(f.attrs)}")
print(f"  Episode attrs: {ep_attrs}")
if demo_object_pos_w is not None:
    print(f"  object_pos_w: {demo_object_pos_w.shape} (per-step world coords available)")
else:
    print(f"  object_pos_w: NOT saved (will reconstruct from obs[21:24])")

# ── Env 생성 ──
if args.skill == "approach_and_grasp":
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = 1
elif args.skill == "carry_and_place":
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    env_cfg = Skill3EnvCfg()
    env_cfg.scene.num_envs = 1
    if args.handoff_buffer:
        env_cfg.handoff_buffer_path = args.handoff_buffer

# HDF5 attrs에서 텔레옵 설정 자동 복원
hdf5_attrs = dict(f.attrs)
env_cfg.enable_domain_randomization = False
env_cfg.arm_limit_write_to_sim = False
env_cfg.grasp_contact_threshold = float(hdf5_attrs.get("grasp_contact_threshold", 0.1))
env_cfg.grasp_max_object_dist = float(hdf5_attrs.get("grasp_max_object_dist", 0.50))
env_cfg.grasp_joint_break_force = 1e8
env_cfg.grasp_joint_break_torque = 1e8
env_cfg.object_mass = float(hdf5_attrs.get("object_mass", 0.3))
env_cfg.arm_action_scale = float(hdf5_attrs.get("arm_action_scale", 1.5))
env_cfg.max_lin_vel = float(hdf5_attrs.get("max_lin_vel", 0.5))
env_cfg.max_ang_vel = float(hdf5_attrs.get("max_ang_vel", 3.0))
env_cfg.episode_length_s = max(len(demo_act) * 0.04 * 2, 300.0)

print(f"  [Config from HDF5] object_mass={env_cfg.object_mass}, "
      f"arm_action_scale={env_cfg.arm_action_scale}")

if args.object_usd:
    env_cfg.object_usd = os.path.expanduser(args.object_usd)
elif "object_usd" in hdf5_attrs:
    env_cfg.object_usd = str(hdf5_attrs["object_usd"])
if args.dest_object_usd:
    env_cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
    env_cfg.arm_limit_json = args.arm_limit_json

if args.skill == "approach_and_grasp":
    env = Skill2Env(cfg=env_cfg)
else:
    env = Skill3Env(cfg=env_cfg)

# ── 재생 ──
print(f"\n{'='*60}")
print(f"  Replaying episode {args.episode} ({len(demo_act)} steps)")
print(f"{'='*60}\n")

obs, _ = env.reset()
device = env.device

# ── 로봇 + 물체 초기 상태를 텔레옵 데이터와 일치시키기 ──
from isaaclab.utils.math import quat_apply

env_id = torch.tensor([0], device=device)

# 1. 로봇 초기 pose 복원
if "robot_init_pos" in ep_attrs and "robot_init_quat" in ep_attrs:
    robot_state = env.robot.data.root_state_w.clone()  # (1, 13)
    robot_pos = torch.tensor(ep_attrs["robot_init_pos"], dtype=torch.float32, device=device)
    robot_quat = torch.tensor(ep_attrs["robot_init_quat"], dtype=torch.float32, device=device)
    robot_state[0, 0:3] = robot_pos
    robot_state[0, 3:7] = robot_quat
    robot_state[0, 7:] = 0.0  # 속도 초기화
    env.robot.write_root_state_to_sim(robot_state, env_id)
    env.home_pos_w[0] = robot_pos
    print(f"  [Robot] Restored init pos={robot_pos.cpu().tolist()}", flush=True)
    print(f"  [Robot] Restored init quat={robot_quat.cpu().tolist()}", flush=True)
else:
    print(f"  [Robot] No init pose saved — using random reset position", flush=True)

# 2. 로봇 관절 초기값 복원 (obs[0, 0:6] = arm_pos + gripper)
init_arm_grip = torch.tensor(demo_obs[0, 0:6], dtype=torch.float32, device=device)
joint_pos = env.robot.data.default_joint_pos[0:1].clone()
joint_pos[0, env.arm_idx] = init_arm_grip
joint_vel = torch.zeros_like(joint_pos)
env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_id)
print(f"  [Robot] Restored joints: {init_arm_grip.cpu().tolist()}", flush=True)

# 3. 물체 위치 복원
if demo_object_pos_w is not None:
    obj_world = torch.tensor(demo_object_pos_w[0], dtype=torch.float32, device=device)
    print(f"  [Object] Using saved object_pos_w (absolute coords)", flush=True)
elif "object_init_pos" in ep_attrs:
    obj_world = torch.tensor(ep_attrs["object_init_pos"], dtype=torch.float32, device=device)
    print(f"  [Object] Using saved object_init_pos attr", flush=True)
else:
    # 레거시: obs[21:24] body-frame 상대좌표에서 역변환
    root_pos_w = env.robot.data.root_pos_w[0]
    root_quat_w = env.robot.data.root_quat_w[0]
    recorded_obj_rel = torch.tensor(demo_obs[0, 21:24], dtype=torch.float32, device=device)
    obj_world = root_pos_w + quat_apply(root_quat_w.unsqueeze(0), recorded_obj_rel.unsqueeze(0)).squeeze(0)
    print(f"  [Object] Reconstructed from obs[21:24] (body-frame inverse)", flush=True)

obj_state = env.object_rigid.data.root_state_w.clone()  # (1, 13)
obj_state[0, 0:3] = obj_world
# 물체 방향 복원
if "object_init_quat" in ep_attrs:
    obj_quat = torch.tensor(ep_attrs["object_init_quat"], dtype=torch.float32, device=device)
    obj_state[0, 3:7] = obj_quat
    print(f"  [Object] Restored init quat={obj_quat.cpu().tolist()}", flush=True)
obj_state[0, 7:] = 0.0  # 속도 초기화
env.object_rigid.write_root_state_to_sim(obj_state, env_id)
env.object_pos_w[0] = obj_world

# 4. 물리 settle
for _ in range(10):
    env.sim.step()
env.robot.update(env.sim.cfg.dt)
env.object_rigid.update(env.sim.cfg.dt)

print(f"  [Object] placed at world = {obj_world.cpu().tolist()}", flush=True)

# Gripper smoothing: 레거시 이진 데이터(0/1)에만 적용, 연속 데이터는 그대로 사용
gripper_values = demo_act[:, 5]
gripper_is_binary = np.all(np.isin(gripper_values, [0.0, 1.0]))
grip_smooth = 0.0
grip_alpha = 0.15
if gripper_is_binary:
    print(f"  [Gripper] Binary detected → EMA smoothing enabled (alpha={grip_alpha})", flush=True)
else:
    print(f"  [Gripper] Continuous detected → no smoothing", flush=True)

for t in range(len(demo_act)):
    if not simulation_app.is_running():
        break

    action = torch.tensor(demo_act[t:t+1], dtype=torch.float32, device=device)

    # gripper: 이진 데이터만 EMA smoothing 적용
    if gripper_is_binary:
        grip_target = action[0, 5].item()
        grip_smooth = grip_alpha * grip_target + (1.0 - grip_alpha) * grip_smooth
        action[0, 5] = grip_smooth

    obs, reward, terminated, truncated, info = env.step(action)

    # 주요 스텝 로그
    if t < 5 or t % 200 == 0 or t == len(demo_act) - 1:
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        a = action[0].cpu().tolist()
        o_arm = obs_t[0, :5].cpu().tolist()
        o_obj = obs_t[0, 21:24].cpu().tolist()
        print(f"  [t={t:>4d}/{len(demo_act)}] "
              f"arm={[f'{x:.3f}' for x in o_arm]} "
              f"obj_rel={[f'{x:.3f}' for x in o_obj]} "
              f"act=[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f},{a[3]:.2f},{a[4]:.2f},"
              f"g={a[5]:.1f},vx={a[6]:.2f},vy={a[7]:.2f},wz={a[8]:.2f}]",
              flush=True)

    if terminated.any() or truncated.any():
        print(f"\n  Episode ended at step {t} (terminated={terminated.any().item()}, "
              f"truncated={truncated.any().item()})", flush=True)
        break

print(f"\n  Replay 완료: {t+1}/{len(demo_act)} steps 재생됨\n", flush=True)

# 종료 대기 (GUI 확인용)
print("  GUI에서 확인 후 창을 닫아주세요...", flush=True)
while simulation_app.is_running():
    env.sim.step()

f.close()
env.close()
simulation_app.close()
