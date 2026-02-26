#!/usr/bin/env python3
"""
BC checkpoint을 Isaac Sim에서 GUI로 돌려보는 간단한 평가 스크립트.
MSE (bc_nav.pt) 및 GMM (bc_nav_gmm.pt) 체크포인트 모두 자동 감지.

Usage:
    # MSE 체크포인트 (랜덤 리셋)
    python eval_bc.py --skill approach_and_grasp \
        --bc_checkpoint checkpoints/skill2/bc_nav.pt

    # GMM 체크포인트 (랜덤 리셋)
    python eval_bc.py --skill approach_and_grasp \
        --bc_checkpoint checkpoints/skill2/bc_nav_gmm.pt

    # HDF5 초기 상태 복원 평가 (수집 환경 그대로 재현)
    python eval_bc.py --skill approach_and_grasp \
        --bc_checkpoint checkpoints/skill2/bc_nav_gmm.pt \
        --demo demos_skill2/combined_skill2_20260226_153002.hdf5

    # Skill-3 (handoff buffer 필요)
    python eval_bc.py --skill carry_and_place \
        --bc_checkpoint checkpoints/skill3/bc_nav.pt \
        --handoff_buffer handoff_buffer.pkl
"""
from __future__ import annotations

import argparse
import os

# ── Args (AppLauncher args 포함) ──
parser = argparse.ArgumentParser(description="BC Eval in Isaac Sim (GUI)")
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place"])
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--demo", type=str, default="",
                    help="HDF5 파일 경로 — 지정 시 에피소드 초기 상태를 복원하여 평가")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")
parser.add_argument("--action_repeat", type=int, default=10,
                    help="BC 출력을 N step 유지 (텔레옵에서 유저가 position 유지하는 것과 동일)")
parser.add_argument("--arm_action_scale", type=float, default=1.0,
                    help="팔 action[0:5] 스케일 (mean regression 보정, 1.2~1.5 권장)")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import h5py
import torch
import numpy as np
from train_bc import BCPolicy, BCPolicyGMM

# ── HDF5 데모 로드 (선택) ──
demo_episodes = []  # list of dicts: {obs, actions, ep_attrs}
demo_file = None
if args.demo and os.path.isfile(args.demo):
    demo_file = h5py.File(args.demo, "r")
    ep_keys = sorted([k for k in demo_file.keys() if k.startswith("episode")],
                     key=lambda k: int(k.split("_")[1]))
    for ek in ep_keys:
        grp = demo_file[ek]
        demo_episodes.append({
            "obs": grp["obs"][:],
            "actions": grp["actions"][:],
            "ep_attrs": dict(grp.attrs),
            "object_pos_w": grp["object_pos_w"][:] if "object_pos_w" in grp else None,
        })
    print(f"\n  Demo loaded: {args.demo} ({len(demo_episodes)} episodes)")
    if args.num_episodes > len(demo_episodes):
        args.num_episodes = len(demo_episodes)
        print(f"  num_episodes clamped to {args.num_episodes}")

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

# 텔레옵과 동일한 설정 (DR 끄기 + grasp 파라미터 일치)
env_cfg.enable_domain_randomization = False
env_cfg.arm_limit_write_to_sim = False
env_cfg.grasp_contact_threshold = 0.1
env_cfg.grasp_max_object_dist = 0.50
env_cfg.grasp_joint_break_force = 1e8
env_cfg.grasp_joint_break_torque = 1e8

# HDF5 attrs에서 config 복원 (있으면)
if demo_file is not None:
    hdf5_attrs = dict(demo_file.attrs)
    env_cfg.object_mass = float(hdf5_attrs.get("object_mass", env_cfg.object_mass))
    env_cfg.arm_action_scale = float(hdf5_attrs.get("arm_action_scale", env_cfg.arm_action_scale))
    env_cfg.max_lin_vel = float(hdf5_attrs.get("max_lin_vel", env_cfg.max_lin_vel))
    env_cfg.max_ang_vel = float(hdf5_attrs.get("max_ang_vel", env_cfg.max_ang_vel))
    if not args.object_usd and "object_usd" in hdf5_attrs:
        env_cfg.object_usd = str(hdf5_attrs["object_usd"])
    print(f"  [Config from HDF5] mass={env_cfg.object_mass}, "
          f"arm_action_scale={env_cfg.arm_action_scale}")
else:
    # 랜덤 리셋 모드: 스폰 각도 좁히기
    env_cfg.spawn_heading_noise_std = 0.1
    env_cfg.spawn_heading_max_rad = 0.26  # ~15deg

env_cfg.episode_length_s = 240.0

# 공통 설정
if args.object_usd:
    env_cfg.object_usd = os.path.expanduser(args.object_usd)
if args.multi_object_json:
    env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
if args.dest_object_usd:
    env_cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
    env_cfg.arm_limit_json = args.arm_limit_json

if args.skill == "approach_and_grasp":
    env = Skill2Env(cfg=env_cfg)
else:
    env = Skill3Env(cfg=env_cfg)

# ── BC Policy 로드 ──
sd = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=True)
obs_dim = sd["net.0.weight"].shape[1]

# GMM vs MSE 자동 감지: pi_layer 키가 있으면 GMM
is_gmm = any(k.startswith("pi_layer") for k in sd.keys())

if is_gmm:
    # n_components 추론: pi_layer.bias shape = (K,)
    n_components = sd["pi_layer.bias"].shape[0]
    policy = BCPolicyGMM(obs_dim=obs_dim, act_dim=9, n_components=n_components)
    policy.load_state_dict(sd)
    policy_type = f"GMM (K={n_components})"
else:
    policy = BCPolicy(obs_dim=obs_dim, act_dim=9)
    policy.load_state_dict(sd)
    policy_type = "MSE"

policy.eval()
policy = policy.to(env.device)

mode_str = "HDF5 초기 상태 복원" if demo_episodes else "랜덤 리셋"
print(f"\n{'='*60}")
print(f"  BC Eval — {args.skill} ({mode_str})")
print(f"  Checkpoint: {args.bc_checkpoint}")
print(f"  Policy: {policy_type}, obs_dim={obs_dim}, act_dim=9")
print(f"  Episodes: {args.num_episodes}")
print(f"{'='*60}\n")


# ── 환경 초기 상태 복원 헬퍼 ──
def _restore_init_state(ep_data):
    """HDF5 에피소드의 초기 상태로 env를 복원."""
    device = env.device
    env_id = torch.tensor([0], device=device)
    ea = ep_data["ep_attrs"]

    # 1. 로봇 위치+방향
    if "robot_init_pos" in ea and "robot_init_quat" in ea:
        rs = env.robot.data.root_state_w.clone()
        rs[0, 0:3] = torch.tensor(ea["robot_init_pos"], dtype=torch.float32, device=device)
        rs[0, 3:7] = torch.tensor(ea["robot_init_quat"], dtype=torch.float32, device=device)
        rs[0, 7:] = 0.0
        env.robot.write_root_state_to_sim(rs, env_id)
        env.home_pos_w[0] = rs[0, :3]

    # 2. 로봇 관절 (obs[0, 0:6] = arm+gripper)
    init_joints = torch.tensor(ep_data["obs"][0, 0:6], dtype=torch.float32, device=device)
    jp = env.robot.data.default_joint_pos[0:1].clone()
    jp[0, env.arm_idx] = init_joints
    jv = torch.zeros_like(jp)
    env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_id)

    # 3. 물체 위치+방향
    obj_state = env.object_rigid.data.root_state_w.clone()
    if ep_data["object_pos_w"] is not None:
        obj_state[0, 0:3] = torch.tensor(ep_data["object_pos_w"][0], dtype=torch.float32, device=device)
    elif "object_init_pos" in ea:
        obj_state[0, 0:3] = torch.tensor(ea["object_init_pos"], dtype=torch.float32, device=device)
    if "object_init_quat" in ea:
        obj_state[0, 3:7] = torch.tensor(ea["object_init_quat"], dtype=torch.float32, device=device)
    obj_state[0, 7:] = 0.0
    env.object_rigid.write_root_state_to_sim(obj_state, env_id)
    env.object_pos_w[0] = obj_state[0, :3]

    # 4. settle
    for _ in range(10):
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)


# ── 실행 루프 ──
episode = 0
successes = 0
step_count = 0
obs, _ = env.reset()

# HDF5 모드: 첫 에피소드 초기 상태 복원
if demo_episodes:
    _restore_init_state(demo_episodes[0])
    # settle 후 obs 다시 읽기
    for _ in range(5):
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)

# 첫 obs 디버그
obs_t = obs["policy"] if isinstance(obs, dict) else obs
print(f"  [DEBUG] obs shape={obs_t.shape}", flush=True)
print(f"  [DEBUG] arm+grip: {obs_t[0,:6].cpu().tolist()}", flush=True)
print(f"  [DEBUG] target_rel: {obs_t[0,21:24].cpu().tolist()}", flush=True)
if obs_t.shape[1] == 30:
    print(f"  [DEBUG] bbox+cat: {obs_t[0,26:30].cpu().tolist()}", flush=True)
elif obs_t.shape[1] == 29:
    print(f"  [DEBUG] grip_force+bbox+cat: {obs_t[0,24:29].cpu().tolist()}", flush=True)
sys.stdout.flush()

action_repeat = args.action_repeat
held_action = None
repeat_counter = 0
print(f"  Action repeat: {action_repeat} steps", flush=True)

while episode < args.num_episodes and simulation_app.is_running():
    # BC forward — action_repeat마다 한 번만 예측
    if held_action is None or repeat_counter >= action_repeat:
        with torch.no_grad():
            obs_t = obs["policy"].to(env.device) if isinstance(obs, dict) else obs.to(env.device)
            held_action = policy.predict(obs_t) if is_gmm else policy(obs_t)
            # mean regression 보정: 팔 action 스케일업
            if args.arm_action_scale != 1.0:
                held_action = held_action.clone()
                held_action[:, :5] = (held_action[:, :5] * args.arm_action_scale).clamp(-1.0, 1.0)
        repeat_counter = 0

        # 매 BC 예측 시 디버그 (처음 5번)
        if step_count < 5 * action_repeat:
            a = held_action[0].cpu().tolist()
            o_arm = obs_t[0, :5].cpu().tolist()
            o_obj = obs_t[0, 21:24].cpu().tolist()
            print(f"  [t={step_count}] arm_obs={[f'{x:.3f}' for x in o_arm]} "
                  f"obj_rel={[f'{x:.3f}' for x in o_obj]} "
                  f"action={[f'{x:.3f}' for x in a]}", flush=True)

    # Env step — held_action 유지
    obs, reward, terminated, truncated, info = env.step(held_action)
    step_count += 1
    repeat_counter += 1

    done = terminated.any() or truncated.any()
    if done:
        episode += 1
        success = info.get("task_success", torch.zeros(1)).any().item()
        if success:
            successes += 1
        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {episode}/{args.num_episodes}: {status} "
              f"({step_count} steps, "
              f"cumulative: {successes}/{episode} = {successes/episode*100:.0f}%)",
              flush=True)
        step_count = 0
        held_action = None
        repeat_counter = 0
        obs, _ = env.reset()

        # 다음 에피소드 초기 상태 복원
        if demo_episodes and episode < len(demo_episodes):
            _restore_init_state(demo_episodes[episode])
            for _ in range(5):
                env.sim.step()
            env.robot.update(env.sim.cfg.dt)
            env.object_rigid.update(env.sim.cfg.dt)

print(f"\n  === 결과: {successes}/{args.num_episodes} 성공 "
      f"({successes/max(episode,1)*100:.0f}%) ===\n")

if demo_file is not None:
    demo_file.close()
env.close()
simulation_app.close()
