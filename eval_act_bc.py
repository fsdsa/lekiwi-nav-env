#!/usr/bin/env python3
"""
ACT BC (fork) 체크포인트을 Isaac Sim에서 GUI로 돌려보는 평가 스크립트.

2가지 실행 모드:
  1. chunk-execute (기본): chunk 전체(K=20 steps)를 순서대로 실행 후 re-plan
  2. temporal-agg: 매 step마다 chunk 예측, 겹치는 예측의 지수가중평균 사용

Usage:
    # Chunk-execute 모드 (기본, 빠름)
    python eval_act_bc.py --skill approach_and_grasp \
        --act_checkpoint checkpoints/act_fork/policy_best.ckpt

    # Temporal ensembling 모드 (더 부드러운 action)
    python eval_act_bc.py --skill approach_and_grasp \
        --act_checkpoint checkpoints/act_fork/policy_best.ckpt \
        --temporal_agg

    # HDF5 초기 상태 복원 평가
    python eval_act_bc.py --skill approach_and_grasp \
        --act_checkpoint checkpoints/act_fork/policy_best.ckpt \
        --demo demos/combined_skill2_20260227_091123.hdf5

    # Skill-3
    python eval_act_bc.py --skill carry_and_place \
        --act_checkpoint checkpoints/act_fork_skill3/policy_best.ckpt \
        --handoff_buffer handoff_buffer.pkl
"""
from __future__ import annotations

import argparse
import os

# ── Args (AppLauncher args 포함) ──
parser = argparse.ArgumentParser(description="ACT BC Eval in Isaac Sim (GUI)")
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place"])
parser.add_argument("--act_checkpoint", type=str, required=True,
                    help="Path to ACT fork checkpoint (.ckpt file). "
                         "config.pkl and dataset_stats.pkl must be in the same directory.")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--demo", type=str, default="",
                    help="HDF5 파일 경로 — 지정 시 에피소드 초기 상태를 복원하여 평가")
parser.add_argument("--temporal_agg", action="store_true",
                    help="Temporal ensembling 사용 (매 step re-plan + 가중평균)")
parser.add_argument("--replan_freq", type=int, default=0,
                    help="chunk-execute 모드에서 re-plan 주기 (0=chunk_size마다, N=N step마다)")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")
parser.add_argument("--arm_action_scale", type=float, default=1.0,
                    help="팔 action[0:5] 스케일 (mean regression 보정, 1.0~1.5)")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import h5py
import pickle
import torch
import numpy as np

# ── ACT fork 경로 추가 ──
ACT_FORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "act_fork")
if ACT_FORK_DIR not in sys.path:
    sys.path.insert(0, ACT_FORK_DIR)

from policy import ACTPolicy


# ── ACT fork 체크포인트 로드 ──
def load_act_fork(ckpt_path):
    """Load ACT fork policy with normalization stats.
    
    Args:
        ckpt_path: Path to .ckpt file. config.pkl and dataset_stats.pkl
                   must be in the same directory.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))

    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    policy_config = {
        'lr': config.get('lr', 1e-4),
        'num_queries': config['chunk_size'],
        'kl_weight': config['kl_weight'],
        'hidden_dim': config['hidden_dim'],
        'dim_feedforward': config['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': config.get('enc_layers', 4),
        'dec_layers': config.get('dec_layers', 7),
        'nheads': config.get('nheads', 8),
        'camera_names': [],
        'state_only': True,
        'state_dim': config['state_dim'],
        'action_dim': config['action_dim'],
        'action_loss_weights': config.get('action_loss_weights', None),
    }

    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=False)
    print(f"Loaded: {ckpt_path} ({loading_status})")
    policy.eval()

    return policy, stats, config


policy, norm_stats, act_config = load_act_fork(args.act_checkpoint)

chunk_size = act_config['chunk_size']
obs_dim = act_config['state_dim']
action_dim = act_config['action_dim']
n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

# Temporal ensembling 설정
if args.temporal_agg:
    exec_mode = "temporal-agg (re-plan every step)"
else:
    exec_mode = f"chunk-execute (K={chunk_size})"

# Re-plan frequency (chunk-execute 모드)
replan_freq = args.replan_freq if args.replan_freq > 0 else chunk_size

# ── HDF5 데모 로드 (선택) ──
demo_episodes = []
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
    env_cfg.spawn_heading_noise_std = 0.1
    env_cfg.spawn_heading_max_rad = 0.26

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

policy = policy.to(env.device)

# ── Normalization stats → GPU tensors ──
obs_mean = torch.from_numpy(norm_stats['obs_mean']).float().to(env.device)
obs_std = torch.from_numpy(norm_stats['obs_std']).float().to(env.device)
action_mean = torch.from_numpy(norm_stats['action_mean']).float().to(env.device)
action_std = torch.from_numpy(norm_stats['action_std']).float().to(env.device)


def normalize_obs(obs):
    """Normalize observation tensor (on GPU)."""
    return (obs - obs_mean) / obs_std


def denormalize_action(action):
    """Denormalize action tensor (on GPU)."""
    return action * action_std + action_mean


# ── 출력 ──
mode_str = "HDF5 초기 상태 복원" if demo_episodes else "랜덤 리셋"
print(f"\n{'='*60}")
print(f"  ACT BC Eval (fork) — {args.skill} ({mode_str})")
print(f"  Checkpoint: {args.act_checkpoint}")
print(f"  Model: {n_params:,} params, state_dim={obs_dim}, action_dim={action_dim}, chunk_size={chunk_size}")
print(f"  Exec mode: {exec_mode}")
print(f"  Episodes: {args.num_episodes}")
print(f"  arm_action_scale: {args.arm_action_scale}")
print(f"  Normalization: obs_mean range [{obs_mean.min():.3f}, {obs_mean.max():.3f}], "
      f"obs_std range [{obs_std.min():.3f}, {obs_std.max():.3f}]")
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


# ── Temporal ensembling state ──
_te_all_time_actions = None  # (max_T, max_T+K, action_dim) normalized
_te_step = 0
_te_k = 0.01  # exponential weight decay


def _te_reset():
    global _te_all_time_actions, _te_step
    _te_all_time_actions = None
    _te_step = 0


def _te_get_action(obs_normalized):
    """Temporal ensembling: predict chunk, store, weighted average.
    Input/output are normalized tensors on GPU."""
    global _te_all_time_actions, _te_step

    K = chunk_size
    t = _te_step

    # Predict chunk (normalized)
    with torch.no_grad():
        all_actions = policy(obs_normalized.unsqueeze(0))  # (1, K, action_dim)

    if _te_all_time_actions is None:
        max_T = 10000
        _te_all_time_actions = torch.zeros(
            max_T, max_T + K, action_dim, device=env.device
        )

    _te_all_time_actions[t, t:t + K] = all_actions[0]
    _te_step += 1

    # Weighted average over overlapping chunks
    actions_for_t = _te_all_time_actions[:t + 1, t]  # (t+1, action_dim)
    actions_populated = torch.all(actions_for_t != 0, dim=1)
    actions_for_t = actions_for_t[actions_populated]

    n = len(actions_for_t)
    exp_weights = torch.exp(-_te_k * torch.arange(n, device=env.device, dtype=torch.float32))
    exp_weights = exp_weights / exp_weights.sum()
    raw_action = (actions_for_t * exp_weights.unsqueeze(1)).sum(dim=0)  # (action_dim,)

    return raw_action  # normalized


# ── 실행 루프 ──
episode = 0
successes = 0
step_count = 0
obs, _ = env.reset()

# HDF5 모드: 첫 에피소드 초기 상태 복원
if demo_episodes:
    _restore_init_state(demo_episodes[0])
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

# State reset
_te_reset()
action_chunk_buffer = None  # (K, action_dim) normalized
chunk_step_idx = 0

while episode < args.num_episodes and simulation_app.is_running():
    obs_t = obs["policy"].to(env.device) if isinstance(obs, dict) else obs.to(env.device)

    # ── Normalize observation ──
    obs_norm = normalize_obs(obs_t[0])  # (obs_dim,)

    if args.temporal_agg:
        # ---- Temporal ensembling 모드: 매 step마다 re-plan ----
        action_norm = _te_get_action(obs_norm)  # (action_dim,) normalized
        action = denormalize_action(action_norm).unsqueeze(0)  # (1, action_dim)
    else:
        # ---- Chunk-execute 모드: chunk 예측 후 순서대로 실행 ----
        if action_chunk_buffer is None or chunk_step_idx >= replan_freq:
            with torch.no_grad():
                all_actions = policy(obs_norm.unsqueeze(0))  # (1, K, action_dim) normalized
                action_chunk_buffer = all_actions[0]  # (K, action_dim) normalized
            chunk_step_idx = 0

            # 디버그: 새 chunk 예측 시 첫 action 출력 (처음 5번만)
            if step_count < 5 * chunk_size:
                o_arm = obs_t[0, :5].cpu().tolist()
                o_obj = obs_t[0, 21:24].cpu().tolist()
                a0_denorm = denormalize_action(action_chunk_buffer[0]).cpu().tolist()
                print(f"  [t={step_count}] NEW CHUNK | "
                      f"arm_obs={[f'{x:.3f}' for x in o_arm]} "
                      f"obj_rel={[f'{x:.3f}' for x in o_obj]} "
                      f"action[0]={[f'{x:.3f}' for x in a0_denorm]}", flush=True)

        # Denormalize single action
        action_norm = action_chunk_buffer[chunk_step_idx]  # (action_dim,) normalized
        action = denormalize_action(action_norm).unsqueeze(0)  # (1, action_dim)
        chunk_step_idx += 1

    # arm action scale 보정
    if args.arm_action_scale != 1.0:
        action = action.clone()
        action[:, :5] = (action[:, :5] * args.arm_action_scale).clamp(-1.0, 1.0)

    # Env step
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

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
        action_chunk_buffer = None
        chunk_step_idx = 0
        _te_reset()
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
