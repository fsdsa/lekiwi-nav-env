#!/usr/bin/env python3
"""
ResiP (Frozen DP + Residual PPO) 체크포인트를 Isaac Sim에서 GUI로 돌려보는 평가 스크립트.

eval_dp_bc.py와 동일한 환경 설정 + train_resip.py의 rollout 로직 결합.

Usage:
    # 랜덤 리셋
    python eval_resip.py --skill approach_and_grasp \
        --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --resip_checkpoint checkpoints/resip/resip_iter660.pt

    # HDF5 초기 상태 복원
    python eval_resip.py --skill approach_and_grasp \
        --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --resip_checkpoint checkpoints/resip/resip_iter660.pt \
        --demo demos/combined_skill2_20260227_091123.hdf5

    # DP BC만 (residual 없이) — resip_checkpoint 생략 시 base policy만 실행
    python eval_resip.py --skill approach_and_grasp \
        --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt
"""
from __future__ import annotations

import argparse
import os

# ── Args (AppLauncher args 포함) ──
parser = argparse.ArgumentParser(description="ResiP Eval in Isaac Sim (GUI)")
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place", "navigate"])
parser.add_argument("--dp_checkpoint", type=str, required=True,
                    help="Path to dp_bc.pt (frozen base policy)")
parser.add_argument("--resip_checkpoint", type=str, default="",
                    help="Path to resip_*.pt (residual policy). 생략 시 base DP만 실행")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--demo", type=str, default="",
                    help="HDF5 파일 경로 — 지정 시 에피소드 초기 상태를 복원하여 평가")
parser.add_argument("--inference_steps", type=int, default=4,
                    help="DDIM inference steps (default=4 for RL speed, 16 for quality)")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")
parser.add_argument("--grip_override", action="store_true", default=False,
                    help="Rule-based gripper override: GT grip curve based on EE distance")
# Navigate obstacle eval options
parser.add_argument("--num_obstacles_min", type=int, default=None,
                    help="Override min obstacles for eval (default: use env config)")
parser.add_argument("--num_obstacles_max", type=int, default=None,
                    help="Override max obstacles for eval (default: use env config)")
parser.add_argument("--obstacle_none_prob", type=float, default=None,
                    help="Override no-obstacle probability (0.0 = always obstacles)")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import h5py
import torch
import numpy as np

from isaaclab.utils.math import quat_apply

from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
if args.skill != "navigate":
    from lekiwi_skill2_env import EE_LOCAL_OFFSET


# ── 모델 로드 ──
def load_frozen_dp(ckpt_path, device, inference_steps=4):
    """Load DP BC checkpoint and freeze."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"],
        action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=inference_steps,
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)

    state_dict = ckpt["model_state_dict"]
    model_state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    norm_state = {k[len("normalizer."):]: v for k, v in state_dict.items() if k.startswith("normalizer.")}
    agent.model.load_state_dict(model_state)
    agent.normalizer.load_state_dict(norm_state, device=device)

    for param in agent.parameters():
        param.requires_grad = False
    agent.eval()

    print(f"Loaded frozen DP: {ckpt_path}")
    print(f"  obs_dim={cfg['obs_dim']}, act_dim={cfg['act_dim']}, "
          f"pred_horizon={cfg['pred_horizon']}, action_horizon={cfg['action_horizon']}")
    print(f"  down_dims={cfg.get('down_dims')}, inference_steps={inference_steps}")
    return agent, cfg


def load_residual_policy(ckpt_path, obs_dim, act_dim, device):
    """Load trained ResidualPolicy from ResiP checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    residual = ResidualPolicy(
        obs_dim=obs_dim,  # ResidualPolicy internally uses obs_dim + action_dim
        action_dim=act_dim,
        actor_hidden_size=saved_args.get("actor_hidden_size", 256),
        actor_num_layers=saved_args.get("actor_num_layers", 2),
        critic_hidden_size=saved_args.get("critic_hidden_size", 256),
        critic_num_layers=saved_args.get("critic_num_layers", 2),
        action_scale=saved_args.get("action_scale", 0.1),
        init_logstd=saved_args.get("init_logstd", -1.0),
        action_head_std=saved_args.get("action_head_std", 0.0),
        learn_std=False,
    ).to(device)

    residual.load_state_dict(ckpt["residual_policy_state_dict"])
    residual.eval()

    n_params = sum(p.numel() for p in residual.parameters())
    print(f"Loaded ResidualPolicy: {ckpt_path}")
    print(f"  params={n_params:,}, action_scale={residual.action_scale}")
    print(f"  iteration={ckpt.get('iteration', '?')}, "
          f"success_rate={ckpt.get('success_rate', '?')}")
    return residual


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

is_navigate = (args.skill == "navigate")

# ── Env 생성 ──
if args.skill == "navigate":
    from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg
    env_cfg = Skill1EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.enable_domain_randomization = False
    env_cfg.arm_limit_write_to_sim = False
    env_cfg.force_tucked_pose = True
    env_cfg.episode_length_s = 30.0
    env_cfg.render_obstacles = True  # GUI: show obstacle cylinders
    # Obstacle overrides for eval
    if args.num_obstacles_min is not None:
        env_cfg.num_obstacles_min = args.num_obstacles_min
    if args.num_obstacles_max is not None:
        env_cfg.num_obstacles_max = args.num_obstacles_max
    if args.obstacle_none_prob is not None:
        env_cfg.obstacle_none_prob = args.obstacle_none_prob
elif args.skill == "approach_and_grasp":
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = 1
elif args.skill == "carry_and_place":
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    env_cfg = Skill3EnvCfg()
    env_cfg.scene.num_envs = 1
    if args.handoff_buffer:
        env_cfg.handoff_buffer_path = args.handoff_buffer

if not is_navigate:
    # 평가 설정 — training과 동일한 grasp 파라미터
    env_cfg.enable_domain_randomization = False
    env_cfg.arm_limit_write_to_sim = False
    env_cfg.grasp_contact_threshold = 0.55
    env_cfg.grasp_require_contact = True
    env_cfg.grasp_gripper_threshold = 0.65
    env_cfg.grasp_max_object_dist = 0.25
    env_cfg.grasp_success_height = 1.00
    env_cfg.grasp_ee_max_dist = 0.10

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

if args.skill == "navigate":
    env = Skill1Env(cfg=env_cfg)
elif args.skill == "approach_and_grasp":
    env = Skill2Env(cfg=env_cfg)
else:
    env = Skill3Env(cfg=env_cfg)

# ── 모델 로드 ──
device = env.device
dp_agent, dp_cfg = load_frozen_dp(args.dp_checkpoint, device, args.inference_steps)
obs_dim = dp_cfg["obs_dim"]
act_dim = dp_cfg["act_dim"]

residual_policy = None
if args.resip_checkpoint and os.path.isfile(args.resip_checkpoint):
    residual_policy = load_residual_policy(args.resip_checkpoint, obs_dim, act_dim, device)

# Per-dim action scale — must match training exactly
per_dim_action_scale = torch.zeros(act_dim, device=device)
if is_navigate:
    # Navigate: arm/gripper=0, base only
    per_dim_action_scale[6:9] = 0.25  # base
else:
    per_dim_action_scale[0:5] = 0.20   # arm
    per_dim_action_scale[5]   = 0.25   # gripper (v7c 학습값)
    per_dim_action_scale[6:9] = 0.35   # base

if args.grip_override:
    mode_str = "DP + Grip Override (GT curve)"
elif residual_policy:
    mode_str = "ResiP (DP + Residual)"
else:
    mode_str = "DP BC only"
demo_str = "HDF5 초기 상태 복원" if demo_episodes else "랜덤 리셋"
print(f"\n{'='*60}")
print(f"  {mode_str} Eval — {args.skill} ({demo_str})")
print(f"  DP: {args.dp_checkpoint}")
if residual_policy:
    print(f"  Residual: {args.resip_checkpoint}")
print(f"  Episodes: {args.num_episodes}")
print(f"{'='*60}\n")


# ── 환경 초기 상태 복원 헬퍼 ──
def _restore_init_state(ep_data):
    env_id = torch.tensor([0], device=device)
    ea = ep_data["ep_attrs"]

    if "robot_init_pos" in ea and "robot_init_quat" in ea:
        rs = env.robot.data.root_state_w.clone()
        rs[0, 0:3] = torch.tensor(ea["robot_init_pos"], dtype=torch.float32, device=device)
        rs[0, 3:7] = torch.tensor(ea["robot_init_quat"], dtype=torch.float32, device=device)
        rs[0, 7:] = 0.0
        env.robot.write_root_state_to_sim(rs, env_id)
        env.home_pos_w[0] = rs[0, :3]

    init_joints = torch.tensor(ep_data["obs"][0, 0:6], dtype=torch.float32, device=device)
    jp = env.robot.data.default_joint_pos[0:1].clone()
    jp[0, env.arm_idx] = init_joints
    jv = torch.zeros_like(jp)
    env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_id)

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

    for _ in range(10):
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)


# Navigate: ordered direction schedule (v3: 4 commands, no strafe)
_NAV_DIR_SCHEDULE = [
    ([0, 1, 0],  "FORWARD"),
    ([0, -1, 0], "BACKWARD"),
    ([0, 0, 0.33],  "TURN LEFT (CCW)"),
    ([0, 0, -0.33], "TURN RIGHT (CW)"),
]
_NAV_DIR_LABELS = {
    (0, 1, 0): "FORWARD", (0, -1, 0): "BACKWARD",
    (0, 0, 1): "TURN LEFT (CCW)", (0, 0, -1): "TURN RIGHT (CW)",
}
_nav_schedule_idx = 0

# ── 실행 루프 ──
episode = 0
successes = 0
step_count = 0
obs, _ = env.reset()

# Navigate: force first direction from schedule
if is_navigate:
    cmd_vec, cmd_label = _NAV_DIR_SCHEDULE[_nav_schedule_idx % len(_NAV_DIR_SCHEDULE)]
    env._direction_cmd[0] = torch.tensor(cmd_vec, dtype=torch.float32, device=device)
    n_obs = env._obstacle_valid[0].sum().item()
    print(f"  [Navigate] direction_cmd: {cmd_label} (schedule {_nav_schedule_idx}) | obstacles={n_obs:.0f}")

if demo_episodes:
    _restore_init_state(demo_episodes[0])
    for _ in range(5):
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)

dp_agent.reset()
def _nav_dir_label(cmd_tensor):
    c = cmd_tensor.cpu().tolist()
    best_label, best_dot = "UNKNOWN", -1.0
    for key, label in _NAV_DIR_LABELS.items():
        dot = sum(a * b for a, b in zip(c, key))
        if dot > best_dot:
            best_dot, best_label = dot, label
    return best_label

if not is_navigate:
    # EE body index + local offset
    fixed_jaw_idx, _ = env.robot.find_bodies(["Wrist_Roll_08c_v1"])
    ee_local_offset = torch.tensor(EE_LOCAL_OFFSET, device=device).unsqueeze(0)  # (1, 3)
    print(f"  EE body: fixed_jaw={fixed_jaw_idx[0]}, offset={EE_LOCAL_OFFSET}")

    def get_distances():
        wrist_pos = env.robot.data.body_pos_w[:, fixed_jaw_idx[0], :]
        wrist_quat = env.robot.data.body_quat_w[:, fixed_jaw_idx[0], :]
        ee_pos = wrist_pos + quat_apply(wrist_quat, ee_local_offset)
        obj_pos = env.object_pos_w
        ee_dist = torch.norm(ee_pos - obj_pos, dim=-1).item()
        base_pos = env.robot.data.root_pos_w[:, :2]
        obj_pos_xy = env.object_pos_w[:, :2]
        base_dist = torch.norm(base_pos - obj_pos_xy, dim=-1).item()
        gripper_pos = env.robot.data.joint_pos[:, env.gripper_idx].item()
        obj_z = env.object_pos_w[:, 2].item()
        env_z = env.scene.env_origins[:, 2].item() if hasattr(env.scene, "env_origins") else 0.0
        ee_z = ee_pos[0, 2].item() - env_z
        return ee_dist, base_dist, gripper_pos, obj_z - env_z, ee_z

    def get_grasp_debug():
        """Grasp 판정에 쓰이는 모든 값 반환."""
        grip = env.robot.data.joint_pos[:, env.gripper_idx].item()
        grip_closed = grip < float(env.cfg.grasp_gripper_threshold)
        contact = 0.0
        if env.contact_sensor is not None:
            contact = env._contact_force_per_env().item()
        has_contact = contact > float(env.cfg.grasp_contact_threshold)
        wrist_pos = env.robot.data.body_pos_w[:, fixed_jaw_idx[0], :]
        wrist_quat = env.robot.data.body_quat_w[:, fixed_jaw_idx[0], :]
        ee_pos = wrist_pos + quat_apply(wrist_quat, ee_local_offset)
        ee_to_obj = torch.norm(ee_pos - env.object_pos_w, dim=-1).item()
        between = ee_to_obj < float(env.cfg.grasp_ee_max_dist)
        grasped = env.object_grasped[0].item()
        gcf = 0.0
        if hasattr(env, 'ground_contact_sensor') and env.ground_contact_sensor is not None:
            gcf = env._ground_contact_force_per_env().item()
        grip_on_ground = gcf > 1.0 and not grasped
        return {
            "grip": grip, "grip_closed": grip_closed,
            "contact": contact, "has_contact": has_contact,
            "ee_to_obj": ee_to_obj, "between_jaws": between,
            "grasped": grasped,
            "grip_on_ground": grip_on_ground, "gcf": gcf,
        }

ep_min_ee = 999.0
ep_min_base = 999.0
ep_min_ee_z = 999.0
window_min_ee_z = 999.0
prev_grasped = False
ep_grasped = False
ep_lifted = False
lift_sustain = 0

# Navigate tracking metrics
nav_lin_errors = []
nav_ang_errors = []
nav_collision_fov_count = 0
nav_collision_any_count = 0
nav_ep_collision_fov = 0
nav_ep_collision_any = 0
nav_total_collisions_fov = 0
nav_total_collisions_any = 0

while episode < args.num_episodes and simulation_app.is_running():
    obs_t = obs["policy"].to(device) if isinstance(obs, dict) else obs.to(device)

    with torch.no_grad():
        # 1. Base action from frozen DP (handles action chunking internally)
        base_naction = dp_agent.base_action_normalized(obs_t)

        if residual_policy is not None:
            # 2. Normalize obs
            nobs = dp_agent.normalizer(obs_t, "obs", forward=True)
            nobs = torch.clamp(nobs, -3, 3)
            nobs = torch.nan_to_num(nobs, nan=0.0)

            # 3. Residual input: [nobs, base_naction]
            residual_nobs = torch.cat([nobs, base_naction], dim=-1)

            # 4. Deterministic residual (raw actor mean, clamped)
            residual_naction = residual_policy.actor_mean(residual_nobs)
            residual_naction = torch.clamp(residual_naction, -1.0, 1.0)

            # 5. Per-dim scale (v26: no post-grasp scale)
            naction = base_naction + residual_naction * per_dim_action_scale
        else:
            naction = base_naction

        # 6. Denormalize → env action
        action = dp_agent.normalizer(naction, "action", forward=False)

    # ── Rule-based gripper override (GT grip curve) ──
    if args.grip_override:
        ee_d_now = get_distances()[0]
        if ee_d_now < 0.05:
            grip_target = 0.30   # tight close for secure grasp
        elif ee_d_now < 0.20:
            t = (ee_d_now - 0.05) / 0.15
            grip_target = 0.30 + t * 1.18  # 0.30→1.48 smooth
        else:
            grip_target = None  # keep DP's gripper
        if grip_target is not None:
            # Blend: 80% override, 20% DP (smooth transition)
            dp_grip = action[0, 5].item()
            action[0, 5] = 0.8 * grip_target + 0.2 * dp_grip

    # 디버그 (처음 5스텝만)
    if step_count < 5:
        a = action[0].cpu().tolist()
        print(f"  [t={step_count}] action={[f'{x:.3f}' for x in a]}", flush=True)

    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    if is_navigate:
        # Navigate: 속도 추종 메트릭
        cmd = env._direction_cmd[0]  # (3,)
        actual_lin = env.robot.data.root_lin_vel_b[0, :2]
        actual_wz = env.robot.data.root_ang_vel_b[0, 2]
        target_lin = cmd[:2] * env.cfg.max_lin_vel
        target_wz = cmd[2] * env.cfg.max_ang_vel
        lin_err = (target_lin - actual_lin).pow(2).sum().sqrt().item()
        ang_err = abs(target_wz.item() - actual_wz.item())
        nav_lin_errors.append(lin_err)
        nav_ang_errors.append(ang_err)

        # Obstacle collision check (FOV vs all)
        metrics = env._compute_metrics()
        min_fov = metrics["min_obs_dist_fov"][0].item()
        min_all = metrics["min_obs_dist_all"][0].item()
        col_dist = float(env.cfg.collision_dist)
        if min_fov < col_dist:
            nav_ep_collision_fov += 1
        if min_all < col_dist:
            nav_ep_collision_any += 1

        if step_count % 50 == 0:
            dir_label = _nav_dir_label(cmd)
            vx = env.robot.data.root_lin_vel_b[0, 0].item()
            vy = env.robot.data.root_lin_vel_b[0, 1].item()
            wz = actual_wz.item()
            pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
            avg_lin = sum(nav_lin_errors[-50:]) / len(nav_lin_errors[-50:])
            avg_ang = sum(nav_ang_errors[-50:]) / len(nav_ang_errors[-50:])
            obs_str = f"obs_fov={min_fov:.2f} obs_all={min_all:.2f}"
            print(f"    [t={step_count:4d}] dir={dir_label} | "
                  f"vel=(vx={vx:+.2f},vy={vy:+.2f},wz={wz:+.2f}) | "
                  f"pos=({pos[0]:+.2f},{pos[1]:+.2f}) | "
                  f"lin_err={avg_lin:.3f} ang_err={avg_ang:.3f} | {obs_str}", flush=True)

        done = terminated.any() or truncated.any()
        term_reason = ""
        if terminated.any() and not truncated.any():
            if min_all < col_dist:
                term_reason = " [COLLISION]"
            else:
                term_reason = " [OOB/FELL]"
        if done:
            episode += 1
            avg_lin = sum(nav_lin_errors) / max(len(nav_lin_errors), 1)
            avg_ang = sum(nav_ang_errors) / max(len(nav_ang_errors), 1)
            dir_label = _nav_dir_label(env._direction_cmd[0])
            nav_total_collisions_fov += nav_ep_collision_fov
            nav_total_collisions_any += nav_ep_collision_any
            survived = "TIMEOUT" if truncated.any() else f"TERMINATED{term_reason}"
            print(f"  Episode {episode}/{args.num_episodes}: {dir_label} | "
                  f"{step_count} steps | {survived} | "
                  f"avg_lin_err={avg_lin:.4f} avg_ang_err={avg_ang:.4f} | "
                  f"col_fov={nav_ep_collision_fov} col_any={nav_ep_collision_any}",
                  flush=True)
            step_count = 0
            nav_lin_errors.clear()
            nav_ang_errors.clear()
            nav_ep_collision_fov = 0
            nav_ep_collision_any = 0
            dp_agent.reset()
            obs, _ = env.reset()
            if episode < args.num_episodes:
                _nav_schedule_idx += 1
                cmd_vec, cmd_label = _NAV_DIR_SCHEDULE[_nav_schedule_idx % len(_NAV_DIR_SCHEDULE)]
                env._direction_cmd[0] = torch.tensor(cmd_vec, dtype=torch.float32, device=device)
                n_obs = env._obstacle_valid[0].sum().item()
                print(f"  [Navigate] direction_cmd: {cmd_label} (schedule {_nav_schedule_idx}) | obstacles={n_obs:.0f}")
    else:
        ee_d, base_d, grip, obj_z, ee_z = get_distances()
        ep_min_ee = min(ep_min_ee, ee_d)
        ep_min_base = min(ep_min_base, base_d)
        ep_min_ee_z = min(ep_min_ee_z, ee_z)
        window_min_ee_z = min(window_min_ee_z, ee_z)

        # Grasp 이벤트 감지 + 상세 출력
        g = get_grasp_debug()
        if g["grasped"] and not prev_grasped:
            ep_grasped = True
            print(f"    ★ GRASP at t={step_count} | "
                  f"grip={g['grip']:.3f}({'✓' if g['grip_closed'] else '✗'}) "
                  f"contact={g['contact']:.3f}({'✓' if g['has_contact'] else '✗'}) "
                  f"ee_to_obj={g['ee_to_obj']:.3f}({'✓' if g['between_jaws'] else '✗'}<{env.cfg.grasp_ee_max_dist})",
                  flush=True)
        prev_grasped = g["grasped"]

        held_now = (obj_z > 0.05 and g["grasped"] and g["grip_closed"] and g["ee_to_obj"] < 0.20)
        if held_now:
            lift_sustain += 1
        else:
            lift_sustain = 0
        if lift_sustain >= 50 and not ep_lifted:
            ep_lifted = True
            print(f"    ★ LIFT at t={step_count} | objZ={obj_z:.3f} ee={g['ee_to_obj']:.3f} grip={g['grip']:.3f} sustain={lift_sustain}", flush=True)

        if step_count % 50 == 0:
            status = ""
            if ep_lifted:
                status = " [LIFTED]"
            elif g["grasped"]:
                status = f" [GRASPED sustain={lift_sustain}]"
            ground_str = " **GROUND**" if g["grip_on_ground"] else ""
            print(f"    [t={step_count:4d}] EE={ee_d:.3f} Base={base_d:.3f} grip={grip:.3f} objZ={obj_z:.3f} "
                  f"eeZ={ee_z:.4f}(min={window_min_ee_z:.4f}) contact={g['contact']:.2f} ee2obj={g['ee_to_obj']:.3f}"
                  f" gcf={g['gcf']:.2f}{ground_str}{status}", flush=True)
            window_min_ee_z = 999.0

        done = terminated.any() or truncated.any()
        if done:
            episode += 1
            success = info.get("task_success", torch.zeros(1)).any().item()
            if success:
                successes += 1
            status = "SUCCESS" if success else "FAIL"
            grasp_lift = f"grasp={'Y' if ep_grasped else 'N'} lift={'Y' if ep_lifted else 'N'}"
            print(f"  Episode {episode}/{args.num_episodes}: {status} "
                  f"({step_count} steps, {grasp_lift}, "
                  f"min_EE={ep_min_ee:.3f} min_Base={ep_min_base:.3f} min_eeZ={ep_min_ee_z:.4f} | "
                  f"cumulative: {successes}/{episode} = {successes/episode*100:.0f}%)",
                  flush=True)
            step_count = 0
            ep_min_ee = 999.0
            ep_min_base = 999.0
            ep_min_ee_z = 999.0
            window_min_ee_z = 999.0
            prev_grasped = False
            ep_grasped = False
            ep_lifted = False
            lift_sustain = 0
            dp_agent.reset()
            obs, _ = env.reset()

            if demo_episodes and episode < len(demo_episodes):
                _restore_init_state(demo_episodes[episode])
                for _ in range(5):
                    env.sim.step()
                env.robot.update(env.sim.cfg.dt)
                env.object_rigid.update(env.sim.cfg.dt)

if is_navigate:
    print(f"\n  === Navigate eval 완료: {episode} episodes ===")
    print(f"  총 충돌: FOV={nav_total_collisions_fov} steps, ALL={nav_total_collisions_any} steps")
    print()
else:
    print(f"\n  === 결과: {successes}/{args.num_episodes} 성공 "
          f"({successes/max(episode,1)*100:.0f}%) ===\n")

if demo_file is not None:
    demo_file.close()
env.close()
simulation_app.close()
