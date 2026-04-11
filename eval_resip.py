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
                    choices=["navigate", "approach_and_grasp", "carry", "carry_and_place"])
parser.add_argument("--dp_checkpoint", type=str, required=True,
                    help="Path to dp_bc.pt (frozen base policy)")
parser.add_argument("--resip_checkpoint", type=str, default="",
                    help="Path to resip_*.pt (residual policy). 생략 시 base DP만 실행")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=6000,
                    help="Max steps per episode (default 6000)")
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
# Carry mode: S2 expert → carry BC
parser.add_argument("--s2_dp_checkpoint", type=str, default="",
                    help="carry: S2 frozen DP checkpoint")
parser.add_argument("--s2_resip_checkpoint", type=str, default="",
                    help="carry: S2 residual policy checkpoint")
parser.add_argument("--carry_steps", type=int, default=600,
                    help="carry: Phase 2 max steps")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# PhysX setDriveTarget 스팸 suppress
import carb
carb.settings.get_settings().set("/physics/suppressReadback", True)
_physx_logger = carb.logging.acquire_logging()
_orig_log_fn = None
def _filter_physx_log(source, level, filename, line, message):
    if "setDriveTarget" in message:
        return
    if _orig_log_fn:
        _orig_log_fn(source, level, filename, line, message)
try:
    import omni.physx
    # PhysX error threshold 높이기
    carb.settings.get_settings().set("/physics/reportKinematicKinematicPairs", False)
    carb.settings.get_settings().set("/physics/suppressReadback", True)
except Exception:
    pass

import sys
import h5py
import torch
import numpy as np

from isaaclab.utils.math import quat_apply

from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
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

# ── Env 생성 ──
if args.skill == "navigate":
    from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg
    env_cfg = Skill1EnvCfg()
    env_cfg.scene.num_envs = getattr(args, 'num_envs', 1) or 1
    env_cfg.force_tucked_pose = True  # arm all-zero로 덮어쓸 예정
elif args.skill in ("approach_and_grasp", "carry"):
    from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = 1
    if args.skill == "carry":
        # train_resip combined_s2_s3와 동일한 config
        env_cfg.grasp_contact_threshold = 0.55
        env_cfg.grasp_gripper_threshold = 0.65
        env_cfg.grasp_max_object_dist = 0.50
        env_cfg.grasp_success_height = 100.0
        env_cfg.lift_hold_steps = 0
        env_cfg.dest_object_fixed = False
        env_cfg.dest_object_scale = 0.56
        env_cfg.dest_object_mass = 50.0
        env_cfg.spawn_heading_noise_std = 0.3
        env_cfg.spawn_heading_max_rad = 0.5
elif args.skill == "carry_and_place":
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    env_cfg = Skill3EnvCfg()
    env_cfg.scene.num_envs = 1
    if args.handoff_buffer:
        env_cfg.handoff_buffer_path = args.handoff_buffer

# 평가 설정 — training과 동일한 grasp 파라미터
env_cfg.enable_domain_randomization = False
env_cfg.arm_limit_write_to_sim = False
env_cfg.grasp_contact_threshold = 0.55   # training env default와 일치
env_cfg.grasp_require_contact = True    # 물리 접촉 기반 grasp 판정
env_cfg.grasp_gripper_threshold = 0.65  # training과 일치
env_cfg.grasp_max_object_dist = 0.25    # training env default와 일치
env_cfg.grasp_success_height = 1.00     # 사실상 비활성화 — lift 중 auto-reset 방지
env_cfg.grasp_ee_max_dist = 0.10        # training env default와 일치

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
    pass  # env default 사용 (spawn_heading_noise_std=0.35, spawn_heading_max_rad=0.76)

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
    env._tucked_pose = torch.zeros(5, dtype=torch.float32, device=env.device)
    import lekiwi_skill1_env as _s1mod
    _s1mod._TUCKED_GRIPPER_RAD = 0.0
elif args.skill in ("approach_and_grasp", "carry"):
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

# Per-dim action scale — must match training exactly (v27: arm+gripper+base)
per_dim_action_scale = torch.zeros(act_dim, device=device)
per_dim_action_scale[0:5] = 0.20   # arm
per_dim_action_scale[5]   = 0.30   # gripper
per_dim_action_scale[6:9] = 0.35   # base

# ── Carry: S2 expert 로드 ──
s2_dp_agent = None
s2_residual = None
s2_scale = None
if args.skill == "carry":
    if not args.s2_dp_checkpoint:
        print("ERROR: --skill carry requires --s2_dp_checkpoint")
        simulation_app.close()
        sys.exit(1)
    s2_dp_agent, s2_cfg = load_frozen_dp(args.s2_dp_checkpoint, device, inference_steps=4)
    s2_scale = torch.zeros(act_dim, device=device)
    s2_scale[0:5] = 0.20; s2_scale[5] = 0.30; s2_scale[6:9] = 0.35
    if args.s2_resip_checkpoint and os.path.isfile(args.s2_resip_checkpoint):
        s2_residual = load_residual_policy(args.s2_resip_checkpoint, s2_cfg["obs_dim"], act_dim, device)

# Carry arm interpolation constants
import numpy as np
S3_ARM_END = np.array([+0.002, -0.193, +0.295, -1.306, +0.006], dtype=np.float64)
S3_GRIP_END = 0.15

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


# ── Navigate 방향 스케줄 ──
_nav_dir_schedule = None
if args.skill == "navigate":
    _nav_dir_schedule = [
        ([0, 1, 0],  "FORWARD"),
        ([0, -1, 0], "BACKWARD"),
        ([-1, 0, 0], "STRAFE LEFT"),
        ([1, 0, 0],  "STRAFE RIGHT"),
        ([0, 0, 1],  "TURN LEFT"),
        ([0, 0, -1], "TURN RIGHT"),
    ]

# ── 실행 루프 ──
episode = 0
successes = 0
step_count = 0
obs, _ = env.reset()
if _nav_dir_schedule and episode < len(_nav_dir_schedule):
    _cmd, _lbl = _nav_dir_schedule[episode]
    env._direction_cmd[0] = torch.tensor(_cmd, dtype=torch.float32, device=device)
    print(f"  [Navigate] direction: {_lbl}")

if demo_episodes:
    _restore_init_state(demo_episodes[0])
    for _ in range(5):
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)

dp_agent.reset()

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
    if args.skill == "navigate":
        return {"grip": 0.0, "grip_closed": False, "contact": 0.0, "has_contact": False,
                "ee_to_obj": 0.0, "between": False, "grasped": False, "gcf": 0.0,
                "grip_on_ground": False}
    grip = env.robot.data.joint_pos[:, env.gripper_idx].item()
    grip_closed = grip < float(env.cfg.grasp_gripper_threshold)
    # Contact force
    contact = 0.0
    if env.contact_sensor is not None:
        contact = env._contact_force_per_env().item()
    has_contact = contact > float(env.cfg.grasp_contact_threshold)
    # EE offset distance
    wrist_pos = env.robot.data.body_pos_w[:, fixed_jaw_idx[0], :]
    wrist_quat = env.robot.data.body_quat_w[:, fixed_jaw_idx[0], :]
    ee_pos = wrist_pos + quat_apply(wrist_quat, ee_local_offset)
    ee_to_obj = torch.norm(ee_pos - env.object_pos_w, dim=-1).item()
    between = ee_to_obj < float(env.cfg.grasp_ee_max_dist)
    grasped = env.object_grasped[0].item()
    # Ground contact (R8과 동일 로직)
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

# Carry state
carry_phase = 1 if args.skill == "carry" else 0  # 1=S2 expert, 2=carry BC
carry_step = 0
carry_lift_counter = 0
S2_LIFT_HOLD = 400
carry_arm_start = None
carry_grip_start = 0.276

# Carry direction schedule (6방향 순서)
_carry_dir_schedule = None
carry_dir_cmd_t = None
if args.skill == "carry":
    _carry_dir_schedule = [
        ([0, 1, 0],  "FORWARD"),
        ([0, -1, 0], "BACKWARD"),
        ([-1, 0, 0], "STRAFE LEFT"),
        ([1, 0, 0],  "STRAFE RIGHT"),
        ([0, 0, 1],  "TURN LEFT"),
        ([0, 0, -1], "TURN RIGHT"),
    ]
    _cmd0, _lbl0 = _carry_dir_schedule[0]
    carry_dir_cmd_t = torch.tensor([_cmd0], dtype=torch.float32, device=device)
    print(f"  [Carry] direction: {_lbl0}")

while episode < args.num_episodes and simulation_app.is_running():
    obs_t = obs["policy"].to(device) if isinstance(obs, dict) else obs.to(device)

    with torch.no_grad():
        # ── Carry Phase 1: S2 expert ──
        if args.skill == "carry" and carry_phase == 1:
            s2_ba = s2_dp_agent.base_action_normalized(obs_t)
            if s2_residual is not None:
                s2_no = s2_dp_agent.normalizer(obs_t, "obs", forward=True)
                s2_no = torch.nan_to_num(torch.clamp(s2_no, -3, 3), nan=0.0)
                s2_ro = torch.cat([s2_no, s2_ba], dim=-1)
                s2_ram = s2_residual.actor_mean(s2_ro)
                s2_ram = torch.clamp(s2_ram, -1.0, 1.0)
                naction = s2_ba + s2_ram * s2_scale
            else:
                naction = s2_ba
            action = s2_dp_agent.normalizer(naction, "action", forward=False)

            # Lift detection
            _objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            _grasped = bool(env.object_grasped[0].item()) if hasattr(env, 'object_grasped') else False
            if _grasped and _objZ > 0.05:
                carry_lift_counter += 1
            else:
                carry_lift_counter = 0

            if step_count % 25 == 0:
                _gp = env.robot.data.joint_pos[0, env.gripper_idx].item()
                print(f"    [S2] step={step_count} grip={_gp:.3f} objZ={_objZ:.3f} lift={carry_lift_counter}/{S2_LIFT_HOLD}", flush=True)

            # Topple
            if _objZ < 0.026 and step_count > 20:
                print(f"    [TOPPLE] objZ={_objZ:.3f} — 리셋")
                carry_lift_counter = 0; carry_phase = 1; carry_step = 0
                s2_dp_agent.reset(); dp_agent.reset()
                obs, _ = env.reset()
                step_count = 0
                continue

            # S2→Carry 전환
            if carry_lift_counter >= S2_LIFT_HOLD:
                carry_arm_start = env.robot.data.joint_pos[0, env.arm_idx][:5].cpu().numpy().astype(np.float64)
                carry_grip_start = env.robot.data.joint_pos[0, env.gripper_idx].item()
                print(f"\n    >>> S2 lift 완료 → Carry BC 시작")
                print(f"        arm={[f'{v:+.3f}' for v in carry_arm_start]} grip={carry_grip_start:.3f}")
                carry_phase = 2; carry_step = 0
                carry_lift_counter = 0
                s2_dp_agent.reset(); dp_agent.reset()

        # ── Carry Phase 2: carry BC ──
        elif args.skill == "carry" and carry_phase == 2:
            # obs 30D → 33D: direction_cmd 추가
            obs_carry = torch.cat([obs_t, carry_dir_cmd_t], dim=-1)
            base_naction = dp_agent.base_action_normalized(obs_carry)
            if residual_policy is not None:
                nobs = dp_agent.normalizer(obs_carry, "obs", forward=True)
                nobs = torch.nan_to_num(torch.clamp(nobs, -3, 3), nan=0.0)
                residual_nobs = torch.cat([nobs, base_naction], dim=-1)
                residual_naction = residual_policy.actor_mean(residual_nobs)
                residual_naction = torch.clamp(residual_naction, -1.0, 1.0)
                naction = base_naction + residual_naction * per_dim_action_scale
            else:
                naction = base_naction
            action = dp_agent.normalizer(naction, "action", forward=False)

            # Arm interpolation override
            carry_step += 1
            t = min(carry_step / args.carry_steps, 1.0)
            arm5 = carry_arm_start * (1 - t) + S3_ARM_END * t
            grip_t = carry_grip_start * (1 - t) + S3_GRIP_END * t
            arm6 = np.concatenate([arm5, [grip_t]])
            # Normalize to action space (env._apply_action과 동일한 리밋 소스)
            override = getattr(env, "_arm_action_limits_override", None)
            if override is not None:
                _lim = override[0].cpu().numpy()
            else:
                _lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx].cpu().numpy()
            _center = 0.5 * (_lim[:, 0] + _lim[:, 1])
            _half = 0.5 * (_lim[:, 1] - _lim[:, 0])
            _finite = np.isfinite(_center) & np.isfinite(_half) & (np.abs(_half) > 1e-6)
            _half = np.where(_finite, _half, 1.0)
            _center = np.where(_finite, _center, 0.0)
            arm_action = np.clip((arm6 - _center) / _half, -1.0, 1.0)
            action[0, :6] = torch.tensor(arm_action, dtype=torch.float32, device=device)
            action = action.clamp(-1.0, 1.0)

            _objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            _gp = env.robot.data.joint_pos[0, env.gripper_idx].item()
            if step_count % 25 == 0:
                _bv = env.robot.data.root_lin_vel_b[0].cpu().tolist()
                _wz = env.robot.data.root_ang_vel_b[0, 2].item()
                print(f"    [Carry] step={carry_step}/{args.carry_steps} interp={t:.2f} grip={_gp:.3f} objZ={_objZ:.3f} vel=(vx={_bv[0]:+.2f} vy={_bv[1]:+.2f} wz={_wz:+.2f})", flush=True)

            # Drop
            if _objZ < 0.05 and carry_step > 10:
                print(f"    [DROP] objZ={_objZ:.3f} — 리셋")
                carry_phase = 1; carry_step = 0; carry_lift_counter = 0
                s2_dp_agent.reset(); dp_agent.reset()
                obs, _ = env.reset()
                step_count = 0
                continue

            # Carry 완료
            if carry_step >= args.carry_steps:
                print(f"    [CARRY DONE] {carry_step} steps 완료")
                episode += 1
                carry_phase = 1; carry_step = 0; carry_lift_counter = 0
                s2_dp_agent.reset(); dp_agent.reset()
                obs, _ = env.reset()
                step_count = 0
                # 다음 방향
                if _carry_dir_schedule and episode < len(_carry_dir_schedule):
                    _cmd, _lbl = _carry_dir_schedule[episode]
                    carry_dir_cmd_t = torch.tensor([_cmd], dtype=torch.float32, device=device)
                    print(f"  [Carry] direction: {_lbl}")
                continue

        # ── 기존 모드 (approach_and_grasp, carry_and_place, navigate) ──
        else:
            base_naction = dp_agent.base_action_normalized(obs_t)

            if residual_policy is not None:
                nobs = dp_agent.normalizer(obs_t, "obs", forward=True)
                nobs = torch.clamp(nobs, -3, 3)
                nobs = torch.nan_to_num(nobs, nan=0.0)
                residual_nobs = torch.cat([nobs, base_naction], dim=-1)
                residual_naction = residual_policy.actor_mean(residual_nobs)
                residual_naction = torch.clamp(residual_naction, -1.0, 1.0)
                naction = base_naction + residual_naction * per_dim_action_scale
            else:
                naction = base_naction

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

    action = action.clamp(-1.0, 1.0)  # PhysX ±2π 에러 방지
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    # Carry: phase 로직이 직접 관리 (위에서 continue로 처리됨)
    if args.skill == "carry":
        continue

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

    # Lift 판정: train_resip.py와 동일 조건
    # held = obj_height > 0.05 AND grasped AND grip < 0.65 AND ee < 0.20
    held_now = (obj_z > 0.05 and g["grasped"] and g["grip_closed"] and g["ee_to_obj"] < 0.20)
    if held_now:
        lift_sustain += 1
    else:
        lift_sustain = 0
    if lift_sustain >= 50 and not ep_lifted:
        ep_lifted = True
        print(f"    ★ LIFT at t={step_count} | objZ={obj_z:.3f} ee={g['ee_to_obj']:.3f} grip={g['grip']:.3f} sustain={lift_sustain}", flush=True)

    if step_count % 50 == 0:
        if args.skill == "navigate":
            # Navigate: 방향 + base velocity
            _dir_cmd = env._direction_cmd[0].cpu().tolist()
            _labels = {(0,1,0): "FWD", (0,-1,0): "BWD", (-1,0,0): "LEFT",
                       (1,0,0): "RIGHT", (0,0,1): "TURN_L", (0,0,-1): "TURN_R"}
            _key = tuple(int(round(x)) for x in _dir_cmd)
            _dir_str = _labels.get(_key, str(_dir_cmd))
            _bv = env.robot.data.root_lin_vel_b[0].cpu().tolist()
            _wz = env.robot.data.root_ang_vel_b[0, 2].item()
            print(f"    [t={step_count:4d}] dir={_dir_str} vel=(vx={_bv[0]:+.2f} vy={_bv[1]:+.2f} wz={_wz:+.2f})", flush=True)
        else:
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

    # 강제 종료: navigate=600, 기타=max_steps
    if args.skill == "navigate" and step_count >= 600:
        done = True
    elif step_count >= args.max_steps:
        done = True
    else:
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
        if _nav_dir_schedule and episode < len(_nav_dir_schedule):
            _cmd, _lbl = _nav_dir_schedule[episode]
            env._direction_cmd[0] = torch.tensor(_cmd, dtype=torch.float32, device=device)
            print(f"  [Navigate] direction: {_lbl}")

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
