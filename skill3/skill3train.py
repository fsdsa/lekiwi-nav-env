#!/usr/bin/env python3
"""
Residual PPO for LeKiwi Skill-3 — CarryAndPlace

BC + Residual RL from step 0 (no warmup). Same approach as Skill-2 v6.8.

Task flow:
    Start: object grasped in gripper
    Phase 1 (Carry):  drive base toward dest (red cup), maintain grasp
    Phase 2 (Place):  extend arm, place object upright near dest
    Phase 3 (Rest):   retract arm, close gripper, rest pose
    Done:  place_grace_done or timeout

Demo statistics (20 episodes, avg 1333 steps):
    robot-dest: 0.70m → 0.35m (approach ~0.35m)
    obj-dest at place: 0.119 ± 0.018m (very close to red cup)
    REST_POSE: [-0.06, -0.21, 0.20, 0.12, 0.05], grip=-0.20
    carry grip ≈ 0.55, place grip opens to ~1.4, rest grip ≈ -0.20

Reward structure:
    ── Carry (grasped) ──
    R1   Hold bonus            +0.05/step    maintain grasp
    R2   Base→dest progress    ×8            delta distance
    R3   Dest proximity        ×2            tanh kernel σ=0.35
    R4   Heading alignment     ×0.3          cos(heading_to_dest)
    R5   Carry grip quality    ×0.5          gaussian(grip, 0.55)
    R6   EE→dest progress      ×15           when dest_d < 0.5m
    ── Place (milestone) ──
    R7   Place preliminary     +150          one-time, obj upright + near dest
    R8   Place final           +300          one-time, at episode end
    ── Rest (after place) ──
    R9   Rest pose shaping     ×5            gaussian(arm, REST_POSE)
    R10  Rest milestone        +80           one-time, sustained 50 steps
    ── Always ──
    R11  Drop penalty          −80           + force terminate
    R12  Time penalty          −0.01
    R13  Action smoothness     −0.3          action delta²
    R14  Dest contact penalty  −5.0          gripper/wrist touching dest object

Usage:
    python train_resip_skill3.py \\
        --bc_checkpoint checkpoints/dp_bc_skill3_aug/dp_bc_epoch300.pt \\
        --skill carry_and_place \\
        --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \\
        --dest_object_usd ~/isaac-objects/.../ACE_Coffee_Mug_.../model_clean.usd \\
        --gripper_contact_prim_path "/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1" \\
        --arm_limit_json calibration/arm_limits_measured.json \\
        --num_envs 64 --total_timesteps 10000000
"""
from __future__ import annotations

import argparse
import os

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Args
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="ResiP Skill-3: CarryAndPlace")

# Environment
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--skill", type=str, default="carry_and_place",
                    choices=["carry_and_place"])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_env_steps", type=int, default=2000,
                    help="Steps per rollout (2000 ≈ 80s at 25Hz)")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")
parser.add_argument("--demo", type=str, default="",
                    help="HDF5 demo file for grasp init (alternative to handoff_buffer)")
parser.add_argument("--grasp_close_steps", type=int, default=240)
parser.add_argument("--grasp_settle_steps", type=int, default=60)
parser.add_argument("--grasp_target_grip", type=float, default=0.45)

# PPO
parser.add_argument("--total_timesteps", type=int, default=10_000_000)
parser.add_argument("--update_epochs", type=int, default=50)
parser.add_argument("--num_minibatches", type=int, default=1)
parser.add_argument("--discount", type=float, default=0.999)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--target_kl", type=float, default=0.1)
parser.add_argument("--ent_coef", type=float, default=0.001)
parser.add_argument("--vf_coef", type=float, default=1.0)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--norm_adv", type=lambda x: x.lower() == "true", default=True)

# LR
parser.add_argument("--lr_actor", type=float, default=3e-4)
parser.add_argument("--lr_critic", type=float, default=5e-3)

# Residual action scale
parser.add_argument("--action_scale_arm", type=float, default=0.20)
parser.add_argument("--action_scale_gripper", type=float, default=0.12,
                    help="Low to prevent accidental drop during carry (BC grip≈0.55)")
parser.add_argument("--action_scale_base", type=float, default=0.35)
parser.add_argument("--action_scale", type=float, default=None,
                    help="Override all scales with single value")
parser.add_argument("--warmup_iters", type=int, default=20,
                    help="Linearly ramp residual scale from 0→1 over N iterations")

# Actor/Critic
parser.add_argument("--actor_hidden_size", type=int, default=256)
parser.add_argument("--actor_num_layers", type=int, default=2)
parser.add_argument("--critic_hidden_size", type=int, default=256)
parser.add_argument("--critic_num_layers", type=int, default=2)
parser.add_argument("--init_logstd", type=float, default=-1.0)
parser.add_argument("--action_head_std", type=float, default=0.0)

# Reward tuning
parser.add_argument("--r_hold", type=float, default=0.05,
                    help="R1: per-step hold bonus")
parser.add_argument("--r_base_progress", type=float, default=8.0,
                    help="R2: base→dest progress scale")
parser.add_argument("--r_proximity", type=float, default=2.0,
                    help="R3: dest proximity tanh scale")
parser.add_argument("--r_proximity_sigma", type=float, default=0.35,
                    help="R3: tanh kernel sigma")
parser.add_argument("--r_heading", type=float, default=0.3,
                    help="R4: heading alignment scale")
parser.add_argument("--r_grip_quality", type=float, default=0.5,
                    help="R5: carry grip quality scale")
parser.add_argument("--r_ee_progress", type=float, default=15.0,
                    help="R6: EE→dest progress scale (when near)")
parser.add_argument("--r_ee_progress_range", type=float, default=0.50,
                    help="R6: activate when dest_d < this")
parser.add_argument("--r_place_prelim", type=float, default=150.0,
                    help="R7: preliminary place milestone")
parser.add_argument("--r_place_final", type=float, default=300.0,
                    help="R8: final place milestone")
parser.add_argument("--r_rest_pose", type=float, default=5.0,
                    help="R9: rest pose shaping scale")
parser.add_argument("--r_rest_milestone", type=float, default=80.0,
                    help="R10: rest pose milestone")
parser.add_argument("--r_rest_sustain", type=int, default=50,
                    help="R10: sustained steps for rest milestone")
parser.add_argument("--r_drop", type=float, default=-80.0,
                    help="R11: drop penalty")
parser.add_argument("--r_time", type=float, default=-0.01,
                    help="R12: time penalty")
parser.add_argument("--r_smoothness", type=float, default=-0.3,
                    help="R13: action smoothness penalty")
parser.add_argument("--r_dest_contact", type=float, default=-5.0,
                    help="R14: penalty when gripper/wrist touches dest object")

# Regularization
parser.add_argument("--residual_l1", type=float, default=0.0)
parser.add_argument("--residual_l2", type=float, default=0.0)
parser.add_argument("--normalize_reward", type=lambda x: x.lower() == "true",
                    default=False)
parser.add_argument("--clip_reward", type=float, default=5.0)

# Eval/save
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--eval_first", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--save_dir", type=str, default="checkpoints/resip_skill3")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--resume_resip", type=str, default=None)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Imports
# ═══════════════════════════════════════════════════════════════════════════════
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import h5py
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Utilities
# ═══════════════════════════════════════════════════════════════════════════════
class RunningMeanStdClip:
    def __init__(self, epsilon=1e-4, shape=(), clip_value=10.0, device="cuda"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.clip_value = clip_value

    def update(self, x):
        bm = torch.mean(x, dim=0)
        bv = torch.var(x, dim=0, unbiased=False)
        bc = x.shape[0]
        d = bm - self.mean
        tc = self.count + bc
        self.mean += d * bc / tc
        self.var = (self.var * self.count + bv * bc + d**2 * self.count * bc / tc) / tc
        self.count = tc

    def __call__(self, x):
        self.update(x)
        return torch.clamp(x / torch.sqrt(self.var + 1e-8),
                           -self.clip_value, self.clip_value)


@torch.no_grad()
def compute_gae(values, nv, rewards, dones, nd, S, gamma, lam):
    adv = torch.zeros_like(rewards)
    lg = 0
    for t in reversed(range(S)):
        nt = 1.0 - (nd.float() if t == S - 1 else dones[t + 1].float())
        nval = nv if t == S - 1 else values[t + 1]
        d = rewards[t] + gamma * nval * nt - values[t]
        adv[t] = lg = d + gamma * lam * nt * lg
    return adv, adv + values


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Environment
# ═══════════════════════════════════════════════════════════════════════════════
class LeKiwiEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

    def reset(self):
        od, _ = self.env.reset()
        o = od["policy"] if isinstance(od, dict) else od
        return o.to(self.device)

    def step(self, action):
        od, r, ter, tru, info = self.env.step(action)
        o = od["policy"] if isinstance(od, dict) else od
        return (o.to(self.device), r.view(-1).to(self.device),
                ter.view(-1).to(self.device), tru.view(-1).to(self.device), info)


def make_env(num_envs, args):
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    cfg = Skill3EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 1.0
    cfg.sim.device = "cuda:0"
    if args.handoff_buffer and not args.demo:
        cfg.handoff_buffer_path = args.handoff_buffer
    # demo 모드: handoff buffer 사용 안 함 (batched_grasp_init으로 대체)
    if args.demo:
        cfg.handoff_buffer_path = ""

    cfg.enable_domain_randomization = False
    cfg.dr_action_delay_steps = 0
    cfg.arm_limit_write_to_sim = False
    cfg.grasp_contact_threshold = 0.55
    cfg.grasp_gripper_threshold = 0.65
    cfg.grasp_max_object_dist = 0.50
    cfg.grasp_success_height = 1.00     # disable parent lift-based success
    cfg.episode_length_s = 100.0        # ~2500 steps max

    # Dest object spawn (train_dppo_skill3.py와 동일)
    cfg.dest_spawn_dist_min = 0.6
    cfg.dest_spawn_dist_max = 0.7
    cfg.dest_spawn_min_separation = 0.3
    # dest_heading_noise_std=0.3, dest_heading_max_rad=0.5 (Skill2EnvCfg 기본값 사용)

    # Place detection thresholds (from demo data analysis)
    cfg.place_radius = 0.172            # obj-dest XY (demo max)
    cfg.place_obj_z_min = 0.032         # upright obj z range
    cfg.place_obj_z_max = 0.034
    cfg.place_grace_steps = 500         # rest pose time after place

    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    if args.multi_object_json:
        cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    cfg.dest_object_fixed = False  # kinematic→dynamic (write_root_state 반영 보장)
    cfg.dest_object_scale = 0.56
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    env = Skill3Env(cfg=cfg)
    print(f"  Env: carry_and_place, n={num_envs}, dev={env.device}")
    print(f"  dest_spawn: {cfg.dest_spawn_dist_min}~{cfg.dest_spawn_dist_max}m")
    print(f"  place: radius={cfg.place_radius}m, obj_z=[{cfg.place_obj_z_min},{cfg.place_obj_z_max}]")
    return LeKiwiEnvWrapper(env)


def load_frozen_dp(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    agent = DiffusionPolicyAgent(
        obs_dim=c["obs_dim"], act_dim=c["act_dim"],
        pred_horizon=c["pred_horizon"], action_horizon=c["action_horizon"],
        num_diffusion_iters=c["num_diffusion_iters"],
        inference_steps=c.get("inference_steps", 16),
        down_dims=c.get("down_dims", [256, 512, 1024]),
    ).to(device)
    sd = ck["model_state_dict"]
    agent.model.load_state_dict(
        {k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    agent.normalizer.load_state_dict(
        {k[11:]: v for k, v in sd.items() if k.startswith("normalizer.")})
    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()
    agent.inference_steps = 4
    print(f"Frozen DP: obs={c['obs_dim']}, act={c['act_dim']}, "
          f"pred_h={c['pred_horizon']}, act_h={c['action_horizon']}")
    return agent, c


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. Demo-based grasp init (from train_dppo_skill3.py)
# ═══════════════════════════════════════════════════════════════════════════════
def load_demo_init_states(path):
    entries = []
    with h5py.File(path, "r") as f:
        ep_keys = sorted([k for k in f.keys() if k.startswith("episode_")],
                         key=lambda x: int(x.split("_")[1]))
        for ek in ep_keys:
            grp = f[ek]; ea = dict(grp.attrs); obs0 = grp["obs"][0]
            entry = {"arm_joints_6": obs0[0:6].astype(np.float32)}
            if "robot_init_pos" in ea:
                entry["robot_init_pos"] = np.array(ea["robot_init_pos"], dtype=np.float32)
            if "robot_init_quat" in ea:
                entry["robot_init_quat"] = np.array(ea["robot_init_quat"], dtype=np.float32)
            entries.append(entry)
    print(f"  Loaded {len(entries)} demo init states from {path}")
    return entries


def batched_grasp_init(env_wrap, demo_entries, grasp_close_steps=240,
                       grasp_settle_steps=60, grasp_target_grip=0.45):
    """Demo init state에서 물리적 grasp 수행. eval_dp_bc.py의 검증된 로직 기반.

    eval_dp_bc.py와 동일:
      1. 로봇 env_origin + 랜덤 yaw 배치
      2. 데모 arm joints + gripper fully open(1.4)
      3. EE에 bbox center 맞춰 물체 배치
      4. 240 step 동안 gripper 닫으면서 매 step 물체를 EE에 텔레포트
      5. 60 step 자유 settle (마찰만으로 유지)
      6. dest object 스폰
    """
    raw = env_wrap.env; dev = raw.device; N = raw.num_envs
    all_ids = torch.arange(N, device=dev)
    n_demos = len(demo_entries)
    chosen = [demo_entries[random.randint(0, n_demos - 1)] for _ in range(N)]

    # ── 1. 로봇 배치: env_origin + 랜덤 yaw ──
    env_origins = raw.scene.env_origins
    root_states = raw.robot.data.default_root_state.clone()
    for i in range(N):
        root_states[i, 0:3] = env_origins[i] + torch.tensor([0.0, 0.0, 0.03], device=dev)
        yaw = random.uniform(-math.pi, math.pi)
        root_states[i, 3] = math.cos(yaw / 2)
        root_states[i, 4:6] = 0.0
        root_states[i, 6] = math.sin(yaw / 2)
    root_states[:, 7:] = 0.0
    raw.robot.write_root_state_to_sim(root_states, all_ids)
    if hasattr(raw, "home_pos_w"): raw.home_pos_w[:] = root_states[:, :3]

    # ── 2. Arm 자세 + gripper 열림 ──
    jp = raw.robot.data.default_joint_pos.clone()
    # 데모에서 읽은 arm 초기 자세 (carry 시작 자세)
    target_grips = torch.zeros(N, device=dev)
    for i, e in enumerate(chosen):
        arm6 = torch.tensor(e["arm_joints_6"], device=dev)
        jp[i, raw.arm_idx[:5]] = arm6[:5]
        target_grips[i] = arm6[5].item()  # 데모 gripper 값 (~0.55)
    jp[:, raw.gripper_idx] = 1.4  # fully open
    if hasattr(raw, "wheel_idx"): jp[:, raw.wheel_idx] = 0.0
    jv = torch.zeros_like(jp)
    raw.robot.write_joint_state_to_sim(jp, jv, env_ids=all_ids)
    raw.robot.set_joint_position_target(jp, env_ids=all_ids)
    raw.robot.set_joint_velocity_target(torch.zeros(N, raw.robot.num_joints, device=dev), env_ids=all_ids)
    for _ in range(10):
        raw.robot.write_data_to_sim(); raw.sim.step()
    raw.robot.update(raw.sim.cfg.dt)

    # ── 3. EE에 물체 배치 (bbox center 보정) ──
    fixed_jaw_idx = raw._fixed_jaw_body_idx
    ee_local = raw._ee_local_offset.view(1, 3).expand(N, -1)  # (N, 3)
    rot70 = torch.tensor([0.8192, -0.5736, 0.0, 0.0], dtype=torch.float32, device=dev)

    wrist_pos = raw.robot.data.body_pos_w[:, fixed_jaw_idx, :]
    wrist_quat = raw.robot.data.body_quat_w[:, fixed_jaw_idx, :]
    ee_pos = wrist_pos + quat_apply(wrist_quat, ee_local)
    obj_quat = quat_mul(wrist_quat, rot70.unsqueeze(0).expand(N, -1))

    # bbox center 보정: bbox center가 EE에 오도록
    obj_bbox = raw.object_bbox  # (N, 3)
    bbox_center_local = torch.zeros(N, 3, device=dev)
    bbox_center_local[:, 2] = obj_bbox[:, 2] / 2.0  # z축 중심
    bbox_center_world = quat_apply(obj_quat, bbox_center_local)
    obj_root_pos = ee_pos - bbox_center_world

    if raw.object_rigid is not None:
        obj_st = raw.object_rigid.data.root_state_w.clone()
        obj_st[:, 0:3] = obj_root_pos
        obj_st[:, 3:7] = obj_quat
        obj_st[:, 7:] = 0.0
        raw.object_rigid.write_root_state_to_sim(obj_st, all_ids)
    raw.object_pos_w[:] = ee_pos

    # ── 4. Gripper 닫으면서 매 step 물체를 EE에 텔레포트 (eval_dp_bc와 동일) ──
    for step_i in range(grasp_close_steps):
        t_frac = (step_i + 1) / grasp_close_steps
        grip_vals = 1.4 + (grasp_target_grip - 1.4) * t_frac  # scalar
        grip_jp = raw.robot.data.joint_pos_target.clone()
        grip_jp[:, raw.gripper_idx] = grip_vals
        raw.robot.set_joint_position_target(grip_jp, env_ids=all_ids)
        raw.robot.write_data_to_sim()

        # 매 step 물체를 현재 EE에 텔레포트 (bbox center 보정)
        if raw.object_rigid is not None:
            w_pos = raw.robot.data.body_pos_w[:, fixed_jaw_idx, :]
            w_quat = raw.robot.data.body_quat_w[:, fixed_jaw_idx, :]
            cur_ee = w_pos + quat_apply(w_quat, ee_local)
            cur_oq = quat_mul(w_quat, rot70.unsqueeze(0).expand(N, -1))
            cur_bbox_w = quat_apply(cur_oq, bbox_center_local)
            os2 = raw.object_rigid.data.root_state_w.clone()
            os2[:, 0:3] = cur_ee - cur_bbox_w
            os2[:, 3:7] = cur_oq
            os2[:, 7:] = 0.0
            raw.object_rigid.write_root_state_to_sim(os2, all_ids)

        raw.sim.step()
        raw.robot.update(raw.sim.cfg.dt)
        if raw.object_rigid is not None: raw.object_rigid.update(raw.sim.cfg.dt)

        if step_i % 60 == 0:
            ag = raw.robot.data.joint_pos[:, raw.gripper_idx]
            oz = raw.object_rigid.data.root_pos_w[:, 2] if raw.object_rigid else None
            print(f"[GRASP] step={step_i}/{grasp_close_steps} "
                  f"target={grip_vals:.3f} actual={ag.mean().item():.3f} "
                  f"obj_z={oz.mean().item():.4f}" if oz is not None else "")

    # ── 5. 자유 settle (grip target 유지하면서 마찰 확인) ──
    # settle 중 grip target을 명시적으로 유지 — 안 하면 gripper가 풀릴 수 있음
    settle_jp = raw.robot.data.joint_pos_target.clone()
    settle_jp[:, raw.gripper_idx] = grasp_target_grip
    print(f"[GRASP] settle {grasp_settle_steps} steps (grip_target={grasp_target_grip})...")
    for si in range(grasp_settle_steps):
        raw.robot.set_joint_position_target(settle_jp, env_ids=all_ids)
        raw.robot.write_data_to_sim(); raw.sim.step()
        raw.robot.update(raw.sim.cfg.dt)
        if raw.object_rigid is not None: raw.object_rigid.update(raw.sim.cfg.dt)
        if si == grasp_settle_steps - 1:
            ag = raw.robot.data.joint_pos[:, raw.gripper_idx]
            oz = raw.object_rigid.data.root_pos_w[:, 2] - env_origins[:, 2] if raw.object_rigid else None
            print(f"[GRASP] settle done: grip={ag.tolist()} obj_z={oz.tolist() if oz is not None else 'N/A'}")

    # ── 5b. Grasp validation ──
    # grip이 target보다 열려있으면 = 물체가 jaw 사이를 blocking = 좋은 grasp
    # cf=0이라도 grip_blocked이면 정상 (contact sensor가 안 잡힐 수 있음)
    cf = raw._contact_force_per_env()
    grip_pos = raw.robot.data.joint_pos[:, raw.gripper_idx].view(-1)
    oz_check = raw.object_rigid.data.root_pos_w[:, 2] - env_origins[:, 2] if raw.object_rigid else None
    grip_blocked = grip_pos > (grasp_target_grip + 0.02)  # object preventing closure
    has_contact = cf > 0.1
    good_grasp = grip_blocked | has_contact
    if oz_check is not None:
        on_ground = oz_check < 0.03
        good_grasp = good_grasp & (~on_ground)
    n_good = good_grasp.sum().item()
    n_bad = N - n_good
    print(f"[GRASP] validation: {n_good}/{N} good "
          f"(grip_blocked={grip_blocked.sum().item()}, has_contact={has_contact.sum().item()}) "
          f"cf={cf.tolist()} grip={grip_pos.tolist()}"
          f"{f' oz={oz_check.tolist()}' if oz_check is not None else ''}")

    # ── 6. 상태 설정 + dest object 스폰 ──
    raw.object_grasped[:] = True; raw.just_dropped[:] = False
    if hasattr(raw, "intentional_placed"): raw.intentional_placed[:] = False
    raw._fallback_teleport_carry[:] = False
    if raw.object_rigid is not None: raw.object_pos_w[:] = raw.object_rigid.data.root_pos_w

    raw._spawn_dest_object(all_ids)
    # dest body 반영을 위한 추가 sim step (grip target 유지)
    for _ in range(10):
        raw.robot.set_joint_position_target(settle_jp, env_ids=all_ids)
        raw.robot.write_data_to_sim(); raw.sim.step()
    raw.robot.update(raw.sim.cfg.dt)
    if raw.object_rigid is not None: raw.object_rigid.update(raw.sim.cfg.dt)
    if raw._dest_object_rigid is not None:
        raw._dest_object_rigid.update(raw.sim.cfg.dt)
        raw.dest_object_pos_w[:] = raw._dest_object_rigid.data.root_pos_w

    raw.task_success[:] = False; raw.just_grasped[:] = False
    raw.place_success_step[:] = 0; raw.preliminary_success[:] = False
    raw.prev_dest_dist[:] = 10.0; raw.prev_object_dist[:] = 10.0
    raw.episode_reward_sum[:] = 0.0; raw.actions[:] = 0.0; raw.prev_actions[:] = 0.0
    # ── Fix: compute action-space values for current arm+grip pose ──
    # arm_action_to_limits: joint_pos = center + action * half
    # Invert: action = (joint_pos - center) / half
    if raw._arm_action_limits_override is not None:
        arm_limits = raw._arm_action_limits_override
    else:
        arm_limits = raw.robot.data.soft_joint_pos_limits[:, raw.arm_idx]
    arm_lo = arm_limits[..., 0]  # (N, 6)
    arm_hi = arm_limits[..., 1]  # (N, 6)
    arm_center = 0.5 * (arm_lo + arm_hi)
    arm_half = 0.5 * (arm_hi - arm_lo)
    # Current joint positions for arm+gripper (6 joints)
    cur_joints = raw.robot.data.joint_pos[:, raw.arm_idx]  # (N, 6)
    # Override gripper target
    cur_joints[:, 5] = grasp_target_grip
    # Invert to action space
    carry_action = (cur_joints - arm_center) / arm_half.clamp(min=1e-6)
    carry_action = carry_action.clamp(-1.0, 1.0)
    # Set actions and delay buffer
    raw.actions[:, 0:6] = carry_action
    raw.prev_actions[:, 0:6] = carry_action
    if raw._action_delay_buf is not None:
        raw._action_delay_buf[:, :, 0:6] = carry_action.unsqueeze(0)
    print(f"    [grasp_init] carry_action[0]={carry_action[0].tolist()}"
          f" delay_buf={'filled' if raw._action_delay_buf is not None else 'None'}")

    # ── 디버그 출력 ──
    grip_sim = raw.robot.data.joint_pos[:, raw.gripper_idx].view(-1)
    obj_z = (raw.object_pos_w[:, 2] - env_origins[:, 2]).view(-1)
    dest_dist = torch.norm(raw.dest_object_pos_w[:, :2] - raw.robot.data.root_pos_w[:, :2], dim=-1)
    dest_pos = raw.dest_object_pos_w[:, :2]
    robot_pos = raw.robot.data.root_pos_w[:, :2]
    print(f"    [grasp_init] grip={grip_sim.tolist()}")
    print(f"    [grasp_init] obj_z={obj_z.tolist()}")
    print(f"    [grasp_init] dest_dist={dest_dist.tolist()}")
    print(f"    [grasp_init] dest_pos={dest_pos.tolist()}")
    print(f"    [grasp_init] robot_pos={robot_pos.tolist()}")

    # ── settle 직후 gripper 물리 상태 전체 덤프 ──
    gi_idx = raw.gripper_idx
    print(f"    [SETTLE_END] grip_pos={raw.robot.data.joint_pos[:, gi_idx].tolist()}")
    print(f"    [SETTLE_END] grip_pos_target={raw.robot.data.joint_pos_target[:, gi_idx].tolist()}")
    print(f"    [SETTLE_END] grip_stiffness={raw.robot.data.joint_stiffness[:, gi_idx].tolist()}")
    print(f"    [SETTLE_END] grip_damping={raw.robot.data.joint_damping[:, gi_idx].tolist()}")
    print(f"    [SETTLE_END] grip_vel={raw.robot.data.joint_vel[:, gi_idx].tolist()}")
    # arm limits 확인
    if raw._arm_action_limits_override is not None:
        grip_lo = raw._arm_action_limits_override[:, 5, 0]
        grip_hi = raw._arm_action_limits_override[:, 5, 1]
    else:
        grip_lo = raw.robot.data.soft_joint_pos_limits[:, raw.arm_idx[5], 0]
        grip_hi = raw.robot.data.soft_joint_pos_limits[:, raw.arm_idx[5], 1]
    print(f"    [SETTLE_END] grip_limits=[{grip_lo[0]:.4f}, {grip_hi[0]:.4f}]"
          f" center={(grip_lo[0]+grip_hi[0])*.5:.4f} half={(grip_hi[0]-grip_lo[0])*.5:.4f}")
    print(f"    [SETTLE_END] carry_action_grip={carry_action[0, 5]:.4f}"
          f" → target={(grip_lo[0]+grip_hi[0])*.5 + carry_action[0,5]*(grip_hi[0]-grip_lo[0])*.5:.4f}")

    # Reset apply_action diagnostic counter so it prints from first BC step
    raw._diag_apply_ct = 0

    obs_dict = raw._get_observations()
    obs_out = (obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict).to(dev)
    return obs_out, carry_action.to(dev)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    seed = args.seed or random.randint(0, 2**32 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"Seed: {seed}")

    # Demo init states (if --demo provided)
    demo_entries = None
    if args.demo:
        demo_path = os.path.expanduser(args.demo)
        demo_entries = load_demo_init_states(demo_path)
        assert len(demo_entries) > 0, f"No demo entries in {demo_path}"

    env = make_env(args.num_envs, args)
    dev = env.device
    N = env.num_envs

    # EE position helper
    from lekiwi_skill2_env import EE_LOCAL_OFFSET
    jaw_idx, _ = env.env.robot.find_bodies(["Wrist_Roll_08c_v1"])
    jaw_idx = jaw_idx[0]
    ee_off = torch.tensor(EE_LOCAL_OFFSET, device=dev).unsqueeze(0)

    # Load frozen BC
    dp, dpc = load_frozen_dp(args.bc_checkpoint, dev)
    OD, AD = dpc["obs_dim"], dpc["act_dim"]

    # Action scale (gripper deliberately low to prevent accidental drop)
    if args.action_scale is not None:
        scale = torch.full((AD,), args.action_scale, device=dev)
    else:
        scale = torch.zeros(AD, device=dev)
        scale[0:5] = args.action_scale_arm
        scale[5]   = args.action_scale_gripper
        scale[6:9] = args.action_scale_base
    print(f"Action scale: {scale.tolist()}")

    # Residual policy
    rpol = ResidualPolicy(
        obs_dim=OD, action_dim=AD,
        actor_hidden_size=args.actor_hidden_size,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_size=args.critic_hidden_size,
        critic_num_layers=args.critic_num_layers,
        actor_activation="ReLU", critic_activation="ReLU",
        init_logstd=args.init_logstd, action_head_std=args.action_head_std,
        action_scale=0.1, learn_std=True,
        critic_last_layer_bias_const=0.25, critic_last_layer_std=0.25,
    ).to(dev)
    print(f"Residual params: {sum(p.numel() for p in rpol.parameters()):,}")

    # Optimizers
    opt_a = optim.AdamW(
        [p for n, p in rpol.named_parameters() if "critic" not in n],
        lr=args.lr_actor, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-6)
    opt_c = optim.AdamW(
        [p for n, p in rpol.named_parameters() if "critic" in n],
        lr=args.lr_critic, eps=1e-5, weight_decay=1e-6)

    S = args.num_env_steps
    B = S * N
    MB = B // args.num_minibatches
    NI = args.total_timesteps // B

    sch_a = optim.lr_scheduler.CosineAnnealingLR(
        opt_a, T_max=NI, eta_min=args.lr_actor * 0.01)
    sch_c = optim.lr_scheduler.CosineAnnealingLR(
        opt_c, T_max=NI, eta_min=args.lr_critic * 0.01)

    rew_norm = RunningMeanStdClip(
        shape=(1,), clip_value=args.clip_reward, device=dev
    ) if args.normalize_reward else None

    # Resume
    gs, gi = 0, 0
    if args.resume_resip:
        ck = torch.load(args.resume_resip, map_location=dev, weights_only=False)
        rpol.load_state_dict(ck["residual_policy_state_dict"])
        if "optimizer_actor_state_dict" in ck:
            opt_a.load_state_dict(ck["optimizer_actor_state_dict"])
        if "optimizer_critic_state_dict" in ck:
            opt_c.load_state_dict(ck["optimizer_critic_state_dict"])
        gs = ck.get("global_step", 0)
        gi = ck.get("iteration", 0)
        print(f"Resumed: iter={gi}, step={gs}")

    # ═══════════════════════════════════════════════════════════════════
    #  Rollout buffers
    # ═══════════════════════════════════════════════════════════════════
    RD = OD + AD
    obs_b  = torch.zeros((S, N, RD), device=dev)
    act_b  = torch.zeros((S, N, AD), device=dev)
    lp_b   = torch.zeros((S, N), device=dev)
    rew_b  = torch.zeros((S, N), device=dev)
    done_b = torch.zeros((S, N), device=dev)
    val_b  = torch.zeros((S, N), device=dev)

    # ═══════════════════════════════════════════════════════════════════
    #  Constants (from demo data analysis)
    # ═══════════════════════════════════════════════════════════════════
    # Rest pose: teleop demo last 50 steps average
    REST_POSE = torch.tensor([-0.06, -0.21, 0.20, 0.12, 0.05], device=dev)
    REST_GRIP = -0.20
    CARRY_GRIP_TARGET = 0.55

    # Place thresholds (from env config)
    PLACE_OBJ_Z_MIN = float(env.env.cfg.place_obj_z_min)
    PLACE_OBJ_Z_MAX = float(env.env.cfg.place_obj_z_max)
    PLACE_RADIUS = float(env.env.cfg.place_radius)

    # ═══════════════════════════════════════════════════════════════════
    #  Per-env state tracking
    # ═══════════════════════════════════════════════════════════════════
    prev_dd      = torch.full((N,), 10.0, device=dev)  # base→dest
    prev_ee_dd   = torch.full((N,), 10.0, device=dev)  # EE→dest XY
    prev_action  = torch.zeros((N, AD), device=dev)

    # Milestones
    ms_place_prelim = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_place_final  = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_rest         = torch.zeros(N, dtype=torch.bool, device=dev)
    rest_sustain    = torch.zeros(N, dtype=torch.long, device=dev)

    # Diagnostics
    diag_n_prelim = torch.zeros(N, dtype=torch.long, device=dev)
    diag_n_final  = torch.zeros(N, dtype=torch.long, device=dev)
    diag_n_drop   = torch.zeros(N, dtype=torch.long, device=dev)
    diag_n_rest   = torch.zeros(N, dtype=torch.long, device=dev)
    diag_min_dd   = torch.full((N,), 99.0, device=dev)
    diag_dest_hit = torch.zeros(N, dtype=torch.long, device=dev)
    diag_prog_sum = 0.0
    diag_clip_ct  = 0
    diag_clip_n   = 0

    # ── Helpers ──
    def ee_pos():
        wp = env.env.robot.data.body_pos_w[:, jaw_idx, :]
        wq = env.env.robot.data.body_quat_w[:, jaw_idx, :]
        return wp + quat_apply(wq, ee_off.expand_as(wp))

    def dest_dist_xy():
        """Robot base → dest XY distance."""
        return torch.nan_to_num(
            torch.norm(env.env.dest_object_pos_w[:, :2]
                       - env.env.robot.data.root_pos_w[:, :2], dim=-1).view(-1),
            nan=1.0)

    def ee_dest_dist_xy():
        """EE → dest XY distance."""
        return torch.nan_to_num(
            torch.norm(ee_pos()[:, :2]
                       - env.env.dest_object_pos_w[:, :2], dim=-1).view(-1),
            nan=1.0)

    def obj_dest_dist_xy():
        """Object → dest XY distance."""
        return torch.nan_to_num(
            torch.norm(env.env.object_pos_w[:, :2]
                       - env.env.dest_object_pos_w[:, :2], dim=-1).view(-1),
            nan=1.0)

    def obj_z():
        """Object height above env floor."""
        env_z = env.env.scene.env_origins[:, 2]
        return (env.env.object_pos_w[:, 2] - env_z).view(-1)

    def heading_to_dest():
        """cos(heading angle to dest), body frame. +y = forward."""
        dd_w = env.env.dest_object_pos_w - env.env.robot.data.root_pos_w
        dd_b = quat_apply_inverse(env.env.robot.data.root_quat_w, dd_w)
        d = torch.norm(dd_b[:, :2], dim=-1, keepdim=True).clamp(min=1e-6)
        return (dd_b[:, 1:2] / d).view(-1)  # cos = forward_component / dist

    def reset_ep(mask):
        prev_dd[mask] = dest_dist_xy()[mask]
        prev_ee_dd[mask] = ee_dest_dist_xy()[mask]
        prev_action[mask] = 0.0
        ms_place_prelim[mask] = False
        ms_place_final[mask] = False
        ms_rest[mask] = False
        rest_sustain[mask] = 0

    # ── Print config ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_sr, best_prelim, best_final = 0.0, 0, 0
    tt = 0
    t0 = time.time()

    print(f"\n{'='*65}")
    print(f"  ResiP Skill-3 — CarryAndPlace")
    print(f"  N={N} S={S} B={B} iters={NI}")
    print(f"  scale: arm={args.action_scale_arm} "
          f"grip={args.action_scale_gripper} base={args.action_scale_base}"
          f" warmup={args.warmup_iters}iters")
    print(f"  lr: a={args.lr_actor} c={args.lr_critic}")
    print(f"  ── Carry ──")
    print(f"  R1 hold={args.r_hold} R2 progress={args.r_base_progress} "
          f"R3 proximity={args.r_proximity}(σ={args.r_proximity_sigma})")
    print(f"  R4 heading={args.r_heading} R5 grip={args.r_grip_quality} "
          f"R6 ee_prog={args.r_ee_progress}(range={args.r_ee_progress_range})")
    print(f"  ── Place ──")
    print(f"  R7 prelim={args.r_place_prelim} R8 final={args.r_place_final}")
    print(f"  ── Rest ──")
    print(f"  R9 rest={args.r_rest_pose} R10 rest_ms={args.r_rest_milestone}"
          f"(sus={args.r_rest_sustain})")
    print(f"  ── Always ──")
    print(f"  R11 drop={args.r_drop} R12 time={args.r_time} "
          f"R13 smooth={args.r_smoothness} R14 dest_contact={args.r_dest_contact}")
    print(f"  REST_POSE={REST_POSE.tolist()} grip={REST_GRIP}")
    print(f"  CARRY_GRIP={CARRY_GRIP_TARGET}")
    print(f"  place: radius={PLACE_RADIUS}m z=[{PLACE_OBJ_Z_MIN},{PLACE_OBJ_Z_MAX}]")
    print(f"{'='*65}\n")

    # ═══════════════════════════════════════════════════════════════════
    #  Training loop
    # ═══════════════════════════════════════════════════════════════════
    carry_action_buf = torch.zeros(N, AD, device=dev)  # from batched_grasp_init

    def reset_all():
        nonlocal carry_action_buf
        obs = env.reset()
        if demo_entries is not None:
            obs, ca = batched_grasp_init(
                env, demo_entries,
                grasp_close_steps=args.grasp_close_steps,
                grasp_settle_steps=args.grasp_settle_steps,
                grasp_target_grip=args.grasp_target_grip)
            carry_action_buf[:, :6] = ca  # arm+grip carry_action
        return obs

    next_obs = reset_all()
    next_done = torch.zeros(N, device=dev)
    dp.reset()

    first_iter = True
    while gs < args.total_timesteps:
        gi += 1
        it0 = time.time()
        ev = (gi - int(args.eval_first)) % args.eval_interval == 0

        # Reset env + state (skip on first iter — already initialized above)
        if not first_iter:
            next_obs = reset_all()
            dp.reset()
        first_iter = False
        next_done = torch.zeros(N, device=dev)
        ms_place_prelim.zero_()
        ms_place_final.zero_()
        ms_rest.zero_()
        rest_sustain.zero_()

        # Init distances
        prev_dd[:] = dest_dist_xy()
        prev_ee_dd[:] = ee_dest_dist_xy()
        prev_action.zero_()
        diag_min_dd.fill_(99.0)
        diag_dest_hit.zero_()
        diag_prog_sum = 0.0

        dd0 = prev_dd.clone()
        print(f"\nIter {gi}/{NI} | {'EVAL' if ev else 'TRAIN'} | "
              f"step={gs} | DD: {dd0.mean():.3f}/{dd0.min():.3f}")

        # ── Rollout ──
        for step in range(S):
            if not ev:
                gs += N

            # BC base action
            with torch.no_grad():
                ba = dp.base_action_normalized(next_obs)
                no = torch.nan_to_num(
                    torch.clamp(dp.normalizer(next_obs, "obs", forward=True),
                                -3, 3), nan=0.0)
                ba = torch.nan_to_num(ba, nan=0.0)

            ro = torch.cat([no, ba], dim=-1)
            done_b[step] = next_done
            obs_b[step] = ro

            # Residual action
            with torch.no_grad():
                ra_s, _, _, val, ra_m = rpol.get_action_and_value(ro)
            ra = ra_m if ev else ra_s
            ra = torch.clamp(ra, -1.0, 1.0)

            if not ev:
                diag_clip_ct += (ra_s.abs() > 0.99).sum().item()
                diag_clip_n += ra_s.numel()

            with torch.no_grad():
                _, lp, _, _, _ = rpol.get_action_and_value(ro, ra)

            # Apply: BC + residual (warmup: linearly ramp scale 0→1)
            wu = min(gi / max(args.warmup_iters, 1), 1.0)
            action = dp.normalizer(ba + ra * scale * wu, "action", forward=False)

            # ── Grip guard: 첫 10 step은 gripper를 carry_action으로 고정 ──
            # BC가 demo carry grip(~0.55)으로 열려는 것을 방지
            # (batched_grasp_init settle grip=0.45 → BC 0.55 = 0.1 opening → object drop)
            GRIP_GUARD_STEPS = 10
            if step < GRIP_GUARD_STEPS and demo_entries is not None:
                blend = step / GRIP_GUARD_STEPS  # 0→1 linear ramp
                carry_grip_a = carry_action_buf[:, 5]  # action-space grip from settle
                bc_grip_a = action[:, 5]
                action = action.clone()
                action[:, 5] = carry_grip_a + blend * (bc_grip_a - carry_grip_a)
                if step < 3:
                    print(f"  [GRIP_GUARD step={step}] carry={carry_grip_a[0]:.4f} "
                          f"bc={bc_grip_a[0]:.4f} blended={action[0,5]:.4f}")

            # ── Diagnostic: first 3 steps of each iteration ──
            if step < 3:
                grip_act = action[:, 5]
                arm_act = action[:, :5]
                base_act = action[:, 6:9]
                _oz = env.env.object_rigid.data.root_pos_w[:, 2] - env.env.scene.env_origins[:, 2]
                _grip = env.env.robot.data.joint_pos[:, env.env.gripper_idx].view(-1)
                print(f"  [STEP {step}] grip_action={grip_act.tolist()}")
                print(f"  [STEP {step}] grip_joint={_grip.tolist()} obj_z={_oz.tolist()}")
                print(f"  [STEP {step}] arm_action={arm_act[0].tolist()}")
                print(f"  [STEP {step}] base_action={base_act[0].tolist()}")

            next_obs, _, ter, tru, info = env.step(action)
            next_obs = torch.nan_to_num(next_obs, nan=0.0)
            done = ter | tru

            # ── env.step 직후 gripper 비교 (settle→step 변화 확인) ──
            if step < 3:
                raw = env.env
                _gp_after = raw.robot.data.joint_pos[:, raw.gripper_idx].view(-1)
                _gt_after = raw.robot.data.joint_pos_target[:, raw.gripper_idx].view(-1)
                _oz_after = raw.object_rigid.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                print(f"  [AFTER_STEP {step}] grip_pos={_gp_after.tolist()} "
                      f"grip_target={_gt_after.tolist()} obj_z={_oz_after.tolist()}")

            # ══════════════════════════════════════════════════════
            #  Read state
            # ══════════════════════════════════════════════════════
            grasped = info.get("object_grasped_mask",
                               env.env.object_grasped).view(-1).bool()
            just_dropped = info.get("just_dropped_mask",
                                     env.env.just_dropped).view(-1).float()
            prelim_success = info.get("preliminary_success",
                                       env.env.preliminary_success).view(-1).bool()
            final_success = info.get("place_success_mask",
                                      env.env.task_success).view(-1).bool()

            dd = dest_dist_xy()
            ee_dd = ee_dest_dist_xy()
            grip = env.env.robot.data.joint_pos[
                :, env.env.gripper_idx].view(-1)
            arm_joints = env.env.robot.data.joint_pos[:, env.env.arm_idx[:5]]
            oz = obj_z()
            cos_h = heading_to_dest()

            rew = torch.zeros(N, device=dev)

            # ══════════════════════════════════════════════════════
            # R1: HOLD BONUS (per-step, grasped)
            # ══════════════════════════════════════════════════════
            rew += grasped.float() * args.r_hold

            # ══════════════════════════════════════════════════════
            # R2: BASE→DEST PROGRESS (grasped, not yet placed)
            # ══════════════════════════════════════════════════════
            carry_mask = grasped & (~ms_place_prelim)
            progress = torch.clamp(prev_dd - dd, -0.2, 0.2)
            rew += carry_mask.float() * progress * args.r_base_progress
            diag_prog_sum += (carry_mask.float() * progress).sum().item()

            # ══════════════════════════════════════════════════════
            # R3: DEST PROXIMITY (tanh kernel, grasped)
            #     Key fix for BC not approaching closely enough
            # ══════════════════════════════════════════════════════
            prox = 1.0 - torch.tanh(dd / args.r_proximity_sigma)
            rew += carry_mask.float() * prox * args.r_proximity

            # ══════════════════════════════════════════════════════
            # R4: HEADING ALIGNMENT (grasped)
            # ══════════════════════════════════════════════════════
            rew += carry_mask.float() * cos_h * args.r_heading

            # ══════════════════════════════════════════════════════
            # R5: CARRY GRIP QUALITY (grasped)
            #     gaussian(grip, 0.55) — prevent accidental open
            # ══════════════════════════════════════════════════════
            grip_q = torch.exp(-((grip - CARRY_GRIP_TARGET) / 0.15) ** 2)
            rew += grasped.float() * grip_q * args.r_grip_quality

            # ══════════════════════════════════════════════════════
            # R6: EE→DEST PROGRESS (when near dest)
            #     Activates when base is close enough, incentivize
            #     arm extension toward dest
            # ══════════════════════════════════════════════════════
            near_dest = dd < args.r_ee_progress_range
            ee_progress = torch.clamp(prev_ee_dd - ee_dd, -0.2, 0.2)
            rew += (carry_mask & near_dest).float() * ee_progress * args.r_ee_progress

            # ══════════════════════════════════════════════════════
            # R7: PLACE PRELIMINARY (+150, one-time)
            #     obj upright + near dest + not grasped
            # ══════════════════════════════════════════════════════
            od_xy = obj_dest_dist_xy()
            upright = (oz >= PLACE_OBJ_Z_MIN) & (oz <= PLACE_OBJ_Z_MAX)
            place_cond = upright & (od_xy < PLACE_RADIUS) & (~grasped)
            new_prelim = place_cond & (~ms_place_prelim)
            rew += new_prelim.float() * args.r_place_prelim
            ms_place_prelim |= place_cond
            diag_n_prelim += new_prelim.long()

            # ══════════════════════════════════════════════════════
            # R8: PLACE FINAL (+300, one-time, at episode end)
            # ══════════════════════════════════════════════════════
            new_final = final_success & (~ms_place_final)
            rew += new_final.float() * args.r_place_final
            ms_place_final |= final_success
            diag_n_final += new_final.long()

            # ══════════════════════════════════════════════════════
            # R9: REST POSE SHAPING (per-step, after preliminary)
            #     arm → REST_POSE, grip → REST_GRIP
            # ══════════════════════════════════════════════════════
            rest_mask = ms_place_prelim & (~grasped)
            if rest_mask.any():
                arm_err = torch.norm(arm_joints - REST_POSE, dim=-1)
                pose_sim = torch.exp(-(arm_err ** 2) / 2.0)  # σ=1.0
                grip_sim = torch.exp(-((grip - REST_GRIP) / 0.30) ** 2)
                rest_reward = 0.7 * pose_sim + 0.3 * grip_sim
                rew += rest_mask.float() * rest_reward * args.r_rest_pose

            # ══════════════════════════════════════════════════════
            # R10: REST MILESTONE (+80, one-time, sustained)
            # ══════════════════════════════════════════════════════
            if rest_mask.any():
                arm_err_small = torch.norm(arm_joints - REST_POSE, dim=-1) < 0.3
                grip_ok = (grip - REST_GRIP).abs() < 0.15
                rest_ok = rest_mask & arm_err_small & grip_ok
                rest_sustain[rest_ok] += 1
                rest_sustain[~rest_ok] = 0

                new_rest_ms = (rest_sustain >= args.r_rest_sustain) & (~ms_rest)
                rew += new_rest_ms.float() * args.r_rest_milestone
                ms_rest |= new_rest_ms
                diag_n_rest += new_rest_ms.long()

            # ══════════════════════════════════════════════════════
            # R11: DROP PENALTY (−80 + force terminate)
            # ══════════════════════════════════════════════════════
            dropped = just_dropped > 0
            rew += dropped.float() * args.r_drop
            diag_n_drop += dropped.long()
            # Force done on drop (env doesn't terminate, we do)
            done = done | dropped

            # ══════════════════════════════════════════════════════
            # R12: TIME PENALTY
            # ══════════════════════════════════════════════════════
            rew += args.r_time

            # ══════════════════════════════════════════════════════
            # R13: ACTION SMOOTHNESS
            # ══════════════════════════════════════════════════════
            action_delta = action - prev_action
            rew += args.r_smoothness * (action_delta ** 2).sum(dim=-1)

            # ══════════════════════════════════════════════════════
            # R14: DEST CONTACT PENALTY (gripper/wrist touching dest)
            # ══════════════════════════════════════════════════════
            dest_cf = info.get("dest_contact_force",
                               torch.zeros(N, device=dev)).view(-1)
            dest_touching = (dest_cf > 0.5).float()
            rew += dest_touching * args.r_dest_contact
            diag_dest_hit += (dest_touching > 0).long()

            # ── Update state ──
            prev_dd[:] = dd
            prev_ee_dd[:] = ee_dd
            prev_action[:] = action.detach()
            diag_min_dd = torch.min(diag_min_dd, dd)

            # ── Done masking ──
            dm = done.view(-1).bool()
            rew[dm] = 0.0
            if dm.any():
                reset_ep(dm)

            rew = torch.nan_to_num(rew, nan=0.0)
            if rew_norm is not None and not ev:
                rew = rew_norm(rew.unsqueeze(-1)).squeeze(-1)

            val_b[step] = val.flatten()
            act_b[step] = ra
            lp_b[step] = lp
            rew_b[step] = rew.view(-1)
            next_done = done.view(-1).float()

        # ══════════════════════════════════════════════════════════════
        #  Summary
        # ══════════════════════════════════════════════════════════════
        tp = diag_n_prelim.sum().item()
        tf = diag_n_final.sum().item()
        td = diag_n_drop.sum().item()
        tr = diag_n_rest.sum().item()
        sr = (diag_n_final > 0).float().mean().item()
        sr_prelim = (diag_n_prelim > 0).float().mean().item()

        dd_now = dest_dist_xy()
        gv = env.env.robot.data.joint_pos[:, env.env.gripper_idx].view(-1)
        od_now = obj_dest_dist_xy()

        fps = S * N / max(time.time() - it0, 1e-6)
        cr = diag_clip_ct / max(diag_clip_n, 1)
        pa = diag_prog_sum / max(S * N, 1)
        dh = diag_dest_hit.sum().item()

        print(f"  SR={sr:.1%}(prelim={sr_prelim:.1%}) | "
              f"Place={tp}/{tf} Drop={td} Rest={tr} DestHit={dh} | "
              f"DD={dd_now.min():.2f}/{dd_now.mean():.2f}"
              f"(min:{diag_min_dd.min():.2f}) | "
              f"OD={od_now.min():.3f}/{od_now.mean():.3f} | "
              f"Grip={gv.min():.2f}/{gv.mean():.2f}/{gv.max():.2f} | "
              f"R={rew_b.sum(0).mean():.1f} | "
              f"FPS={fps:.0f} Prog={pa:.4f} Clip={cr:.3f} WU={min(gi/max(args.warmup_iters,1),1.0):.2f}")

        # Reset diagnostics
        diag_n_prelim.zero_()
        diag_n_final.zero_()
        diag_n_drop.zero_()
        diag_n_rest.zero_()
        diag_dest_hit.zero_()
        diag_min_dd.fill_(99.0)
        diag_prog_sum = 0.0
        diag_clip_ct = 0
        diag_clip_n = 0

        # ══════════════════════════════════════════════════════════════
        #  Eval save
        # ══════════════════════════════════════════════════════════════
        if ev:
            improved = False
            if sr > best_sr or (sr == best_sr and tf > best_final):
                best_sr = sr
                best_final = tf
                torch.save({
                    "residual_policy_state_dict": rpol.state_dict(),
                    "dp_checkpoint": args.bc_checkpoint,
                    "dp_config": dpc,
                    "success_rate": sr,
                    "prelim_places": tp,
                    "final_places": tf,
                    "rest_milestones": tr,
                    "iteration": gi,
                    "global_step": gs,
                    "args": vars(args),
                }, save_dir / "resip_best.pt")
                print(f"  ★ Best SR={sr:.1%} Final={tf}")
                improved = True

            if tp > best_prelim:
                best_prelim = tp
                if not improved:
                    torch.save({
                        "residual_policy_state_dict": rpol.state_dict(),
                        "dp_checkpoint": args.bc_checkpoint,
                        "dp_config": dpc,
                        "success_rate": sr,
                        "prelim_places": tp,
                        "iteration": gi,
                        "global_step": gs,
                        "args": vars(args),
                    }, save_dir / "resip_best_prelim.pt")
                    print(f"  ★ Best Prelim={tp}")

            if gi % 10 == 0 or gi <= 10:
                torch.save({
                    "residual_policy_state_dict": rpol.state_dict(),
                    "dp_checkpoint": args.bc_checkpoint,
                    "dp_config": dpc,
                    "iteration": gi,
                    "args": vars(args),
                }, save_dir / f"resip_iter{gi}.pt")

            print(f"  Best: SR={best_sr:.1%} Prelim={best_prelim} Final={best_final}")
            continue

        # ══════════════════════════════════════════════════════════════
        #  PPO Update
        # ══════════════════════════════════════════════════════════════
        with torch.no_grad():
            ba2 = dp.base_action_normalized(next_obs)
            no2 = torch.clamp(
                dp.normalizer(next_obs, "obs", forward=True), -3, 3)
            nv = rpol.get_value(torch.cat([no2, ba2], dim=-1)).flatten()

        adv, ret = compute_gae(
            val_b, nv, rew_b, done_b, next_done,
            S, args.discount, args.gae_lambda)

        f = lambda t, *s: t.reshape(-1, *s) if s else t.reshape(-1)
        bo, ba_, blp = f(obs_b, RD), f(act_b, AD), f(lp_b)
        bv, badv, bret = f(val_b), f(adv), f(ret)

        idx = np.arange(B)
        cfs = []

        for ep in range(args.update_epochs):
            stop = False
            np.random.shuffle(idx)
            for i0 in range(0, B, MB):
                mi = idx[i0:i0 + MB]
                _, nlp, ent, nv2, am = rpol.get_action_and_value(bo[mi], ba_[mi])
                lr = nlp - blp[mi]
                ratio = lr.exp()

                with torch.no_grad():
                    kl = ((ratio - 1) - lr).mean()
                    cfs.append(
                        ((ratio - 1).abs() > args.clip_coef).float().mean().item())

                ma = badv[mi]
                if args.norm_adv:
                    ma = (ma - ma.mean()) / (ma.std() + 1e-8)

                pg = torch.max(
                    -ma * ratio,
                    -ma * ratio.clamp(1 - args.clip_coef,
                                      1 + args.clip_coef)
                ).mean()
                vl = 0.5 * ((nv2.view(-1) - bret[mi]) ** 2).mean()
                el = ent.mean() * args.ent_coef

                loss = (pg - el
                        + args.residual_l1 * am.abs().mean()
                        + args.residual_l2 * (am ** 2).mean()
                        + vl * args.vf_coef)

                opt_a.zero_grad()
                opt_c.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(rpol.parameters(), args.max_grad_norm)
                opt_a.step()
                opt_c.step()

                if args.target_kl and kl > args.target_kl:
                    print(f"    KL stop ep{ep}: {kl:.4f}>{args.target_kl}")
                    stop = True
                    break
            if stop:
                break

        sch_a.step()
        sch_c.step()

        yp, yt = bv.cpu().numpy(), bret.cpu().numpy()
        vy = np.var(yt)
        ev2 = np.nan if vy == 0 else 1 - np.var(yt - yp) / vy
        tt += time.time() - it0
        sps = int(gs / tt) if tt > 0 else 0

        print(f"  pg={pg.item():.4f} v={vl.item():.4f} "
              f"ent={ent.mean().item():.4f} kl={kl.item():.4f} "
              f"clip={np.mean(cfs):.3f} ev={ev2:.3f} SPS={sps}")

        if gi % 10 == 0:
            torch.save({
                "residual_policy_state_dict": rpol.state_dict(),
                "optimizer_actor_state_dict": opt_a.state_dict(),
                "optimizer_critic_state_dict": opt_c.state_dict(),
                "dp_checkpoint": args.bc_checkpoint,
                "dp_config": dpc,
                "iteration": gi,
                "global_step": gs,
                "args": vars(args),
            }, save_dir / f"resip_iter{gi}.pt")

    # ═══════════════════════════════════════════════════════════════════
    #  Final
    # ═══════════════════════════════════════════════════════════════════
    print(f"\nDone in {time.time()-t0:.0f}s | "
          f"Best: SR={best_sr:.1%} Prelim={best_prelim} Final={best_final}")
    torch.save({
        "residual_policy_state_dict": rpol.state_dict(),
        "dp_checkpoint": args.bc_checkpoint,
        "dp_config": dpc,
        "best_eval_success_rate": best_sr,
        "iteration": gi,
        "global_step": gs,
        "args": vars(args),
    }, save_dir / "resip_final.pt")
    env.env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
