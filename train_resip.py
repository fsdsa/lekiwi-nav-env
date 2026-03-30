#!/usr/bin/env python3
"""
Residual PPO for LeKiwi — v6.4

v6 original + 3 changes only:
  1. R4b Lifted pose (×30, Gaussian) — per-step during lift
  2. R8  Ground contact penalty (−0.5)   — blocks floor-press exploit
  3. Warmup: 500 steps, decay 60 iters

All v6 rewards UNCHANGED:
  R1  Gripper open milestone     +10    one-time, gates R2/R3
  R2  Arm approach (XY)          ×30    delta, base-subtracted
  R3  Verified grasp             +25    one-time, 10-step sustained (env eg flag)
  R3b Verified lift              +100   one-time, held sustain >=25
  R4  Lift height                ×200   per-step, sustain≥3, grip closed, ee<0.20
  R4c Lifted pose return         +300   one-time, held sustain >=25 + pose hold
  R5  Sustained lift bonus       ×50    per-step after 15 steps held
  R6  Soft-lift milestone        +100   one-time after 15 steps held
  R7  Time penalty               −0.01
"""
from __future__ import annotations

import argparse
import os

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Args
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="ResiP v6.4")

parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place", "combined_s2_s3", "carry", "navigate"])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_env_steps", type=int, default=700)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")

# Combined S2→S3
parser.add_argument("--s2_resip_checkpoint", type=str, default="",
                    help="combined: Skill-2 ResiP checkpoint (frozen expert)")
parser.add_argument("--s3_bc_checkpoint", type=str, default="",
                    help="combined: Skill-3 BC checkpoint for warmup")
parser.add_argument("--s2_lift_hold_steps", type=int, default=400,
                    help="combined: Skill-2 lift success 판정 step 수")
parser.add_argument("--s3_dest_spawn_dist_min", type=float, default=0.6)
parser.add_argument("--s3_dest_spawn_dist_max", type=float, default=0.9)
parser.add_argument("--s3_dest_heading_max_rad", type=float, default=0.5)

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
parser.add_argument("--clip_vloss", type=lambda x: x.lower() == "true", default=False)

# LR
parser.add_argument("--lr_actor", type=float, default=3e-4)
parser.add_argument("--lr_critic", type=float, default=5e-3)

# Residual
parser.add_argument("--action_scale_arm", type=float, default=0.20)
parser.add_argument("--action_scale_gripper", type=float, default=0.30)
parser.add_argument("--action_scale_base", type=float, default=0.35)
parser.add_argument("--action_scale", type=float, default=None)
parser.add_argument("--actor_hidden_size", type=int, default=256)
parser.add_argument("--actor_num_layers", type=int, default=2)
parser.add_argument("--critic_hidden_size", type=int, default=256)
parser.add_argument("--critic_num_layers", type=int, default=2)
parser.add_argument("--init_logstd", type=float, default=-1.0)
parser.add_argument("--action_head_std", type=float, default=0.0)

# Warmup — keep short so residual RL can intervene before BC commits to bad approach
parser.add_argument("--warmup_steps_initial", type=int, default=500)
parser.add_argument("--warmup_steps_final", type=int, default=0)
parser.add_argument("--warmup_decay_iters", type=int, default=30)

# Reward
parser.add_argument("--normalize_reward", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--clip_reward", type=float, default=5.0)

# Regularization
parser.add_argument("--residual_l1", type=float, default=0.0)
parser.add_argument("--residual_l2", type=float, default=0.0)

# Reward tuning
parser.add_argument("--grasp_verify_steps", type=int, default=5)
parser.add_argument("--lift_min_sustain", type=int, default=3)
parser.add_argument("--lift_milestone_steps", type=int, default=15)
parser.add_argument("--verified_grasp_bonus", type=float, default=25.0)
parser.add_argument("--verified_lift_bonus", type=float, default=100.0)
parser.add_argument("--gripper_open_threshold", type=float, default=1.0)
parser.add_argument("--held_ee_max_dist", type=float, default=0.10)
parser.add_argument("--verified_lift_steps", type=int, default=25)
parser.add_argument("--lifted_pose_return_bonus", type=float, default=300.0)
parser.add_argument("--lifted_pose_return_sim_thresh", type=float, default=0.80)
parser.add_argument("--lifted_pose_return_hold_steps", type=int, default=8)
parser.add_argument("--lifted_pose_pre_vlift_frac", type=float, default=0.15)
parser.add_argument("--r4b_scale", type=float, default=160.0)
parser.add_argument("--r8_penalty", type=float, default=-2.0)

# Eval/save
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--eval_first", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--save_dir", type=str, default="checkpoints/resip")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--resume_resip", type=str, default=None)
parser.add_argument("--resume_actor_only", type=lambda x: x.lower() == "true", default=False,
                    help="Actor weights만 로드, critic/optimizer는 새로 초기화")
parser.add_argument("--enable_domain_randomization", type=lambda x: x.lower() == "true", default=False)
parser.add_argument("--_test_phase_a_arm_override", type=lambda x: x.lower() == "true", default=False,
                    help="TEST: Phase A에서 arm을 보간으로 강제. drop 원인 파악용.")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Imports
# ═══════════════════════════════════════════════════════════════════════════════
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from isaaclab.utils.math import quat_apply
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


def make_env(skill, num_envs, args):
    if skill in ("approach_and_grasp", "combined_s2_s3", "carry"):
        from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
        cfg = Skill2EnvCfg()
        cfg.scene.num_envs = num_envs
    elif skill == "carry_and_place":
        from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
        cfg = Skill3EnvCfg()
        cfg.scene.num_envs = num_envs
        if args.handoff_buffer:
            cfg.handoff_buffer_path = args.handoff_buffer
    else:
        raise ValueError(skill)

    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = args.enable_domain_randomization
    cfg.arm_limit_write_to_sim = False
    cfg.grasp_contact_threshold = 0.55
    cfg.grasp_gripper_threshold = 0.65
    cfg.grasp_max_object_dist = 0.50
    cfg.episode_length_s = 300.0
    cfg.spawn_heading_noise_std = 0.3
    cfg.spawn_heading_max_rad = 0.5
    cfg.dr_object_static_friction_scale_range = (1.0, 1.5)
    cfg.dr_object_dynamic_friction_scale_range = (1.0, 1.5)

    if skill == "combined_s2_s3":
        # S2 phase: lift success로 자동 종료 안 함 (수동 판정)
        cfg.grasp_success_height = 100.0
        cfg.lift_hold_steps = 0
        # dest object 설정 (S3 phase에서 사용)
        cfg.dest_spawn_dist_min = args.s3_dest_spawn_dist_min
        cfg.dest_spawn_dist_max = args.s3_dest_spawn_dist_max
        cfg.dest_heading_noise_std = 0.3
        cfg.dest_heading_max_rad = args.s3_dest_heading_max_rad
        cfg.dest_object_fixed = False
        cfg.dest_object_scale = 0.56
        cfg.dest_object_mass = 50.0  # 절대 안 밀리게
    elif skill == "carry":
        cfg.grasp_success_height = 100.0
        cfg.lift_hold_steps = 0
        cfg.max_dist_from_origin = 50.0  # carry: 이동 중 oob 방지
    else:
        cfg.grasp_success_height = 1.00

    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.multi_object_json:
        cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if args.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    if skill in ("approach_and_grasp", "combined_s2_s3", "carry"):
        env = Skill2Env(cfg=cfg)
    else:
        env = Skill3Env(cfg=cfg)
    print(f"  Env: {skill}, n={num_envs}, dev={env.device}")
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
    agent.model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    agent.normalizer.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("normalizer.")})
    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()
    agent.inference_steps = 4
    print(f"Frozen DP: obs={c['obs_dim']}, act={c['act_dim']}")
    return agent, c


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if args.skill == "combined_s2_s3":
        return main_combined()
    if args.skill == "carry":
        return main_carry()
    if args.skill == "navigate":
        return main_navigate()

    seed = args.seed or random.randint(0, 2**32 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"Seed: {seed}")

    env = make_env(args.skill, args.num_envs, args)
    dev = env.device
    N = env.num_envs

    from lekiwi_skill2_eval import EE_LOCAL_OFFSET
    jaw_idx, _ = env.env.robot.find_bodies(["Wrist_Roll_08c_v1"])
    jaw_idx = jaw_idx[0]
    ee_off = torch.tensor(EE_LOCAL_OFFSET, device=dev).unsqueeze(0)

    dp, dpc = load_frozen_dp(args.bc_checkpoint, dev)
    OD, AD = dpc["obs_dim"], dpc["act_dim"]

    # Scale
    if args.action_scale is not None:
        scale = torch.full((AD,), args.action_scale, device=dev)
    else:
        scale = torch.zeros(AD, device=dev)
        scale[0:5] = args.action_scale_arm
        scale[5]   = args.action_scale_gripper
        scale[6:9] = args.action_scale_base
    print(f"Scale: {scale.tolist()}")

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

    opt_a = optim.AdamW([p for n, p in rpol.named_parameters() if "critic" not in n],
                        lr=args.lr_actor, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-6)
    opt_c = optim.AdamW([p for n, p in rpol.named_parameters() if "critic" in n],
                        lr=args.lr_critic, eps=1e-5, weight_decay=1e-6)

    S = args.num_env_steps
    B = S * N
    MB = B // args.num_minibatches
    NI = args.total_timesteps // B

    sch_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=NI, eta_min=args.lr_actor * 0.01)
    sch_c = optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=NI, eta_min=args.lr_critic * 0.01)

    rew_norm = RunningMeanStdClip(shape=(1,), clip_value=args.clip_reward, device=dev) \
               if args.normalize_reward else None

    # Resume
    gs, gi = 0, 0
    if args.resume_resip:
        ck = torch.load(args.resume_resip, map_location=dev, weights_only=False)
        if args.resume_actor_only:
            # Actor weights만 로드, critic/optimizer는 새로 초기화 (MDP 변경 시 안전)
            actor_sd = {k: v for k, v in ck["residual_policy_state_dict"].items()
                        if "critic" not in k}
            rpol.load_state_dict(actor_sd, strict=False)
            gs = ck.get("global_step", 0); gi = ck.get("iteration", 0)
            print(f"Resumed ACTOR ONLY from: {args.resume_resip} "
                  f"(iter={gi}, step={gs}, critic/optimizer reset)")
        else:
            rpol.load_state_dict(ck["residual_policy_state_dict"])
            if "optimizer_actor_state_dict" in ck: opt_a.load_state_dict(ck["optimizer_actor_state_dict"])
            if "optimizer_critic_state_dict" in ck: opt_c.load_state_dict(ck["optimizer_critic_state_dict"])
            gs = ck.get("global_step", 0); gi = ck.get("iteration", 0)
            print(f"Resumed: iter={gi}, step={gs}")

    # Buffers
    RD = OD + AD
    obs_b  = torch.zeros((S, N, RD), device=dev)
    act_b  = torch.zeros((S, N, AD), device=dev)
    lp_b   = torch.zeros((S, N), device=dev)
    rew_b  = torch.zeros((S, N), device=dev)
    done_b = torch.zeros((S, N), device=dev)
    val_b  = torch.zeros((S, N), device=dev)

    # Constants (v6 identical)
    GV  = args.grasp_verify_steps
    LMS = args.lift_min_sustain
    LMI = args.lift_milestone_steps
    VLS = args.verified_lift_steps
    LHT = 0.05
    HELD_EE_MAX = args.held_ee_max_dist
    OPEN_T = args.gripper_open_threshold

    # Lift reward target: 4-12 cm object height and return-to-carry pose.
    LIFTED_POSE = torch.tensor([-0.045, -0.194, 0.277, -0.908, 0.020], device=dev)
    LIFT_REWARD_MIN_H = 0.04
    LIFT_REWARD_MAX_H = 0.12
    LIFT_POSE_SIGMA = 0.35

    # Per-env state (v6 identical)
    ms_go   = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_gr   = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_vl   = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_pr   = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_li   = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_sl   = torch.zeros(N, dtype=torch.bool, device=dev)
    g_sus   = torch.zeros(N, dtype=torch.long, device=dev)
    l_sus   = torch.zeros(N, dtype=torch.long, device=dev)
    p_sus   = torch.zeros(N, dtype=torch.long, device=dev)
    p_ee_xy = torch.zeros(N, device=dev)
    p_bs_xy = torch.zeros(N, device=dev)

    # Diagnostics
    r_gr     = torch.zeros(N, dtype=torch.long, device=dev)
    r_vl     = torch.zeros(N, dtype=torch.long, device=dev)
    r_pr     = torch.zeros(N, dtype=torch.long, device=dev)
    r_egr    = torch.zeros(N, dtype=torch.long, device=dev)
    r_li     = torch.zeros(N, dtype=torch.long, device=dev)
    r_sl     = torch.zeros(N, dtype=torch.long, device=dev)
    r_moz    = torch.zeros(N, device=dev)
    r_mgs    = torch.zeros(N, dtype=torch.long, device=dev)
    r_mls    = torch.zeros(N, dtype=torch.long, device=dev)
    r_mcf    = torch.zeros(N, device=dev)
    r_cgs    = torch.zeros(1, device=dev)
    r_cgn    = torch.zeros(1, dtype=torch.long, device=dev)
    r_ggs    = torch.zeros(1, device=dev)
    r_ggn    = torch.zeros(1, dtype=torch.long, device=dev)
    r_bgs    = torch.zeros(1, device=dev)
    r_bgn    = torch.zeros(1, dtype=torch.long, device=dev)
    r_bls    = torch.zeros(1, device=dev)
    r_bln    = torch.zeros(1, dtype=torch.long, device=dev)
    _ldbg    = 0
    _r2_sum  = 0.0
    _open_ct = 0
    _clip_ct = 0
    _clip_n  = 0
    _r4b_sum = 0.0
    _r8_n    = 0

    # ── Helpers ──
    def ee_pos():
        wp = env.env.robot.data.body_pos_w[:, jaw_idx, :]
        wq = env.env.robot.data.body_quat_w[:, jaw_idx, :]
        return wp + quat_apply(wq, ee_off.expand_as(wp))

    def ee_obj_dist_3d():
        d = torch.norm(ee_pos() - env.env.object_pos_w, dim=-1).view(-1)
        return torch.nan_to_num(d, nan=1.0)

    def ee_obj_dist_xy():
        d = torch.norm(ee_pos()[:, :2] - env.env.object_pos_w[:, :2], dim=-1).view(-1)
        return torch.nan_to_num(d, nan=1.0)

    def base_obj_dist_xy():
        d = torch.norm(env.env.robot.data.root_pos_w[:, :2]
                       - env.env.object_pos_w[:, :2], dim=-1).view(-1)
        return torch.nan_to_num(d, nan=1.0)

    def reset_ep(mask):
        ms_go[mask] = False; ms_gr[mask] = False
        ms_vl[mask] = False; ms_pr[mask] = False; ms_li[mask] = False; ms_sl[mask] = False
        g_sus[mask] = 0; l_sus[mask] = 0; p_sus[mask] = 0
        p_ee_xy[mask] = ee_obj_dist_xy()[mask]
        p_bs_xy[mask] = base_obj_dist_xy()[mask]

    # ── Print config ──
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    bsr, bsl, bgr = 0.0, 0, 0
    tt = 0; t0 = time.time()
    next_obs = env.reset(); next_done = torch.zeros(N, device=dev); dp.reset()

    print(f"\n{'='*60}")
    print(f"  ResiP v6.4 — {args.skill}")
    print(f"  N={N} S={S} B={B} iters={NI}")
    print(f"  scale: arm={args.action_scale_arm} grip={args.action_scale_gripper} base={args.action_scale_base}")
    print(f"  lr: a={args.lr_actor} c={args.lr_critic} kl={args.target_kl} ent={args.ent_coef}")
    print(f"  DR={'ON' if args.enable_domain_randomization else 'OFF'}")
    print(f"  rew_norm={'ON' if args.normalize_reward else 'OFF'}")
    print(f"  R1=GripOpen(+10) R2=ArmXY(×30,base-sub)")
    print(
        f"  R3=VGrasp(+{args.verified_grasp_bonus:.0f},{GV}s) "
        f"R3b=VLift(+{args.verified_lift_bonus:.0f},{VLS}s,objZ>{LHT:.2f},ee<{HELD_EE_MAX:.2f}) "
        f"R4=Lift(×200,{LIFT_REWARD_MIN_H:.2f}~{LIFT_REWARD_MAX_H:.2f}m,sus≥{LMS},ee<{HELD_EE_MAX})"
    )
    print(
        f"  R4b=LiftPose(×{args.r4b_scale},σ={LIFT_POSE_SIGMA},pre={args.lifted_pose_pre_vlift_frac:.2f}) "
        f"R4c=PoseReturn(+{args.lifted_pose_return_bonus:.0f},sim>{args.lifted_pose_return_sim_thresh:.2f},{args.lifted_pose_return_hold_steps}s)"
    )
    print(f"  R5=SustBonus(×50,{LMI}s) R6=SoftLift(+100) R7=Time(-0.01)")
    print(f"  R8=GCF({args.r8_penalty})")
    print(f"  warmup: {args.warmup_steps_initial}→{args.warmup_steps_final}/{args.warmup_decay_iters}")
    print(f"{'='*60}\n")

    # ═════════════════════════════════════════════════════════════════════════
    while gs < args.total_timesteps:
        gi += 1; it0 = time.time()
        ev = (gi - int(args.eval_first)) % args.eval_interval == 0

        next_obs = env.reset(); dp.reset()
        next_done = torch.zeros(N, device=dev)
        ms_go.zero_(); ms_gr.zero_(); ms_vl.zero_(); ms_pr.zero_(); ms_li.zero_(); ms_sl.zero_()
        g_sus.zero_(); l_sus.zero_(); p_sus.zero_()

        # Warmup
        prog = min(1.0, (gi - 1) / max(1, args.warmup_decay_iters))
        ws = max(0, int(args.warmup_steps_initial
                        + (args.warmup_steps_final - args.warmup_steps_initial) * prog))
        # Warmup with periodic reset: keep BC handoff short so RL can correct failures early.
        WU_RESET = 500
        for wi in range(ws):
            with torch.no_grad():
                a = dp.normalizer(dp.base_action_normalized(next_obs), "action", forward=False)
            next_obs, _, ter, tru, _ = env.step(a)
            next_done = (ter | tru).view(-1).float()
            if (wi + 1) % WU_RESET == 0 and wi < ws - 1:
                next_obs = env.reset(); dp.reset()
                next_done = torch.zeros(N, device=dev)

        p_ee_xy = ee_obj_dist_xy()
        p_bs_xy = base_obj_dist_xy()

        print(f"\nIter {gi}/{NI} | {'EVAL' if ev else 'TRAIN'} | "
              f"step={gs} | wu={ws} | "
              f"EE: {p_ee_xy.mean():.3f}/{p_ee_xy.min():.3f}")

        # ── Rollout ──
        for step in range(S):
            if not ev: gs += N

            with torch.no_grad():
                ba = dp.base_action_normalized(next_obs)
                no = torch.nan_to_num(torch.clamp(
                    dp.normalizer(next_obs, "obs", forward=True), -3, 3), nan=0.0)
                ba = torch.nan_to_num(ba, nan=0.0)

            ro = torch.cat([no, ba], dim=-1)
            done_b[step] = next_done; obs_b[step] = ro

            with torch.no_grad():
                ra_s, _, _, val, ra_m = rpol.get_action_and_value(ro)
            ra = ra_m if ev else ra_s
            ra = torch.clamp(ra, -1.0, 1.0)
            if not ev:
                _clip_ct += (ra_s.abs() > 0.99).sum().item()
                _clip_n  += ra_s.numel()
            with torch.no_grad():
                _, lp, _, _, _ = rpol.get_action_and_value(ro, ra)

            action = dp.normalizer(ba + ra * scale, "action", forward=False)
            next_obs, _, ter, tru, info = env.step(action)
            next_obs = torch.nan_to_num(next_obs, nan=0.0)
            done = ter | tru

            # ── Read state ──
            ee_xy = ee_obj_dist_xy()
            ee_3d = ee_obj_dist_3d()
            bs_xy = base_obj_dist_xy()
            grip = torch.nan_to_num(env.env.robot.data.joint_pos[:, env.env.gripper_idx].view(-1),
                                    nan=0.0, posinf=1.5, neginf=-0.2)
            eg = info.get("object_grasped_mask", env.env.object_grasped).view(-1)
            jg = info.get("just_grasped_mask", env.env.just_grasped).view(-1).float()
            oh = info.get("object_height_mask",
                (env.env.object_pos_w[:, 2] - env.env.scene.env_origins[:, 2])).view(-1)
            cf = info.get("contact_force_raw", torch.zeros(N, device=dev)).view(-1)
            gcf = info.get("ground_contact_force_raw", torch.zeros(N, device=dev)).view(-1)
            dual_contact = info.get("has_contact_mask", torch.zeros(N, dtype=torch.bool, device=dev)).view(-1).bool()
            ee_inside_bbox = info.get("ee_inside_bbox_mask", torch.zeros(N, dtype=torch.bool, device=dev)).view(-1).bool()
            object_standing = info.get("object_standing_mask", torch.ones(N, dtype=torch.bool, device=dev)).view(-1).bool()

            rew = torch.zeros(N, device=dev)

            # ══════════════════════════════════════════════════════
            # R1: GRIPPER OPEN MILESTONE (+10) — v6 identical
            # ══════════════════════════════════════════════════════
            nop = (grip > OPEN_T) & (~ms_go) & (~ms_gr)
            rew += nop.float() * 10.0
            ms_go |= nop

            # ══════════════════════════════════════════════════════
            # R2: ARM APPROACH XY (×30, base-subtracted) — v6 identical
            # ══════════════════════════════════════════════════════
            aok = (~ms_gr) & ms_go
            ee_p  = p_ee_xy - ee_xy
            bs_p  = p_bs_xy - bs_xy
            arm_p = ee_p - bs_p
            _r2_val = aok.float() * torch.clamp(arm_p * 30.0, -1.0, 3.0)
            rew += _r2_val
            _r2_sum += torch.nan_to_num(_r2_val, nan=0.0).sum().item()
            _open_ct += nop.sum().item()

            # ══════════════════════════════════════════════════════
            # R3: VERIFIED GRASP (reduced one-time bonus)
            # ══════════════════════════════════════════════════════
            live_grasp = eg & dual_contact & ee_inside_bbox & object_standing & (grip < float(env.env.cfg.grasp_gripper_threshold))
            gc = live_grasp & ms_go & (~ms_gr)
            g_sus[gc] += 1
            g_sus[~gc & (~ms_gr)] = 0
            r_mgs = torch.max(r_mgs, g_sus)

            vg = (g_sus >= GV) & (~ms_gr)
            rew += vg.float() * args.verified_grasp_bonus
            ms_gr |= vg
            r_gr += vg.long()
            r_egr += (jg > 0).long()

            if vg.any():
                r_ggs += grip[vg].sum()
                r_ggn += vg.sum()
                r_bgs += bs_xy[vg].sum()
                r_bgn += vg.sum()

            # ══════════════════════════════════════════════════════
            # R4: LIFT HEIGHT (×200, sustain≥3, ee<0.20) — v6 identical
            # ══════════════════════════════════════════════════════
            live_hold = live_grasp & (ee_3d < HELD_EE_MAX)
            held = (oh > LHT) & ms_gr & live_hold

            l_sus[held] += 1
            l_sus[~held] = 0
            r_mls = torch.max(r_mls, l_sus)

            arm_joints = torch.nan_to_num(env.env.robot.data.joint_pos[:, :5], nan=0.0)
            joint_err = torch.norm(arm_joints - LIFTED_POSE, dim=-1)
            pose_sim = torch.exp(-(joint_err ** 2) / (2.0 * (LIFT_POSE_SIGMA ** 2)))
            pose_sim = torch.nan_to_num(pose_sim, nan=0.0)

            hp = torch.clamp(
                (oh - LIFT_REWARD_MIN_H) / (LIFT_REWARD_MAX_H - LIFT_REWARD_MIN_H),
                0.0,
                1.0,
            )
            lok = l_sus >= LMS

            # R3b: first verified lift after sustained hold
            vl = held & (l_sus >= VLS) & (~ms_vl)
            rew += vl.float() * args.verified_lift_bonus
            ms_vl |= vl
            r_vl += vl.long()

            rew += ms_gr.float() * held.float() * lok.float() * hp * 200.0

            # ══════════════════════════════════════════════════════
            # R4b: two-stage lifted-pose shaping
            # ══════════════════════════════════════════════════════
            pose_stage = torch.where(
                ms_vl,
                torch.ones_like(pose_sim),
                torch.full_like(pose_sim, float(args.lifted_pose_pre_vlift_frac)),
            )
            r4b_r = ms_gr.float() * held.float() * lok.float() * pose_stage * pose_sim * args.r4b_scale
            r4b_r = torch.nan_to_num(r4b_r, nan=0.0)
            rew += r4b_r
            _r4b_sum += r4b_r.sum().item()

            # R4c: lifted-pose return milestone after stable lift and pose hold
            pose_ready = held & ms_vl & (pose_sim > args.lifted_pose_return_sim_thresh)
            p_sus[pose_ready] += 1
            p_sus[~pose_ready] = 0
            pose_return = pose_ready & (p_sus >= int(args.lifted_pose_return_hold_steps)) & (~ms_pr)
            rew += pose_return.float() * args.lifted_pose_return_bonus
            ms_pr |= pose_return
            r_pr += pose_return.long()

            # ══════════════════════════════════════════════════════
            # R5: SUSTAINED LIFT BONUS (×50, 15+ steps) — v6 identical
            # ══════════════════════════════════════════════════════
            gq = torch.exp(-((grip - 0.50) / 0.20) ** 2)
            sus = held & (l_sus >= LMI)
            rew += sus.float() * gq * 50.0

            # ══════════════════════════════════════════════════════
            # R6: SOFT-LIFT MILESTONE (+100) — v6 identical
            # ══════════════════════════════════════════════════════
            sl = sus & (gq > 0.3) & (~ms_sl)
            rew += sl.float() * 100.0
            ms_sl |= sl
            r_sl += sl.long()

            # ══════════════════════════════════════════════════════
            # R7: TIME PENALTY — v6 identical
            # ══════════════════════════════════════════════════════
            rew -= 0.01

            # ══════════════════════════════════════════════════════
            # R8: GRIPPER-GROUND CONTACT PENALTY [v6.4 NEW]
            # ══════════════════════════════════════════════════════
            grip_on_ground = (gcf > 1.0) & (oh < 0.05)
            rew += grip_on_ground.float() * args.r8_penalty
            _r8_n += grip_on_ground.sum().item()

            # ── Diagnostics ──
            nl = (l_sus >= LMI) & (~ms_li)
            r_li += nl.long(); ms_li |= nl
            if nl.any():
                r_bls += bs_xy[nl].sum()
                r_bln += nl.sum()

            _hi = oh > LHT
            if _hi.any() and _ldbg % 200 == 0:
                print(f"  [LIFT] n={_hi.sum().item()} cf={cf[_hi].min():.1f}~{cf[_hi].max():.1f} "
                      f"grip={grip[_hi].min():.2f}~{grip[_hi].max():.2f} "
                      f"ee3d={ee_3d[_hi].min():.3f} held={held.sum().item()} mg={ms_gr[_hi].sum().item()}")
            if _hi.any(): _ldbg += 1

            r_mcf = torch.max(r_mcf, cf)
            if eg.any():
                r_cgs += cf[eg].sum(); r_cgn += eg.sum()
                r_moz[eg] = torch.max(r_moz[eg], oh[eg])

            p_ee_xy = ee_xy.clone()
            p_bs_xy = bs_xy.clone()

            # ── Done masking ──
            dm = done.view(-1).bool()
            rew[dm] = 0.0

            if dm.any(): reset_ep(dm)

            rew = torch.nan_to_num(rew, nan=0.0)
            if rew_norm is not None and not ev:
                rew = rew_norm(rew.unsqueeze(-1)).squeeze(-1)

            val_b[step] = val.flatten()
            act_b[step] = ra; lp_b[step] = lp
            rew_b[step] = rew.view(-1)
            next_done = done.view(-1).float()

        # ── Summary ──
        sr = (r_li > 0).float().mean().item()
        tg, tvl, tpr, teg = r_gr.sum().item(), r_vl.sum().item(), r_pr.sum().item(), r_egr.sum().item()
        tl, tsl = r_li.sum().item(), r_sl.sum().item()

        fed = torch.nan_to_num(ee_obj_dist_3d(), nan=9.99)
        bd = torch.nan_to_num(base_obj_dist_xy(), nan=9.99)
        gv = torch.nan_to_num(env.env.robot.data.joint_pos[:, env.env.gripper_idx].view(-1), nan=0.5)

        d = ""
        eg2 = r_moz > 0
        if eg2.any(): d += f" | ObjZ:{r_moz[eg2].max():.3f}/{r_moz[eg2].mean():.3f} n={eg2.sum().item()}"
        d += f" | GSus:{r_mgs.max().item()} LSus:{r_mls.max().item()} n{LMI}+={(r_mls >= LMI).sum().item()}"
        mc = r_mcf.max().item()
        ac = r_mcf[r_mcf > 0].mean().item() if (r_mcf > 0).any() else 0
        d += f" | CF:{mc:.0f}/{ac:.0f} n={(r_mcf > 0).sum().item()}"
        if r_cgn.item() > 0: d += f" cf@g={(r_cgs / r_cgn).item():.3f}"
        if r_ggn.item() > 0: d += f" | GAG:{(r_ggs / r_ggn).item():.3f}(n={r_ggn.item()})"
        if r_bgn.item() > 0: d += f" B@G:{(r_bgs / r_bgn).item():.3f}"
        if r_bln.item() > 0: d += f" B@L:{(r_bls / r_bln).item():.3f}"

        for t in [r_gr, r_vl, r_pr, r_egr, r_li, r_sl, r_moz, r_mgs, r_mls, r_mcf,
                  r_cgs, r_cgn, r_ggs, r_ggn, r_bgs, r_bgn, r_bls, r_bln]:
            t.zero_()

        fps = S * N / max(time.time() - it0, 1e-6)
        cr = _clip_ct / max(_clip_n, 1)
        r2a = _r2_sum / max(S * N, 1)
        r4ba = _r4b_sum / max(S * N, 1)
        diag2 = (f" | R2avg={r2a:.3f} Opens={_open_ct} ClipR={cr:.3f}"
                 f" R4b={r4ba:.3f} R8=({_r8_n})")
        _r2_sum = 0.0; _open_ct = 0; _clip_ct = 0; _clip_n = 0
        _r4b_sum = 0.0; _r8_n = 0

        print(f"  SR={sr:.2%} | G={tg}(env:{teg}) | VL={tvl} | PR={tpr} | L={tl} | SL={tsl} | "
              f"EE={fed.min():.3f}({fed.mean():.3f}) | "
              f"Base={bd.min():.3f}({bd.mean():.3f}) | "
              f"Grip={gv.min():.2f}/{gv.mean():.2f}/{gv.max():.2f} | "
              f"R={rew_b.sum(0).mean():.1f} | FPS={fps:.0f}{d}{diag2}")

        if ev:
            if sr > bsr: bsr = sr
            if tsl > bsl:
                bsl = tsl
                torch.save({"residual_policy_state_dict": rpol.state_dict(),
                    "dp_checkpoint": args.bc_checkpoint, "dp_config": dpc,
                    "success_rate": sr, "soft_lifts": tsl, "grasps": tg,
                    "iteration": gi, "global_step": gs, "args": vars(args)},
                    save_dir / "resip_best.pt")
                print(f"  ★ Best SL={tsl}")
            if tg > bgr:
                bgr = tg
                torch.save({"residual_policy_state_dict": rpol.state_dict(),
                    "dp_checkpoint": args.bc_checkpoint, "dp_config": dpc,
                    "success_rate": sr, "grasps": tg,
                    "iteration": gi, "global_step": gs, "args": vars(args)},
                    save_dir / "resip_best_grasp.pt")
                print(f"  ★ Best G={tg}")
            if gi % 10 == 0 or gi <= 10:
                torch.save({"residual_policy_state_dict": rpol.state_dict(),
                    "dp_checkpoint": args.bc_checkpoint, "dp_config": dpc,
                    "iteration": gi, "args": vars(args)},
                    save_dir / f"resip_iter{gi}.pt")
            print(f"  Best: SR={bsr:.2%} SL={bsl} G={bgr}")
            continue

        # ═════════════════════════════════════════════════════════
        # PPO
        # ═════════════════════════════════════════════════════════
        with torch.no_grad():
            ba2 = dp.base_action_normalized(next_obs)
            no2 = torch.clamp(dp.normalizer(next_obs, "obs", forward=True), -3, 3)
            nv = rpol.get_value(torch.cat([no2, ba2], dim=-1)).flatten()

        adv, ret = compute_gae(val_b, nv, rew_b, done_b, next_done,
                               S, args.discount, args.gae_lambda)

        f = lambda t, *s: t.reshape(-1, *s) if s else t.reshape(-1)
        bo, ba_, blp = f(obs_b, RD), f(act_b, AD), f(lp_b)
        bv, badv, bret = f(val_b), f(adv), f(ret)

        idx = np.arange(B); cfs = []

        for ep in range(args.update_epochs):
            stop = False; np.random.shuffle(idx)
            for i0 in range(0, B, MB):
                mi = idx[i0:i0 + MB]
                _, nlp, ent, nv2, am = rpol.get_action_and_value(bo[mi], ba_[mi])
                lr = nlp - blp[mi]; ratio = lr.exp()

                with torch.no_grad():
                    kl = ((ratio - 1) - lr).mean()
                    cfs.append(((ratio - 1).abs() > args.clip_coef).float().mean().item())

                ma = badv[mi]
                if args.norm_adv: ma = (ma - ma.mean()) / (ma.std() + 1e-8)

                pg = torch.max(-ma * ratio,
                               -ma * ratio.clamp(1 - args.clip_coef,
                                                  1 + args.clip_coef)).mean()
                vl = 0.5 * ((nv2.view(-1) - bret[mi]) ** 2).mean()
                el = ent.mean() * args.ent_coef

                loss = (pg - el
                        + args.residual_l1 * am.abs().mean()
                        + args.residual_l2 * (am ** 2).mean()
                        + vl * args.vf_coef)

                opt_a.zero_grad(); opt_c.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(rpol.parameters(), args.max_grad_norm)
                opt_a.step(); opt_c.step()

                if args.target_kl and kl > args.target_kl:
                    print(f"    KL stop ep{ep}: {kl:.4f}>{args.target_kl}")
                    stop = True; break
            if stop: break

        sch_a.step(); sch_c.step()

        yp, yt = bv.cpu().numpy(), bret.cpu().numpy()
        vy = np.var(yt)
        ev2 = np.nan if vy == 0 else 1 - np.var(yt - yp) / vy

        tt += time.time() - it0
        sps = int(gs / tt) if tt > 0 else 0

        print(f"  pg={pg.item():.4f} v={vl.item():.4f} ent={ent.mean().item():.4f} "
              f"kl={kl.item():.4f} clip={np.mean(cfs):.3f} ev={ev2:.3f} SPS={sps}")

        if gi % 10 == 0:
            torch.save({"residual_policy_state_dict": rpol.state_dict(),
                "optimizer_actor_state_dict": opt_a.state_dict(),
                "optimizer_critic_state_dict": opt_c.state_dict(),
                "dp_checkpoint": args.bc_checkpoint, "dp_config": dpc,
                "iteration": gi, "global_step": gs, "args": vars(args)},
                save_dir / f"resip_iter{gi}.pt")

    print(f"\nDone in {time.time()-t0:.0f}s | Best: SR={bsr:.2%} SL={bsl} G={bgr}")
    torch.save({"residual_policy_state_dict": rpol.state_dict(),
        "dp_checkpoint": args.bc_checkpoint, "dp_config": dpc,
        "best_eval_success_rate": bsr, "iteration": gi, "global_step": gs,
        "args": vars(args)}, save_dir / "resip_final.pt")
    env.env.close(); simulation_app.close()


def main_combined():
    """Combined S2→S3: Skill-2 expert (frozen) → lift success → spawn dest → Skill-3 BC+Residual PPO."""
    seed = args.seed or random.randint(0, 2**32 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"Seed: {seed}")

    env = make_env("combined_s2_s3", args.num_envs, args)
    dev = env.device
    N = env.num_envs

    from lekiwi_skill2_eval import EE_LOCAL_OFFSET
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    jaw_idx, _ = env.env.robot.find_bodies(["Wrist_Roll_08c_v1"])
    jaw_idx = jaw_idx[0]
    ee_off = torch.tensor(EE_LOCAL_OFFSET, device=dev).unsqueeze(0)

    # ── Load S2 expert (frozen DP + ResiP) ──
    s2_dp, s2_dpc = load_frozen_dp(args.bc_checkpoint, dev)
    S2_OD, S2_AD = s2_dpc["obs_dim"], s2_dpc["act_dim"]
    s2_rpol = None
    if args.s2_resip_checkpoint and os.path.isfile(args.s2_resip_checkpoint):
        s2_ck = torch.load(args.s2_resip_checkpoint, map_location=dev, weights_only=False)
        s2_rpol = ResidualPolicy(
            obs_dim=S2_OD, action_dim=S2_AD,
            actor_hidden_size=args.actor_hidden_size,
            actor_num_layers=args.actor_num_layers,
            init_logstd=args.init_logstd, action_head_std=args.action_head_std,
            action_scale=0.1, learn_std=True,
        ).to(dev)
        s2_rpol.load_state_dict(s2_ck["residual_policy_state_dict"])
        s2_rpol.eval()
        for p in s2_rpol.parameters():
            p.requires_grad = False
        print(f"  [S2] ResiP loaded: {args.s2_resip_checkpoint}")
    s2_scale = torch.zeros(S2_AD, device=dev)
    s2_scale[0:5] = 0.20; s2_scale[5] = 0.25; s2_scale[6:9] = 0.35

    # ── Load S3 BC (frozen) ──
    s3_dp, s3_dpc = load_frozen_dp(args.s3_bc_checkpoint, dev)
    S3_OD, S3_AD = s3_dpc["obs_dim"], s3_dpc["act_dim"]
    print(f"  [S3] BC loaded: {args.s3_bc_checkpoint} (obs={S3_OD}D, act={S3_AD}D)")

    # ── S3 Residual Policy (trainable) ──

    rpol = ResidualPolicy(
        obs_dim=S3_OD, action_dim=S3_AD,
        actor_hidden_size=args.actor_hidden_size,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_size=args.critic_hidden_size,
        critic_num_layers=args.critic_num_layers,
        init_logstd=args.init_logstd, action_head_std=args.action_head_std,
        action_scale=0.1, learn_std=True,
        critic_last_layer_bias_const=0.25, critic_last_layer_std=0.25,
    ).to(dev)
    print(f"  [S3] Residual params: {sum(p.numel() for p in rpol.parameters()):,}")

    opt_a = optim.AdamW([p for n, p in rpol.named_parameters() if "critic" not in n],
                        lr=args.lr_actor, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-6)
    opt_c = optim.AdamW([p for n, p in rpol.named_parameters() if "critic" in n],
                        lr=args.lr_critic, eps=1e-5, weight_decay=1e-6)

    # Resume S3 residual
    gs, gi = 0, 0
    if args.resume_resip:
        ck = torch.load(args.resume_resip, map_location=dev, weights_only=False)
        if args.resume_actor_only:
            actor_sd = {k: v for k, v in ck["residual_policy_state_dict"].items() if "critic" not in k}
            rpol.load_state_dict(actor_sd, strict=False)
            gs = ck.get("global_step", 0); gi = ck.get("iteration", 0)
            print(f"  [S3] Resumed ACTOR ONLY (iter={gi}, step={gs})")
        else:
            rpol.load_state_dict(ck["residual_policy_state_dict"])
            if "optimizer_actor_state_dict" in ck: opt_a.load_state_dict(ck["optimizer_actor_state_dict"])
            if "optimizer_critic_state_dict" in ck: opt_c.load_state_dict(ck["optimizer_critic_state_dict"])
            gs = ck.get("global_step", 0); gi = ck.get("iteration", 0)
            print(f"  [S3] Resumed: iter={gi}, step={gs}")

    S = args.num_env_steps
    B = S * N
    MB = B // args.num_minibatches
    NI = args.total_timesteps // B
    rew_norm = RunningMeanStdClip(shape=(1,), clip_value=args.clip_reward, device=dev) \
               if args.normalize_reward else None

    sch_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=NI, eta_min=args.lr_actor * 0.01)
    sch_c = optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=NI, eta_min=args.lr_critic * 0.01)

    # Rollout buffers (S3 phase only)
    RD = S3_OD + S3_AD
    obs_b  = torch.zeros((S, N, RD), device=dev)
    act_b  = torch.zeros((S, N, S3_AD), device=dev)
    lp_b   = torch.zeros((S, N), device=dev)
    rew_b  = torch.zeros((S, N), device=dev)
    done_b = torch.zeros((S, N), device=dev)
    val_b  = torch.zeros((S, N), device=dev)

    # Per-env phase: 0=S2 (expert), 1=S3 (trainable)
    phase = torch.zeros(N, dtype=torch.long, device=dev)
    s2_lift_counter = torch.zeros(N, dtype=torch.long, device=dev)
    ep_step_counter = torch.zeros(N, dtype=torch.long, device=dev)  # per-env step counter
    S2_LIFT_HOLD = args.s2_lift_hold_steps  # 500
    S2_MAX_STEPS = 2000  # S2 timeout
    S2_NOLIFT_STEP = 700  # 이 step까지 못 들면 fail

    # ── S3 milestone / reward state ──
    ms_place = torch.zeros(N, dtype=torch.bool, device=dev)       # place 성공 milestone
    s3_no_contact_counter = torch.zeros(N, dtype=torch.long, device=dev)  # consecutive no-contact steps
    s3_wedged_counter = torch.zeros(N, dtype=torch.long, device=dev)  # wedge detection counter
    prev_base_dst_xy = torch.zeros(N, device=dev)                 # R1 delta 계산용
    prev_src_dst_xy = torch.zeros(N, device=dev)                  # Phase B src→dst delta 계산용
    prev_src_h = torch.zeros(N, device=dev)                       # R_lower delta 계산용
    prev_arm1 = torch.zeros(N, device=dev)                         # R_arm_lower delta 계산용
    s3_step_counter = torch.zeros(N, dtype=torch.long, device=dev)  # S3 phase step counter
    carry_arm_start_buf = torch.zeros(N, 5, device=dev)  # Phase A arm override test용
    carry_grip_start_buf = torch.zeros(N, device=dev)
    s3_init_pose6 = torch.zeros(N, 6, device=dev)  # S2→S3 전환 시 arm5+grip1 (36D obs용)
    s3_phase_a_latch = torch.ones(N, dtype=torch.bool, device=dev)  # Phase A latch: 한번 B면 복귀 안 함
    s3_arm1_at_phase_b_entry = torch.zeros(N, device=dev)  # Phase B 진입 시 arm1 값
    s3_objZ_at_phase_b_entry = torch.zeros(N, device=dev)  # Phase B 진입 시 objZ
    S3_NO_CONTACT_STEPS = 8   # consecutive steps without contact = drop check (Phase A only)
    s3_topple_counter = torch.zeros(N, dtype=torch.long, device=dev)  # objZ < 0.029 연속 카운터
    S3_PLACE_RADIUS = 0.14    # source↔dest XY distance for place success
    S3_DEST_CONTACT_PENALTY = -1.0   # dest 접촉 패널티 (place 시도 억제 방지)
    S3_PHASE_B_DIST = 0.42   # base→dest 이 거리 이하면 Phase B (팔 뻗기)
    S3_MAX_STEPS = 2000       # S3 phase timeout
    S3_REST_POSE = torch.tensor([-0.027, -0.207, 0.203, 0.123, 0.034], device=dev)

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    bsr, bsl, bgr = 0.0, 0, 0
    tt = 0; t0 = time.time()
    s2_success_total = 0; s3_success_total = 0; s3_place_total = 0; s3_drop_total = 0; total_episodes = 0
    s3_place_real_total = 0; s3_place_suspect_total = 0

    next_obs = env.reset(); s2_dp.reset(); s3_dp.reset()
    next_done = torch.zeros(N, device=dev)
    phase.zero_(); s2_lift_counter.zero_()

    print(f"\n{'='*60}")
    print(f"  ResiP Combined S2→S3")
    print(f"  N={N} S={S} B={B} iters={NI}")
    print(f"  S2 expert: {args.s2_resip_checkpoint or 'BC only'}")
    print(f"  S3 BC: {args.s3_bc_checkpoint}")
    print(f"  S2 lift hold: {S2_LIFT_HOLD} steps")
    print(f"  S3 dest spawn: {args.s3_dest_spawn_dist_min}~{args.s3_dest_spawn_dist_max}m")
    print(f"{'='*60}\n")

    # ═════════════════════════════════════════════════════════════════════════
    while gs < args.total_timesteps:
        gi += 1; it0 = time.time()
        ev = (gi - int(args.eval_first)) % args.eval_interval == 0

        # Phase별 residual scale (Phase A: BC 보존, Phase B: RL 주도)
        s3_scale_a = torch.zeros(S3_AD, device=dev)
        s3_scale_a[0:5] = 0.10    # arm: BC 보정 여지 확대 (early drop 보완)
        s3_scale_a[5] = 0.10       # grip: BC 보정 여지 확대
        s3_scale_a[6:9] = 0.10     # base: 약간 보정

        s3_scale_b = torch.zeros(S3_AD, device=dev)
        s3_scale_b[0:5] = 0.30     # arm: RL이 하강 동작 주도
        s3_scale_b[5] = 1.50        # grip: RL이 열기 주도 (BC=-0.573 상쇄 필요)
        s3_scale_b[6:9] = 0.20      # base: Phase B 미세 접근

        # 전체 reset 안 함 — S3 env는 유지, S2 env는 자연스럽게 진행
        # 첫 iter이거나 모든 env가 S2일 때만 reset
        if gi == 1:
            next_obs = env.reset(); s2_dp.reset(); s3_dp.reset()
            next_done = torch.zeros(N, device=dev)
            phase.zero_(); s2_lift_counter.zero_(); ep_step_counter.zero_()

        print(f"\nIter {gi}/{NI} | {'EVAL' if ev else 'TRAIN'} | step={gs} | "
              f"S2:{(phase==0).sum().item()} S3:{(phase==1).sum().item()}")

        # ── Rollout ──
        s3_step_count = torch.zeros(N, dtype=torch.long, device=dev)
        s3_valid = torch.zeros((S, N), dtype=torch.bool, device=dev)
        for step in range(S):
            if not ev: gs += N

            actor_obs = next_obs  # (N, 30D) from Skill2Env

            # ── Per-env action based on phase ──
            with torch.no_grad():
                # S2 expert action (for all envs, will mask later)
                s2_ba = s2_dp.base_action_normalized(actor_obs)
                if s2_rpol is not None:
                    s2_no = torch.nan_to_num(torch.clamp(
                        s2_dp.normalizer(actor_obs, "obs", forward=True), -3, 3), nan=0.0)
                    s2_ro = torch.cat([s2_no, s2_ba], dim=-1)
                    s2_ra, _, _, _, s2_ram = s2_rpol.get_action_and_value(s2_ro)
                    s2_ra = s2_ram  # deterministic for expert
                    s2_action = s2_dp.normalizer(s2_ba + torch.clamp(s2_ra, -1, 1) * s2_scale, "action", forward=False)
                else:
                    s2_action = s2_dp.normalizer(s2_ba, "action", forward=False)

                # S3 obs conversion: S2 30D → S3 29D (직접 조립)
                dest_pos = env.env.dest_object_pos_w
                robot_pos = env.env.robot.data.root_pos_w
                robot_quat = env.env.robot.data.root_quat_w
                rel_w = dest_pos - robot_pos
                dest_rel_body = quat_apply_inverse(robot_quat, rel_w)
                # contact force (연속값) — S3 obs[24]
                contact_force = env.env._contact_force_per_env().unsqueeze(-1)  # (N, 1)
                s3_obs29 = torch.cat([
                    actor_obs[:, 0:21],      # arm(5)+grip(1)+base_vel(3)+lin_vel(3)+ang_vel(3)+arm_vel(6) = 21D
                    dest_rel_body,            # [21:24] dest relative pos 3D
                    contact_force,            # [24:25] grip force 1D (연속값)
                    actor_obs[:, 26:29],      # [25:28] bbox 3D
                    actor_obs[:, 29:30],      # [28:29] category 1D
                ], dim=-1)  # 29D
                # 36D = 29D + init_pose(6D) + phase_a_flag(1D, latch)
                bdxy = torch.norm(
                    env.env.robot.data.root_pos_w[:, :2] - env.env.dest_object_pos_w[:, :2], dim=-1)
                # Latch: 한번 0.40m 이하 진입 시 영구 Phase B
                entered_b = (phase == 1) & s3_phase_a_latch & (bdxy <= S3_PHASE_B_DIST)
                if entered_b.any():
                    arm_joints_at_entry = env.env.robot.data.joint_pos[entered_b][:, env.env.arm_idx[:5]]
                    s3_arm1_at_phase_b_entry[entered_b] = arm_joints_at_entry[:, 1]
                    _src_h_local = env.env.object_pos_w[:, 2] - env.env.scene.env_origins[:, 2]
                    s3_objZ_at_phase_b_entry[entered_b] = _src_h_local[entered_b]
                s3_phase_a_latch[entered_b] = False
                phase_a_flag = s3_phase_a_latch.float().unsqueeze(-1)  # (N, 1)
                s3_obs = torch.cat([s3_obs29, s3_init_pose6, phase_a_flag], dim=-1)  # 36D

                # S3 BC base action
                s3_ba = s3_dp.base_action_normalized(s3_obs)
                s3_no = torch.nan_to_num(torch.clamp(
                    s3_dp.normalizer(s3_obs, "obs", forward=True), -3, 3), nan=0.0)
                s3_ba = torch.nan_to_num(s3_ba, nan=0.0)
                s3_ro = torch.cat([s3_no, s3_ba], dim=-1)

            # S3 residual (trainable)
            with torch.no_grad():
                s3_ra_s, _, _, s3_val, s3_ra_m = rpol.get_action_and_value(s3_ro)
            s3_ra = s3_ra_m if ev else s3_ra_s
            s3_ra = torch.clamp(s3_ra, -1.0, 1.0)
            S3_BC_WARMUP_ITERS = 10
            residual_alpha = min(1.0, gi / S3_BC_WARMUP_ITERS)
            s3_ra = s3_ra * residual_alpha
            with torch.no_grad():
                _, s3_lp, _, _, _ = rpol.get_action_and_value(s3_ro, s3_ra)
            s3_scale = torch.where(
                s3_phase_a_latch.unsqueeze(-1),
                s3_scale_a.unsqueeze(0).expand(N, -1),
                s3_scale_b.unsqueeze(0).expand(N, -1),
            )
            combined = s3_ba + s3_ra * s3_scale
            combined[:, 5] = torch.clamp(combined[:, 5], -0.45, 1.0)  # gripper clamp: grip_pos >= ~0.26 (끼임 방지)
            s3_action = s3_dp.normalizer(combined, "action", forward=False)

            # Merge action by phase
            is_s2 = (phase == 0)
            is_s3 = (phase == 1)
            action = torch.where(is_s2.unsqueeze(-1), s2_action, s3_action)

            # ★ TEST: Phase A arm override — carry 이동 중 arm을 보간으로 강제
            # 이 블록을 제거하면 BC arm으로 복귀. drop 비교용.
            if getattr(args, '_test_phase_a_arm_override', False) and is_s3.any():
                _dst_pos = env.env.dest_object_pos_w
                _base_pos = env.env.robot.data.root_pos_w
                _bdist = torch.norm(_base_pos[:, :2] - _dst_pos[:, :2], dim=-1)
                _in_phase_a = is_s3 & (_bdist > S3_PHASE_B_DIST)
                if _in_phase_a.any():
                    # S3 시작 arm pose → S3_ARM_END 보간
                    _t = (s3_step_counter.float() / 600.0).clamp(0, 1).unsqueeze(-1)
                    _S3_END = torch.tensor([+0.002, -0.193, +0.295, -1.306, +0.006], device=dev)
                    _arm_t = carry_arm_start_buf * (1 - _t) + _S3_END.unsqueeze(0) * _t
                    _grip_t = carry_grip_start_buf * (1 - _t.squeeze(-1)) + 0.15 * _t.squeeze(-1)
                    _arm6 = torch.cat([_arm_t, _grip_t.unsqueeze(-1)], dim=-1)
                    # Normalize
                    _lim = getattr(env.env, "_arm_action_limits_override", None)
                    if _lim is None:
                        _lim = env.env.robot.data.soft_joint_pos_limits[:, env.env.arm_idx]
                    _lo, _hi = _lim[..., 0], _lim[..., 1]
                    _ctr = 0.5 * (_lo + _hi); _hlf = 0.5 * (_hi - _lo)
                    _fin = torch.isfinite(_ctr) & torch.isfinite(_hlf) & (_hlf.abs() > 1e-6)
                    _hlf = torch.where(_fin, _hlf, torch.ones_like(_hlf))
                    _ctr = torch.where(_fin, _ctr, torch.zeros_like(_ctr))
                    _arm_norm = ((_arm6 - _ctr) / _hlf).clamp(-1, 1)
                    action[_in_phase_a, :6] = _arm_norm[_in_phase_a]

            # Store S3 transitions
            obs_b[step] = s3_ro
            act_b[step] = s3_ra
            lp_b[step] = s3_lp.view(-1)
            val_b[step] = s3_val.view(-1)
            done_b[step] = next_done
            # Mark S2 phase steps as done (won't contribute to GAE)
            done_b[step][is_s2] = 1.0
            s3_valid[step] = is_s3

            # DEBUG: gripper 추적 + S3 hold env jaw/wrist 분석
            if N <= 4:
                _gp = env.env.robot.data.joint_pos[0, env.env.gripper_idx].item()
                _ph = phase[0].item()
                _s2g = s2_action[0, 5].item()
                _s3g = s3_action[0, 5].item()
                _act_g = action[0, 5].item()
                _oh = (env.env.object_pos_w[0, 2] - env.env.scene.env_origins[0, 2]).item()
                _s3s = s3_step_counter[0].item()
                _show = (step % 10 == 0) or (_ph == 1 and _s3s <= 15)
                if _show:
                    _jaw = env.env._contact_force_per_env()[0].item()
                    _wrist = env.env._wrist_contact_force_per_env()[0].item()
                    print(f"    [DBG] step={step} {'S2' if _ph==0 else f'S3({_s3s})'} grip_pos={_gp:.3f} s2g={_s2g:.3f} s3g={_s3g:.3f} act={_act_g:.3f} objZ={_oh:.3f} jaw={_jaw:.2f} wrist={_wrist:.2f}")
            # S3 hold env jaw vs wrist 집계 (50 step 마다)
            if is_s3.any() and step % 50 == 0:
                _jaw_all = env.env._contact_force_per_env()
                _wrist_all = env.env._wrist_contact_force_per_env()
                _s3m = is_s3
                _jaw_s3 = _jaw_all[_s3m]
                _wrist_s3 = _wrist_all[_s3m]
                _jaw_only = ((_jaw_s3 > 0.3) & (_wrist_s3 <= 0.3)).sum().item()
                _wrist_only = ((_jaw_s3 <= 0.3) & (_wrist_s3 > 0.3)).sum().item()
                _both = ((_jaw_s3 > 0.3) & (_wrist_s3 > 0.3)).sum().item()
                _none = ((_jaw_s3 <= 0.3) & (_wrist_s3 <= 0.3)).sum().item()
                _oh_s3 = (env.env.object_pos_w[_s3m, 2] - env.env.scene.env_origins[_s3m, 2])
                _held = (_oh_s3 > 0.04).sum().item()
                _phb = (~s3_phase_a_latch[_s3m]).sum().item()
                _arm1_s3 = env.env.robot.data.joint_pos[_s3m][:, env.env.arm_idx[1]]
                _arm1_hi = (_arm1_s3 > 2.0).sum().item()
                _grip_s3 = env.env.robot.data.joint_pos[_s3m, env.env.gripper_idx]
                _grip_open = (_grip_s3 > 0.40).sum().item()
                _src_dst_s3 = torch.norm(env.env.object_pos_w[_s3m, :2] - env.env.dest_object_pos_w[_s3m, :2], dim=-1)
                _near = (_src_dst_s3 < 0.14).sum().item()
                print(f"    [CONTACT] step={step} S3={_s3m.sum().item()} phB={_phb} | ct: jaw={_jaw_only} wrist={_wrist_only} both={_both} none={_none} | held={_held} arm1>2={_arm1_hi} grip>0.4={_grip_open} near_dest={_near}")

            # Step env
            next_obs, _, ter, tru, info = env.step(action)
            next_obs = torch.nan_to_num(next_obs, nan=0.0)
            done = ter | tru
            ep_step_counter += 1

            # ── S2 phase: lift detection + early termination ──
            oh = info.get("object_height_mask",
                (env.env.object_pos_w[:, 2] - env.env.scene.env_origins[:, 2])).view(-1)
            eg = info.get("object_grasped_mask", env.env.object_grasped).view(-1)

            # S2 early fail: topple (objZ < 0.026)
            s2_topple = is_s2 & (oh < 0.026)
            # S2 early fail: 700step까지 못 들었으면 (objZ < 0.04)
            s2_nolift = is_s2 & (ep_step_counter == S2_NOLIFT_STEP) & (oh < 0.04)
            # S2 timeout
            s2_timeout = is_s2 & (ep_step_counter >= S2_MAX_STEPS)
            # S2 실패 → 강제 reset
            s2_fail = s2_topple | s2_nolift | s2_timeout
            if s2_fail.any():
                fail_ids = s2_fail.nonzero(as_tuple=False).squeeze(-1)
                env.env._reset_idx(fail_ids)
                next_obs_new = env.env._get_observations()
                policy_obs = next_obs_new["policy"]
                next_obs[fail_ids] = policy_obs[fail_ids]
                phase[fail_ids] = 0
                s2_lift_counter[fail_ids] = 0
                ep_step_counter[fail_ids] = 0
                done[fail_ids.unsqueeze(-1) if fail_ids.dim() == 1 else fail_ids] = True

            # S2 lift detection
            lifted = is_s2 & eg & (oh > 0.05) & (~s2_fail)
            s2_lift_counter[lifted] += 1
            s2_lift_counter[is_s2 & ~lifted & ~s2_fail] = 0

            # Transition S2→S3: lift held for S2_LIFT_HOLD steps
            transition = is_s2 & (s2_lift_counter >= S2_LIFT_HOLD) & (~s2_fail)
            if transition.any():
                t_ids = transition.nonzero(as_tuple=False).squeeze(-1)
                # 로봇 전방 기준 dest 스폰
                rpos = env.env.robot.data.root_pos_w[t_ids]
                rquat = env.env.robot.data.root_quat_w[t_ids]
                fwd = quat_apply(rquat, torch.tensor([[0, 1, 0]], dtype=torch.float32, device=dev).expand(len(t_ids), -1))
                # Curriculum: iter 진행에 따라 max dist 점진 확장
                curr_dist_max = min(
                    args.s3_dest_spawn_dist_min + 0.01 * gi,  # iter당 +0.01m
                    args.s3_dest_spawn_dist_max,
                )
                dist = torch.rand(len(t_ids), device=dev) * (curr_dist_max - args.s3_dest_spawn_dist_min) + args.s3_dest_spawn_dist_min
                # heading noise (±dest_heading_max_rad)
                angle_noise = (torch.rand(len(t_ids), device=dev) * 2 - 1) * args.s3_dest_heading_max_rad
                cos_n = torch.cos(angle_noise); sin_n = torch.sin(angle_noise)
                fwd_x = fwd[:, 0] * cos_n - fwd[:, 1] * sin_n
                fwd_y = fwd[:, 0] * sin_n + fwd[:, 1] * cos_n
                dest_x = rpos[:, 0] + fwd_x * dist
                dest_y = rpos[:, 1] + fwd_y * dist
                dest_z = rpos[:, 2] - 0.03
                # dest object 위치 설정
                dest_rigid = env.env._dest_object_rigid
                if dest_rigid is not None:
                    pose = dest_rigid.data.default_root_state[t_ids, :7].clone()
                    pose[:, 0] = dest_x; pose[:, 1] = dest_y; pose[:, 2] = dest_z
                    yaw = torch.rand(len(t_ids), device=dev) * 2 * 3.14159 - 3.14159
                    pose[:, 3] = torch.cos(yaw * 0.5); pose[:, 6] = torch.sin(yaw * 0.5)
                    pose[:, 4] = 0; pose[:, 5] = 0
                    dest_rigid.write_root_pose_to_sim(pose, env_ids=t_ids)
                    env.env.dest_object_pos_w[t_ids, 0] = dest_x
                    env.env.dest_object_pos_w[t_ids, 1] = dest_y
                    env.env.dest_object_pos_w[t_ids, 2] = dest_z
                phase[t_ids] = 1
                ep_step_counter[t_ids] = 0
                ms_place[t_ids] = False
                s3_no_contact_counter[t_ids] = 0
                s3_step_counter[t_ids] = 0
                s3_wedged_counter[t_ids] = 0
                s3_phase_a_latch[t_ids] = True  # Phase A로 시작
                s3_arm1_at_phase_b_entry[t_ids] = 0.0
                s3_objZ_at_phase_b_entry[t_ids] = 0.0
                # S2→S3 전환 시 arm pose 캡처 (Phase A arm override + 36D obs)
                carry_arm_start_buf[t_ids] = env.env.robot.data.joint_pos[t_ids][:, env.env.arm_idx[:5]]
                carry_grip_start_buf[t_ids] = env.env.robot.data.joint_pos[t_ids, env.env.gripper_idx]
                s3_init_pose6[t_ids, :5] = carry_arm_start_buf[t_ids]
                s3_init_pose6[t_ids, 5] = carry_grip_start_buf[t_ids]
                # Initialize prev distance for delta reward
                prev_base_dst_xy[t_ids] = torch.norm(
                    rpos[:, :2] - torch.stack([dest_x, dest_y], dim=-1), dim=-1)
                prev_src_dst_xy[t_ids] = torch.norm(
                    env.env.object_pos_w[t_ids, :2] - torch.stack([dest_x, dest_y], dim=-1), dim=-1)
                prev_src_h[t_ids] = (env.env.object_pos_w[t_ids, 2] - env.env.scene.env_origins[t_ids, 2])
                prev_arm1[t_ids] = env.env.robot.data.joint_pos[t_ids][:, env.env.arm_idx[1]]
                s2_success_total += t_ids.shape[0]
                s3_dp.reset()  # S3 BC deque 클리어 (S2 obs 기반 stale action 제거)
                # DEBUG: S2→S3 전환 시점 gripper 상태
                _grip = env.env.robot.data.joint_pos[t_ids, env.env.gripper_idx]
                _arm = env.env.robot.data.joint_pos[t_ids][:, env.env.arm_idx]
                _objz = (env.env.object_pos_w[t_ids, 2] - env.env.scene.env_origins[t_ids, 2])
                # S2 vs S3 arm action delta
                _s2a = s2_action[t_ids, :6]
                _s3a = s3_action[t_ids, :6]
                _delta = _s3a - _s2a
                print(f"    [S2→S3] {t_ids.shape[0]} envs at step {step} | grip={_grip[:3].tolist()} arm3={_arm[:3, 3].tolist()} objZ={_objz[:3].tolist()}")
                print(f"      s2_act[0:6]={_s2a[0].tolist()}")
                print(f"      s3_act[0:6]={_s3a[0].tolist()}")
                print(f"      delta[0:6] ={_delta[0].tolist()}")

            # ── S3 reward (R0~R5) ──
            rew = torch.zeros(N, device=dev)
            s3_drop = torch.zeros(N, dtype=torch.bool, device=dev)
            s3_timeout = torch.zeros(N, dtype=torch.bool, device=dev)
            s3_wedge_fail = torch.zeros(N, dtype=torch.bool, device=dev)
            if is_s3.any():
                s3m = is_s3
                s3_step_counter[s3m] += 1
                # Source object & dest object positions
                src_pos = env.env.object_pos_w       # (N, 3) source 약병
                dst_pos = env.env.dest_object_pos_w  # (N, 3) dest 컵
                env_z = env.env.scene.env_origins[:, 2]
                src_h = src_pos[:, 2] - env_z        # source object height
                src_dst_xy = torch.norm(src_pos[:, :2] - dst_pos[:, :2], dim=-1)  # XY distance

                # Contact forces
                jaw_cf = env.env._contact_force_per_env()          # jaw↔source
                wrist_cf = env.env._wrist_contact_force_per_env()  # wrist↔source
                has_contact = (jaw_cf > 0.3) & (wrist_cf > 0.3)   # Phase A: 양쪽 모두 (엄격)
                has_contact_phb = (jaw_cf > 0.3) | (wrist_cf > 0.3)  # Phase B: 한쪽이라도 (완화)

                # Robot base position
                base_pos = env.env.robot.data.root_pos_w  # (N, 3)
                base_dst_xy = torch.norm(base_pos[:, :2] - dst_pos[:, :2], dim=-1)

                # Gripper pos & arm joints
                grip_pos = env.env.robot.data.joint_pos[:, env.env.gripper_idx]
                arm_joints = env.env.robot.data.joint_pos[:, env.env.arm_idx]

                # Gripper closed check (S2 grasp와 동일 기준)
                gripper_closed = grip_pos < float(env.env.cfg.grasp_gripper_threshold)

                # S3 hold = contact + gripper_closed (between_jaws는 carry 중 EE drift로 false drop 유발하므로 제외)
                is_holding = has_contact & gripper_closed & (grip_pos >= 0.20)
                is_holding_phb = has_contact_phb & gripper_closed & (grip_pos >= 0.15)

                # ── R0: Drop detection (topple + contact 기반) ──
                # Phase A: 0.04 (carrying height), Phase B: 0.029 (topple)
                s3_drop_thresh = torch.where(s3_phase_a_latch, torch.tensor(0.04, device=dev), torch.tensor(0.029, device=dev))
                s3_topple_counter[s3m & (src_h < s3_drop_thresh)] += 1
                s3_topple_counter[s3m & (src_h >= s3_drop_thresh)] = 0
                s3_topple_drop = s3m & (s3_topple_counter >= 8) & (~ms_place)

                # Contact loss: Phase A 전용 (Phase B에서는 place 시 contact 사라지므로 무시)
                s3_no_contact_counter[s3m & is_holding] = 0
                s3_no_contact_counter[s3m & ~is_holding & s3_phase_a_latch] += 1  # Phase A만
                s3_no_contact_counter[s3m & ~s3_phase_a_latch] = 0                # Phase B 리셋
                s3_contact_drop = s3m & (s3_no_contact_counter >= S3_NO_CONTACT_STEPS) & (~ms_place)

                s3_drop = s3_topple_drop | s3_contact_drop
                # S3 timeout
                s3_timeout = s3m & (s3_step_counter >= S3_MAX_STEPS) & (~ms_place)
                # Wedge detection: objZ > 0.10인데 is_holding 아니고 gripper 닫힌 상태 지속
                wedged = s3m & (~is_holding) & (src_h > 0.10) & gripper_closed & s3_phase_a_latch  # Phase A에서만
                s3_wedged_counter[wedged] += 1
                s3_wedged_counter[~wedged] = 0
                s3_wedge_fail = s3m & (s3_wedged_counter >= 30)
                s3_fail = s3_drop | s3_timeout | s3_wedge_fail

                # ── Phase 판정 (latch, obs의 phase_a_flag와 동일) ──
                phase_a = s3m & (~ms_place) & (~s3_fail) & s3_phase_a_latch
                phase_b = s3m & (~ms_place) & (~s3_fail) & (~s3_phase_a_latch)
                phase_c = s3m & ms_place

                # ── R_hold: Phase A에서 hold 보상 (hold-forever 방지: 0.05/step) ──
                hold = phase_a & is_holding & (src_h > 0.033)
                rew[hold] += 0.05

                # ── R_arm_maintain: Phase A에서 init_arm_pose 유지 보상 ──
                if phase_a.any():
                    arm_jp = env.env.robot.data.joint_pos[:, env.env.arm_idx[:5]]
                    grip_jp = env.env.robot.data.joint_pos[:, env.env.arm_idx[5:6]]
                    arm_err = (arm_jp - s3_init_pose6[:, :5]).pow(2).sum(dim=-1)
                    grip_err = (grip_jp.squeeze(-1) - s3_init_pose6[:, 5]).pow(2)
                    r_arm = 0.10 * torch.exp(-arm_err / (0.3 ** 2)) + 0.05 * torch.exp(-grip_err / (0.2 ** 2))
                    rew[phase_a] += r_arm[phase_a]

                # ── R1: Phase A — base → dest 접근 (delta × 30, 잡고 있을 때만) ──
                R1_TARGET = S3_PHASE_B_DIST  # 0.42 — Phase A 끝까지 접근 보상
                prev_err = (prev_base_dst_xy - R1_TARGET).abs()
                curr_err = (base_dst_xy - R1_TARGET).abs()
                approach_delta = torch.clamp(prev_err - curr_err, -0.05, 0.05)
                r1 = approach_delta * 30.0   # 0.5m 접근 → +15 total
                r1_mask = phase_a & is_holding
                rew[r1_mask] += r1[r1_mask]

                # ── R_arm: Phase B — src↔dst XY delta (팔 뻗기 + base 접근, is_holding) ──
                src_dst_delta = torch.clamp(prev_src_dst_xy - src_dst_xy, -0.05, 0.05)
                r_arm = src_dst_delta * 80.0  # balanced with R_lower
                r_arm_mask = phase_b & is_holding_phb
                rew[r_arm_mask] += r_arm[r_arm_mask]

                # ── R_base_approach: Phase B — base → dest 미세 접근 ──
                base_dst_delta = torch.clamp(prev_base_dst_xy - base_dst_xy, -0.02, 0.02)
                r_base_approach = base_dst_delta * 15.0
                rew[phase_b] += r_base_approach[phase_b]

                # ── R_lower: Phase B — objZ 내리기 (src_dst < 0.30) ──
                near_dest = phase_b & (src_dst_xy < 0.30)
                if near_dest.any():
                    objz_delta = torch.clamp(prev_src_h - src_h, -0.01, 0.01)
                    rew[near_dest] += (objz_delta * 80.0)[near_dest]  # balanced with R_arm

                # ── R_arm_lower: Phase B — arm1 하강 delta 보상 ──
                # 데모: arm1 총 Δ=3.21 over ~583스텝, per-step mean=0.00554
                # delta×30 → 에피소드 총 ~96 (stall 시 0). Place(400)의 24%.
                if near_dest.any():
                    arm1 = arm_joints[:, 1]
                    arm1_delta = torch.clamp(arm1 - prev_arm1, 0.0, 0.05)  # 상승만 보상, per-step max 0.05
                    rew[near_dest] += (arm1_delta * 30.0)[near_dest]

                # ── R_release: Phase B — grip 열기 (arm1 > 2.0 + dest 근처, Gaussian proximity) ──
                # 기존 src_h<0.10은 물체를 쥐고 있으면 불가능 (EP17: arm1=2.75 → objZ=0.135)
                release_ready = phase_b & (arm_joints[:, 1] > 2.0) & (src_dst_xy < 0.20)
                if release_ready.any():
                    proximity = torch.exp(-0.5 * (src_dst_xy[release_ready] / 0.15) ** 2)
                    grip_open_progress = torch.clamp((grip_pos[release_ready] - 0.25) / 0.30, 0.0, 1.0)
                    rew[release_ready] += grip_open_progress * proximity * 8.0

                # Update prev distances
                prev_base_dst_xy[s3m] = base_dst_xy[s3m]
                prev_src_dst_xy[s3m] = src_dst_xy[s3m]
                prev_src_h[s3m] = src_h[s3m]
                prev_arm1[s3m] = arm_joints[s3m, 1]

                # ── R2: dest contact penalty (Phase B에서 완화) ──
                dest_cf = env.env._dest_contact_force_per_env()
                dest_touching = (dest_cf > 0.3) & s3m
                dest_penalty = torch.where(phase_b, torch.tensor(-0.1, device=dev), torch.tensor(-1.0, device=dev))
                rew[dest_touching] += dest_penalty[dest_touching]

                # ── R3: Place success (+200) — contact 기반 (물체 바닥에 서있음 + dest 근처 + contact 없음) ──
                place_cond = (
                    s3m & ~ms_place
                    & (src_dst_xy < S3_PLACE_RADIUS)
                    & (~has_contact)
                    & (src_h > 0.029) & (src_h < 0.05)  # 바닥에 서있는 상태
                    & ~s3_fail
                )
                if place_cond.any():
                    ms_place[place_cond] = True
                    rew[place_cond] += 300.0
                    # REAL place 보너스: arm이 실제로 내려가서 place한 경우
                    real_place = place_cond & (
                        (arm_joints[:, 1] - s3_arm1_at_phase_b_entry) > 1.5
                    ) & (s3_step_counter > 100)
                    rew[real_place] += 100.0
                    s3_place_total += place_cond.sum().item()
                    # Detailed PLACE log: arm1 trajectory, Phase B duration, quality
                    # (real/suspect counters updated after computation below)
                    _p_arm1_now = arm_joints[place_cond, 1]
                    _p_arm1_entry = s3_arm1_at_phase_b_entry[place_cond]
                    _p_arm1_delta = _p_arm1_now - _p_arm1_entry
                    _p_objZ_entry = s3_objZ_at_phase_b_entry[place_cond]
                    _p_objZ_now = src_h[place_cond]
                    _p_s3step = s3_step_counter[place_cond]
                    _p_real = ((_p_arm1_delta > 1.0) & (_p_s3step > 50)).sum().item()
                    _p_suspect = place_cond.sum().item() - _p_real
                    s3_place_real_total += _p_real
                    s3_place_suspect_total += _p_suspect
                    print(f"    [S3] PLACE! {place_cond.sum().item()} envs (REAL={_p_real} SUSPECT={_p_suspect}) step={step} "
                          f"s3_step={_p_s3step.tolist()} | "
                          f"arm1: entry={_p_arm1_entry.tolist()}→now={_p_arm1_now.tolist()} Δ={_p_arm1_delta.tolist()} | "
                          f"objZ: entry={_p_objZ_entry.tolist()}→now={_p_objZ_now.tolist()} | "
                          f"grip={grip_pos[place_cond].tolist()} src_dst={src_dst_xy[place_cond].tolist()}")

                # ── R4: Phase C — rest pose + gripper close ──
                # 데모: place 후 grip→-0.20(닫힘) + arm→rest pose
                if phase_c.any():
                    pose_err = torch.norm(arm_joints[phase_c, :5] - S3_REST_POSE, dim=-1)
                    r4_pose = torch.exp(-0.5 * (pose_err / 0.3) ** 2) * 0.5
                    # grip close: 0.5→-0.20 (데모 최종 grip=-0.20)
                    grip_close_progress = torch.clamp((0.5 - grip_pos[phase_c]) / 0.7, 0.0, 1.0)
                    r4_grip = grip_close_progress * 0.3
                    rew[phase_c] += r4_pose + r4_grip

                # ── R5: Time penalty ──
                rew[s3m] += -0.01

                # ── R0/timeout: drop 패널티 없음, reset만 ──
                rew[s3_timeout] = -5.0

            rew_b[step] = rew

            # ── S3 fail (drop / timeout / wedge) → force reset to S2 ──
            s3_fail = s3_drop | s3_timeout | s3_wedge_fail
            if s3_drop.any():
                _drop_steps = s3_step_counter[s3_drop]
                s3_drop_total += s3_drop.sum().item()
                s3_drop_early = (_drop_steps < 50).sum().item()
                s3_drop_late = (_drop_steps >= 50).sum().item()
                _jaw_d = jaw_cf[s3_drop]
                _wrist_d = wrist_cf[s3_drop]
                _grip_d = grip_pos[s3_drop]
                _oh_d = src_h[s3_drop]
                _arm_d = arm_joints[s3_drop, :5]
                _d_arm1_entry = s3_arm1_at_phase_b_entry[s3_drop]
                _d_arm1_now = arm_joints[s3_drop, 1]
                _d_phase_a = s3_phase_a_latch[s3_drop]  # True=Phase A에서 drop
                _d_in_a = _d_phase_a.sum().item()
                _d_in_b = s3_drop.sum().item() - _d_in_a
                if s3_drop_early + s3_drop_late > 0:
                    _wedged = (_oh_d > 0.04).sum().item()
                    _real = (_oh_d <= 0.04).sum().item()
                    print(f"    [DROP] phA={_d_in_a} phB={_d_in_b} early={s3_drop_early} late={s3_drop_late} "
                          f"avg_step={_drop_steps.float().mean():.0f} | "
                          f"grip={_grip_d.mean():.3f} objZ={_oh_d.mean():.3f} | "
                          f"real={_real} wedged={_wedged} | "
                          f"arm1: entry={_d_arm1_entry.mean():.3f}→now={_d_arm1_now.mean():.3f}")
                    if _wedged > 0:
                        _w_mask = _oh_d > 0.04
                        print(f"      [WEDGED] grip={_grip_d[_w_mask].mean():.3f} objZ={_oh_d[_w_mask].mean():.3f} arm={_arm_d[_w_mask].mean(dim=0).tolist()}")
            if s3_fail.any():
                fail_ids = s3_fail.nonzero(as_tuple=False).squeeze(-1)
                env.env._reset_idx(fail_ids)
                next_obs_new = env.env._get_observations()
                policy_obs = next_obs_new["policy"]
                next_obs[fail_ids] = policy_obs[fail_ids]
                phase[fail_ids] = 0
                s2_lift_counter[fail_ids] = 0
                ep_step_counter[fail_ids] = 0
                ms_place[fail_ids] = False
                s3_no_contact_counter[fail_ids] = 0
                s3_topple_counter[fail_ids] = 0
                s3_step_counter[fail_ids] = 0
                s3_wedged_counter[fail_ids] = 0
                s3_init_pose6[fail_ids] = 0.0
                s3_phase_a_latch[fail_ids] = True
                s3_arm1_at_phase_b_entry[fail_ids] = 0.0
                s3_objZ_at_phase_b_entry[fail_ids] = 0.0
                prev_base_dst_xy[fail_ids] = 0.0
                prev_src_dst_xy[fail_ids] = 0.0
                prev_src_h[fail_ids] = 0.0
                prev_arm1[fail_ids] = 0.0

            # Handle env auto-resets (terminated/truncated by env)
            next_done = (ter | tru).view(-1).float()
            # S2 강제 fail도 done으로 처리
            next_done[s2_fail] = 1.0
            next_done[s3_fail] = 1.0
            reset_mask = (done.view(-1) | s2_fail | s3_fail)
            if reset_mask.any():
                phase[reset_mask] = 0
                s2_lift_counter[reset_mask] = 0
                ep_step_counter[reset_mask] = 0
                ms_place[reset_mask] = False
                s3_no_contact_counter[reset_mask] = 0
                s3_topple_counter[reset_mask] = 0
                s3_step_counter[reset_mask] = 0
                s3_wedged_counter[reset_mask] = 0
                s3_init_pose6[reset_mask] = 0.0
                s3_phase_a_latch[reset_mask] = True
                s3_arm1_at_phase_b_entry[reset_mask] = 0.0
                s3_objZ_at_phase_b_entry[reset_mask] = 0.0
                prev_base_dst_xy[reset_mask] = 0.0
                prev_src_dst_xy[reset_mask] = 0.0
                prev_src_h[reset_mask] = 0.0
                prev_arm1[reset_mask] = 0.0
                total_episodes += reset_mask.sum().item()

            s3_step_count[is_s3] += 1

        # ── PPO Update (S3 transitions only) ──
        if not ev:
            with torch.no_grad():
                dest_pos_f = env.env.dest_object_pos_w
                robot_pos_f = env.env.robot.data.root_pos_w
                robot_quat_f = env.env.robot.data.root_quat_w
                rel_w_f = dest_pos_f - robot_pos_f
                dest_rel_f = quat_apply_inverse(robot_quat_f, rel_w_f)
                cf_f = env.env._contact_force_per_env().unsqueeze(-1)
                s3_obs29_f = torch.cat([
                    next_obs[:, 0:21], dest_rel_f, cf_f,
                    next_obs[:, 26:29], next_obs[:, 29:30],
                ], dim=-1)
                # bootstrap: latch 상태 그대로 사용 (rollout 마지막 시점의 phase)
                phase_a_flag_f = s3_phase_a_latch.float().unsqueeze(-1)
                s3_obs_final = torch.cat([s3_obs29_f, s3_init_pose6, phase_a_flag_f], dim=-1)  # 36D
                s3_no_f = torch.nan_to_num(torch.clamp(
                    s3_dp.normalizer(s3_obs_final, "obs", forward=True), -3, 3), nan=0.0)
                s3_ba_f = torch.nan_to_num(s3_dp.base_action_normalized(s3_obs_final), nan=0.0)
                s3_ro_f = torch.cat([s3_no_f, s3_ba_f], dim=-1)
                nv = rpol.get_value(s3_ro_f).view(-1)

            badv, bret = compute_gae(val_b, nv, rew_b, done_b,
                                     next_done, S, args.discount, args.gae_lambda)
            badv = (badv - badv.mean()) / (badv.std() + 1e-8)

            # Filter to S3 phase transitions only (carry와 동일)
            valid_mask = s3_valid.reshape(-1)
            valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            BV = len(valid_idx)

            if BV == 0:
                print(f"    [PPO] No valid S3 steps, skipping PPO update")
            else:
                bobs = obs_b.view(-1, RD)[valid_idx]
                bact = act_b.view(-1, S3_AD)[valid_idx]
                blp  = lp_b.view(-1)[valid_idx]
                bret = bret.view(-1)[valid_idx]
                badv = badv.view(-1)[valid_idx]
                bv   = val_b.view(-1)[valid_idx]
                B = BV
                MB = max(B // max(args.num_minibatches, 1), 1)

            if BV > 0:
              idx = torch.randperm(B, device=dev)
              for ep in range(args.update_epochs):
                stop = False
                for st in range(0, B, MB):
                    mb = idx[st:st+MB]
                    _, nlp, ent, nv2, _ = rpol.get_action_and_value(bobs[mb], bact[mb])
                    ratio = (nlp.view(-1) - blp[mb]).exp()
                    pg1 = -badv[mb] * ratio
                    pg2 = -badv[mb] * ratio.clamp(1-0.2, 1+0.2)
                    pg = torch.max(pg1, pg2).mean()
                    vl = 0.5 * ((nv2.view(-1) - bret[mb])**2).mean()
                    loss = pg + vl * 0.5 - ent.mean() * args.ent_coef
                    opt_a.zero_grad(); opt_c.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(rpol.parameters(), 0.5)
                    opt_a.step(); opt_c.step()
                    kl = (blp[mb] - nlp.view(-1)).mean()
                    if args.target_kl and kl > args.target_kl:
                        stop = True; break
                if stop: break
            sch_a.step(); sch_c.step()

        # ── Logging ──
        s3_envs = (phase == 1).sum().item()
        s3_steps = s3_step_count.sum().item()
        # S3 phase 상태 추적
        s3_mask = (phase == 1)
        if s3_mask.any():
            s3_oh = (env.env.object_pos_w[s3_mask, 2] - env.env.scene.env_origins[s3_mask, 2])
            # is_holding 기반 판정 (contact AND + gripper_closed + between_jaws)
            _jaw = env.env._contact_force_per_env()[s3_mask]
            _wrist = env.env._wrist_contact_force_per_env()[s3_mask]
            _grip = env.env.robot.data.joint_pos[s3_mask, env.env.gripper_idx]
            _has_ct = (_jaw > 0.3) & (_wrist > 0.3)
            _grip_closed = _grip < float(env.env.cfg.grasp_gripper_threshold)
            s3_holding = (_has_ct & _grip_closed & (s3_oh > 0.033)).sum().item()
            s3_placed = ms_place[s3_mask].sum().item()
            s3_dropped = s3_mask.sum().item() - s3_holding - s3_placed
            s3_src_dst = torch.norm(
                env.env.object_pos_w[s3_mask, :2] - env.env.dest_object_pos_w[s3_mask, :2], dim=-1)
            s3_base_dst = torch.norm(
                env.env.robot.data.root_pos_w[s3_mask, :2] - env.env.dest_object_pos_w[s3_mask, :2], dim=-1)
            print(f"  S2→S3: {s2_success_total} total | S3 envs: {s3_envs} (hold={s3_holding}, placed={s3_placed}, drop={s3_dropped}) | place_total={s3_place_total} (REAL={s3_place_real_total} SUSPECT={s3_place_suspect_total}) drop_total={s3_drop_total}")
            print(f"  S3 objZ: min={s3_oh.min():.3f} mean={s3_oh.mean():.3f} max={s3_oh.max():.3f}")
            print(f"  S3 src→dst: min={s3_src_dst.min():.3f} mean={s3_src_dst.mean():.3f} | base→dst: min={s3_base_dst.min():.3f} mean={s3_base_dst.mean():.3f}")
            # Phase B detailed stats
            _phb_mask = s3_mask & (~s3_phase_a_latch)
            if _phb_mask.any():
                _phb_arm1 = env.env.robot.data.joint_pos[_phb_mask][:, env.env.arm_idx[1]]
                _phb_grip = env.env.robot.data.joint_pos[_phb_mask, env.env.gripper_idx]
                _phb_src_dst = torch.norm(env.env.object_pos_w[_phb_mask, :2] - env.env.dest_object_pos_w[_phb_mask, :2], dim=-1)
                _a1_lo = (_phb_arm1 < 0.5).sum().item()
                _a1_mid = ((_phb_arm1 >= 0.5) & (_phb_arm1 < 2.0)).sum().item()
                _a1_hi = (_phb_arm1 >= 2.0).sum().item()
                _g_closed = (_phb_grip < 0.30).sum().item()
                _g_open = (_phb_grip >= 0.40).sum().item()
                print(f"  PhB: n={_phb_mask.sum().item()} arm1(<0.5/{0.5}~2/>2)={_a1_lo}/{_a1_mid}/{_a1_hi} "
                      f"grip(<0.3/>0.4)={_g_closed}/{_g_open} src_dst<0.14={(_phb_src_dst < 0.14).sum().item()}")
        else:
            print(f"  S2→S3: {s2_success_total} total | S3 envs: 0 | place={s3_place_total}")
        print(f"  Total episodes: {total_episodes} | S3 steps: {s3_steps}")

        # Save
        if gi % 10 == 0:
            torch.save({
                "residual_policy_state_dict": rpol.state_dict(),
                "optimizer_actor_state_dict": opt_a.state_dict(),
                "optimizer_critic_state_dict": opt_c.state_dict(),
                "s2_resip_checkpoint": args.s2_resip_checkpoint,
                "s3_bc_checkpoint": args.s3_bc_checkpoint,
                "iteration": gi, "global_step": gs, "args": vars(args),
            }, save_dir / f"resip_iter{gi}.pt")

    print(f"\nDone in {time.time()-t0:.0f}s")
    torch.save({
        "residual_policy_state_dict": rpol.state_dict(),
        "s2_resip_checkpoint": args.s2_resip_checkpoint,
        "s3_bc_checkpoint": args.s3_bc_checkpoint,
        "iteration": gi, "global_step": gs, "args": vars(args),
    }, save_dir / "resip_final.pt")
    env.env.close(); simulation_app.close()


def main_carry():
    """Carry-phase ResiP: S2 expert (frozen) lifts → direction-conditioned BC+Residual for base carry."""
    seed = args.seed or random.randint(0, 2**32 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"Seed: {seed}")

    env = make_env("carry", args.num_envs, args)
    dev = env.device
    N = env.num_envs

    from lekiwi_skill2_eval import EE_LOCAL_OFFSET
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    jaw_idx, _ = env.env.robot.find_bodies(["Wrist_Roll_08c_v1"])
    jaw_idx = jaw_idx[0]
    ee_off = torch.tensor(EE_LOCAL_OFFSET, device=dev).unsqueeze(0)

    # ── Load S2 expert (frozen DP + ResiP) ──
    s2_dp, s2_dpc = load_frozen_dp(args.bc_checkpoint, dev)
    S2_OD, S2_AD = s2_dpc["obs_dim"], s2_dpc["act_dim"]
    s2_rpol = None
    if args.s2_resip_checkpoint and os.path.isfile(args.s2_resip_checkpoint):
        s2_ck = torch.load(args.s2_resip_checkpoint, map_location=dev, weights_only=False)
        s2_rpol = ResidualPolicy(
            obs_dim=S2_OD, action_dim=S2_AD,
            actor_hidden_size=args.actor_hidden_size,
            actor_num_layers=args.actor_num_layers,
            init_logstd=args.init_logstd, action_head_std=args.action_head_std,
            action_scale=0.1, learn_std=True,
        ).to(dev)
        s2_rpol.load_state_dict(s2_ck["residual_policy_state_dict"])
        s2_rpol.eval()
        for p in s2_rpol.parameters():
            p.requires_grad = False
        print(f"  [S2] ResiP loaded: {args.s2_resip_checkpoint}")
    s2_scale = torch.zeros(S2_AD, device=dev)
    s2_scale[0:5] = 0.20; s2_scale[5] = 0.25; s2_scale[6:9] = 0.35

    # ── Load carry BC (frozen, 39D obs) ──
    s3_dp, s3_dpc = load_frozen_dp(args.s3_bc_checkpoint, dev)
    S3_OD = s3_dpc["obs_dim"]  # expected 39 (30D + dir_cmd 3D + init_arm_pose 6D)
    S3_AD = s3_dpc["act_dim"]  # 9
    print(f"  [Carry] BC loaded: {args.s3_bc_checkpoint} (obs={S3_OD}D, act={S3_AD}D)")

    # ── Carry residual policy (trainable): obs=33D → internal 33+9=42D input ──
    rpol = ResidualPolicy(
        obs_dim=S3_OD, action_dim=S3_AD,
        actor_hidden_size=args.actor_hidden_size,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_size=args.critic_hidden_size,
        critic_num_layers=args.critic_num_layers,
        init_logstd=args.init_logstd, action_head_std=args.action_head_std,
        action_scale=0.1, learn_std=True,
        critic_last_layer_bias_const=0.25, critic_last_layer_std=0.25,
    ).to(dev)
    print(f"  [Carry] Residual params: {sum(p.numel() for p in rpol.parameters()):,}")

    # Scale: arm=0.02 (small correction), base=action_scale_base
    carry_scale = torch.zeros(S3_AD, device=dev)
    carry_scale[0:5] = 0.05   # arm: RL이 pose drift 보정
    carry_scale[5] = 0.05      # gripper
    carry_scale[6:9] = 0.0     # base: BC 그대로 (navigate v5와 동일)

    opt_a = optim.AdamW([p for n, p in rpol.named_parameters() if "critic" not in n],
                        lr=args.lr_actor, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-6)
    opt_c = optim.AdamW([p for n, p in rpol.named_parameters() if "critic" in n],
                        lr=args.lr_critic, eps=1e-5, weight_decay=1e-6)

    # Resume
    gs, gi = 0, 0
    if args.resume_resip:
        ck = torch.load(args.resume_resip, map_location=dev, weights_only=False)
        if args.resume_actor_only:
            actor_sd = {k: v for k, v in ck["residual_policy_state_dict"].items() if "critic" not in k}
            rpol.load_state_dict(actor_sd, strict=False)
            gs = ck.get("global_step", 0); gi = ck.get("iteration", 0)
            print(f"  [Carry] Resumed ACTOR ONLY (iter={gi}, step={gs})")
        else:
            rpol.load_state_dict(ck["residual_policy_state_dict"])
            if "optimizer_actor_state_dict" in ck: opt_a.load_state_dict(ck["optimizer_actor_state_dict"])
            if "optimizer_critic_state_dict" in ck: opt_c.load_state_dict(ck["optimizer_critic_state_dict"])
            gs = ck.get("global_step", 0); gi = ck.get("iteration", 0)
            print(f"  [Carry] Resumed: iter={gi}, step={gs}")

    S = args.num_env_steps
    B = S * N
    MB = B // args.num_minibatches
    NI = args.total_timesteps // B
    rew_norm = RunningMeanStdClip(shape=(1,), clip_value=args.clip_reward, device=dev) \
               if args.normalize_reward else None

    sch_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=NI, eta_min=args.lr_actor * 0.01)
    sch_c = optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=NI, eta_min=args.lr_critic * 0.01)

    # Rollout buffers (carry phase only)
    RD = S3_OD + S3_AD  # 33 + 9 = 42 (normalized obs + base action)
    obs_b  = torch.zeros((S, N, RD), device=dev)
    act_b  = torch.zeros((S, N, S3_AD), device=dev)
    lp_b   = torch.zeros((S, N), device=dev)
    rew_b  = torch.zeros((S, N), device=dev)
    done_b = torch.zeros((S, N), device=dev)
    val_b  = torch.zeros((S, N), device=dev)

    # Per-env phase: 0=S2 (expert), 1=carry (trainable)
    phase = torch.zeros(N, dtype=torch.long, device=dev)
    s2_lift_counter = torch.zeros(N, dtype=torch.long, device=dev)
    ep_step_counter = torch.zeros(N, dtype=torch.long, device=dev)
    S2_LIFT_HOLD = args.s2_lift_hold_steps
    S2_MAX_STEPS = 2000
    S2_NOLIFT_STEP = 700

    # 6 direction commands (body frame): fwd/bwd/left/right/turn_L/turn_R
    DIR_CMDS = torch.tensor([
        [0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=torch.float32, device=dev)
    direction_cmd = DIR_CMDS[torch.randint(6, (N,), device=dev)]

    # Arm interpolation: captured arm pose → S3_ARM_END over CARRY_INTERP_STEPS
    S3_ARM_END = torch.tensor([+0.002, -0.193, +0.295, -1.306, +0.006], device=dev)
    # Kiwi IK for wheel-level reward
    import math as _math
    _angles = [a * _math.pi / 180.0 for a in [-30.0, -150.0, 90.0]]  # FL, FR, Back
    KIWI_M_T = torch.tensor([
        [_math.cos(_angles[0]), _math.sin(_angles[0]), 0.1085],
        [_math.cos(_angles[1]), _math.sin(_angles[1]), 0.1085],
        [_math.cos(_angles[2]), _math.sin(_angles[2]), 0.1085],
    ], dtype=torch.float32, device=dev)
    WHEEL_R = 0.049
    S3_GRIP_END = 0.15
    CARRY_INTERP_STEPS = 600
    carry_arm_start = torch.zeros(N, 5, device=dev)  # captured at S2→carry transition
    carry_grip_start = torch.zeros(N, device=dev)
    carry_step_counter = torch.zeros(N, dtype=torch.long, device=dev)
    # init_arm_pose: S2→carry 전환 시점의 arm pose (39D obs용, 6D)
    carry_init_arm_pose = torch.zeros(N, 6, device=dev)

    prev_action = torch.zeros(N, S3_AD, device=dev)  # for smoothness reward

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    bsr, bsl, bgr = 0.0, 0, 0
    tt = 0; t0 = time.time()
    s2_success_total = 0; carry_drop_total = 0; total_episodes = 0
    # Per-iter reward accumulators
    iter_rew_lin = 0.0; iter_rew_ang = 0.0; iter_rew_hold = 0.0
    iter_rew_total = 0.0; iter_carry_steps = 0
    iter_lin_err = 0.0; iter_ang_err = 0.0
    # PPO loss accumulators
    iter_pg_loss = 0.0; iter_vf_loss = 0.0; iter_entropy = 0.0; iter_kl = 0.0
    iter_ppo_updates = 0
    # Per-direction tracking (6 directions)
    DIR_LABELS = ["FWD", "BWD", "LEFT", "RIGHT", "TL", "TR"]
    dir_ang_err = torch.zeros(6, device=dev)
    dir_lin_err = torch.zeros(6, device=dev)
    dir_steps = torch.zeros(6, device=dev)

    next_obs = env.reset(); s2_dp.reset(); s3_dp.reset()
    next_done = torch.zeros(N, device=dev)
    phase.zero_(); s2_lift_counter.zero_()
    env_origins = env.env.scene.env_origins

    print(f"\n{'='*60}")
    print(f"  ResiP Carry (direction-conditioned)")
    print(f"  N={N} S={S} B={B} iters={NI}")
    print(f"  S2 expert: {args.s2_resip_checkpoint or 'BC only'}")
    print(f"  Carry BC: {args.s3_bc_checkpoint} ({S3_OD}D obs)")
    print(f"  S2 lift hold: {S2_LIFT_HOLD} steps")
    print(f"  Carry residual scale: {carry_scale.tolist()}")
    print(f"{'='*60}\n")

    # ═════════════════════════════════════════════════════════════════════════
    while gs < args.total_timesteps:
        gi += 1; it0 = time.time()
        ev = (gi - int(args.eval_first)) % args.eval_interval == 0

        if gi == 1:
            next_obs = env.reset(); s2_dp.reset(); s3_dp.reset()
            next_done = torch.zeros(N, device=dev)
            phase.zero_(); s2_lift_counter.zero_(); ep_step_counter.zero_()
            direction_cmd = DIR_CMDS[torch.randint(6, (N,), device=dev)]

        print(f"\nIter {gi}/{NI} | {'EVAL' if ev else 'TRAIN'} | step={gs} | "
              f"S2:{(phase==0).sum().item()} Carry:{(phase==1).sum().item()}")

        carry_step_count = torch.zeros(N, dtype=torch.long, device=dev)
        carry_valid = torch.zeros((S, N), dtype=torch.bool, device=dev)
        for step in range(S):
            if not ev: gs += N

            actor_obs = next_obs  # (N, 30D) from Skill2Env

            with torch.no_grad():
                # ── S2 expert action ──
                s2_ba = s2_dp.base_action_normalized(actor_obs)
                if s2_rpol is not None:
                    s2_no = torch.nan_to_num(torch.clamp(
                        s2_dp.normalizer(actor_obs, "obs", forward=True), -3, 3), nan=0.0)
                    s2_ro = torch.cat([s2_no, s2_ba], dim=-1)
                    s2_ra, _, _, _, s2_ram = s2_rpol.get_action_and_value(s2_ro)
                    s2_ra = s2_ram  # deterministic
                    s2_action = s2_dp.normalizer(s2_ba + torch.clamp(s2_ra, -1, 1) * s2_scale, "action", forward=False)
                else:
                    s2_action = s2_dp.normalizer(s2_ba, "action", forward=False)

                # ── Carry obs: env 30D + direction_cmd 3D + init_arm_pose 6D → 39D ──
                carry_obs = torch.cat([actor_obs, direction_cmd, carry_init_arm_pose], dim=-1)  # (N, 39)
                # Lookup table base action + velocity ramp (50step 가속)
                vel_ramp = (carry_step_counter.float() / 50.0).clamp(max=1.0)
                carry_ba = torch.zeros(N, S3_AD, device=dev)
                carry_ba[:, 6] = direction_cmd[:, 0] * (0.15 / env.env.cfg.max_lin_vel) * vel_ramp
                carry_ba[:, 7] = direction_cmd[:, 1] * (0.15 / env.env.cfg.max_lin_vel) * vel_ramp
                carry_ba[:, 8] = direction_cmd[:, 2] * (1.0 / env.env.cfg.max_ang_vel) * vel_ramp
                carry_no = torch.nan_to_num(torch.clamp(
                    s3_dp.normalizer(carry_obs, "obs", forward=True), -3, 3), nan=0.0)
                carry_ro = torch.cat([carry_no, carry_ba], dim=-1)  # (N, 42)

            # Carry residual (trainable)
            with torch.no_grad():
                carry_ra_s, _, _, carry_val, carry_ra_m = rpol.get_action_and_value(carry_ro)
            carry_ra = carry_ra_m if ev else carry_ra_s
            carry_ra = torch.clamp(carry_ra, -1.0, 1.0)
            CARRY_BC_WARMUP_ITERS = 5
            residual_alpha = min(1.0, gi / CARRY_BC_WARMUP_ITERS)
            carry_ra = carry_ra * residual_alpha
            with torch.no_grad():
                _, carry_lp, _, _, _ = rpol.get_action_and_value(carry_ro, carry_ra)
            combined = carry_ba + carry_ra * carry_scale
            # 회전 명령(TL/TR)일 때 residual 비활성화 — BC만 사용
            is_turn = (direction_cmd[:, 2].abs() > 0.5)  # TL=[0,0,1], TR=[0,0,-1]
            if is_turn.any():
                combined[is_turn] = carry_ba[is_turn]  # residual 제거, BC만
            carry_action_raw = combined.clone()  # 이미 env action space [-1, 1]

            # ── Arm interpolation: override action[0:6] for carry envs ──
            is_carry = (phase == 1)
            if is_carry.any():
                t_interp = (carry_step_counter[is_carry].float() / CARRY_INTERP_STEPS).clamp(max=1.0)
                arm_target = carry_arm_start[is_carry] * (1 - t_interp.unsqueeze(-1)) + S3_ARM_END * t_interp.unsqueeze(-1)
                grip_target = carry_grip_start[is_carry] * (1 - t_interp) + S3_GRIP_END * t_interp
                # Convert joint targets to action space: action = (target - center) / half_range
                override = getattr(env.env, "_arm_action_limits_override", None)
                if override is not None:
                    lim = override
                else:
                    lim = env.env.robot.data.soft_joint_pos_limits[:, env.env.arm_idx]
                lo, hi = lim[..., 0], lim[..., 1]
                center = 0.5 * (lo + hi)
                half = 0.5 * (hi - lo)
                finite = torch.isfinite(center) & torch.isfinite(half) & (half.abs() > 1e-6)
                half = torch.where(finite, half, torch.ones_like(half))
                center = torch.where(finite, center, torch.zeros_like(center))
                # arm_idx covers 6 joints (arm[0:5] + gripper[5])
                arm6_target = torch.cat([arm_target, grip_target.unsqueeze(-1)], dim=-1)  # (M, 6)
                # Use first env's limits (shared across envs)
                arm_action_norm = ((arm6_target - center[0]) / half[0]).clamp(-1, 1)
                carry_action_raw[is_carry, 0:5] = arm_action_norm[:, :5]
                carry_action_raw[is_carry, 5] = arm_action_norm[:, 5]

            # Merge action by phase
            is_s2 = (phase == 0)
            action = torch.where(is_s2.unsqueeze(-1), s2_action, carry_action_raw)

            # Store carry transitions
            obs_b[step] = carry_ro
            act_b[step] = carry_ra
            lp_b[step] = carry_lp.view(-1)
            val_b[step] = carry_val.view(-1)
            done_b[step] = next_done
            done_b[step][is_s2] = 1.0  # S2 steps don't contribute to GAE
            _is_turn = (direction_cmd[:, 2].abs() > 0.5)
            carry_valid[step] = is_carry & (~_is_turn)  # Turn env 제외, linear만 PPO

            # Step env
            next_obs, _, ter, tru, info = env.step(action)
            next_obs = torch.nan_to_num(next_obs, nan=0.0)
            done = ter | tru
            ep_step_counter += 1
            carry_step_counter[is_carry] += 1

            # ── S2 phase: lift detection + early termination ──
            oh = info.get("object_height_mask",
                (env.env.object_pos_w[:, 2] - env_origins[:, 2])).view(-1)
            eg = info.get("object_grasped_mask", env.env.object_grasped).view(-1)

            s2_topple = is_s2 & (oh < 0.026)
            s2_nolift = is_s2 & (ep_step_counter == S2_NOLIFT_STEP) & (oh < 0.04)
            s2_timeout = is_s2 & (ep_step_counter >= S2_MAX_STEPS)
            s2_fail = s2_topple | s2_nolift | s2_timeout
            if s2_fail.any():
                fail_ids = s2_fail.nonzero(as_tuple=False).squeeze(-1)
                env.env._reset_idx(fail_ids)
                next_obs_new = env.env._get_observations()
                policy_obs = next_obs_new["policy"]
                next_obs[fail_ids] = policy_obs[fail_ids]
                phase[fail_ids] = 0
                s2_lift_counter[fail_ids] = 0
                ep_step_counter[fail_ids] = 0
                direction_cmd[fail_ids] = DIR_CMDS[torch.randint(6, (len(fail_ids),), device=dev)]
                done[fail_ids.unsqueeze(-1) if fail_ids.dim() == 1 else fail_ids] = True

            # S2 lift detection
            lifted = is_s2 & eg & (oh > 0.05) & (~s2_fail)
            s2_lift_counter[lifted] += 1
            s2_lift_counter[is_s2 & ~lifted & ~s2_fail] = 0

            # Transition S2→Carry
            transition = is_s2 & (s2_lift_counter >= S2_LIFT_HOLD) & (~s2_fail)
            if transition.any():
                t_ids = transition.nonzero(as_tuple=False).squeeze(-1)
                # Capture arm pose at transition
                carry_arm_start[t_ids] = env.env.robot.data.joint_pos[t_ids][:, env.env.arm_idx][:, :5]
                carry_grip_start[t_ids] = env.env.robot.data.joint_pos[t_ids, env.env.gripper_idx]
                # init_arm_pose for 39D obs
                carry_init_arm_pose[t_ids, :5] = carry_arm_start[t_ids]
                carry_init_arm_pose[t_ids, 5] = carry_grip_start[t_ids]
                phase[t_ids] = 1
                ep_step_counter[t_ids] = 0
                carry_step_counter[t_ids] = 0
                # Sample fresh direction cmd for carry phase
                direction_cmd[t_ids] = DIR_CMDS[torch.randint(6, (len(t_ids),), device=dev)]
                s2_success_total += t_ids.shape[0]
                s3_dp.reset()
                print(f"    [S2→Carry] {t_ids.shape[0]} envs at step {step}")

            # ── Carry reward (body-velocity tracking — navigate 동일 구조) ──
            rew = torch.zeros(N, device=dev)
            carry_drop = torch.zeros(N, dtype=torch.bool, device=dev)
            if is_carry.any():
                cm = is_carry
                body_vel = env.env.robot.data.root_lin_vel_b   # (N, 3)
                body_wz = env.env.robot.data.root_ang_vel_b[:, 2]  # (N,)

                # Velocity ramp (action ramp와 동일)
                _vel_ramp = (carry_step_counter.float() / 50.0).clamp(max=1.0)

                # Body velocity targets (ramp 적용)
                target_vx = direction_cmd[:, 0] * 0.15 * _vel_ramp
                target_vy = direction_cmd[:, 1] * 0.15 * _vel_ramp
                target_wz = direction_cmd[:, 2] * 1.0 * _vel_ramp  # 0.33 × max_ang_vel = 1.0

                lin_err = (body_vel[:, 0] - target_vx)**2 + (body_vel[:, 1] - target_vy)**2
                ang_err = (body_wz - target_wz)**2

                # Gaussian kernel (navigate와 동일, σ² 분모)
                CARRY_LIN_STD = 0.075
                CARRY_ANG_STD = 0.10
                rew_lin = 1.5 * torch.exp(-lin_err / (CARRY_LIN_STD ** 2))
                rew_ang = 1.5 * torch.exp(-ang_err / (CARRY_ANG_STD ** 2))

                # Smoothness (base + arm)
                delta_base = combined[:, 6:9] - prev_action[:, 6:9]
                delta_arm = combined[:, 0:6] - prev_action[:, 0:6]
                rew_smooth = -0.005 * (delta_base ** 2).sum(dim=-1) - 0.01 * (delta_arm ** 2).sum(dim=-1)

                # Hold bonus
                objZ = env.env.object_pos_w[:, 2] - env_origins[:, 2]
                rew_hold = 0.05 * (objZ > 0.05).float()

                # Arm pose maintenance reward (init_arm_pose 유지)
                jp = env.env.robot.data.joint_pos
                cur_arm = jp[:, env.env.arm_idx[:5]]
                cur_grip = jp[:, env.env.arm_idx[5:6]]
                arm_err = (cur_arm - carry_init_arm_pose[:, :5]).pow(2).sum(dim=-1)
                grip_err = (cur_grip.squeeze(-1) - carry_init_arm_pose[:, 5]).pow(2)
                rew_arm_hold = 2.0 * torch.exp(-arm_err / (0.3 ** 2)) + 0.5 * torch.exp(-grip_err / (0.2 ** 2))

                # Time penalty
                rew_time = -0.01

                # Drop detection
                dropped = objZ < 0.03
                rew_drop = -5.0 * dropped.float()

                rew[cm] = (rew_lin + rew_ang + rew_smooth + rew_hold + rew_arm_hold + rew_time + rew_drop)[cm]

                # Accumulate for logging
                n_carry = cm.sum().item()
                if n_carry > 0:
                    iter_rew_lin += rew_lin[cm].sum().item()
                    iter_rew_ang += rew_ang[cm].sum().item()
                    iter_rew_hold += rew_hold[cm].sum().item()
                    iter_rew_total += rew[cm].sum().item()
                    iter_carry_steps += n_carry
                    iter_lin_err += lin_err[cm].sum().item()
                    iter_ang_err += ang_err[cm].sum().item()
                    # Per-direction tracking
                    for di in range(6):
                        d_mask = cm & (direction_cmd == DIR_CMDS[di]).all(dim=-1)
                        if d_mask.any():
                            dir_ang_err[di] += ang_err[d_mask].sum().item()
                            dir_lin_err[di] += lin_err[d_mask].sum().item()
                            dir_steps[di] += d_mask.sum().item()

                carry_drop = cm & dropped
                if carry_drop.any():
                    carry_drop_total += carry_drop.sum().item()

            rew_b[step] = rew
            prev_action = combined.detach().clone()

            # ── Carry fail (drop) → force reset to S2 ──
            carry_timeout = is_carry & (ep_step_counter >= 3000)
            carry_fail = carry_drop | carry_timeout
            if carry_fail.any():
                fail_ids = carry_fail.nonzero(as_tuple=False).squeeze(-1)
                env.env._reset_idx(fail_ids)
                next_obs_new = env.env._get_observations()
                policy_obs = next_obs_new["policy"]
                next_obs[fail_ids] = policy_obs[fail_ids]
                phase[fail_ids] = 0
                s2_lift_counter[fail_ids] = 0
                ep_step_counter[fail_ids] = 0
                carry_step_counter[fail_ids] = 0
                direction_cmd[fail_ids] = DIR_CMDS[torch.randint(6, (len(fail_ids),), device=dev)]

            # Handle env auto-resets
            next_done = (ter | tru).view(-1).float()
            next_done[s2_fail] = 1.0
            next_done[carry_fail] = 1.0
            reset_mask = (done.view(-1) | s2_fail | carry_fail)
            if reset_mask.any():
                phase[reset_mask] = 0
                s2_lift_counter[reset_mask] = 0
                ep_step_counter[reset_mask] = 0
                carry_step_counter[reset_mask] = 0
                direction_cmd[reset_mask] = DIR_CMDS[torch.randint(6, (reset_mask.sum().item(),), device=dev)]
                total_episodes += reset_mask.sum().item()

            carry_step_count[is_carry] += 1

        # ── PPO Update (carry transitions only, navigate-identical structure) ──
        if not ev:
            with torch.no_grad():
                carry_obs_f = torch.cat([next_obs, direction_cmd, carry_init_arm_pose], dim=-1)
                carry_no_f = torch.nan_to_num(torch.clamp(
                    s3_dp.normalizer(carry_obs_f, "obs", forward=True), -3, 3), nan=0.0)
                vel_ramp_f = (carry_step_counter.float() / 50.0).clamp(max=1.0)
                carry_ba_f = torch.zeros(N, S3_AD, device=dev)
                carry_ba_f[:, 6] = direction_cmd[:, 0] * (0.15 / env.env.cfg.max_lin_vel) * vel_ramp_f
                carry_ba_f[:, 7] = direction_cmd[:, 1] * (0.15 / env.env.cfg.max_lin_vel) * vel_ramp_f
                carry_ba_f[:, 8] = direction_cmd[:, 2] * (1.0 / env.env.cfg.max_ang_vel) * vel_ramp_f
                carry_ro_f = torch.cat([carry_no_f, carry_ba_f], dim=-1)
                nv = rpol.get_value(carry_ro_f).view(-1)

            adv, ret = compute_gae(val_b, nv, rew_b, done_b, next_done,
                                   S, args.discount, args.gae_lambda)

            # Flatten all buffers
            f = lambda t, *s: t.reshape(-1, *s) if s else t.reshape(-1)
            bo_all, ba_all, blp_all = f(obs_b, RD), f(act_b, S3_AD), f(lp_b)
            bv_all, badv_all, bret_all = f(val_b), f(adv), f(ret)
            valid_mask = carry_valid.reshape(-1)  # (S*N,) bool

            # Filter to carry-only transitions
            valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            BV = len(valid_idx)

            if BV > 0:
                bo   = bo_all[valid_idx]
                ba_  = ba_all[valid_idx]
                blp  = blp_all[valid_idx]
                bv   = bv_all[valid_idx]
                badv = badv_all[valid_idx]
                bret = bret_all[valid_idx]

                MBV = max(BV // max(args.num_minibatches, 1), 1)
                idx = np.arange(BV); cfs = []

                for ep in range(args.update_epochs):
                    stop = False; np.random.shuffle(idx)
                    for i0 in range(0, BV, MBV):
                        mi = idx[i0:i0 + MBV]
                        _, nlp, ent, nv2, am = rpol.get_action_and_value(bo[mi], ba_[mi])
                        lr = nlp - blp[mi]; ratio = lr.exp()

                        with torch.no_grad():
                            kl = ((ratio - 1) - lr).mean()
                            cfs.append(((ratio - 1).abs() > args.clip_coef).float().mean().item())

                        ma = badv[mi]
                        if args.norm_adv:
                            ma = (ma - ma.mean()) / (ma.std() + 1e-8)

                        pg = torch.max(-ma * ratio,
                                       -ma * ratio.clamp(1 - args.clip_coef,
                                                          1 + args.clip_coef)).mean()
                        vl = 0.5 * ((nv2.view(-1) - bret[mi]) ** 2).mean()
                        el = ent.mean() * args.ent_coef

                        loss = (pg - el
                                + args.residual_l1 * am.abs().mean()
                                + args.residual_l2 * (am ** 2).mean()
                                + vl * args.vf_coef)

                        opt_a.zero_grad(); opt_c.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(rpol.parameters(), args.max_grad_norm)
                        opt_a.step(); opt_c.step()

                        iter_pg_loss += pg.item()
                        iter_vf_loss += vl.item()
                        iter_entropy += ent.mean().item()
                        iter_kl += kl.item()
                        iter_ppo_updates += 1

                        if args.target_kl and kl > args.target_kl:
                            print(f"    KL stop ep{ep}: {kl:.4f}>{args.target_kl}")
                            stop = True; break
                    if stop: break

                print(f"  PPO: {BV}/{B} valid carry steps ({100*BV/B:.0f}%), clip={np.mean(cfs):.3f}")

            sch_a.step(); sch_c.step()

        # ── Comprehensive Logging ──
        carry_envs = (phase == 1).sum().item()
        carry_steps_now = carry_step_count.sum().item()
        carry_mask = (phase == 1)

        # Velocity tracking (현재 snapshot)
        body_vel_snap = env.env.robot.data.root_lin_vel_b  # (N, 3)
        body_wz_snap = env.env.robot.data.root_ang_vel_b[:, 2]
        if carry_mask.any():
            c_vx = body_vel_snap[carry_mask, 0].mean().item()
            c_vy = body_vel_snap[carry_mask, 1].mean().item()
            c_wz = body_wz_snap[carry_mask].mean().item()
            c_oh = (env.env.object_pos_w[carry_mask, 2] - env_origins[carry_mask, 2])
            c_held = (c_oh > 0.04).sum().item()
            c_grip = env.env.robot.data.joint_pos[carry_mask, env.env.gripper_idx].mean().item()
        else:
            c_vx = c_vy = c_wz = c_grip = 0.0
            c_held = 0

        # Per-iter averages
        n_cs = max(iter_carry_steps, 1)
        avg_rl = iter_rew_lin / n_cs
        avg_ra = iter_rew_ang / n_cs
        avg_rh = iter_rew_hold / n_cs
        avg_rt = iter_rew_total / n_cs
        avg_le = (iter_lin_err / n_cs) ** 0.5  # RMSE
        avg_ae = (iter_ang_err / n_cs) ** 0.5

        # PPO averages
        n_ppo = max(iter_ppo_updates, 1)
        avg_pg = iter_pg_loss / n_ppo
        avg_vf = iter_vf_loss / n_ppo
        avg_ent = iter_entropy / n_ppo
        avg_kl_val = iter_kl / n_ppo

        alpha = min(1.0, gi / 5.0)

        print(f"\n{'='*80}")
        print(f"  [CARRY ITER {gi}] alpha={alpha:.2f} lr={sch_a.get_last_lr()[0]:.2e}")
        print(f"  Reward: total={avg_rt:.3f} | lin={avg_rl:.3f} ang={avg_ra:.3f} hold={avg_rh:.3f}")
        print(f"  Tracking: lin_rmse={avg_le:.3f} ang_rmse={avg_ae:.3f}")
        print(f"  Velocity: vx={c_vx:+.3f} vy={c_vy:+.3f} wz={c_wz:+.3f} grip={c_grip:.3f}")
        print(f"  PPO: pg_loss={avg_pg:.4f} vf_loss={avg_vf:.4f} entropy={avg_ent:.4f} kl={avg_kl_val:.4f}")
        print(f"  S2→Carry: {s2_success_total} total | Carry envs: {carry_envs} (held={c_held}) | drop={carry_drop_total}")
        print(f"  Carry steps: {iter_carry_steps} | Episodes: {total_episodes}")
        # Per-direction breakdown
        dir_parts = []
        for di in range(6):
            ds = max(dir_steps[di].item(), 1)
            da = (dir_ang_err[di].item() / ds) ** 0.5
            dl = (dir_lin_err[di].item() / ds) ** 0.5
            dir_parts.append(f"{DIR_LABELS[di]}:a={da:.3f},l={dl:.3f}")
        print(f"  PerDir: {' | '.join(dir_parts)}")
        print(f"  Time: {time.time()-it0:.1f}s iter, {time.time()-t0:.0f}s total")
        print(f"{'='*80}")

        # Reset per-iter accumulators
        iter_rew_lin = 0.0; iter_rew_ang = 0.0; iter_rew_hold = 0.0
        iter_rew_total = 0.0; iter_carry_steps = 0
        iter_lin_err = 0.0; iter_ang_err = 0.0
        iter_pg_loss = 0.0; iter_vf_loss = 0.0; iter_entropy = 0.0
        iter_kl = 0.0; iter_ppo_updates = 0
        dir_ang_err.zero_(); dir_lin_err.zero_(); dir_steps.zero_()

        # Save
        if gi % 10 == 0:
            torch.save({
                "residual_policy_state_dict": rpol.state_dict(),
                "optimizer_actor_state_dict": opt_a.state_dict(),
                "optimizer_critic_state_dict": opt_c.state_dict(),
                "s2_resip_checkpoint": args.s2_resip_checkpoint,
                "s3_bc_checkpoint": args.s3_bc_checkpoint,
                "iteration": gi, "global_step": gs, "args": vars(args),
            }, save_dir / f"resip_carry_iter{gi}.pt")

    print(f"\nDone in {time.time()-t0:.0f}s")
    torch.save({
        "residual_policy_state_dict": rpol.state_dict(),
        "s2_resip_checkpoint": args.s2_resip_checkpoint,
        "s3_bc_checkpoint": args.s3_bc_checkpoint,
        "iteration": gi, "global_step": gs, "args": vars(args),
    }, save_dir / "resip_carry_final.pt")
    env.env.close(); simulation_app.close()


def main_navigate():
    """Navigate ResiP on Skill2EvalEnv (cuboid ground, friction=0.5)."""
    seed = args.seed or random.randint(0, 2**32 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"Seed: {seed}")

    # ── Env (Skill2EvalEnv) ──
    from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.episode_length_s = float(args.num_env_steps) * cfg.sim.dt * cfg.decimation + 1.0
    cfg.max_dist_from_origin = 50.0
    cfg.dr_action_delay_steps = 0
    cfg.grasp_success_height = 100.0  # task_success 비활성화 (navigate에 물체 없음)
    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    from isaaclab.envs import DirectRLEnvCfg
    env = Skill2Env(cfg=cfg)
    dev = env.device
    N = env.num_envs

    # env 기본 _apply_action 사용 (arm_action_to_limits 매핑 포함)
    # navigate에서도 approach_lift와 동일한 [-1,1] action space
    # Tucked pose in normalized [-1,1] space (reward 계산용)
    _TUCKED_ARM_T = torch.tensor([-0.02966, -0.213839, 0.09066, -0.4, 0.058418], device=dev)
    _TUCKED_GRIP_V = -0.201554

    # terminate 비활성화 (task_success 오판 방지, timeout만 유지)
    _orig_dones = env._get_dones
    def _nav_dones():
        t, tr = _orig_dones()
        t[:] = False  # out_of_bounds, fell, topple 비활성화
        tr[:] = env.episode_length_buf >= (env.max_episode_length - 1)  # timeout만
        return t, tr
    env._get_dones = _nav_dones

    # direction_cmd 버퍼
    direction_cmd = torch.zeros(N, 3, device=dev)

    # 4 linear directions only (turn 제외 — BC로 충분)
    DIR_CMDS = torch.tensor([
        [0.0, 1.0, 0.0],    # forward
        [0.0, -1.0, 0.0],   # backward
        [-1.0, 0.0, 0.0],   # strafe left
        [1.0, 0.0, 0.0],    # strafe right
    ], device=dev)

    # ── Policy ──
    from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
    dp_ckpt = torch.load(args.bc_checkpoint, map_location=dev, weights_only=False)
    dp_cfg = dp_ckpt["config"]
    OD = dp_cfg["obs_dim"]  # 20
    AD = dp_cfg["act_dim"]  # 9

    dp_agent = DiffusionPolicyAgent(
        obs_dim=OD, act_dim=AD,
        pred_horizon=dp_cfg["pred_horizon"],
        action_horizon=dp_cfg["action_horizon"],
        num_diffusion_iters=dp_cfg["num_diffusion_iters"],
        inference_steps=4,
        down_dims=dp_cfg.get("down_dims", [256, 512, 1024]),
    ).to(dev)
    sd = dp_ckpt["model_state_dict"]
    dp_agent.model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
    dp_agent.normalizer.load_state_dict(
        {k[len("normalizer."):]: v for k, v in sd.items() if k.startswith("normalizer.")}, device=dev)
    dp_agent.eval()
    for p in dp_agent.parameters():
        p.requires_grad = False

    residual = ResidualPolicy(
        obs_dim=OD, action_dim=AD,
        actor_hidden_size=args.actor_hidden_size,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_size=args.critic_hidden_size,
        critic_num_layers=args.critic_num_layers,
        action_scale=0.1,
        init_logstd=args.init_logstd,
        action_head_std=args.action_head_std,
        learn_std=True,
    ).to(dev)

    per_dim = torch.zeros(AD, device=dev)
    per_dim[0:5] = 0.05   # arm: RL이 pose drift 보정
    per_dim[5] = 0.05      # gripper
    per_dim[6:9] = 0.0     # base: BC 그대로 (이미 최적)

    opt_actor = torch.optim.AdamW(
        [p for n, p in residual.named_parameters() if "critic" not in n],
        lr=args.lr_actor, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-6)
    opt_critic = torch.optim.AdamW(
        [p for n, p in residual.named_parameters() if "critic" in n],
        lr=args.lr_critic, eps=1e-5, weight_decay=1e-6)

    # ── Obs helper ──
    # init_arm_pose: 에피소드 시작 시점의 arm pose (reset 시 캡처)
    init_arm_pose = torch.zeros(N, 6, device=dev)  # (N, arm5+grip1)

    def capture_init_arm_pose(env_ids=None):
        """Reset된 env의 현재 arm pose를 init_arm_pose에 저장."""
        jp = env.robot.data.joint_pos
        if env_ids is None:
            init_arm_pose[:, :5] = jp[:, env.arm_idx[:5]]
            init_arm_pose[:, 5:] = jp[:, env.arm_idx[5:6]]
        else:
            init_arm_pose[env_ids, :5] = jp[env_ids][:, env.arm_idx[:5]]
            init_arm_pose[env_ids, 5:] = jp[env_ids][:, env.arm_idx[5:6]]

    def build_nav_obs():
        jp = env.robot.data.joint_pos
        arm = jp[:, env.arm_idx[:5]]     # (N, 5) 실제 관절 위치
        grip = jp[:, env.arm_idx[5:6]]   # (N, 1)
        bv = env.robot.data.root_lin_vel_b[:, :2]
        wz = env.robot.data.root_ang_vel_b[:, 2:3]
        base_vel = torch.cat([bv, wz], dim=-1)
        lidar = torch.ones(N, 8, device=dev)
        # 26D: arm(5)+grip(1)+base_vel(3)+dir_cmd(3)+lidar(8)+init_arm(5)+init_grip(1)
        return torch.cat([arm, grip, base_vel, direction_cmd, lidar, init_arm_pose], dim=-1)

    # ── Save dir ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    rollout_steps = args.num_env_steps
    total_iters = args.total_timesteps // (N * rollout_steps)
    print(f"\n  Navigate ResiP on Skill2EvalEnv")
    print(f"  BC: {args.bc_checkpoint}")
    print(f"  obs={OD}, act={AD}, envs={N}, rollout={rollout_steps}")
    print(f"  per_dim_scale: base={args.action_scale_base}")
    print(f"  total_iters={total_iters}\n")

    # Buffers — obs_buf stores pre-computed res_input (OD + AD), not raw obs
    RES_DIM = OD + AD  # normalized obs + base action
    obs_buf = torch.zeros(rollout_steps, N, RES_DIM, device=dev)
    act_buf = torch.zeros(rollout_steps, N, AD, device=dev)
    logp_buf = torch.zeros(rollout_steps, N, device=dev)
    val_buf = torch.zeros(rollout_steps, N, device=dev)
    rew_buf = torch.zeros(rollout_steps, N, device=dev)
    done_buf = torch.zeros(rollout_steps, N, device=dev)

    obs, _ = env.reset()
    capture_init_arm_pose()  # 초기 arm pose 캡처
    # 균등 분배: N envs를 4방향으로 나눔
    n_dirs = len(DIR_CMDS)
    idx = torch.arange(N, device=dev) % n_dirs
    direction_cmd[:] = DIR_CMDS[idx]
    dp_agent.reset()

    for gi in range(total_iters):
        dp_agent.reset()

        for step in range(rollout_steps):
            nav_obs = build_nav_obs()

            # BC + Residual
            with torch.no_grad():
                base_nact = dp_agent.base_action_normalized(nav_obs)
                nobs = dp_agent.normalizer(nav_obs, "obs", forward=True)
                nobs = torch.clamp(nobs, -3, 3)
                nobs = torch.nan_to_num(nobs, nan=0.0)
            res_input = torch.cat([nobs, base_nact], dim=-1)
            obs_buf[step] = res_input  # pre-computed, PPO에서 재사용
            with torch.no_grad():
                res_action, res_logp, res_entropy, res_val, res_mean = \
                    residual.get_action_and_value(res_input)
            nact = base_nact + res_action * per_dim
            action = dp_agent.normalizer(nact, "action", forward=False)
            action = action.clamp(-1, 1)

            act_buf[step] = res_action
            logp_buf[step] = res_logp
            val_buf[step] = res_val.squeeze(-1)

            # Step env
            next_obs, rew_env, ter, tru, info = env.step(action)
            done = ter | tru

            # ── Navigate Reward ──
            base_vel = env.robot.data.root_lin_vel_b  # (N, 3)
            target_lin = direction_cmd[:, :2] * env.cfg.max_lin_vel
            target_wz = direction_cmd[:, 2] * env.cfg.max_ang_vel
            actual_lin = base_vel[:, :2]
            actual_wz = base_vel[:, 2]

            lin_err = (target_lin - actual_lin).pow(2).sum(dim=-1)
            ang_err = (target_wz - actual_wz).pow(2)
            rew_lin = 1.5 * torch.exp(-lin_err / (0.25 ** 2))
            rew_ang = 1.5 * torch.exp(-ang_err / (0.25 ** 2))

            prev = env.prev_actions if hasattr(env, 'prev_actions') else torch.zeros_like(action)
            delta_base = action[:, 6:9] - prev[:, 6:9]
            delta_arm = action[:, 0:6] - prev[:, 0:6]
            rew_smooth = -0.005 * (delta_base ** 2).sum(dim=-1) - 0.01 * (delta_arm ** 2).sum(dim=-1)
            rew_time = torch.full((N,), -0.01, device=dev)

            # ── Arm pose maintenance reward: init_arm_pose 유지 ──
            jp = env.robot.data.joint_pos
            cur_arm = jp[:, env.arm_idx[:5]]         # (N, 5)
            cur_grip = jp[:, env.arm_idx[5:6]]       # (N, 1)
            arm_err = (cur_arm - init_arm_pose[:, :5]).pow(2).sum(dim=-1)
            grip_err = (cur_grip.squeeze(-1) - init_arm_pose[:, 5]).pow(2)
            rew_tucked = 2.0 * torch.exp(-arm_err / (0.3 ** 2)) + 0.5 * torch.exp(-grip_err / (0.2 ** 2))

            reward = rew_lin + rew_ang + rew_tucked + rew_smooth + rew_time
            reward = torch.nan_to_num(reward, nan=0.0)  # NaN 방어
            rew_buf[step] = reward
            done_buf[step] = done.float().squeeze(-1) if done.dim() > 1 else done.float()

            # Reset done envs: 균등 분배 + init_arm_pose 재캡처
            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.dim() == 0:
                    done_ids = done_ids.unsqueeze(0)
                new_idx = done_ids % n_dirs
                direction_cmd[done_ids] = DIR_CMDS[new_idx]
                capture_init_arm_pose(done_ids)
                dp_agent.reset()

            obs = next_obs

        # ── PPO Update ──
        with torch.no_grad():
            last_obs = build_nav_obs()
            base_nact_last = dp_agent.base_action_normalized(last_obs)
            nobs_last = dp_agent.normalizer(last_obs, "obs", forward=True).clamp(-3, 3)
            nobs_last = torch.nan_to_num(nobs_last, nan=0.0)
            res_input_last = torch.cat([nobs_last, base_nact_last], dim=-1)
            last_val = residual.get_value(res_input_last).squeeze(-1)

        dp_agent.reset()

        # GAE
        advantages = torch.zeros_like(rew_buf)
        lastgaelam = 0
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_val = last_val
                next_nonterminal = 1.0 - done_buf[t]
            else:
                next_val = val_buf[t + 1]
                next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.discount * next_val * next_nonterminal - val_buf[t]
            advantages[t] = lastgaelam = delta + args.discount * args.gae_lambda * next_nonterminal * lastgaelam
        returns = advantages + val_buf

        # Flatten — obs_buf already contains pre-computed res_input
        b_obs = obs_buf.reshape(-1, RES_DIM)
        b_act = act_buf.reshape(-1, AD)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)
        B = b_obs.shape[0]

        if args.norm_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Mini-batch PPO
        for epoch in range(args.update_epochs):
            idx = torch.randperm(B, device=dev)
            mb_size = B // max(args.num_minibatches, 1)
            for start in range(0, B, mb_size):
                end = start + mb_size
                mb = idx[start:end]

                # b_obs[mb] is already pre-computed res_input — no dp_agent call needed
                _, new_logp, new_entropy_mb, new_val_t, _ = \
                    residual.get_action_and_value(b_obs[mb], b_act[mb])
                new_val = new_val_t.squeeze(-1)

                ratio = (new_logp - b_logp[mb]).exp()
                surr1 = ratio * b_adv[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * b_adv[mb]
                pg_loss = -torch.min(surr1, surr2).mean()

                v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()

                entropy = new_entropy_mb.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy

                opt_actor.zero_grad()
                opt_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(residual.parameters(), args.max_grad_norm)
                opt_actor.step()
                opt_critic.step()

            # KL early stopping — b_obs is already pre-computed res_input
            with torch.no_grad():
                _, new_logp_all, _, _, _ = residual.get_action_and_value(b_obs, b_act)
                log_ratio = new_logp_all - b_logp
                kl = ((log_ratio.exp() - 1) - log_ratio).mean()
            if kl > args.target_kl:
                break

        # Logging
        avg_rew = rew_buf.mean().item()
        avg_lin = rew_lin.mean().item() if 'rew_lin' in dir() else 0
        avg_ang = rew_ang.mean().item() if 'rew_ang' in dir() else 0
        avg_tuck = rew_tucked.mean().item() if 'rew_tucked' in dir() else 0
        print(f"  iter={gi:4d} rew={avg_rew:.3f} lin={avg_lin:.3f} ang={avg_ang:.3f} tuck={avg_tuck:.3f} "
              f"kl={kl:.4f} v_loss={v_loss:.4f} entropy={entropy:.4f}")

        # Save
        if (gi + 1) % 10 == 0 or gi == total_iters - 1:
            torch.save({
                "residual_policy_state_dict": residual.state_dict(),
                "iteration": gi,
                "args": vars(args),
            }, save_dir / f"resip_nav_iter{gi}.pt")
            print(f"  → Saved resip_nav_iter{gi}.pt")

    # Final save
    torch.save({
        "residual_policy_state_dict": residual.state_dict(),
        "iteration": total_iters - 1,
        "args": vars(args),
    }, save_dir / "resip_nav_best.pt")
    print(f"\nDone. Final: {save_dir / 'resip_nav_best.pt'}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

