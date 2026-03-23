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
                    choices=["approach_and_grasp", "carry_and_place", "combined_s2_s3"])
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
parser.add_argument("--s3_dest_heading_max_rad", type=float, default=0.76)

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
    if skill in ("approach_and_grasp", "combined_s2_s3"):
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

    if skill in ("approach_and_grasp", "combined_s2_s3"):
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
    s3_scale = torch.zeros(S3_AD, device=dev)
    s3_scale[0:5] = args.action_scale_arm
    s3_scale[5] = args.action_scale_gripper
    s3_scale[6:9] = args.action_scale_base

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
    prev_base_dst_xy = torch.zeros(N, device=dev)                 # R1 delta 계산용
    prev_src_dst_xy = torch.zeros(N, device=dev)                  # Phase B src→dst delta 계산용
    s3_step_counter = torch.zeros(N, dtype=torch.long, device=dev)  # S3 phase step counter
    S3_NO_CONTACT_STEPS = 8   # consecutive steps without contact = drop check
    S3_PLACE_RADIUS = 0.172   # source↔dest XY distance for place success
    S3_DEST_CONTACT_PENALTY = -1.0   # dest 접촉 패널티 (place 시도 억제 방지)
    S3_PHASE_B_DIST = 0.45   # base→dest 이 거리 이하면 Phase B (팔 뻗기)
    S3_MAX_STEPS = 2000       # S3 phase timeout
    S3_REST_POSE = torch.tensor([0.025, 0.000, 0.001, 0.003, 0.040], device=dev)

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    bsr, bsl, bgr = 0.0, 0, 0
    tt = 0; t0 = time.time()
    s2_success_total = 0; s3_success_total = 0; s3_place_total = 0; s3_drop_total = 0; total_episodes = 0

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

        # Scale scheduling: base는 처음부터, arm/grip은 점진적
        arm_alpha = min(1.0, gi / 40)
        grip_alpha = min(1.0, gi / 60)
        s3_scale = torch.zeros(S3_AD, device=dev)
        s3_scale[0:5] = args.action_scale_arm * arm_alpha
        s3_scale[5] = args.action_scale_gripper * grip_alpha
        s3_scale[6:9] = args.action_scale_base

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
                s3_obs = torch.cat([
                    actor_obs[:, 0:21],      # arm(5)+grip(1)+base_vel(3)+lin_vel(3)+ang_vel(3)+arm_vel(6) = 21D
                    dest_rel_body,            # [21:24] dest relative pos 3D
                    contact_force,            # [24:25] grip force 1D (연속값)
                    actor_obs[:, 26:29],      # [25:28] bbox 3D
                    actor_obs[:, 29:30],      # [28:29] category 1D
                ], dim=-1)  # 29D

                # S3 BC base action
                s3_ba = s3_dp.base_action_normalized(s3_obs)
                s3_no = torch.nan_to_num(torch.clamp(
                    s3_dp.normalizer(s3_obs, "obs", forward=True), -3, 3), nan=0.0)
                s3_ba = torch.nan_to_num(s3_ba, nan=0.0)
                s3_ro = torch.cat([s3_no, s3_ba], dim=-1)

            # S3 residual (trainable) — 60iter에 걸쳐 BC→residual 점진 전환
            with torch.no_grad():
                s3_ra_s, _, _, s3_val, s3_ra_m = rpol.get_action_and_value(s3_ro)
            s3_ra = s3_ra_m if ev else s3_ra_s
            s3_ra = torch.clamp(s3_ra, -1.0, 1.0)
            with torch.no_grad():
                _, s3_lp, _, _, _ = rpol.get_action_and_value(s3_ro, s3_ra)
            combined = s3_ba + s3_ra * s3_scale
            combined[:, 5] = torch.clamp(combined[:, 5], -0.40, 1.0)  # gripper action clamp
            s3_action = s3_dp.normalizer(combined, "action", forward=False)

            # Merge action by phase
            is_s2 = (phase == 0)
            is_s3 = (phase == 1)
            action = torch.where(is_s2.unsqueeze(-1), s2_action, s3_action)

            # Store S3 transitions
            obs_b[step] = s3_ro
            act_b[step] = s3_ra
            lp_b[step] = s3_lp.view(-1)
            val_b[step] = s3_val.view(-1)
            done_b[step] = next_done
            # Mark S2 phase steps as done (won't contribute to GAE)
            done_b[step][is_s2] = 1.0

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
                print(f"    [CONTACT] step={step} S3={_s3m.sum().item()} jaw_only={_jaw_only} wrist_only={_wrist_only} both={_both} none={_none} held(objZ>0.04)={_held}")

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
                # Initialize prev distance for delta reward
                prev_base_dst_xy[t_ids] = torch.norm(
                    rpos[:, :2] - torch.stack([dest_x, dest_y], dim=-1), dim=-1)
                prev_src_dst_xy[t_ids] = torch.norm(
                    env.env.object_pos_w[t_ids, :2] - torch.stack([dest_x, dest_y], dim=-1), dim=-1)
                s2_success_total += t_ids.shape[0]
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
                has_contact = (jaw_cf > 0.3) & (wrist_cf > 0.3)   # 양쪽 gripper 모두 접촉

                # Robot base position
                base_pos = env.env.robot.data.root_pos_w  # (N, 3)
                base_dst_xy = torch.norm(base_pos[:, :2] - dst_pos[:, :2], dim=-1)

                # Gripper pos & arm joints
                grip_pos = env.env.robot.data.joint_pos[:, env.env.gripper_idx]
                arm_joints = env.env.robot.data.joint_pos[:, env.env.arm_idx]

                # Gripper closed check (S2 grasp와 동일 기준)
                gripper_closed = grip_pos < float(env.env.cfg.grasp_gripper_threshold)

                # S3 hold = contact + gripper_closed (between_jaws는 carry 중 EE drift로 false drop 유발하므로 제외)
                is_holding = has_contact & gripper_closed & (grip_pos > 0.25)

                # ── R0: Drop detection ──
                # Contact lost → increment counter; contact present → reset
                # Place 시도 중 (gripper 열림 + base 가까움)이면 drop 판정 제외
                attempting_place = (grip_pos > 0.45) & (base_dst_xy < S3_PHASE_B_DIST) & (src_h > 0.025)
                s3_no_contact_counter[s3m & is_holding] = 0
                s3_no_contact_counter[s3m & ~is_holding & ~attempting_place] += 1
                s3_no_contact_counter[s3m & attempting_place] = 0  # place 시도 중 counter 리셋
                # Drop = contact lost for S3_NO_CONTACT_STEPS + not placed (거리 무관, 재파지 불가)
                s3_drop = s3m & (s3_no_contact_counter >= S3_NO_CONTACT_STEPS) & (~ms_place)
                # S3 timeout
                s3_timeout = s3m & (s3_step_counter >= S3_MAX_STEPS) & (~ms_place)
                s3_fail = s3_drop | s3_timeout

                # ── Phase 판정 ──
                phase_a = s3m & (~ms_place) & (~s3_fail) & (base_dst_xy > S3_PHASE_B_DIST)
                phase_b = s3m & (~ms_place) & (~s3_fail) & (base_dst_xy <= S3_PHASE_B_DIST)
                phase_c = s3m & ms_place

                # ── R_hold: Phase A에서 hold 보상 (hold-forever 방지: 0.05/step) ──
                hold = phase_a & is_holding & (src_h > 0.033)
                rew[hold] += 0.05

                # ── R1: Phase A — base → dest 접근 (delta × 30, 잡고 있을 때만) ──
                R1_TARGET = 0.35
                prev_err = (prev_base_dst_xy - R1_TARGET).abs()
                curr_err = (base_dst_xy - R1_TARGET).abs()
                approach_delta = torch.clamp(prev_err - curr_err, -0.05, 0.05)
                r1 = approach_delta * 30.0   # 0.5m 접근 → +15 total
                r1_mask = phase_a & is_holding
                rew[r1_mask] += r1[r1_mask]

                # ── R_arm: Phase B — src↔dst delta shaping (팔 뻗기 유도, 잡고 있을 때만) ──
                src_dst_delta = torch.clamp(prev_src_dst_xy - src_dst_xy, -0.05, 0.05)
                r_arm = src_dst_delta * 30.0
                r_arm_mask = phase_b & is_holding
                rew[r_arm_mask] += r_arm[r_arm_mask]

                # Update prev distances
                prev_base_dst_xy[s3m] = base_dst_xy[s3m]
                prev_src_dst_xy[s3m] = src_dst_xy[s3m]

                # ── R2: Phase B — dest contact penalty ──
                dest_cf = env.env._dest_contact_force_per_env()
                dest_touching = (dest_cf > 0.3) & s3m
                rew[dest_touching] += S3_DEST_CONTACT_PENALTY  # -1.0

                # ── R3: Place success (one-time milestone +200) ──
                place_cond = (
                    s3m & ~ms_place
                    & (src_h > 0.025)           # 쓰러지지 않음
                    & (src_h < 0.04)            # 바닥에 놓여있음 (carry 중 제외)
                    & (src_dst_xy < S3_PLACE_RADIUS)
                    & (grip_pos > 0.5)          # gripper 열림
                    & ~s3_fail
                )
                if place_cond.any():
                    ms_place[place_cond] = True
                    rew[place_cond] += 200.0
                    s3_place_total += place_cond.sum().item()
                    print(f"    [S3] PLACE! {place_cond.sum().item()} envs at step {step} base_dst={base_dst_xy[place_cond].tolist()} src_dst={src_dst_xy[place_cond].tolist()} grip={grip_pos[place_cond].tolist()} s3_step={s3_step_counter[place_cond].tolist()}")

                # ── R4: Phase C — rest pose + gripper open 유도 (축소) ──
                if phase_c.any():
                    pose_err = torch.norm(arm_joints[phase_c, :5] - S3_REST_POSE, dim=-1)
                    r4 = torch.exp(-0.5 * (pose_err / 0.3) ** 2) * 0.1
                    grip_open_rew = torch.clamp(grip_pos[phase_c] / 0.9, 0.0, 1.0) * 0.05
                    rew[phase_c] += r4 + grip_open_rew

                # ── R5: Time penalty ──
                rew[s3m] += -0.01

                # ── R0/timeout: drop 패널티 없음, reset만 ──
                rew[s3_timeout] = -5.0

            rew_b[step] = rew

            # ── S3 fail (drop / timeout) → force reset to S2 ──
            s3_fail = s3_drop | s3_timeout
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
                if s3_drop_early + s3_drop_late > 0:
                    _wedged = (_oh_d > 0.04).sum().item()
                    _real = (_oh_d <= 0.04).sum().item()
                    print(f"    [DROP] early={s3_drop_early} late={s3_drop_late} avg_step={_drop_steps.float().mean():.0f} | jaw={_jaw_d.mean():.2f} wrist={_wrist_d.mean():.2f} grip={_grip_d.mean():.3f} objZ={_oh_d.mean():.3f} | real={_real} wedged={_wedged}")
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
                s3_step_counter[fail_ids] = 0
                prev_base_dst_xy[fail_ids] = 0.0
                prev_src_dst_xy[fail_ids] = 0.0

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
                s3_step_counter[reset_mask] = 0
                prev_base_dst_xy[reset_mask] = 0.0
                prev_src_dst_xy[reset_mask] = 0.0
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
                s3_obs_final = torch.cat([
                    next_obs[:, 0:21], dest_rel_f, cf_f,
                    next_obs[:, 26:29], next_obs[:, 29:30],
                ], dim=-1)
                s3_no_f = torch.nan_to_num(torch.clamp(
                    s3_dp.normalizer(s3_obs_final, "obs", forward=True), -3, 3), nan=0.0)
                s3_ba_f = torch.nan_to_num(s3_dp.base_action_normalized(s3_obs_final), nan=0.0)
                s3_ro_f = torch.cat([s3_no_f, s3_ba_f], dim=-1)
                nv = rpol.get_value(s3_ro_f).view(-1)

            bret, badv = compute_gae(val_b, nv, rew_b, done_b,
                                     next_done, S, args.discount, args.gae_lambda)
            badv = (badv - badv.mean()) / (badv.std() + 1e-8)
            bobs = obs_b.view(-1, RD)
            bact = act_b.view(-1, S3_AD)
            blp  = lp_b.view(-1)
            bret = bret.view(-1)
            badv = badv.view(-1)
            bv   = val_b.view(-1)

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
            print(f"  S2→S3: {s2_success_total} total | S3 envs: {s3_envs} (hold={s3_holding}, placed={s3_placed}, drop={s3_dropped}) | place_total={s3_place_total} drop_total={s3_drop_total}")
            print(f"  S3 objZ: min={s3_oh.min():.3f} mean={s3_oh.mean():.3f} max={s3_oh.max():.3f}")
            print(f"  S3 src→dst: min={s3_src_dst.min():.3f} mean={s3_src_dst.mean():.3f} | base→dst: min={s3_base_dst.min():.3f} mean={s3_base_dst.mean():.3f}")
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


if __name__ == "__main__":
    main()

