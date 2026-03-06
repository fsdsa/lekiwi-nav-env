#!/usr/bin/env python3
"""
Residual PPO for LeKiwi — v7.1

═══════════════════════════════════════════════════════════════════════════
LESSON LOG (v2–v6)
═══════════════════════════════════════════════════════════════════════════

v2  Continuous gripper-open reward      → agent keeps gripper open forever
v3  Close-gripper-near per-step reward  → agent closes gripper, rams object
    1-frame contact spike               → fake grasp (+100 bonus)
    EE distance rewards base forward    → base rams (1 DOF > 5 DOF)
v4  EE delta still includes base move  → same base-forward exploit
v5  R2 grip>0.5 gate                   → toggle exploit (open=approach,
                                          close=retreat without penalty)
    EE 3D vs base XY mismatch          → base rotation fakes arm progress
    Lift: no EE proximity check         → bounced object triggers lift reward
    Reward normalization                → crushes milestone scale differences,
                                          +100 grasp ≈ +10 gripper open after norm
    ent_coef=0.01                       → entropy diverges (3.9→5.4)

═══════════════════════════════════════════════════════════════════════════
V7 REWARD DESIGN — based on actual v6 + two additions
═══════════════════════════════════════════════════════════════════════════

v6 실기 검증 결과 발견된 두 가지 문제:
  1. 그리퍼가 바닥을 밀면서 팔 위치를 잡음 → R8 추가
  2. 물체를 든 후 하늘로 팔을 쭉 뻗음 (lifted pose 아님) → R4b 추가

Changes from v6:
  R4b Lifted pose approach       ×30    Gaussian σ=2.0, per-step during lift
  R8  Gripper-ground contact     −5.0   gcf sensor, pre-grasp only

Everything else identical to v6:
  R1  Gripper open milestone     +10    one-time, gates R2/R3
  R2  EE proximity (3D)          ×3     tanh σ=0.20, budget=80 per approach
  R3  Verified grasp             +100   one-time, 5-step sustained (env sticky)
  R4  Lift height                ×200   per-step, sustain≥3, grip closed, ee<0.20
  R5  Sustained lift bonus       ×50    per-step after 15 steps held
  R6  Soft-lift milestone        +100   one-time after 15 steps held
  R7  Time penalty               −0.01

═══════════════════════════════════════════════════════════════════════════
V7.1 — Drop detection & re-grasp
═══════════════════════════════════════════════════════════════════════════

  물체를 잡은 후 떨어뜨리면(oh<0.04 & ee_3d>0.15):
  - R3 보상 회수 (−100)
  - env.object_grasped sticky 리셋
  - milestone 리셋 (ms_gr, ms_li, ms_sl, g_sus, l_sus)
  - R2 재활성화 → 재접근 → 재파지(+100) → 재lift 경로 확보
  - ms_go는 유지 (그리퍼 열기는 이미 학습됨)

  R2를 delta-based base-subtraction에서 absolute EE proximity로 교체:
  - 기존: (p_ee_xy - ee_xy) - (p_bs_xy - bs_xy) → BC arm+base 동시 이동 시 ~0
  - 신규: (1 - tanh(ee_3d / 0.20)) × 3.0 → 가까울수록 강한 per-step 보상
  - cumulative budget=80: approach당 최대 80까지만 누적 (R3 +100 > R2 max)
    → 물체 옆에 가만히 있는 exploit 방지, grasp로 넘어가는 게 항상 이득
"""
from __future__ import annotations

import argparse
import os

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Args
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="ResiP v7.1")

parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place"])
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

# Warmup
parser.add_argument("--warmup_steps_initial", type=int, default=500)
parser.add_argument("--warmup_steps_final", type=int, default=0)
parser.add_argument("--warmup_decay_iters", type=int, default=100)

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
parser.add_argument("--r4b_scale", type=float, default=30.0)        # v7 NEW
parser.add_argument("--r8_penalty", type=float, default=-5.0)       # v7 NEW

# Eval/save
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--eval_first", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--save_dir", type=str, default="checkpoints/resip")
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
    if skill == "approach_and_grasp":
        from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
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
    cfg.enable_domain_randomization = True
    cfg.arm_limit_write_to_sim = False
    cfg.grasp_contact_threshold = 0.55
    cfg.grasp_gripper_threshold = 0.65
    cfg.grasp_max_object_dist = 0.50
    cfg.grasp_success_height = 1.00
    cfg.episode_length_s = 300.0
    cfg.spawn_heading_noise_std = 0.3
    cfg.spawn_heading_max_rad = 0.5
    cfg.dr_object_static_friction_scale_range = (1.0, 1.5)
    cfg.dr_object_dynamic_friction_scale_range = (1.0, 1.5)

    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.multi_object_json:
        cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if args.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    if skill == "approach_and_grasp":
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
    seed = args.seed or random.randint(0, 2**32 - 1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"Seed: {seed}")

    env = make_env(args.skill, args.num_envs, args)
    dev = env.device
    N = env.num_envs

    from lekiwi_skill2_env import EE_LOCAL_OFFSET
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

    # Constants
    GV  = args.grasp_verify_steps
    LMS = args.lift_min_sustain
    LMI = args.lift_milestone_steps
    LHT = 0.05
    HELD_EE_MAX = 0.20  # max ee-object distance for lift reward

    # Drop detection constants (v7.1)
    DROP_OH_THRESH = 0.04   # 약통 서있을 때 0.033, 넘어졌을 때 0.020
    DROP_EE_THRESH = 0.15   # 물체 들고 있으면 ee_3d < 0.15 이므로 정상 파지와 구분

    # R2 budget: approach당 최대 누적 보상 (R3 +100보다 작아야 grasp 동기 유지)
    R2_MAX_BUDGET = 80.0

    # v7 NEW: lifted pose target (proper carry position, not arm-straight-up)
    LIFTED_POSE = torch.tensor([-0.02, -1.00, 1.00, 0.30, -0.55], device=dev)

    # Per-env state
    ms_go   = torch.zeros(N, dtype=torch.bool, device=dev)   # gripper opened
    ms_gr   = torch.zeros(N, dtype=torch.bool, device=dev)   # grasped (verified)
    ms_li   = torch.zeros(N, dtype=torch.bool, device=dev)   # lifted
    ms_sl   = torch.zeros(N, dtype=torch.bool, device=dev)   # soft-lifted
    g_sus   = torch.zeros(N, dtype=torch.long, device=dev)   # grasp sustain counter
    l_sus   = torch.zeros(N, dtype=torch.long, device=dev)   # lift sustain counter
    r2_bud  = torch.zeros(N, device=dev)                     # R2 cumulative budget spent

    # Diagnostics
    r_gr     = torch.zeros(N, dtype=torch.long, device=dev)
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
    r_bgs    = torch.zeros(1, device=dev)   # base dist at grasp sum
    r_bgn    = torch.zeros(1, dtype=torch.long, device=dev)
    r_bls    = torch.zeros(1, device=dev)   # base dist at lift sum
    r_bln    = torch.zeros(1, dtype=torch.long, device=dev)
    _ldbg    = 0
    _r2_sum  = 0.0    # R2 reward accumulator
    _open_ct = 0      # ms_go trigger count
    _clip_ct = 0      # residual clamp hit count
    _clip_n  = 0      # total samples for clip rate
    _r4b_sum = 0.0    # v7 NEW: R4b reward sum
    _r8_n    = 0      # v7 NEW: R8 ground contact count
    _drop_n  = 0      # v7.1: drop detection count

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
        ms_li[mask] = False; ms_sl[mask] = False
        g_sus[mask] = 0; l_sus[mask] = 0
        r2_bud[mask] = 0.0

    # ── Print config ──
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    bsr, bsl, bgr = 0.0, 0, 0
    tt = 0; t0 = time.time()
    next_obs = env.reset(); next_done = torch.zeros(N, device=dev); dp.reset()

    print(f"\n{'='*60}")
    print(f"  ResiP v7.1 — {args.skill}")
    print(f"  N={N} S={S} B={B} iters={NI}")
    print(f"  scale: arm={args.action_scale_arm} grip={args.action_scale_gripper} base={args.action_scale_base}")
    print(f"  lr: a={args.lr_actor} c={args.lr_critic} kl={args.target_kl} ent={args.ent_coef}")
    print(f"  rew_norm={'ON' if args.normalize_reward else 'OFF'}")
    print(f"  R1=GripOpen(+10) R2=EEprox(×3,tanh σ=0.20,budget={R2_MAX_BUDGET})")
    print(f"  R3=VGrasp(+100,{GV}s) R4=Lift(×200,sus≥{LMS},ee<{HELD_EE_MAX})")
    print(f"  R4b=LiftPose(×{args.r4b_scale},σ=2.0) [v7 NEW]")
    print(f"  R5=SustBonus(×50,{LMI}s) R6=SoftLift(+100) R7=Time(-0.01)")
    print(f"  R8=GCF({args.r8_penalty}) [v7 NEW]")
    print(f"  DROP: oh<{DROP_OH_THRESH} & ee>{DROP_EE_THRESH} → -100, milestone reset [v7.1]")
    print(f"  warmup: {args.warmup_steps_initial}→{args.warmup_steps_final}/{args.warmup_decay_iters}")
    print(f"{'='*60}\n")

    # ═════════════════════════════════════════════════════════════════════════
    while gs < args.total_timesteps:
        gi += 1; it0 = time.time()
        ev = (gi - int(args.eval_first)) % args.eval_interval == 0

        next_obs = env.reset(); dp.reset()
        next_done = torch.zeros(N, device=dev)
        ms_go.zero_(); ms_gr.zero_(); ms_li.zero_(); ms_sl.zero_()
        g_sus.zero_(); l_sus.zero_(); r2_bud.zero_()

        # Warmup
        prog = min(1.0, (gi - 1) / max(1, args.warmup_decay_iters))
        ws = max(0, int(args.warmup_steps_initial
                        + (args.warmup_steps_final - args.warmup_steps_initial) * prog))
        for _ in range(ws):
            with torch.no_grad():
                a = dp.normalizer(dp.base_action_normalized(next_obs), "action", forward=False)
            next_obs, _, ter, tru, _ = env.step(a)
            next_done = (ter | tru).view(-1).float()

        print(f"\nIter {gi}/{NI} | {'EVAL' if ev else 'TRAIN'} | "
              f"step={gs} | wu={ws} | "
              f"EE: {ee_obj_dist_xy().mean():.3f}/{ee_obj_dist_xy().min():.3f}")

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
            ee_3d = ee_obj_dist_3d()
            bs_xy = base_obj_dist_xy()
            grip = env.env.robot.data.joint_pos[:, env.env.gripper_idx].view(-1)
            eg = info.get("object_grasped_mask", env.env.object_grasped).view(-1)
            jg = info.get("just_grasped_mask", env.env.just_grasped).view(-1).float()
            oh = info.get("object_height_mask",
                (env.env.object_pos_w[:, 2] - env.env.scene.env_origins[:, 2])).view(-1)
            cf = info.get("contact_force_raw", torch.zeros(N, device=dev)).view(-1)
            gcf = info.get("ground_contact_force_raw", torch.zeros(N, device=dev)).view(-1)

            rew = torch.zeros(N, device=dev)

            # ══════════════════════════════════════════════════════
            # DROP DETECTION & MILESTONE RESET (v7.1)
            # 물체를 잡은 후(ms_gr=True) 떨어뜨림을 감지:
            #   oh < 0.04: 물체가 바닥으로 돌아감 (서있으면 0.033, 넘어지면 0.020)
            #   ee_3d > 0.15: 그리퍼가 물체에서 멀어짐 (정상 파지 중 불가능)
            # ══════════════════════════════════════════════════════
            dropped = ms_gr & (oh < DROP_OH_THRESH) & (ee_3d > DROP_EE_THRESH)
            if dropped.any():
                # R3 보상 회수
                rew[dropped] -= 100.0
                # env sticky object_grasped 리셋 (안 하면 g_sus가 바로 올라감)
                env.env.object_grasped[dropped] = False
                # milestone 리셋 (ms_go는 유지: 그리퍼 열기는 이미 학습됨)
                ms_gr[dropped] = False
                ms_li[dropped] = False
                ms_sl[dropped] = False
                g_sus[dropped] = 0
                l_sus[dropped] = 0
                r2_bud[dropped] = 0.0  # 새 approach phase → R2 budget 리셋
                _drop_n += dropped.sum().item()

            # ══════════════════════════════════════════════════════
            # R1: GRIPPER OPEN MILESTONE (+10)
            # ══════════════════════════════════════════════════════
            nop = (grip > 0.8) & (~ms_go) & (~ms_gr)
            rew += nop.float() * 10.0
            ms_go |= nop

            # ══════════════════════════════════════════════════════
            # R2: EE PROXIMITY (tanh σ=0.20, absolute, budgeted)
            # [v7.1] absolute proximity + cumulative budget
            # budget < R3(+100) → grasp가 항상 이득
            # budget 소진 후 유일한 보상 경로 = grasp → lift
            # ══════════════════════════════════════════════════════
            aok = (~ms_gr) & ms_go
            ee_prox = 1.0 - torch.tanh(ee_3d / 0.20)
            r2_raw = aok.float() * ee_prox * 3.0
            r2_remaining = torch.clamp(R2_MAX_BUDGET - r2_bud, min=0.0)
            _r2_val = torch.min(r2_raw, r2_remaining)
            r2_bud += _r2_val
            rew += _r2_val
            _r2_sum += torch.nan_to_num(_r2_val, nan=0.0).sum().item()
            _open_ct += nop.sum().item()

            # ══════════════════════════════════════════════════════
            # R3: VERIFIED GRASP (+100, 5-step sustained)
            # (v6 방식: env sticky object_grasped 사용)
            # ══════════════════════════════════════════════════════
            gc = eg & ms_go & (~ms_gr)
            g_sus[gc] += 1
            g_sus[~gc & (~ms_gr)] = 0
            r_mgs = torch.max(r_mgs, g_sus)

            vg = (g_sus >= GV) & (~ms_gr)
            rew += vg.float() * 100.0
            ms_gr |= vg
            r_gr += vg.long()
            r_egr += (jg > 0).long()

            if vg.any():
                r_ggs += grip[vg].sum()
                r_ggn += vg.sum()
                r_bgs += bs_xy[vg].sum()
                r_bgn += vg.sum()

            # ══════════════════════════════════════════════════════
            # R4: LIFT HEIGHT (×200, sustain≥3, ee<0.20)
            # ══════════════════════════════════════════════════════
            gc2 = grip < float(env.env.cfg.grasp_gripper_threshold)
            held = (oh > LHT) & ms_gr & gc2 & (ee_3d < HELD_EE_MAX)

            l_sus[held] += 1
            l_sus[~held] = 0
            r_mls = torch.max(r_mls, l_sus)

            hp = torch.clamp((oh - 0.05) / (0.17 - 0.05), 0.0, 1.0)
            lok = l_sus >= LMS
            rew += ms_gr.float() * held.float() * lok.float() * hp * 200.0

            # ══════════════════════════════════════════════════════
            # R4b: LIFTED POSE APPROACH (Gaussian σ=2.0)
            # [v7 NEW] 물체를 든 상태에서 하늘로 팔을 쭉 뻗는 대신
            # 적절한 carry pose를 유지하도록 유도
            # ══════════════════════════════════════════════════════
            arm_joints = env.env.robot.data.joint_pos[:, :5]
            joint_err = torch.norm(arm_joints - LIFTED_POSE, dim=-1)
            pose_sim = torch.exp(-(joint_err ** 2) / 8.0)   # σ=2.0 → /8.0
            r4b_r = ms_gr.float() * held.float() * lok.float() * pose_sim * args.r4b_scale
            rew += r4b_r
            _r4b_sum += r4b_r.sum().item()

            # ══════════════════════════════════════════════════════
            # R5: SUSTAINED LIFT BONUS (×50, 15+ steps)
            # ══════════════════════════════════════════════════════
            gq = torch.exp(-((grip - 0.50) / 0.20) ** 2)
            sus = held & (l_sus >= LMI)
            rew += sus.float() * gq * 50.0

            # ══════════════════════════════════════════════════════
            # R6: SOFT-LIFT MILESTONE (+100)
            # ══════════════════════════════════════════════════════
            sl = sus & (gq > 0.3) & (~ms_sl)
            rew += sl.float() * 100.0
            ms_sl |= sl
            r_sl += sl.long()

            # ══════════════════════════════════════════════════════
            # R7: TIME PENALTY
            # ══════════════════════════════════════════════════════
            rew -= 0.01

            # ══════════════════════════════════════════════════════
            # R8: GRIPPER-GROUND CONTACT PENALTY
            # [v7 NEW] 바닥을 그리퍼로 밀면서 위치를 잡는 행동 억제
            # 물체가 5cm 이상 안 올라갔으면 바닥 접촉 시 페널티
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
        tg, teg = r_gr.sum().item(), r_egr.sum().item()
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

        for t in [r_gr, r_egr, r_li, r_sl, r_moz, r_mgs, r_mls, r_mcf,
                  r_cgs, r_cgn, r_ggs, r_ggn, r_bgs, r_bgn, r_bls, r_bln]:
            t.zero_()

        fps = S * N / max(time.time() - it0, 1e-6)
        cr = _clip_ct / max(_clip_n, 1)
        r2a = _r2_sum / max(S * N, 1)
        r4ba = _r4b_sum / max(S * N, 1)
        diag2 = (f" | R2avg={r2a:.3f} Opens={_open_ct} ClipR={cr:.3f}"
                 f" R4b={r4ba:.3f} R8=({_r8_n}) Drop={_drop_n}")
        _r2_sum = 0.0; _open_ct = 0; _clip_ct = 0; _clip_n = 0
        _r4b_sum = 0.0; _r8_n = 0; _drop_n = 0

        print(f"  SR={sr:.2%} | G={tg}(env:{teg}) | L={tl} | SL={tsl} | "
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


if __name__ == "__main__":
    main()
