#!/usr/bin/env python3
"""
DPPO: Diffusion Policy Policy Optimization for LeKiwi Skill-2.

Replaces train_resip.py. Fine-tunes the Diffusion Policy (BC) UNet directly
using PPO on the denoising process, preserving action chunking.

Reward Design (7 components, chunk-level):
  R1  Approach progress    ×15     dense, before grasp only
  R2  Verified grasp       +200    sparse, one-time (5-step sustained)
  R3  Lift height          ×100    dense, during lift (height-proportional)
  R4  Sustained lift       +500    sparse, one-time (100+ steps stable lift)
  R5  Drop penalty         −200    sparse + episode terminate
  R6  Ground contact       −2.0    dense, per chunk
  R7  Time penalty         −0.1    dense, per chunk

Design philosophy:
  - DPPO preserves action chunking → BC already knows HOW (sequence structure)
  - Reward only tells WHICH outcomes are good → milestone-focused
  - No gripper timing shaping (v6.7 lesson: per-step gripper RL fails)
  - No arm direction shaping (DPPO handles this through UNet fine-tuning)
  - No lifted pose reward (BC provides pose, DPPO improves precision)
  - Running reward normalization handles scale automatically

Usage:
    cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env && \
    PYTHONUNBUFFERED=1 LEKIWI_USD_PATH=/home/jovyan/Downloads/lekiwi_robot.usd \
    python train_dppo.py \
      --skill approach_and_grasp \
      --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
      --object_usd /path/to/object.usd \
      --num_envs 1024 --headless \
      --save_dir checkpoints/dppo_v1
"""
from __future__ import annotations

import argparse
import os

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Args
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="DPPO for LeKiwi")
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp"])
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")

# DPPO specific
parser.add_argument("--ddim_steps", type=int, default=4)
parser.add_argument("--ft_denoising_steps", type=int, default=4)
parser.add_argument("--min_sampling_std", type=float, default=0.08)
parser.add_argument("--min_logprob_std", type=float, default=0.08)
parser.add_argument("--gamma_denoising", type=float, default=0.9)
parser.add_argument("--clip_ploss_coef", type=float, default=0.01)
parser.add_argument("--clip_ploss_coef_base", type=float, default=0.001)
parser.add_argument("--clip_ploss_coef_rate", type=float, default=3.0)
parser.add_argument("--denoised_clip_value", type=float, default=1.0)
parser.add_argument("--randn_clip_value", type=float, default=3.0)
parser.add_argument("--final_action_clip_value", type=float, default=1.0)
parser.add_argument("--eta", type=float, default=1.0)

# PPO
parser.add_argument("--n_steps", type=int, default=100)
parser.add_argument("--total_iters", type=int, default=1000)
parser.add_argument("--update_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=20000)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--target_kl", type=float, default=1.0)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--norm_adv", type=lambda x: x.lower() == "true", default=True)

# LR
parser.add_argument("--actor_lr", type=float, default=1e-5)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--n_critic_warmup_itr", type=int, default=1)

# Reward
parser.add_argument("--reward_scale_running", type=lambda x: x.lower() == "true",
                    default=True)
parser.add_argument("--r_approach_scale", type=float, default=15.0)
parser.add_argument("--r_grasp_bonus", type=float, default=200.0)
parser.add_argument("--r_lift_scale", type=float, default=100.0)
parser.add_argument("--r_success_bonus", type=float, default=500.0)
parser.add_argument("--r_drop_penalty", type=float, default=-200.0)
parser.add_argument("--r_ground_penalty", type=float, default=-2.0)
parser.add_argument("--r_time_penalty", type=float, default=-0.1)
parser.add_argument("--grasp_verify_steps", type=int, default=5)
parser.add_argument("--lift_sustain_threshold", type=int, default=100)

# Warmup
parser.add_argument("--warmup_steps", type=int, default=660)

# Eval/save
parser.add_argument("--eval_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--save_dir", type=str, default="checkpoints/dppo")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None)

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

from diffusion_policy import DiffusionPolicyAgent
from dppo_model import DPPODiffusion, RunningRewardScaler

from lekiwi_skill2_env import EE_LOCAL_OFFSET


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Environment
# ═══════════════════════════════════════════════════════════════════════════════

class LeKiwiEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

    def reset(self):
        od, _ = self.env.reset()
        return (od["policy"] if isinstance(od, dict) else od).to(self.device)

    def step(self, action):
        od, r, ter, tru, info = self.env.step(action)
        o = (od["policy"] if isinstance(od, dict) else od).to(self.device)
        return o, r.view(-1).to(self.device), ter.view(-1).to(self.device), \
               tru.view(-1).to(self.device), info


def make_env(skill, num_envs, args_):
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.grasp_contact_threshold = 0.55
    cfg.grasp_gripper_threshold = 0.65
    cfg.grasp_max_object_dist = 0.50
    cfg.episode_length_s = 300.0
    cfg.spawn_heading_noise_std = 0.3
    cfg.spawn_heading_max_rad = 0.5
    cfg.grasp_success_height = 1.00     # disable env auto-reset on lift
    cfg.lift_success_sustain_steps = 0   # we handle success ourselves

    if args_.object_usd:
        cfg.object_usd = os.path.expanduser(args_.object_usd)
    if args_.multi_object_json:
        cfg.multi_object_json = os.path.expanduser(args_.multi_object_json)
    if args_.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args_.dest_object_usd)
    cfg.gripper_contact_prim_path = args_.gripper_contact_prim_path
    if args_.arm_limit_json and os.path.isfile(args_.arm_limit_json):
        cfg.arm_limit_json = args_.arm_limit_json

    env = Skill2Env(cfg=cfg)
    print(f"  Env: {skill}, n={num_envs}, dev={env.device}")
    return LeKiwiEnvWrapper(env)


def load_bc_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"], action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=cfg.get("inference_steps", 16),
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)
    sd = ck["model_state_dict"]
    agent.model.load_state_dict(
        {k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    agent.normalizer.load_state_dict(
        {k[11:]: v for k, v in sd.items() if k.startswith("normalizer.")},
        device=device)
    print(f"  BC: obs={cfg['obs_dim']} act={cfg['act_dim']} "
          f"pred_h={cfg['pred_horizon']} act_h={cfg['action_horizon']}")
    return agent.model, agent.normalizer, cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.skill, args.num_envs, args)
    dev = env.device
    N = env.num_envs

    # EE offset
    jaw_idx, _ = env.env.robot.find_bodies(["Wrist_Roll_08c_v1"])
    jaw_idx = jaw_idx[0]
    ee_off = torch.tensor(EE_LOCAL_OFFSET, device=dev).unsqueeze(0)

    # Load BC
    unet, normalizer, bc_cfg = load_bc_checkpoint(args.bc_checkpoint, dev)
    OD, AD = bc_cfg["obs_dim"], bc_cfg["act_dim"]
    PRED_H = bc_cfg["pred_horizon"]
    ACT_H = bc_cfg["action_horizon"]

    # Create DPPO
    dppo = DPPODiffusion(
        unet_pretrained=unet, normalizer=normalizer,
        obs_dim=OD, act_dim=AD,
        pred_horizon=PRED_H, act_steps=ACT_H,
        denoising_steps=bc_cfg["num_diffusion_iters"],
        ddim_steps=args.ddim_steps, ft_denoising_steps=args.ft_denoising_steps,
        min_sampling_denoising_std=args.min_sampling_std,
        min_logprob_denoising_std=args.min_logprob_std,
        gamma_denoising=args.gamma_denoising,
        clip_ploss_coef=args.clip_ploss_coef,
        clip_ploss_coef_base=args.clip_ploss_coef_base,
        clip_ploss_coef_rate=args.clip_ploss_coef_rate,
        denoised_clip_value=args.denoised_clip_value,
        randn_clip_value=args.randn_clip_value,
        final_action_clip_value=args.final_action_clip_value,
        eta=args.eta, norm_adv=args.norm_adv,
        device=str(dev),
    ).to(dev)

    # Optimizers
    actor_opt = optim.AdamW(dppo.actor_ft.parameters(), lr=args.actor_lr, weight_decay=0)
    critic_opt = optim.AdamW(dppo.critic.parameters(), lr=args.critic_lr, weight_decay=0)
    actor_sched = optim.lr_scheduler.CosineAnnealingLR(
        actor_opt, T_max=args.total_iters, eta_min=args.actor_lr * 0.01)
    critic_sched = optim.lr_scheduler.CosineAnnealingLR(
        critic_opt, T_max=args.total_iters, eta_min=args.critic_lr * 0.1)

    reward_scaler = RunningRewardScaler(N) if args.reward_scale_running else None

    start_itr = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        dppo.actor_ft.load_state_dict(ck["actor_ft_state_dict"])
        dppo.critic.load_state_dict(ck["critic_state_dict"])
        if "actor_opt" in ck: actor_opt.load_state_dict(ck["actor_opt"])
        if "critic_opt" in ck: critic_opt.load_state_dict(ck["critic_opt"])
        start_itr = ck.get("iteration", 0)
        print(f"  Resumed iter={start_itr}")

    S = args.n_steps
    K = args.ft_denoising_steps
    WU = args.warmup_steps
    WU_CALLS = WU // ACT_H
    GV = args.grasp_verify_steps
    LMI = args.lift_sustain_threshold
    HELD_EE_MAX = 0.20
    LIFT_H_MIN = 0.05

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_sr, best_sl = 0.0, 0
    t_start = time.time()

    # Per-env reward state
    ms_grasp = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_success = torch.zeros(N, dtype=torch.bool, device=dev)
    g_sus = torch.zeros(N, dtype=torch.long, device=dev)
    l_sus = torch.zeros(N, dtype=torch.long, device=dev)
    prev_bd = torch.zeros(N, device=dev)

    def ee_pos():
        wp = env.env.robot.data.body_pos_w[:, jaw_idx, :]
        wq = env.env.robot.data.body_quat_w[:, jaw_idx, :]
        return wp + quat_apply(wq, ee_off.expand_as(wp))

    def ee_obj_3d():
        return torch.norm(ee_pos() - env.env.object_pos_w, dim=-1).view(-1)

    def base_obj_xy():
        return torch.norm(env.env.robot.data.root_pos_w[:, :2]
                          - env.env.object_pos_w[:, :2], dim=-1).view(-1)

    def obj_h():
        return (env.env.object_pos_w[:, 2] - env.env.scene.env_origins[:, 2]).view(-1)

    def reset_rs(mask):
        ms_grasp[mask] = False; ms_success[mask] = False
        g_sus[mask] = 0; l_sus[mask] = 0
        prev_bd[mask] = base_obj_xy()[mask]

    print(f"\n{'='*70}")
    print(f"  DPPO — {args.skill}")
    print(f"  N={N} S={S} ACT_H={ACT_H} K={K}")
    print(f"  env_steps/iter = {S * ACT_H} + warmup {WU}")
    print(f"  actor_lr={args.actor_lr} critic_lr={args.critic_lr}")
    print(f"  gamma={args.gamma} gamma_d={args.gamma_denoising}")
    print(f"  clip={args.clip_ploss_coef}/{args.clip_ploss_coef_base}")
    print(f"  min_std={args.min_sampling_std} eta={args.eta}")
    print(f"  ── Reward ──")
    print(f"  R1 Approach ×{args.r_approach_scale}  R2 Grasp +{args.r_grasp_bonus}")
    print(f"  R3 Lift ×{args.r_lift_scale}  R4 Success +{args.r_success_bonus} (sus≥{LMI})")
    print(f"  R5 Drop {args.r_drop_penalty}+term  R6 Ground {args.r_ground_penalty}")
    print(f"  R7 Time {args.r_time_penalty}")
    print(f"{'='*70}\n")

    # ═══════════════════════════════════════════════════════════════════
    for itr in range(start_itr, args.total_iters):
        itr_t0 = time.time()
        ev = (itr % args.eval_interval == 0)
        if ev: dppo.eval()
        else: dppo.train(); dppo.actor.eval()

        obs = env.reset()
        ms_grasp.zero_(); ms_success.zero_(); g_sus.zero_(); l_sus.zero_()

        # Warmup
        if WU_CALLS > 0:
            for wi in range(WU_CALLS):
                with torch.no_grad():
                    an, _ = dppo.sample_actions(obs, deterministic=True)
                    act = dppo.normalizer(
                        an[:, :ACT_H].reshape(-1, AD), "action", forward=False
                    ).reshape(N, ACT_H, AD)
                for ai in range(ACT_H):
                    obs, _, ter, tru, _ = env.step(act[:, ai])
                    if (ter | tru).any(): break
        prev_bd[:] = base_obj_xy()

        # Buffers
        obs_b = torch.zeros(S, N, OD, device="cpu")
        chain_b = torch.zeros(S, N, K + 1, PRED_H, AD, device="cpu")
        rew_b = torch.zeros(S, N, device="cpu")
        term_b = torch.zeros(S, N, device="cpu")
        first_b = torch.zeros(S + 1, N, device="cpu")
        first_b[0] = 1.0

        dg, dl, dd, dgcf, dml = 0, 0, 0, 0, 0  # diagnostics

        # ═══ Rollout ═══
        for step in range(S):
            obs_b[step] = obs.cpu()
            with torch.no_grad():
                an, ch = dppo.sample_actions(obs, deterministic=ev)
                act = dppo.normalizer(
                    an[:, :ACT_H].reshape(-1, AD), "action", forward=False
                ).reshape(N, ACT_H, AD)
            chain_b[step] = ch.cpu()

            cr = torch.zeros(N, device=dev)
            ct = torch.zeros(N, dtype=torch.bool, device=dev)
            cd = torch.zeros(N, dtype=torch.bool, device=dev)

            for ai in range(ACT_H):
                obs_n, _, ter, tru, info = env.step(act[:, ai])
                alive = ~cd

                grip = env.env.robot.data.joint_pos[:, env.env.gripper_idx].view(-1)
                gc = grip < float(env.env.cfg.grasp_gripper_threshold)
                eg = info.get("object_grasped_mask", env.env.object_grasped).view(-1)
                jd = info.get("just_dropped_mask",
                              torch.zeros(N, dtype=torch.bool, device=dev)).view(-1)
                oh = obj_h()
                ed = ee_obj_3d()
                gcf = info.get("ground_contact_force_raw",
                               torch.zeros(N, device=dev)).view(-1)

                # R6: Ground contact
                gog = (gcf > 1.0) & (oh < 0.05) & alive
                cr += gog.float() * (args.r_ground_penalty / ACT_H)
                dgcf += gog.sum().item()

                # Grasp sustain
                gc_ok = eg & (~ms_grasp) & alive
                g_sus[gc_ok] += 1
                g_sus[~gc_ok & (~ms_grasp)] = 0

                # R2: Verified grasp
                vg = (g_sus >= GV) & (~ms_grasp) & alive
                if vg.any():
                    cr[vg] += args.r_grasp_bonus
                    ms_grasp |= vg
                    dg += vg.sum().item()

                # Lift tracking
                held = (oh > LIFT_H_MIN) & ms_grasp & gc & (ed < HELD_EE_MAX) & alive
                l_sus[held] += 1
                l_sus[~held & ms_grasp] = 0
                dml = max(dml, l_sus.max().item())

                # R3: Lift height
                hp = torch.clamp((oh - 0.03) / 0.15, 0.0, 1.0)
                cr += held.float() * hp * (args.r_lift_scale / ACT_H)

                # R4: Sustained lift success
                ns = (l_sus >= LMI) & (~ms_success) & alive
                if ns.any():
                    cr[ns] += args.r_success_bonus
                    ms_success |= ns
                    dl += ns.sum().item()

                # R5: Drop penalty + terminate
                dropped = jd.bool() & ms_grasp & alive
                if dropped.any():
                    cr[dropped] += args.r_drop_penalty
                    ct |= dropped
                    ms_grasp[dropped] = False
                    g_sus[dropped] = 0; l_sus[dropped] = 0
                    dd += dropped.sum().item()

                ct |= ter.bool()
                cd |= (ter | tru).view(-1).bool()
                obs = obs_n

            # R1: Approach progress (chunk-level)
            cbd = base_obj_xy()
            am = (~ms_grasp) & (~cd)
            ad = torch.clamp(prev_bd - cbd, -0.1, 0.1)
            cr += am.float() * ad * args.r_approach_scale
            prev_bd[:] = cbd

            # R7: Time
            cr += (~cd).float() * args.r_time_penalty

            # Reset done envs
            if cd.any(): reset_rs(cd)

            rew_b[step] = cr.cpu()
            term_b[step] = ct.float().cpu()
            first_b[step + 1] = cd.float().cpu()

        # ═══ Values + Logprobs ═══
        with torch.no_grad():
            val_b = torch.zeros(S, N, device="cpu")
            for s in range(S):
                val_b[s] = dppo.get_value(obs_b[s].to(dev)).cpu()
            nv = dppo.get_value(obs).cpu()

            lp_b = torch.zeros(S, N, K, PRED_H, AD, device="cpu")
            for s in range(S):
                lp_b[s] = dppo.get_logprobs_all(obs_b[s].to(dev), chain_b[s].to(dev)).cpu()

        # Reward scaling
        rn = rew_b.numpy()
        if reward_scaler is not None and not ev:
            fn = first_b[:-1].numpy()
            for s in range(S):
                rn[s] = reward_scaler(rn[s], fn[s])
            rew_b = torch.from_numpy(rn).float()

        # GAE
        adv_b = torch.zeros(S, N)
        lg = torch.zeros(N)
        for t in reversed(range(S)):
            nxt = nv.view(N) if t == S - 1 else val_b[t + 1]
            nt = 1.0 - term_b[t]
            d = rew_b[t] + args.gamma * nxt * nt - val_b[t]
            lg = d + args.gamma * args.gae_lambda * nt * lg
            adv_b[t] = lg
        ret_b = adv_b + val_b

        sr = dl / N
        fps = S * ACT_H * N / max(time.time() - itr_t0, 1e-6)
        print(f"\nIter {itr}/{args.total_iters} | {'EVAL' if ev else 'TRAIN'} | "
              f"SR={sr:.2%} | R={rew_b.sum(0).mean():.1f} | "
              f"G={dg} L={dl} D={dd} GCF={dgcf} MaxLSus={dml} | FPS={fps:.0f}")

        if ev:
            if sr > best_sr:
                best_sr = sr
                torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
                    "critic_state_dict": dppo.critic.state_dict(),
                    "normalizer_state_dict": dppo.normalizer.state_dict(),
                    "bc_config": bc_cfg, "iteration": itr,
                    "success_rate": sr, "args": vars(args)}, save_dir / "dppo_best.pt")
                print(f"  ★ Best SR={sr:.2%}")
            if dl > best_sl:
                best_sl = dl
                torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
                    "critic_state_dict": dppo.critic.state_dict(),
                    "normalizer_state_dict": dppo.normalizer.state_dict(),
                    "bc_config": bc_cfg, "iteration": itr,
                    "lifts": dl, "args": vars(args)}, save_dir / "dppo_best_lift.pt")
                print(f"  ★ Best L={dl}")
            print(f"  Best: SR={best_sr:.2%} L={best_sl}")
            continue

        # ═══ PPO Update ═══
        of = obs_b.reshape(S * N, OD)
        cf = chain_b.reshape(S * N, K + 1, PRED_H, AD)
        rf = ret_b.reshape(S * N)
        vf = val_b.reshape(S * N)
        af = adv_b.reshape(S * N)
        lpf = lp_b.reshape(S * N, K, PRED_H, AD)

        tot = S * N * K
        bs = min(args.batch_size, tot)
        cfs = []

        for ep in range(args.update_epochs):
            brk = False
            perm = torch.randperm(tot, device="cpu")
            nb = max(1, tot // bs)
            for bi in range(nb):
                i0, i1 = bi * bs, min((bi + 1) * bs, tot)
                idx = perm[i0:i1]
                bi_ = idx // K
                di_ = idx % K

                o = of[bi_].to(dev)
                cp = cf[bi_, di_].to(dev)
                cn = cf[bi_, di_ + 1].to(dev)
                r_ = rf[bi_].to(dev)
                v_ = vf[bi_].to(dev)
                a_ = af[bi_].to(dev)
                lp = lpf[bi_, di_].to(dev)
                di_d = di_.to(dev)

                pg, vl, kl, clf = dppo.loss_ppo(
                    o, cp, cn, di_d, r_, v_, a_, lp, reward_horizon=ACT_H)
                loss = pg + vl * args.vf_coef
                cfs.append(clf)

                actor_opt.zero_grad(); critic_opt.zero_grad()
                loss.backward()
                if itr >= args.n_critic_warmup_itr:
                    if args.max_grad_norm:
                        nn.utils.clip_grad_norm_(dppo.actor_ft.parameters(), args.max_grad_norm)
                    actor_opt.step()
                critic_opt.step()

                if args.target_kl and kl > args.target_kl:
                    print(f"    KL stop ep{ep}: {kl:.4f}>{args.target_kl}")
                    brk = True; break
            if brk: break

        actor_sched.step(); critic_sched.step()
        vy = np.var(rf.numpy())
        evr = np.nan if vy == 0 else 1 - np.var(rf.numpy() - vf.numpy()) / vy
        sps = int(S * ACT_H * N * (itr - start_itr + 1) / max(time.time() - t_start, 1))
        print(f"  pg={pg.item():.4f} v={vl.item():.4f} kl={kl:.4f} "
              f"clip={np.mean(cfs):.3f} ev={evr:.3f} SPS={sps}")

        if (itr + 1) % args.save_interval == 0 or itr == args.total_iters - 1:
            torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
                "critic_state_dict": dppo.critic.state_dict(),
                "normalizer_state_dict": dppo.normalizer.state_dict(),
                "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                "bc_config": bc_cfg, "iteration": itr + 1,
                "args": vars(args)}, save_dir / f"dppo_iter{itr+1}.pt")
            print(f"  Saved iter {itr+1}")

    print(f"\nDone in {time.time()-t_start:.0f}s | Best SR={best_sr:.2%} L={best_sl}")
    torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
        "critic_state_dict": dppo.critic.state_dict(),
        "normalizer_state_dict": dppo.normalizer.state_dict(),
        "bc_config": bc_cfg, "iteration": args.total_iters,
        "best_success_rate": best_sr, "args": vars(args)}, save_dir / "dppo_final.pt")
    env.env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
