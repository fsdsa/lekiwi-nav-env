#!/usr/bin/env python3
"""Carry BC eval — S2 expert lift → carry BC with 39D obs.

Usage:
    python eval_carry.py \
        --carry_bc_checkpoint checkpoints/dp_bc_carry_v4/dp_bc_epoch300.pt \
        --s2_bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --s2_resip_checkpoint backup/appoachandlift/resip64%.pt \
        --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \
        --num_episodes 6
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--carry_bc_checkpoint", type=str, required=True)
parser.add_argument("--carry_resip_checkpoint", type=str, default="",
                    help="Carry ResiP checkpoint (optional, BC only if omitted)")
parser.add_argument("--s2_bc_checkpoint", type=str, required=True)
parser.add_argument("--s2_resip_checkpoint", type=str, default="")
parser.add_argument("--object_usd", type=str, required=True)
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--object_scale_phys", type=float, default=0.7)
parser.add_argument("--num_episodes", type=int, default=6)
parser.add_argument("--carry_steps", type=int, default=200)
parser.add_argument("--s2_max_steps", type=int, default=800)
parser.add_argument("--s2_lift_hold", type=int, default=200)
parser.add_argument("--inference_steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import torch, numpy as np
from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg

# ── Env ──
cfg = Skill2EnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
cfg.enable_domain_randomization = False
cfg.arm_limit_write_to_sim = False
cfg.episode_length_s = 3600.0
cfg.max_dist_from_origin = 50.0
cfg.dr_action_delay_steps = 0
cfg.object_usd = os.path.expanduser(args.object_usd)
cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
cfg.object_scale = args.object_scale_phys
cfg.grasp_success_height = 100.0

env = Skill2Env(cfg=cfg)
dev = env.device

# ── S2 Expert ──
s2_ckpt = torch.load(args.s2_bc_checkpoint, map_location=dev, weights_only=False)
s2_cfg = s2_ckpt["config"]
s2_dp = DiffusionPolicyAgent(
    obs_dim=s2_cfg["obs_dim"], act_dim=s2_cfg["act_dim"],
    pred_horizon=s2_cfg["pred_horizon"],
    action_horizon=s2_cfg["action_horizon"],
    num_diffusion_iters=s2_cfg["num_diffusion_iters"],
    inference_steps=4,
    down_dims=s2_cfg.get("down_dims", [64, 128, 256]),
).to(dev)
sd = s2_ckpt["model_state_dict"]
s2_dp.model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
s2_dp.normalizer.load_state_dict(
    {k[len("normalizer."):]: v for k, v in sd.items() if k.startswith("normalizer.")}, device=dev)
s2_dp.eval()
for p in s2_dp.parameters():
    p.requires_grad = False

s2_resip = None
if args.s2_resip_checkpoint:
    rp_ckpt = torch.load(args.s2_resip_checkpoint, map_location=dev, weights_only=False)
    s2_resip = ResidualPolicy(
        obs_dim=s2_cfg["obs_dim"], action_dim=s2_cfg["act_dim"],
        action_scale=0.1, learn_std=True,
    ).to(dev)
    s2_resip.load_state_dict(rp_ckpt["residual_policy_state_dict"])
    s2_resip.eval()
    s2_per_dim = torch.zeros(s2_cfg["act_dim"], device=dev)
    s2_per_dim[0:5] = 0.20; s2_per_dim[5] = 0.25; s2_per_dim[6:9] = 0.35

# ── Carry BC ──
carry_ckpt = torch.load(args.carry_bc_checkpoint, map_location=dev, weights_only=False)
carry_cfg = carry_ckpt["config"]
carry_dp = DiffusionPolicyAgent(
    obs_dim=carry_cfg["obs_dim"], act_dim=carry_cfg["act_dim"],
    pred_horizon=carry_cfg["pred_horizon"],
    action_horizon=carry_cfg["action_horizon"],
    num_diffusion_iters=carry_cfg["num_diffusion_iters"],
    inference_steps=args.inference_steps,
    down_dims=carry_cfg.get("down_dims", [64, 128, 256]),
).to(dev)
sd = carry_ckpt["model_state_dict"]
carry_dp.model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
carry_dp.normalizer.load_state_dict(
    {k[len("normalizer."):]: v for k, v in sd.items() if k.startswith("normalizer.")}, device=dev)
carry_dp.eval()
for p in carry_dp.parameters():
    p.requires_grad = False

# ── Carry ResiP (optional) ──
carry_resip = None
carry_per_dim = None
if args.carry_resip_checkpoint:
    rp_ckpt = torch.load(args.carry_resip_checkpoint, map_location=dev, weights_only=False)
    carry_resip = ResidualPolicy(
        obs_dim=carry_cfg["obs_dim"], action_dim=carry_cfg["act_dim"],
        action_scale=0.1, learn_std=True,
    ).to(dev)
    carry_resip.load_state_dict(rp_ckpt["residual_policy_state_dict"])
    carry_resip.eval()
    for p in carry_resip.parameters():
        p.requires_grad = False
    carry_per_dim = torch.zeros(carry_cfg["act_dim"], device=dev)
    carry_per_dim[0:5] = 0.05; carry_per_dim[5] = 0.05  # arm only, base=0
    print(f"  Carry ResiP: {args.carry_resip_checkpoint}")

mode_str = "BC+RL" if carry_resip else "BC only"
print(f"\n  Carry Eval ({mode_str})")
print(f"  S2: {args.s2_bc_checkpoint}")
print(f"  Carry BC: {args.carry_bc_checkpoint} (obs={carry_cfg['obs_dim']}D)")
print(f"  Episodes: {args.num_episodes}\n")

# ── Directions ──
DIRECTIONS = {
    "FORWARD":      (0.0, 0.7, 0.0),
    "BACKWARD":     (0.0, -0.7, 0.0),
    "STRAFE LEFT":  (-0.7, 0.0, 0.0),
    "STRAFE RIGHT": (0.7, 0.0, 0.0),
    "TURN LEFT":    (0.0, 0.0, -0.33),
    "TURN RIGHT":   (0.0, 0.0, 0.33),
}
DIR_CMD = {
    "FORWARD":      [0.0, 1.0, 0.0],
    "BACKWARD":     [0.0, -1.0, 0.0],
    "STRAFE LEFT":  [-1.0, 0.0, 0.0],
    "STRAFE RIGHT": [1.0, 0.0, 0.0],
    "TURN LEFT":    [0.0, 0.0, 1.0],
    "TURN RIGHT":   [0.0, 0.0, -1.0],
}
dir_schedule = list(DIRECTIONS.keys())

# ── arm_action_to_limits mapping ──
override = getattr(env, "_arm_action_limits_override", None)
if override is not None:
    lim = override[0].detach().cpu().numpy()
else:
    lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx].detach().cpu().numpy()
arm_center = 0.5 * (lim[:, 0] + lim[:, 1])
arm_half_range = 0.5 * (lim[:, 1] - lim[:, 0])
arm_half_range = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)

# ── Helpers ──
def get_s2_action(obs_30d):
    with torch.no_grad():
        base_nact = s2_dp.base_action_normalized(obs_30d)
        if s2_resip is not None:
            nobs = s2_dp.normalizer(obs_30d, "obs", forward=True).clamp(-3, 3)
            nobs = torch.nan_to_num(nobs, nan=0.0)
            ri = torch.cat([nobs, base_nact], dim=-1)
            _, _, _, _, ra_mean = s2_resip.get_action_and_value(ri)
            nact = base_nact + ra_mean * s2_per_dim
        else:
            nact = base_nact
        action = s2_dp.normalizer(nact, "action", forward=False)
    return action.clamp(-1, 1)

# ── Main loop ──
for ep in range(args.num_episodes):
    label = dir_schedule[ep % len(dir_schedule)]
    dir_cmd_np = np.array(DIR_CMD[label], dtype=np.float32)
    bvx, bvy, bwz = DIRECTIONS[label]

    print(f"  [Episode {ep+1}/{args.num_episodes}] {label}")

    # Phase 1: S2 expert lift (최대 5회 retry)
    obs, _ = env.reset()
    s2_dp.reset()
    lift_counter = 0
    max_retries = 5
    for attempt in range(max_retries):
        s2_step = 0
        lifted = False
        for s2_step in range(args.s2_max_steps):
            obs_30d = obs["policy"] if isinstance(obs, dict) else obs
            action = get_s2_action(obs_30d)
            obs, _, ter, tru, _ = env.step(action)

            grip_pos = env.robot.data.joint_pos[0, env.arm_idx[5]].item()
            grip_closed = grip_pos < float(env.cfg.grasp_gripper_threshold)
            has_contact = False
            if env.contact_sensor is not None:
                cf = env._contact_force_per_env()[0].item()
                has_contact = cf > float(env.cfg.grasp_contact_threshold)
            objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()

            if grip_closed and has_contact and objZ > 0.05:
                lift_counter += 1
            else:
                lift_counter = 0

            # Topple
            if objZ < 0.026 and s2_step > 20:
                print(f"    [S2 FAIL] objZ={objZ:.3f} step={s2_step} — retry ({attempt+1}/{max_retries})")
                obs, _ = env.reset()
                s2_dp.reset()
                lift_counter = 0
                break

            if s2_step % 100 == 0:
                print(f"    [S2] step={s2_step} objZ={objZ:.3f} grip={grip_pos:.3f} lift={lift_counter}/{args.s2_lift_hold}")

            if lift_counter >= args.s2_lift_hold:
                lifted = True
                break

        if lifted:
            break

    if not lifted:
        print(f"    [S2 FAIL] Could not lift after {max_retries} retries — skip")
        continue

    # Capture init_arm_pose
    jp = env.robot.data.joint_pos[0]
    init_arm = jp[env.arm_idx[:5]].cpu().numpy().astype(np.float32)
    init_grip = np.array([jp[env.arm_idx[5]].item()], dtype=np.float32)
    init_arm_pose = np.concatenate([init_arm, init_grip])
    print(f"    [S2→Carry] arm={[f'{v:+.3f}' for v in init_arm]} grip={init_grip[0]:.3f}")

    # Phase 2: Carry BC
    carry_dp.reset()
    lin_errors, ang_errors = [], []

    for step in range(args.carry_steps):
        obs_30d = obs["policy"] if isinstance(obs, dict) else obs
        obs_39 = torch.cat([
            obs_30d[0],
            torch.tensor(dir_cmd_np, device=dev),
            torch.tensor(init_arm_pose, device=dev),
        ]).unsqueeze(0)  # (1, 39)

        with torch.no_grad():
            base_nact = carry_dp.base_action_normalized(obs_39)
            if carry_resip is not None:
                nobs = carry_dp.normalizer(obs_39, "obs", forward=True).clamp(-3, 3)
                nobs = torch.nan_to_num(nobs, nan=0.0)
                ri = torch.cat([nobs, base_nact], dim=-1)
                _, _, _, _, ra_mean = carry_resip.get_action_and_value(ri)
                nact = base_nact + ra_mean * carry_per_dim
            else:
                nact = base_nact
            action = carry_dp.normalizer(nact, "action", forward=False)
        action = action.clamp(-1, 1)

        obs, _, ter, tru, _ = env.step(action)

        # Metrics — target = action * max_vel
        target_vx = bvx * env.cfg.max_lin_vel
        target_vy = bvy * env.cfg.max_lin_vel
        target_wz = bwz * env.cfg.max_ang_vel
        actual_vx = env.robot.data.root_lin_vel_b[0, 0].item()
        actual_vy = env.robot.data.root_lin_vel_b[0, 1].item()
        actual_wz = env.robot.data.root_ang_vel_b[0, 2].item()
        lin_err = ((target_vx - actual_vx)**2 + (target_vy - actual_vy)**2)**0.5
        ang_err = abs(target_wz - actual_wz)
        lin_errors.append(lin_err)
        ang_errors.append(ang_err)

        objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
        jp = env.robot.data.joint_pos[0]
        arm_jp = jp[env.arm_idx[:5]].cpu().tolist()
        grip_jp = jp[env.arm_idx[5]].item()

        if step % 50 == 0:
            arm_str = ",".join(f"{v:+.3f}" for v in arm_jp)
            print(f"    [t={step:4d}] {label} | vel=({actual_vx:+.2f},{actual_vy:+.2f},{actual_wz:+.2f}) | "
                  f"objZ={objZ:.3f} | arm=[{arm_str}] grip={grip_jp:+.3f} | "
                  f"lin_err={np.mean(lin_errors[-50:]):.3f}", flush=True)

        if objZ < 0.03 and step > 10:
            print(f"    [DROP] objZ={objZ:.3f} at step={step}")
            break

    avg_lin = np.mean(lin_errors) if lin_errors else 0
    avg_ang = np.mean(ang_errors) if ang_errors else 0
    print(f"  Episode {ep+1}: {label} | {len(lin_errors)} steps | lin_err={avg_lin:.4f} ang_err={avg_ang:.4f}\n")

print("Done.")
sim_app.close()
