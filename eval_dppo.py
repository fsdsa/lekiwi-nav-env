#!/usr/bin/env python3
"""
DPPO 체크포인트 평가 스크립트.

fine-tuned UNet을 로드해서 BC warmup + DPPO policy로 GUI 평가.
ResiP의 eval_resip.py와 동일한 환경 설정.

Usage:
    python eval_dppo.py \
        --skill approach_and_grasp \
        --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --dppo_checkpoint checkpoints/dppo_v1/dppo_best.pt \
        --object_usd /path/to/object.usd \
        --num_episodes 20
"""
from __future__ import annotations
import argparse, os

parser = argparse.ArgumentParser(description="DPPO Eval (GUI)")
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp"])
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--dppo_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--warmup_steps", type=int, default=660)
parser.add_argument("--lift_threshold", type=int, default=100)
parser.add_argument("--ddim_steps", type=int, default=4)
parser.add_argument("--ft_denoising_steps", type=int, default=4)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import copy, math, torch
import numpy as np
from isaaclab.utils.math import quat_apply
from diffusion_policy import DiffusionPolicyAgent, LinearNormalizer

# Import DPPO model classes (no Isaac Lab dependency)
from dppo_model import DPPODiffusion

from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg, EE_LOCAL_OFFSET

# ── Env setup ──
env_cfg = Skill2EnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.enable_domain_randomization = False
env_cfg.arm_limit_write_to_sim = False
env_cfg.grasp_contact_threshold = 0.55
env_cfg.grasp_gripper_threshold = 0.65
env_cfg.grasp_max_object_dist = 0.50
env_cfg.episode_length_s = 300.0
env_cfg.spawn_heading_noise_std = 0.3
env_cfg.spawn_heading_max_rad = 0.5
env_cfg.grasp_success_height = 0.05

if args.object_usd:
    env_cfg.object_usd = os.path.expanduser(args.object_usd)
env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
    env_cfg.arm_limit_json = args.arm_limit_json

env = Skill2Env(cfg=env_cfg)
device = env.device

# ── Load BC checkpoint ──
bc_ck = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
bc_cfg = bc_ck["config"]
bc_agent = DiffusionPolicyAgent(
    obs_dim=bc_cfg["obs_dim"], act_dim=bc_cfg["act_dim"],
    pred_horizon=bc_cfg["pred_horizon"], action_horizon=bc_cfg["action_horizon"],
    num_diffusion_iters=bc_cfg["num_diffusion_iters"],
    inference_steps=bc_cfg.get("inference_steps", 16),
    down_dims=bc_cfg.get("down_dims", [256, 512, 1024]),
).to(device)
sd = bc_ck["model_state_dict"]
bc_agent.model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
bc_agent.normalizer.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("normalizer.")}, device=device)

OD, AD = bc_cfg["obs_dim"], bc_cfg["act_dim"]
PRED_H = bc_cfg["pred_horizon"]
ACT_H = bc_cfg["action_horizon"]

# ── Create DPPO model and load fine-tuned weights ──
dppo = DPPODiffusion(
    unet_pretrained=bc_agent.model,
    normalizer=bc_agent.normalizer,
    obs_dim=OD, act_dim=AD,
    pred_horizon=PRED_H, act_steps=ACT_H,
    denoising_steps=bc_cfg["num_diffusion_iters"],
    ddim_steps=args.ddim_steps,
    ft_denoising_steps=args.ft_denoising_steps,
    device=str(device),
).to(device)

dppo_ck = torch.load(args.dppo_checkpoint, map_location=device, weights_only=False)
dppo.actor_ft.load_state_dict(dppo_ck["actor_ft_state_dict"])
if "critic_state_dict" in dppo_ck:
    dppo.critic.load_state_dict(dppo_ck["critic_state_dict"])
dppo.eval()
print(f"DPPO checkpoint: {args.dppo_checkpoint} (iter={dppo_ck.get('iteration','?')})")

# ── Helpers ──
fixed_jaw_idx, _ = env.robot.find_bodies(["Wrist_Roll_08c_v1"])
ee_local_offset = torch.tensor(EE_LOCAL_OFFSET, device=device).unsqueeze(0)

def get_distances():
    wp = env.robot.data.body_pos_w[:, fixed_jaw_idx[0], :]
    wq = env.robot.data.body_quat_w[:, fixed_jaw_idx[0], :]
    ee = wp + quat_apply(wq, ee_local_offset)
    ee_d = torch.norm(ee - env.object_pos_w, dim=-1).item()
    base_d = torch.norm(env.robot.data.root_pos_w[:, :2] - env.object_pos_w[:, :2], dim=-1).item()
    grip = env.robot.data.joint_pos[:, env.gripper_idx].item()
    env_z = env.scene.env_origins[:, 2].item()
    obj_z = env.object_pos_w[:, 2].item() - env_z
    return ee_d, base_d, grip, obj_z

LMI = args.lift_threshold
WU = args.warmup_steps
WU_CALLS = WU // ACT_H

print(f"\n{'='*60}")
print(f"  DPPO Eval — {args.skill}")
print(f"  warmup={WU} ({WU_CALLS} calls), lift_threshold={LMI}")
print(f"  DDIM={args.ddim_steps}, ft_steps={args.ft_denoising_steps}")
print(f"{'='*60}\n")

# ── Eval loop ──
episode = 0
successes = 0
step_count = 0
ep_grasped = False
ep_lifted = False
grasp_sustain = 0
lift_sustain = 0
ms_gr = False

obs_dict, _ = env.reset()
obs = obs_dict["policy"].to(device)

while episode < args.num_episodes and simulation_app.is_running():
    # ── BC warmup ──
    if step_count == 0 and WU_CALLS > 0:
        print(f"  Running BC warmup: {WU} steps ({WU_CALLS} calls)...")
        for wi in range(WU_CALLS):
            with torch.no_grad():
                actions_norm, _ = dppo.sample_actions(obs, deterministic=True)
                actions = dppo.normalizer(
                    actions_norm[:, :ACT_H].reshape(-1, AD), "action", forward=False
                ).reshape(1, ACT_H, AD)
            for ai in range(ACT_H):
                obs_dict, _, ter, tru, _ = env.step(actions[:, ai])
                obs = obs_dict["policy"].to(device)
                if (ter | tru).any():
                    break
        ee_d, base_d, grip, obj_z = get_distances()
        print(f"  Warmup done: EE={ee_d:.3f} Base={base_d:.3f} grip={grip:.3f} objZ={obj_z:.3f}")

    # ── DPPO action chunk ──
    with torch.no_grad():
        actions_norm, _ = dppo.sample_actions(obs, deterministic=True)
        actions = dppo.normalizer(
            actions_norm[:, :ACT_H].reshape(-1, AD), "action", forward=False
        ).reshape(1, ACT_H, AD)

    chunk_done = False
    for ai in range(ACT_H):
        obs_dict, reward, terminated, truncated, info = env.step(actions[:, ai])
        obs = obs_dict["policy"].to(device)
        step_count += 1

        # Grasp/lift tracking
        ee_d, base_d, grip, obj_z = get_distances()
        grasped = env.object_grasped[0].item()
        gc = grip < float(env.cfg.grasp_gripper_threshold)

        if grasped and not ms_gr:
            grasp_sustain += 1
        elif not grasped and not ms_gr:
            grasp_sustain = 0
        if grasp_sustain >= 5 and not ms_gr:
            ms_gr = True
            ep_grasped = True
            print(f"    ★ VERIFIED GRASP at t={step_count}")

        # Lift
        held = obj_z > 0.05 and ms_gr and gc and ee_d < 0.20
        if held:
            lift_sustain += 1
        else:
            lift_sustain = 0
        if lift_sustain >= LMI and not ep_lifted:
            ep_lifted = True
            print(f"    ★ LIFT at t={step_count} | objZ={obj_z:.3f} sustain={lift_sustain}")

        # Drop
        if ms_gr and not grasped:
            print(f"    ✗ DROP at t={step_count}")
            ms_gr = False
            grasp_sustain = 0
            lift_sustain = 0

        if step_count % 50 == 0:
            status = ""
            if ep_lifted: status = f" [LIFTED sus={lift_sustain}]"
            elif ms_gr: status = f" [GRASPED l_sus={lift_sustain}]"
            print(f"    [t={step_count:4d}] EE={ee_d:.3f} Base={base_d:.3f} "
                  f"grip={grip:.3f} objZ={obj_z:.3f}{status}")

        done = terminated.any() or truncated.any()
        if done:
            chunk_done = True
            break

    if chunk_done or (terminated.any() or truncated.any()):
        episode += 1
        if ep_lifted:
            successes += 1
        status = "SUCCESS" if ep_lifted else "FAIL"
        grasp_str = f"grasp={'Y' if ep_grasped else 'N'} lift={'Y' if ep_lifted else 'N'}"
        print(f"  Episode {episode}/{args.num_episodes}: {status} "
              f"({step_count} steps, {grasp_str} | "
              f"cumulative: {successes}/{episode} = {successes/episode*100:.0f}%)")

        # Reset
        step_count = 0
        ep_grasped = False
        ep_lifted = False
        grasp_sustain = 0
        lift_sustain = 0
        ms_gr = False

        obs_dict, _ = env.reset()
        obs = obs_dict["policy"].to(device)

print(f"\n  === 결과: {successes}/{args.num_episodes} 성공 "
      f"({successes/max(episode,1)*100:.0f}%) ===")
print(f"  (LMI={LMI}, warmup={WU})\n")

env.close()
simulation_app.close()
