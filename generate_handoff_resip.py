#!/usr/bin/env python3
"""
ResiP (DP + Residual) 기반 Handoff Buffer 생성.

Skill-2 ResiP 체크포인트로 approach_and_grasp를 실행하여
성공적으로 물체를 lift한 상태를 수집 → Skill-3 초기 상태로 사용.

Usage:
    python generate_handoff_resip.py \
        --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --resip_checkpoint checkpoints/resip_v66c/resip_best.pt \
        --object_usd /path/to/object.usd \
        --num_entries 500 --num_envs 64 \
        --output handoff_buffer.pkl \
        --headless
"""
from __future__ import annotations

import argparse
import os
import pickle

parser = argparse.ArgumentParser(description="Generate Handoff Buffer (ResiP)")
parser.add_argument("--dp_checkpoint", type=str, required=True)
parser.add_argument("--resip_checkpoint", type=str, default="")
parser.add_argument("--num_entries", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--output", type=str, default="handoff_buffer.pkl")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--inference_steps", type=int, default=4)
parser.add_argument("--warmup_steps", type=int, default=660)
parser.add_argument("--lift_height", type=float, default=0.05,
                    help="Minimum object height to consider as lifted")
parser.add_argument("--lift_sustain", type=int, default=50,
                    help="Steps object must be held above lift_height")
# Action scale
parser.add_argument("--action_scale_arm", type=float, default=0.20)
parser.add_argument("--action_scale_gripper", type=float, default=0.30)
parser.add_argument("--action_scale_base", type=float, default=0.35)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg


def main():
    # ── Env ──
    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.grasp_contact_threshold = 0.55
    cfg.grasp_gripper_threshold = 0.65
    cfg.grasp_max_object_dist = 0.50
    cfg.episode_length_s = 300.0
    cfg.spawn_heading_noise_std = 0.3
    cfg.spawn_heading_max_rad = 0.5
    cfg.lift_success_sustain_steps = 0
    cfg.grasp_success_height = 1.00
    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.multi_object_json:
        cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    env = Skill2Env(cfg=cfg)
    device = env.device
    N = env.num_envs

    # ── Models ──
    ck = torch.load(args.dp_checkpoint, map_location=device, weights_only=False)
    c = ck["config"]
    dp = DiffusionPolicyAgent(
        obs_dim=c["obs_dim"], act_dim=c["act_dim"],
        pred_horizon=c["pred_horizon"], action_horizon=c["action_horizon"],
        num_diffusion_iters=c["num_diffusion_iters"],
        inference_steps=args.inference_steps,
        down_dims=c.get("down_dims", [256, 512, 1024]),
    ).to(device)
    sd = ck["model_state_dict"]
    dp.model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    dp.normalizer.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("normalizer.")})
    for p in dp.parameters():
        p.requires_grad = False
    dp.eval()
    OD, AD = c["obs_dim"], c["act_dim"]
    print(f"Frozen DP: obs={OD}, act={AD}")

    rpol = None
    if args.resip_checkpoint and os.path.isfile(args.resip_checkpoint):
        rck = torch.load(args.resip_checkpoint, map_location=device, weights_only=False)
        sa = rck.get("args", {})
        rpol = ResidualPolicy(
            obs_dim=OD, action_dim=AD,
            actor_hidden_size=sa.get("actor_hidden_size", 256),
            actor_num_layers=sa.get("actor_num_layers", 2),
            critic_hidden_size=sa.get("critic_hidden_size", 256),
            critic_num_layers=sa.get("critic_num_layers", 2),
            action_scale=sa.get("action_scale", 0.1),
            init_logstd=sa.get("init_logstd", -1.0),
            action_head_std=sa.get("action_head_std", 0.0),
            learn_std=False,
        ).to(device)
        rpol.load_state_dict(rck["residual_policy_state_dict"])
        rpol.eval()
        print(f"Residual: {args.resip_checkpoint} (iter={rck.get('iteration','?')})")

    scale = torch.zeros(AD, device=device)
    scale[0:5] = args.action_scale_arm
    scale[5] = args.action_scale_gripper
    scale[6:9] = args.action_scale_base

    # ── Collect ──
    entries = []
    lift_counter = torch.zeros(N, dtype=torch.long, device=device)
    collected_this_ep = torch.zeros(N, dtype=torch.bool, device=device)
    total_steps = 0

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    dp.reset()

    # Verify env_origins are non-zero (origin subtraction depends on this)
    _eo = env.scene.env_origins
    if _eo.abs().sum() < 0.01:
        print("  [WARN] env_origins are all zeros! Buffer will contain absolute coordinates.")
    else:
        print(f"  env_origins range: x[{_eo[:,0].min():.1f},{_eo[:,0].max():.1f}] y[{_eo[:,1].min():.1f},{_eo[:,1].max():.1f}]")

    # Warmup
    if args.warmup_steps > 0:
        print(f"  BC warmup: {args.warmup_steps} steps...")
        for wi in range(args.warmup_steps):
            with torch.no_grad():
                a = dp.normalizer(dp.base_action_normalized(obs), "action", forward=False)
            obs_dict, _, ter, tru, _ = env.step(a)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            if (ter | tru).any():
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
                dp.reset()

    print(f"\n  Collecting {args.num_entries} handoff entries (N={N})...\n")

    while len(entries) < args.num_entries and simulation_app.is_running():
        with torch.no_grad():
            ba = dp.base_action_normalized(obs)
            if rpol is not None:
                no = torch.nan_to_num(torch.clamp(
                    dp.normalizer(obs, "obs", forward=True), -3, 3), nan=0.0)
                ro = torch.cat([no, ba], dim=-1)
                ra = rpol.actor_mean(ro)
                ra = torch.clamp(ra, -1.0, 1.0)
                action = dp.normalizer(ba + ra * scale, "action", forward=False)
            else:
                action = dp.normalizer(ba, "action", forward=False)

        obs_dict, _, ter, tru, info = env.step(action)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        total_steps += 1

        # Check lift state
        obj_z = env.object_pos_w[:, 2] - env.scene.env_origins[:, 2]
        grasped = env.object_grasped
        grip = env.robot.data.joint_pos[:, env.gripper_idx]
        grip_closed = grip < float(env.cfg.grasp_gripper_threshold)

        held = (obj_z > args.lift_height) & grasped & grip_closed
        lift_counter[held] += 1
        lift_counter[~held] = 0

        # Collect entries for envs that sustained lift
        ready = (lift_counter >= args.lift_sustain) & (~collected_this_ep)
        if ready.any():
            rids = ready.nonzero(as_tuple=False).squeeze(-1)
            origins = env.scene.env_origins[rids]
            for rid in rids:
                i = rid.item()
                oi = int(env.active_object_idx[i].item())
                if env._multi_object and oi < len(env.object_rigids):
                    obj_quat = env.object_rigids[oi].data.root_quat_w[i].cpu().tolist()
                elif env.object_rigid is not None:
                    obj_quat = env.object_rigid.data.root_quat_w[i].cpu().tolist()
                else:
                    obj_quat = [1.0, 0.0, 0.0, 0.0]

                origin = env.scene.env_origins[i]
                entry = {
                    "base_pos": (env.robot.data.root_pos_w[i] - origin).cpu().tolist(),
                    "base_ori": env.robot.data.root_quat_w[i].cpu().tolist(),
                    "arm_joints": env.robot.data.joint_pos[i, env.arm_idx[:5]].cpu().tolist(),
                    "gripper_state": env.robot.data.joint_pos[i, env.arm_idx[5]].item(),
                    "object_pos": (env.object_pos_w[i] - origin).cpu().tolist(),
                    "object_ori": obj_quat,
                    "object_type_idx": env.active_object_idx[i].item(),
                }
                entries.append(entry)
                collected_this_ep[i] = True

            prev_count = len(entries) - rids.numel()
            if len(entries) // 50 > prev_count // 50:
                print(f"    {len(entries)}/{args.num_entries} ({total_steps} steps)")

        # Reset done envs
        done = (ter | tru).view(-1).bool()
        if done.any():
            lift_counter[done] = 0
            collected_this_ep[done] = False
            # env auto-resets, just update obs
            dp.reset()

    entries = entries[:args.num_entries]
    with open(args.output, "wb") as f:
        pickle.dump(entries, f)
    print(f"\n  Saved {len(entries)} entries to {args.output} ({total_steps} total steps)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
