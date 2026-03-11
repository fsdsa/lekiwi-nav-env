#!/usr/bin/env python3
"""
DPPO Skill-3 (CarryAndPlace) 평가 스크립트.

BC eval 스크립트(eval_dp_bc.py)의 초기상태 복원 로직 + DPPODiffusion 추론.

초기상태 복원 방식:
  1) --demo HDF5에서 로봇/물체 초기 상태 읽기
  2) gripper를 240 step에 걸쳐 점진적으로 닫으면서 매 step 물체를 EE에 텔레포트
  3) 120 step 자유 settle (마찰만으로 유지)
  → handoff_buffer 방식의 문제 (friction만으로 hold 불가) 우회

추론:
  DPPODiffusion.sample_actions() → deterministic DDIM → denormalize → env.step()

Usage:
    # BC (diffusion policy) 평가
    python eval_dppo_skill3.py \\
      --dp_checkpoint checkpoints/dp_bc_skill3_aug/dp_bc_epoch200.pt \\
      --mode bc \\
      --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \\
      --demo demos_skill3/combined_skill3_20260227_091123.hdf5 \\
      --num_episodes 10

    # DPPO fine-tuned 평가
    python eval_dppo_skill3.py \\
      --dp_checkpoint checkpoints/dp_bc_skill3_aug/dp_bc_epoch200.pt \\
      --dppo_checkpoint checkpoints/dppo_skill3/dppo_best.pt \\
      --mode dppo \\
      --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \\
      --demo demos_skill3/combined_skill3_20260227_091123.hdf5 \\
      --num_episodes 10
"""
from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="DPPO / BC Eval — Skill-3 CarryAndPlace")

# Model
parser.add_argument("--dp_checkpoint", type=str, required=True,
                    help="BC Diffusion Policy checkpoint (always needed for normalizer)")
parser.add_argument("--dppo_checkpoint", type=str, default="",
                    help="DPPO fine-tuned checkpoint (actor_ft weights)")
parser.add_argument("--mode", type=str, default="auto", choices=["auto", "bc", "dppo"],
                    help="auto: dppo if --dppo_checkpoint given, else bc")
parser.add_argument("--inference_steps", type=int, default=16,
                    help="DDIM inference steps for BC mode")
parser.add_argument("--ddim_steps", type=int, default=4,
                    help="DDIM steps for DPPO mode")
parser.add_argument("--ft_denoising_steps", type=int, default=4,
                    help="Fine-tuned denoising steps for DPPO")

# Env
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--demo", type=str, default="",
                    help="HDF5 demo file for initial state restoration")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")

# DPPO model params
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

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py
import torch
import numpy as np

from diffusion_policy import DiffusionPolicyAgent
from dppo_model import DPPODiffusion


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_bc_agent(path, device, inference_steps=16):
    """Load BC DiffusionPolicyAgent (for BC mode or as DPPO base)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"], action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=inference_steps,
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)
    sd = ck["model_state_dict"]
    agent.model.load_state_dict(
        {k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    agent.normalizer.load_state_dict(
        {k[11:]: v for k, v in sd.items() if k.startswith("normalizer.")},
        device=device)
    agent.eval()
    print(f"  BC loaded: obs={cfg['obs_dim']} act={cfg['act_dim']} "
          f"pred_h={cfg['pred_horizon']} act_h={cfg['action_horizon']}")
    return agent, cfg


def load_dppo_model(bc_path, dppo_path, device, args_):
    """Load DPPO model: BC UNet + fine-tuned actor_ft + normalizer."""
    # Load BC for UNet + normalizer
    bc_agent, bc_cfg = load_bc_agent(bc_path, device)

    dppo = DPPODiffusion(
        unet_pretrained=bc_agent.model,
        normalizer=bc_agent.normalizer,
        obs_dim=bc_cfg["obs_dim"], act_dim=bc_cfg["act_dim"],
        pred_horizon=bc_cfg["pred_horizon"],
        act_steps=bc_cfg["action_horizon"],
        denoising_steps=bc_cfg["num_diffusion_iters"],
        ddim_steps=args_.ddim_steps,
        ft_denoising_steps=args_.ft_denoising_steps,
        min_sampling_denoising_std=args_.min_sampling_std,
        min_logprob_denoising_std=args_.min_logprob_std,
        gamma_denoising=args_.gamma_denoising,
        clip_ploss_coef=args_.clip_ploss_coef,
        clip_ploss_coef_base=args_.clip_ploss_coef_base,
        clip_ploss_coef_rate=args_.clip_ploss_coef_rate,
        denoised_clip_value=args_.denoised_clip_value,
        randn_clip_value=args_.randn_clip_value,
        final_action_clip_value=args_.final_action_clip_value,
        eta=args_.eta,
        device=str(device),
    ).to(device)

    # Load fine-tuned actor weights
    ck = torch.load(dppo_path, map_location=device, weights_only=False)
    dppo.actor_ft.load_state_dict(ck["actor_ft_state_dict"])
    if "normalizer_state_dict" in ck:
        dppo.normalizer.load_state_dict(ck["normalizer_state_dict"], device=device)
    dppo.eval()

    itr = ck.get("iteration", "?")
    sr = ck.get("success_rate", ck.get("best_success_rate", "?"))
    print(f"  DPPO loaded: iter={itr} SR={sr}")
    print(f"    ddim={args_.ddim_steps} ft_steps={args_.ft_denoising_steps}")
    return dppo, bc_cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  Initial state restoration (from eval_dp_bc.py)
# ═══════════════════════════════════════════════════════════════════════════════

def restore_init_state_s3(env, ep_data):
    """HDF5 에피소드 초기 상태로 env 복원.

    1. Robot base → HDF5 위치
    2. Arm → S3 시작 자세, gripper open
    3. Object → EE 위치에 배치
    4. 240 step: gripper 점진 닫기 + 매 step 물체 EE 텔레포트
    5. 120 step: 자유 settle (friction grasp 확인)
    """
    from isaaclab.utils.math import quat_apply, quat_mul

    device = env.device
    env_id = torch.tensor([0], device=device)
    ea = ep_data["ep_attrs"]

    # 1. Robot base 위치
    if "robot_init_pos" in ea and "robot_init_quat" in ea:
        rs = env.robot.data.root_state_w.clone()
        rs[0, 0:3] = torch.tensor(ea["robot_init_pos"], dtype=torch.float32, device=device)
        rs[0, 3:7] = torch.tensor(ea["robot_init_quat"], dtype=torch.float32, device=device)
        rs[0, 7:] = 0.0
        env.robot.write_root_state_to_sim(rs, env_id)
        env.home_pos_w[0] = rs[0, :3]

    init_joints = torch.tensor(ep_data["obs"][0, 0:6], dtype=torch.float32, device=device)
    target_grip = init_joints[5].item()

    # 2. Arm pose + gripper open
    jp = env.robot.data.default_joint_pos[0:1].clone()
    jp[0, env.arm_idx[:5]] = init_joints[:5]
    jp[0, env.gripper_idx] = 1.4  # max open
    jp[0, env.wheel_idx] = 0.0
    jv = torch.zeros_like(jp)
    env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_id)
    env.robot.set_joint_position_target(jp, env_ids=env_id)
    vel_target = torch.zeros(1, env.robot.num_joints, device=device)
    env.robot.set_joint_velocity_target(vel_target, env_ids=env_id)

    for _ in range(10):
        env.robot.write_data_to_sim()
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)

    # 3. Object at EE
    wrist_pos = env.robot.data.body_pos_w[0, env._fixed_jaw_body_idx, :]
    wrist_quat = env.robot.data.body_quat_w[0, env._fixed_jaw_body_idx, :]
    rot90_local = torch.tensor([0.8192, -0.5736, 0.0, 0.0],
                               dtype=torch.float32, device=device)
    obj_quat = quat_mul(wrist_quat.unsqueeze(0), rot90_local.unsqueeze(0))[0]
    ee_pos = wrist_pos + quat_apply(wrist_quat.unsqueeze(0), env._ee_local_offset)[0]

    obj_bbox = env.object_bbox[0]
    bbox_center_local = torch.tensor([0.0, 0.0, obj_bbox[2].item() / 2.0],
                                     dtype=torch.float32, device=device)
    bbox_center_world = quat_apply(obj_quat.unsqueeze(0),
                                   bbox_center_local.unsqueeze(0))[0]
    obj_root_pos = ee_pos - bbox_center_world

    obj_state = env.object_rigid.data.root_state_w.clone()
    obj_state[0, 0:3] = obj_root_pos
    obj_state[0, 3:7] = obj_quat
    obj_state[0, 7:] = 0.0
    env.object_rigid.write_root_state_to_sim(obj_state, env_id)
    env.object_pos_w[0] = ee_pos

    # 4. Gradual gripper close (240 steps) with object teleport
    grasp_grip = 0.45
    n_close = 240
    for i in range(n_close):
        t_frac = (i + 1) / n_close
        grip_val = target_grip + (grasp_grip - target_grip) * t_frac
        grip_jp = env.robot.data.joint_pos_target[0:1].clone()
        grip_jp[0, env.gripper_idx] = grip_val
        env.robot.set_joint_position_target(grip_jp, env_ids=env_id)
        env.robot.write_data_to_sim()

        w_pos = env.robot.data.body_pos_w[0, env._fixed_jaw_body_idx, :]
        w_quat = env.robot.data.body_quat_w[0, env._fixed_jaw_body_idx, :]
        cur_ee = w_pos + quat_apply(w_quat.unsqueeze(0), env._ee_local_offset)[0]
        cur_obj_quat = quat_mul(w_quat.unsqueeze(0), rot90_local.unsqueeze(0))[0]
        cur_bbox_w = quat_apply(cur_obj_quat.unsqueeze(0),
                                bbox_center_local.unsqueeze(0))[0]
        obj_st = env.object_rigid.data.root_state_w.clone()
        obj_st[0, 0:3] = cur_ee - cur_bbox_w
        obj_st[0, 3:7] = cur_obj_quat
        obj_st[0, 7:] = 0.0
        env.object_rigid.write_root_state_to_sim(obj_st, env_id)

        env.sim.step()
        env.robot.update(env.sim.cfg.dt)
        env.object_rigid.update(env.sim.cfg.dt)

    # 5. Free settle
    for _ in range(120):
        env.robot.write_data_to_sim()
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)

    grip_sim = env.robot.data.joint_pos[0, env.gripper_idx].item()
    obj_z = env.object_rigid.data.root_pos_w[0, 2].item()
    print(f"    [restore] grip={grip_sim:.3f} obj_z={obj_z:.3f}")

    # Internal state
    env.object_grasped[0] = True
    env.just_dropped[0] = False
    env.intentional_placed[0] = False
    env._fallback_teleport_carry[0] = False
    env.object_pos_w[0] = env.object_rigid.data.root_pos_w[0]

    # Dest object spawn
    env._spawn_dest_object(env_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda:0"

    # ── Determine mode ──
    mode = args.mode
    if mode == "auto":
        mode = "dppo" if args.dppo_checkpoint else "bc"
    print(f"\n  Mode: {mode.upper()}")

    # ── Load demo HDF5 ──
    demo_episodes = []
    demo_file = None
    if args.demo and os.path.isfile(args.demo):
        demo_file = h5py.File(args.demo, "r")
        ep_keys = sorted(
            [k for k in demo_file.keys() if k.startswith("episode")],
            key=lambda k: int(k.split("_")[1]))
        for ek in ep_keys:
            grp = demo_file[ek]
            demo_episodes.append({
                "obs": grp["obs"][:],
                "actions": grp["actions"][:],
                "ep_attrs": dict(grp.attrs),
                "object_pos_w": grp["object_pos_w"][:] if "object_pos_w" in grp else None,
                "object_quat_w": grp["object_quat_w"][:] if "object_quat_w" in grp else None,
            })
        print(f"  Demo: {args.demo} ({len(demo_episodes)} episodes)")
        if args.num_episodes > len(demo_episodes):
            args.num_episodes = len(demo_episodes)
    else:
        print("  [WARN] No --demo provided: using env random reset "
              "(object may not be grasped at start)")

    # ── Create env ──
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    env_cfg = Skill3EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.enable_domain_randomization = False
    env_cfg.arm_limit_write_to_sim = False
    env_cfg.dr_action_delay_steps = 0
    env_cfg.grasp_contact_threshold = 0.1
    env_cfg.grasp_max_object_dist = 0.50
    env_cfg.episode_length_s = 240.0
    env_cfg.dest_spawn_dist_min = 0.6
    env_cfg.dest_spawn_dist_max = 0.7
    env_cfg.dest_spawn_min_separation = 0.3

    if demo_file is not None:
        ha = dict(demo_file.attrs)
        env_cfg.object_mass = float(ha.get("object_mass", env_cfg.object_mass))
        env_cfg.object_scale = float(ha.get("object_scale_phys", env_cfg.object_scale))
        env_cfg.arm_action_scale = float(ha.get("arm_action_scale", env_cfg.arm_action_scale))
        env_cfg.max_lin_vel = float(ha.get("max_lin_vel", env_cfg.max_lin_vel))
        env_cfg.max_ang_vel = float(ha.get("max_ang_vel", env_cfg.max_ang_vel))
        if not args.object_usd and "object_usd" in ha:
            env_cfg.object_usd = str(ha["object_usd"])

    if args.object_usd:
        env_cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.multi_object_json:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if args.dest_object_usd:
        env_cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        env_cfg.arm_limit_json = args.arm_limit_json

    env = Skill3Env(cfg=env_cfg)

    # ── Load model ──
    if mode == "dppo":
        dppo, bc_cfg = load_dppo_model(
            args.dp_checkpoint, args.dppo_checkpoint, env.device, args)
        action_horizon = bc_cfg["action_horizon"]
        obs_dim = bc_cfg["obs_dim"]
        act_dim = bc_cfg["act_dim"]
    else:
        bc_agent, bc_cfg = load_bc_agent(
            args.dp_checkpoint, env.device, args.inference_steps)
        action_horizon = bc_cfg["action_horizon"]
        obs_dim = bc_cfg["obs_dim"]
        act_dim = bc_cfg["act_dim"]

    # ── Disable env auto-termination (manual place check) ──
    _original_get_dones = env._get_dones

    def _eval_get_dones():
        terminated, truncated = _original_get_dones()
        # Keep place_grace_done but prevent premature termination
        return terminated, truncated

    # env._get_dones = _eval_get_dones  # uncomment if needed

    print(f"\n{'='*60}")
    print(f"  DPPO/BC Skill-3 Eval — CarryAndPlace")
    print(f"  Mode: {mode.upper()}")
    print(f"  obs_dim={obs_dim} act_dim={act_dim} act_h={action_horizon}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Init: {'Demo HDF5' if demo_episodes else 'Random reset'}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════
    #  Eval loop
    # ═══════════════════════════════════════════════════════════════
    episode = 0
    successes = 0
    prelim_successes = 0
    step_count = 0
    action_queue = []

    # Reset
    obs, _ = env.reset()
    if demo_episodes:
        restore_init_state_s3(env, demo_episodes[0])
        obs = env._get_observations()
        env.robot.update(env.sim.cfg.dt)
        env.object_rigid.update(env.sim.cfg.dt)

    if mode == "dppo":
        dppo.prev_naction = None
    else:
        bc_agent.reset()

    while episode < args.num_episodes and simulation_app.is_running():
        obs_t = obs["policy"].to(env.device) if isinstance(obs, dict) else obs.to(env.device)

        # ── Get action from queue or predict new chunk ──
        if len(action_queue) == 0:
            with torch.no_grad():
                if mode == "dppo":
                    # DPPO: deterministic DDIM → denormalize
                    an, _ = dppo.sample_actions(obs_t, deterministic=True)
                    act_chunk = dppo.normalizer(
                        an[:, :action_horizon].reshape(-1, act_dim),
                        "action", forward=False
                    ).reshape(1, action_horizon, act_dim)
                else:
                    # BC: base_action_normalized → denormalize (action_horizon steps)
                    base_naction = bc_agent.base_action_normalized(obs_t)
                    action = bc_agent.normalizer(base_naction, "action", forward=False)
                    act_chunk = action.unsqueeze(0)  # (1, 1, act_dim) — single step

            # Queue up actions
            if act_chunk.dim() == 3:
                for ai in range(act_chunk.shape[1]):
                    action_queue.append(act_chunk[:, ai, :])
            else:
                action_queue.append(act_chunk)

        action = action_queue.pop(0)

        # ── Status logging ──
        if step_count % 50 == 0:
            o = obs_t[0]
            grip = o[5].item()
            dest_rel = o[21:24].cpu().tolist()
            dest_d = (dest_rel[0]**2 + dest_rel[1]**2)**0.5
            grip_f = o[24].item() if obs_t.shape[1] > 24 else 0.0
            grasped = env.object_grasped[0].item()
            obj_z_abs = env.object_rigid.data.root_pos_w[0, 2].item()
            env_z = env.scene.env_origins[0, 2].item()
            obj_h = obj_z_abs - env_z
            # Object-dest XY distance
            obj_dest = torch.norm(
                env.object_pos_w[0, :2] - env.dest_object_pos_w[0, :2]).item()
            a = action[0].cpu().tolist()
            print(f"  [t={step_count:4d}] grip={grip:.3f} obj_h={obj_h:.3f} "
                  f"dest_d={dest_d:.3f} obj_dest={obj_dest:.3f} "
                  f"grip_f={grip_f:.1f} grasped={grasped} "
                  f"base=[{a[6]:.2f},{a[7]:.2f},{a[8]:.2f}] "
                  f"arm=[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f},{a[3]:.2f},{a[4]:.2f}] "
                  f"grip_act={a[5]:.2f}",
                  flush=True)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        done = terminated.any() or truncated.any()
        if done:
            episode += 1
            action_queue.clear()

            # Success check
            _ps = info.get("place_success_mask", env.task_success)
            success = _ps.any().item() if hasattr(_ps, 'any') else bool(_ps)
            _prelim = info.get("preliminary_success",
                               env.preliminary_success)
            prelim = _prelim.any().item() if hasattr(_prelim, 'any') else bool(_prelim)

            if success:
                successes += 1
            if prelim:
                prelim_successes += 1

            # Termination reason
            reasons = []
            if terminated.any():
                if hasattr(env, 'just_dropped') and env.just_dropped[0].item():
                    reasons.append("DROP")
                root_z = env.robot.data.root_pos_w[0, 2].item()
                env_z = env.scene.env_origins[0, 2].item()
                if abs(root_z - env_z) > 0.5:
                    reasons.append("FELL")
            if truncated.any():
                if success:
                    reasons.append("PLACE_SUCCESS")
                elif prelim:
                    reasons.append("PLACE_GRACE")
                else:
                    reasons.append("TIMEOUT")
            reason_str = "+".join(reasons) if reasons else "UNKNOWN"

            # Final state
            fin_grip = env.robot.data.joint_pos[0, env.gripper_idx].item()
            fin_obj_h = env.object_rigid.data.root_pos_w[0, 2].item() - env.scene.env_origins[0, 2].item()
            fin_dest = torch.norm(
                env.dest_object_pos_w[0, :2] - env.robot.data.root_pos_w[0, :2]
            ).item()
            fin_obj_dest = torch.norm(
                env.object_pos_w[0, :2] - env.dest_object_pos_w[0, :2]
            ).item()

            status = "SUCCESS" if success else ("PRELIM" if prelim else "FAIL")
            print(f"\n  Episode {episode}/{args.num_episodes}: {status} ({reason_str}) "
                  f"| {step_count} steps "
                  f"| grip={fin_grip:.3f} obj_h={fin_obj_h:.3f} "
                  f"dest_d={fin_dest:.3f} obj_dest_d={fin_obj_dest:.3f}"
                  f"\n  Cumulative: {successes}/{episode} = "
                  f"{successes/episode*100:.0f}% "
                  f"(prelim: {prelim_successes}/{episode})\n",
                  flush=True)

            step_count = 0
            if mode == "dppo":
                dppo.prev_naction = None
            else:
                bc_agent.reset()

            if episode < args.num_episodes:
                obs, _ = env.reset()
                if demo_episodes and episode < len(demo_episodes):
                    restore_init_state_s3(env, demo_episodes[episode])
                    obs = env._get_observations()
                    env.robot.update(env.sim.cfg.dt)
                    env.object_rigid.update(env.sim.cfg.dt)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Results ({mode.upper()}):")
    print(f"    Final success: {successes}/{max(episode,1)} "
          f"= {successes/max(episode,1)*100:.0f}%")
    print(f"    Prelim success: {prelim_successes}/{max(episode,1)} "
          f"= {prelim_successes/max(episode,1)*100:.0f}%")
    print(f"{'='*60}\n")

    if demo_file is not None:
        demo_file.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
