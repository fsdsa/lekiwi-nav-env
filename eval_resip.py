#!/usr/bin/env python3
"""
ResiP (Frozen DP + Residual PPO) 체크포인트를 Isaac Sim에서 GUI로 돌려보는 평가 스크립트.

v6.6 학습 환경과 완전히 동일한 설정:
  - DR=OFF, grasp_success_height=1.00, lift_success_sustain_steps=0
  - grasp_ee_max_dist=env default (0.07) — 학습과 동일
  - BC warmup 660 steps → 학습 시 RL이 보는 것과 동일한 시작 상태
  - 성공 판정: lift_sustain >= 100 (4초 안정 lift) — train의 LMI=100과 동일

Usage:
    python eval_resip.py --skill approach_and_grasp \
        --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --resip_checkpoint checkpoints/resip_v66/resip_best.pt \
        --object_usd /path/to/object.usd \
        --num_episodes 20
"""
from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description="ResiP Eval v6.6 (GUI)")
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place"])
parser.add_argument("--dp_checkpoint", type=str, required=True)
parser.add_argument("--resip_checkpoint", type=str, default="")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--demo", type=str, default="")
parser.add_argument("--inference_steps", type=int, default=4)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")
parser.add_argument("--warmup_steps", type=int, default=660,
                    help="BC warmup steps (학습과 동일하게 660)")
parser.add_argument("--lift_threshold", type=int, default=100,
                    help="Lift sustain threshold (학습 LMI와 동일하게 100)")
parser.add_argument("--grip_override", action="store_true", default=False)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
import h5py
import torch
import numpy as np

from isaaclab.utils.math import quat_apply, quat_apply_inverse
from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
from lekiwi_skill2_env import EE_LOCAL_OFFSET


def load_frozen_dp(ckpt_path, device, inference_steps=4):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"], action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=inference_steps,
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)
    state_dict = ckpt["model_state_dict"]
    agent.model.load_state_dict({k[6:]: v for k, v in state_dict.items() if k.startswith("model.")})
    agent.normalizer.load_state_dict({k[11:]: v for k, v in state_dict.items() if k.startswith("normalizer.")}, device=device)
    for p in agent.parameters(): p.requires_grad = False
    agent.eval()
    print(f"Frozen DP: {ckpt_path} (obs={cfg['obs_dim']}, act={cfg['act_dim']}, steps={inference_steps})")
    return agent, cfg


def load_residual_policy(ckpt_path, obs_dim, act_dim, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    residual = ResidualPolicy(
        obs_dim=obs_dim, action_dim=act_dim,
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
    print(f"Residual: {ckpt_path} (iter={ckpt.get('iteration','?')}, sr={ckpt.get('success_rate','?')})")
    return residual


# ── HDF5 데모 로드 ──
demo_episodes = []
demo_file = None
if args.demo and os.path.isfile(args.demo):
    demo_file = h5py.File(args.demo, "r")
    ep_keys = sorted([k for k in demo_file.keys() if k.startswith("episode")],
                     key=lambda k: int(k.split("_")[1]))
    for ek in ep_keys:
        grp = demo_file[ek]
        demo_episodes.append({
            "obs": grp["obs"][:], "actions": grp["actions"][:],
            "ep_attrs": dict(grp.attrs),
            "object_pos_w": grp["object_pos_w"][:] if "object_pos_w" in grp else None,
        })
    print(f"  Demo: {args.demo} ({len(demo_episodes)} episodes)")
    if args.num_episodes > len(demo_episodes):
        args.num_episodes = len(demo_episodes)

# ── Env 생성 — 학습 make_env와 완전 동일 ──
if args.skill == "approach_and_grasp":
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = 1
elif args.skill == "carry_and_place":
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    env_cfg = Skill3EnvCfg()
    env_cfg.scene.num_envs = 1
    if args.handoff_buffer: env_cfg.handoff_buffer_path = args.handoff_buffer

# ★ 학습 make_env와 완전히 동일한 설정
env_cfg.enable_domain_randomization = False       # 학습과 동일: DR=OFF
env_cfg.arm_limit_write_to_sim = False            # 학습과 동일
env_cfg.grasp_contact_threshold = 0.55            # 학습과 동일
env_cfg.grasp_gripper_threshold = 0.65            # 학습과 동일
env_cfg.grasp_max_object_dist = 0.50              # 학습과 동일
env_cfg.episode_length_s = 300.0                  # 학습과 동일
env_cfg.spawn_heading_noise_std = 0.3             # 학습과 동일
env_cfg.spawn_heading_max_rad = 0.5               # 학습과 동일
env_cfg.lift_success_sustain_steps = 0            # 학습과 동일: env 성공 종료 없음
env_cfg.grasp_success_height = 1.00               # 학습과 동일: 사실상 비활성화
# grasp_ee_max_dist: 설정 안 함 → env 기본값 0.07 사용 (학습과 동일!)

# HDF5 demo에서 설정 복원
if demo_file is not None:
    hdf5_attrs = dict(demo_file.attrs)
    env_cfg.object_mass = float(hdf5_attrs.get("object_mass", env_cfg.object_mass))
    env_cfg.arm_action_scale = float(hdf5_attrs.get("arm_action_scale", env_cfg.arm_action_scale))
    env_cfg.max_lin_vel = float(hdf5_attrs.get("max_lin_vel", env_cfg.max_lin_vel))
    env_cfg.max_ang_vel = float(hdf5_attrs.get("max_ang_vel", env_cfg.max_ang_vel))
    if not args.object_usd and "object_usd" in hdf5_attrs:
        env_cfg.object_usd = str(hdf5_attrs["object_usd"])

if args.object_usd: env_cfg.object_usd = os.path.expanduser(args.object_usd)
if args.multi_object_json: env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
if args.dest_object_usd: env_cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
    env_cfg.arm_limit_json = args.arm_limit_json

if args.skill == "approach_and_grasp":
    env = Skill2Env(cfg=env_cfg)
else:
    env = Skill3Env(cfg=env_cfg)

# ── 모델 로드 ──
device = env.device
dp_agent, dp_cfg = load_frozen_dp(args.dp_checkpoint, device, args.inference_steps)
obs_dim, act_dim = dp_cfg["obs_dim"], dp_cfg["act_dim"]

residual_policy = None
if args.resip_checkpoint and os.path.isfile(args.resip_checkpoint):
    residual_policy = load_residual_policy(args.resip_checkpoint, obs_dim, act_dim, device)

# ★ 학습과 완전히 동일한 per-dim scale
per_dim_action_scale = torch.zeros(act_dim, device=device)
per_dim_action_scale[0:5] = 0.20   # arm
per_dim_action_scale[5]   = 0.30   # gripper
per_dim_action_scale[6:9] = 0.35   # base

mode_str = "ResiP (DP + Residual)" if residual_policy else "DP BC only"
demo_str = "HDF5 복원" if demo_episodes else "랜덤 리셋"
print(f"\n{'='*60}")
print(f"  {mode_str} Eval — {args.skill} ({demo_str})")
print(f"  warmup={args.warmup_steps} steps, lift_threshold={args.lift_threshold}")
print(f"  grasp_ee_max_dist={env_cfg.grasp_ee_max_dist} (env default, 학습과 동일)")
print(f"{'='*60}\n")

# ── 헬퍼 ──
fixed_jaw_idx, _ = env.robot.find_bodies(["Wrist_Roll_08c_v1"])
ee_local_offset = torch.tensor(EE_LOCAL_OFFSET, device=device).unsqueeze(0)

def get_distances():
    wp = env.robot.data.body_pos_w[:, fixed_jaw_idx[0], :]
    wq = env.robot.data.body_quat_w[:, fixed_jaw_idx[0], :]
    ee = wp + quat_apply(wq, ee_local_offset)
    ee_d = torch.norm(ee - env.object_pos_w, dim=-1).item()
    base_d = torch.norm(env.robot.data.root_pos_w[:, :2] - env.object_pos_w[:, :2], dim=-1).item()
    grip = env.robot.data.joint_pos[:, env.gripper_idx].item()
    env_z = env.scene.env_origins[:, 2].item() if hasattr(env.scene, "env_origins") else 0.0
    obj_z = env.object_pos_w[:, 2].item() - env_z
    ee_z = ee[0, 2].item() - env_z
    return ee_d, base_d, grip, obj_z, ee_z

def get_grasp_debug():
    grip = env.robot.data.joint_pos[:, env.gripper_idx].item()
    contact = env._contact_force_per_env().item() if env.contact_sensor else 0.0
    wp = env.robot.data.body_pos_w[:, fixed_jaw_idx[0], :]
    wq = env.robot.data.body_quat_w[:, fixed_jaw_idx[0], :]
    ee = wp + quat_apply(wq, ee_local_offset)
    ee_to_obj = torch.norm(ee - env.object_pos_w, dim=-1).item()
    grasped = env.object_grasped[0].item()
    gcf = env._ground_contact_force_per_env().item() if hasattr(env, 'ground_contact_sensor') and env.ground_contact_sensor else 0.0
    return {
        "grip": grip, "grip_closed": grip < float(env.cfg.grasp_gripper_threshold),
        "contact": contact, "has_contact": contact > float(env.cfg.grasp_contact_threshold),
        "ee_to_obj": ee_to_obj, "between_jaws": ee_to_obj < float(env.cfg.grasp_ee_max_dist),
        "grasped": grasped, "gcf": gcf,
        "grip_on_ground": gcf > 1.0 and not grasped,
    }

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
    env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_id)
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
    for _ in range(10): env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)


# ── 실행 루프 ──
IS_S3 = args.skill == "carry_and_place"
LMI = args.lift_threshold  # 학습의 lift_milestone_steps와 동일
WU = args.warmup_steps     # 학습의 warmup_steps_final과 동일

episode = 0; successes = 0; step_count = 0
ep_min_ee = 999.0; ep_min_base = 999.0; ep_min_ee_z = 999.0; window_min_ee_z = 999.0
prev_grasped = False; ep_grasped = False; ep_lifted = False; lift_sustain = 0
grasp_sustain = 0; ms_gr = False
# Skill-3 state
ep_placed = False; ep_dropped = False; ep_min_dd = 999.0

def get_dest_dist():
    """Skill-3: dest object까지 XY 거리 + heading."""
    root_pos = env.robot.data.root_pos_w
    root_quat = env.robot.data.root_quat_w
    dest_delta = env.dest_object_pos_w - root_pos
    dest_b = quat_apply_inverse(root_quat, dest_delta)
    dd = torch.norm(dest_b[:, :2], dim=-1).item()
    cos_h = (dest_b[0, 1] / (dd + 1e-6)).item()  # forward = +Y body
    return dd, cos_h

obs, _ = env.reset()

if demo_episodes:
    _restore_init_state(demo_episodes[0])
    for _ in range(5): env.sim.step()
    env.robot.update(env.sim.cfg.dt); env.object_rigid.update(env.sim.cfg.dt)

dp_agent.reset()

# ★ BC warmup — 학습과 동일하게 660 steps BC만 실행
if WU > 0:
    print(f"  Running BC warmup: {WU} steps...")
    for wi in range(WU):
        obs_t = obs["policy"].to(device) if isinstance(obs, dict) else obs.to(device)
        with torch.no_grad():
            a = dp_agent.normalizer(dp_agent.base_action_normalized(obs_t), "action", forward=False)
        obs, _, ter, tru, _ = env.step(a)
        if (ter.any() or tru.any()):
            obs, _ = env.reset(); dp_agent.reset()
            if demo_episodes and episode < len(demo_episodes):
                _restore_init_state(demo_episodes[episode])
                for _ in range(5): env.sim.step()
                env.robot.update(env.sim.cfg.dt); env.object_rigid.update(env.sim.cfg.dt)
    if IS_S3:
        dd0, cos0 = get_dest_dist()
        grip = env.robot.data.joint_pos[:, env.gripper_idx].item()
        print(f"  Warmup done: DD={dd0:.3f} cosH={cos0:.3f} grip={grip:.3f} grasped={env.object_grasped[0].item()}")
    else:
        ee_d, base_d, grip, obj_z, ee_z = get_distances()
        print(f"  Warmup done: EE={ee_d:.3f} Base={base_d:.3f} grip={grip:.3f} objZ={obj_z:.3f}")

if IS_S3:
    print(f"\n  Starting Skill-3 evaluation ({args.num_episodes} episodes)...\n")
else:
    print(f"\n  Starting evaluation ({args.num_episodes} episodes, LMI={LMI})...\n")

while episode < args.num_episodes and simulation_app.is_running():
    obs_t = obs["policy"].to(device) if isinstance(obs, dict) else obs.to(device)

    with torch.no_grad():
        base_naction = dp_agent.base_action_normalized(obs_t)

        if residual_policy is not None:
            nobs = torch.nan_to_num(torch.clamp(
                dp_agent.normalizer(obs_t, "obs", forward=True), -3, 3), nan=0.0)
            residual_nobs = torch.cat([nobs, base_naction], dim=-1)
            residual_naction = residual_policy.actor_mean(residual_nobs)
            residual_naction = torch.clamp(residual_naction, -1.0, 1.0)
            naction = base_naction + residual_naction * per_dim_action_scale
        else:
            naction = base_naction

        action = dp_agent.normalizer(naction, "action", forward=False)

    if not IS_S3 and args.grip_override:
        ee_d_now = get_distances()[0]
        if ee_d_now < 0.05:
            grip_target = 0.30
        elif ee_d_now < 0.20:
            grip_target = 0.30 + (ee_d_now - 0.05) / 0.15 * 1.18
        else:
            grip_target = None
        if grip_target is not None:
            action[0, 5] = 0.8 * grip_target + 0.2 * action[0, 5].item()

    if step_count < 5:
        a = action[0].cpu().tolist()
        print(f"  [t={step_count}] action={[f'{x:.3f}' for x in a]}", flush=True)

    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    if IS_S3:
        # ── Skill-3: CarryAndPlace 평가 ──
        dd, cos_h = get_dest_dist()
        ep_min_dd = min(ep_min_dd, dd)
        grip = env.robot.data.joint_pos[:, env.gripper_idx].item()
        grasped = env.object_grasped[0].item()
        obj_z = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()

        # Place detection (env의 task_success)
        ps = info.get("place_success_mask", env.task_success)
        if hasattr(ps, 'item'):
            ps_val = ps[0].item() if ps.numel() > 1 else ps.item()
        else:
            ps_val = bool(ps)
        if ps_val and not ep_placed:
            ep_placed = True
            print(f"    ★ PLACE at t={step_count} | DD={dd:.3f} grip={grip:.3f} objZ={obj_z:.3f}", flush=True)

        # Drop detection
        jd = info.get("just_dropped_mask", env.just_dropped)
        if hasattr(jd, 'item'):
            jd_val = jd[0].item() if jd.numel() > 1 else jd.item()
        else:
            jd_val = bool(jd)
        if jd_val and not ep_dropped:
            ep_dropped = True
            print(f"    ✗ DROP at t={step_count} | DD={dd:.3f} grip={grip:.3f} objZ={obj_z:.3f}", flush=True)

        if step_count % 50 == 0:
            status = ""
            if ep_placed: status = " [PLACED]"
            elif ep_dropped: status = " [DROPPED]"
            elif grasped: status = " [CARRYING]"
            print(f"    [t={step_count:4d}] DD={dd:.3f}(min={ep_min_dd:.3f}) cosH={cos_h:.3f} "
                  f"grip={grip:.3f} objZ={obj_z:.3f} grasped={grasped}{status}", flush=True)

    else:
        # ── Skill-2: ApproachAndGrasp 평가 ──
        ee_d, base_d, grip, obj_z, ee_z = get_distances()
        ep_min_ee = min(ep_min_ee, ee_d)
        ep_min_base = min(ep_min_base, base_d)
        ep_min_ee_z = min(ep_min_ee_z, ee_z)
        window_min_ee_z = min(window_min_ee_z, ee_z)

        g = get_grasp_debug()

        # ★ Verified grasp: train의 g_sus/ms_gr 로직과 동일 (5-step sustained)
        if g["grasped"] and not ms_gr:
            grasp_sustain += 1
        elif not g["grasped"] and not ms_gr:
            grasp_sustain = 0
        if grasp_sustain >= 5 and not ms_gr:
            ms_gr = True; ep_grasped = True
            print(f"    ★ VERIFIED GRASP at t={step_count} (sustain={grasp_sustain}) | "
                  f"grip={g['grip']:.3f} contact={g['contact']:.3f} ee={g['ee_to_obj']:.3f}", flush=True)

        if g["grasped"] and not prev_grasped and not ms_gr:
            print(f"    ◇ env grasp at t={step_count} | grip={g['grip']:.3f} contact={g['contact']:.3f} ee={g['ee_to_obj']:.3f}", flush=True)
        prev_grasped = g["grasped"]

        # Drop detection
        if ms_gr and not g["grasped"]:
            print(f"    ✗ DROP at t={step_count} | grip={g['grip']:.3f} ee={g['ee_to_obj']:.3f}", flush=True)
            ms_gr = False; grasp_sustain = 0; lift_sustain = 0

        # ★ Lift 판정: train의 held/l_sus 로직과 완전 동일
        held_now = (obj_z > 0.05 and ms_gr and g["grip_closed"] and g["ee_to_obj"] < 0.20)
        if held_now:
            lift_sustain += 1
        else:
            lift_sustain = 0
        if lift_sustain >= LMI and not ep_lifted:
            ep_lifted = True
            print(f"    ★ LIFT at t={step_count} | objZ={obj_z:.3f} ee={g['ee_to_obj']:.3f} "
                  f"grip={g['grip']:.3f} sustain={lift_sustain}", flush=True)

        if step_count % 50 == 0:
            status = ""
            if ep_lifted: status = f" [LIFTED sustain={lift_sustain}]"
            elif ms_gr: status = f" [VGRASP lift_sus={lift_sustain}]"
            elif g["grasped"]: status = f" [GRASP g_sus={grasp_sustain}]"
            ground_str = " **GROUND**" if g["grip_on_ground"] else ""
            print(f"    [t={step_count:4d}] EE={ee_d:.3f} Base={base_d:.3f} grip={grip:.3f} objZ={obj_z:.3f} "
                  f"eeZ={ee_z:.4f}(min={window_min_ee_z:.4f}) contact={g['contact']:.2f} ee2obj={g['ee_to_obj']:.3f}"
                  f" gcf={g['gcf']:.2f}{ground_str}{status}", flush=True)
            window_min_ee_z = 999.0

    done = terminated.any() or truncated.any()
    if done:
        episode += 1
        if IS_S3:
            # ★ Skill-3 성공 판정: place success
            if ep_placed: successes += 1
            status = "SUCCESS" if ep_placed else ("DROPPED" if ep_dropped else "FAIL")
            print(f"  Episode {episode}/{args.num_episodes}: {status} "
                  f"({step_count} steps, min_DD={ep_min_dd:.3f} | "
                  f"cumulative: {successes}/{episode} = {successes/episode*100:.0f}%)", flush=True)
        else:
            # ★ Skill-2 성공 판정: ep_lifted (l_sus >= LMI)
            if ep_lifted: successes += 1
            status = "SUCCESS" if ep_lifted else "FAIL"
            grasp_lift = f"grasp={'Y' if ep_grasped else 'N'} lift={'Y' if ep_lifted else 'N'}"
            print(f"  Episode {episode}/{args.num_episodes}: {status} "
                  f"({step_count} steps, {grasp_lift}, "
                  f"min_EE={ep_min_ee:.3f} min_Base={ep_min_base:.3f} | "
                  f"cumulative: {successes}/{episode} = {successes/episode*100:.0f}%)", flush=True)

        step_count = 0; ep_min_ee = 999.0; ep_min_base = 999.0
        ep_min_ee_z = 999.0; window_min_ee_z = 999.0
        prev_grasped = False; ep_grasped = False; ep_lifted = False
        lift_sustain = 0; grasp_sustain = 0; ms_gr = False
        ep_placed = False; ep_dropped = False; ep_min_dd = 999.0
        dp_agent.reset()
        obs, _ = env.reset()

        if demo_episodes and episode < len(demo_episodes):
            _restore_init_state(demo_episodes[episode])
            for _ in range(5): env.sim.step()
            env.robot.update(env.sim.cfg.dt); env.object_rigid.update(env.sim.cfg.dt)

        # ★ 매 에피소드 BC warmup 재실행
        if WU > 0 and episode < args.num_episodes:
            for wi in range(WU):
                obs_t = obs["policy"].to(device) if isinstance(obs, dict) else obs.to(device)
                with torch.no_grad():
                    a = dp_agent.normalizer(dp_agent.base_action_normalized(obs_t), "action", forward=False)
                obs, _, ter, tru, _ = env.step(a)
                if ter.any() or tru.any():
                    obs, _ = env.reset(); dp_agent.reset()
                    if demo_episodes and episode < len(demo_episodes):
                        _restore_init_state(demo_episodes[episode])
                        for _ in range(5): env.sim.step()
                        env.robot.update(env.sim.cfg.dt); env.object_rigid.update(env.sim.cfg.dt)

print(f"\n  === 결과: {successes}/{args.num_episodes} 성공 "
      f"({successes/max(episode,1)*100:.0f}%) ===")
if IS_S3:
    print(f"  (warmup={WU})\n")
else:
    print(f"  (LMI={LMI}, warmup={WU}, grasp_ee_max_dist={env_cfg.grasp_ee_max_dist})\n")

if demo_file is not None: demo_file.close()
env.close(); simulation_app.close()
