#!/usr/bin/env python3
"""S3 (Approach & Place) BC/BC+RL eval.

S2 expert lift → S3 BC (36D obs) carry → place.

Usage (BC only):
    python eval_s3.py \
        --s3_bc_checkpoint checkpoints/dp_bc_skill3_36d/dp_bc_epoch300.pt \
        --s2_bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --s2_resip_checkpoint backup/appoachandlift/resip64%.pt \
        --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \
        --dest_object_usd ~/isaac-objects/.../ACE_Coffee_Mug_.../model_clean.usd \
        --num_episodes 10
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--s3_bc_checkpoint", type=str, required=True)
parser.add_argument("--s3_resip_checkpoint", type=str, default="")
parser.add_argument("--s2_bc_checkpoint", type=str, required=True)
parser.add_argument("--s2_resip_checkpoint", type=str, default="")
parser.add_argument("--object_usd", type=str, required=True)
parser.add_argument("--dest_object_usd", type=str, required=True)
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--object_scale_phys", type=float, default=0.7)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--s2_max_steps", type=int, default=800)
parser.add_argument("--s2_lift_hold", type=int, default=200)
parser.add_argument("--s3_max_steps", type=int, default=3000)
parser.add_argument("--inference_steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import torch, numpy as np, math
from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
from isaaclab.utils.math import quat_apply_inverse

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
cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
cfg.object_scale = args.object_scale_phys
cfg.grasp_success_height = 100.0
cfg.dest_spawn_dist_min = 0.6
cfg.dest_spawn_dist_max = 0.9

env = Skill2Env(cfg=cfg)
dev = env.device

S3_PHASE_B_DIST = 0.42
S3_PLACE_RADIUS = 0.14

# ── S2 Expert ──
s2_ckpt = torch.load(args.s2_bc_checkpoint, map_location=dev, weights_only=False)
s2_cfg = s2_ckpt["config"]
s2_dp = DiffusionPolicyAgent(
    obs_dim=s2_cfg["obs_dim"], act_dim=s2_cfg["act_dim"],
    pred_horizon=s2_cfg["pred_horizon"], action_horizon=s2_cfg["action_horizon"],
    num_diffusion_iters=s2_cfg["num_diffusion_iters"], inference_steps=4,
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

# ── S3 BC ──
s3_ckpt = torch.load(args.s3_bc_checkpoint, map_location=dev, weights_only=False)
s3_cfg = s3_ckpt["config"]
s3_dp = DiffusionPolicyAgent(
    obs_dim=s3_cfg["obs_dim"], act_dim=s3_cfg["act_dim"],
    pred_horizon=s3_cfg["pred_horizon"], action_horizon=s3_cfg["action_horizon"],
    num_diffusion_iters=s3_cfg["num_diffusion_iters"],
    inference_steps=args.inference_steps,
    down_dims=s3_cfg.get("down_dims", [64, 128, 256]),
).to(dev)
sd = s3_ckpt["model_state_dict"]
s3_dp.model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
s3_dp.normalizer.load_state_dict(
    {k[len("normalizer."):]: v for k, v in sd.items() if k.startswith("normalizer.")}, device=dev)
s3_dp.eval()
for p in s3_dp.parameters():
    p.requires_grad = False

# ── S3 ResiP (optional) ──
s3_resip = None
if args.s3_resip_checkpoint:
    rp_ckpt = torch.load(args.s3_resip_checkpoint, map_location=dev, weights_only=False)
    s3_resip = ResidualPolicy(
        obs_dim=s3_cfg["obs_dim"], action_dim=s3_cfg["act_dim"],
        action_scale=0.1, learn_std=True,
    ).to(dev)
    s3_resip.load_state_dict(rp_ckpt["residual_policy_state_dict"])
    s3_resip.eval()

mode_str = "BC+RL" if s3_resip else "BC only"
print(f"\n  S3 Eval ({mode_str})")
print(f"  S2: {args.s2_bc_checkpoint}")
print(f"  S3 BC: {args.s3_bc_checkpoint} (obs={s3_cfg['obs_dim']}D)")
if s3_resip:
    print(f"  S3 ResiP: {args.s3_resip_checkpoint}")
print(f"  Episodes: {args.num_episodes}\n")

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

def build_s3_obs(obs_30d, init_pose6, phase_a_flag_val):
    """Build 36D S3 obs from 30D env obs."""
    robot_pos = env.robot.data.root_pos_w[0]
    robot_quat = env.robot.data.root_quat_w[0]
    dest_pos = env.dest_object_pos_w[0]
    rel_w = dest_pos - robot_pos
    dest_rel_body = quat_apply_inverse(robot_quat.unsqueeze(0), rel_w.unsqueeze(0))[0]
    cf = env._contact_force_per_env().unsqueeze(-1)  # (1, 1)

    s3_obs29 = torch.cat([
        obs_30d[0, 0:21],
        dest_rel_body,
        cf[0],
        obs_30d[0, 26:29],
        obs_30d[0, 29:30],
    ]).unsqueeze(0)  # (1, 29)

    flag = torch.tensor([[phase_a_flag_val]], device=dev)
    s3_obs = torch.cat([s3_obs29, init_pose6.unsqueeze(0), flag], dim=-1)  # (1, 36)
    return s3_obs

# Phase-wise residual scale (must match train_resip.py main_combined)
s3_scale_a = torch.zeros(s3_cfg["act_dim"], device=dev)
s3_scale_a[0:5] = 0.05; s3_scale_a[5] = 0.05; s3_scale_a[6:9] = 0.10
s3_scale_b = torch.zeros(s3_cfg["act_dim"], device=dev)
s3_scale_b[0:5] = 0.30; s3_scale_b[5] = 1.20; s3_scale_b[6:9] = 0.20

def get_s3_action(s3_obs, phase_a=True):
    with torch.no_grad():
        base_nact = s3_dp.base_action_normalized(s3_obs)
        base_nact = torch.nan_to_num(base_nact, nan=0.0)
        if s3_resip is not None:
            nobs = torch.nan_to_num(
                s3_dp.normalizer(s3_obs, "obs", forward=True).clamp(-3, 3), nan=0.0)
            ri = torch.cat([nobs, base_nact], dim=-1)
            _, _, _, _, ra_mean = s3_resip.get_action_and_value(ri)
            ra_mean = torch.clamp(ra_mean, -1.0, 1.0)
            s3_scale = s3_scale_a if phase_a else s3_scale_b
            nact = base_nact + ra_mean * s3_scale
        else:
            nact = base_nact
        nact[:, 5] = torch.clamp(nact[:, 5], -0.45, 1.0)  # gripper clamp: grip_pos >= ~0.26 (끼임 방지)
        action = s3_dp.normalizer(nact, "action", forward=False)
    return action.clamp(-1, 1)

# ── Main loop ──
results = []
for ep in range(args.num_episodes):
    print(f"  [Episode {ep+1}/{args.num_episodes}]")

    # Phase 1: S2 expert lift
    obs, _ = env.reset()
    s2_dp.reset()
    lift_counter = 0
    lifted = False
    max_retries = 5
    for attempt in range(max_retries):
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

            if (objZ < 0.026 and s2_step > 20) or (s2_step >= args.s2_max_steps - 1):
                reason = f"objZ={objZ:.3f}" if objZ < 0.026 else "timeout"
                print(f"    [S2 FAIL] {reason} step={s2_step} — retry ({attempt+1}/{max_retries})")
                obs, _ = env.reset()
                s2_dp.reset()
                lift_counter = 0
                break

            if s2_step % 200 == 0:
                print(f"    [S2] step={s2_step} objZ={objZ:.3f} grip={grip_pos:.3f} lift={lift_counter}/{args.s2_lift_hold}")

            if lift_counter >= args.s2_lift_hold:
                lifted = True
                break
        if lifted:
            break

    if not lifted:
        print(f"    [S2 FAIL] Could not lift — skip")
        results.append({"status": "s2_fail"})
        continue

    # Capture init_arm_pose
    jp = env.robot.data.joint_pos[0]
    init_pose6 = torch.cat([jp[env.arm_idx[:5]], jp[env.arm_idx[5:6]]]).to(dev)
    init_arm_np = jp[env.arm_idx[:5]].cpu().tolist()
    init_grip = jp[env.arm_idx[5]].item()
    print(f"    [S2→S3] arm3={init_arm_np[3]:+.3f} grip={init_grip:.3f}")

    # Dest 재스폰: 로봇 전방 0.6~0.9m (train_resip.py와 동일)
    from isaaclab.utils.math import quat_apply as _qa
    rpos = env.robot.data.root_pos_w[0:1]
    rquat = env.robot.data.root_quat_w[0:1]
    fwd = _qa(rquat, torch.tensor([[0, 1, 0]], dtype=torch.float32, device=dev))
    dist = torch.rand(1, device=dev) * (0.9 - 0.6) + 0.6
    angle_noise = (torch.rand(1, device=dev) * 2 - 1) * 0.5  # ±0.5 rad
    cos_n = torch.cos(angle_noise); sin_n = torch.sin(angle_noise)
    fwd_x = fwd[0, 0] * cos_n - fwd[0, 1] * sin_n
    fwd_y = fwd[0, 0] * sin_n + fwd[0, 1] * cos_n
    dest_x = rpos[0, 0] + fwd_x * dist
    dest_y = rpos[0, 1] + fwd_y * dist
    dest_z = rpos[0, 2] - 0.03
    dest_rigid = env._dest_object_rigid
    if dest_rigid is not None:
        pose = dest_rigid.data.default_root_state[0:1, :7].clone()
        pose[0, 0] = dest_x.item(); pose[0, 1] = dest_y.item(); pose[0, 2] = dest_z.item()
        yaw = torch.rand(1, device=dev) * 2 * 3.14159 - 3.14159
        pose[0, 3] = torch.cos(yaw * 0.5).item(); pose[0, 6] = torch.sin(yaw * 0.5).item()
        pose[0, 4] = 0; pose[0, 5] = 0
        dest_rigid.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=dev))
        env.dest_object_pos_w[0, 0] = dest_x.item()
        env.dest_object_pos_w[0, 1] = dest_y.item()
        env.dest_object_pos_w[0, 2] = dest_z.item()
    _bdist = torch.norm(rpos[0, :2] - torch.tensor([dest_x.item(), dest_y.item()], device=dev)).item()
    print(f"    [Dest spawn] base_dst={_bdist:.3f}m")

    # Phase 2: S3 (approach + place)
    s3_dp.reset()
    phase_a_active = True
    dropped = False
    placed = False
    phase_a_steps = 0
    phase_b_step_start = -1
    min_objZ = 1.0
    max_grip_open = 0.0
    min_src_dst = 10.0
    arm_init_err_sum = 0.0
    arm_init_err_count = 0
    prev_action = None

    # Phase A ghost carry tracking
    phase_a_cf_zero_count = 0
    phase_a_cf_zero_max = 0

    # Phase B arm trajectory tracking
    arm1_at_phase_b_entry = None
    arm1_max_in_phase_b = -999.0
    objZ_at_phase_b_entry = None
    objZ_min_in_phase_b = 999.0
    grip_at_phase_b_entry = None

    base_dst_start = torch.norm(env.robot.data.root_pos_w[0, :2] - env.dest_object_pos_w[0, :2]).item()

    for s3_step in range(args.s3_max_steps):
        obs_30d = obs["policy"] if isinstance(obs, dict) else obs

        # Phase A→B latch
        base_dst = torch.norm(env.robot.data.root_pos_w[0, :2] - env.dest_object_pos_w[0, :2]).item()
        if phase_a_active and base_dst <= S3_PHASE_B_DIST:
            phase_a_active = False
            phase_b_step_start = s3_step
            jp_at_trans = env.robot.data.joint_pos[0]
            arm_at_trans = jp_at_trans[env.arm_idx[:5]].cpu().tolist()
            grip_at_trans = jp_at_trans[env.arm_idx[5]].item()
            arm_trans_str = ",".join(f"{v:+.3f}" for v in arm_at_trans)
            # Phase B entry tracking
            arm1_at_phase_b_entry = arm_at_trans[1]
            objZ_at_phase_b_entry = objZ
            grip_at_phase_b_entry = grip_at_trans
            ghost_str = f" ⚠ghost_cf0={phase_a_cf_zero_max}" if phase_a_cf_zero_max >= 5 else ""
            print(f"    [Phase A→B] step={s3_step} base_dst={base_dst:.3f} arm=[{arm_trans_str}] "
                  f"grip={grip_at_trans:.3f} objZ={objZ:.3f} cf={cf_val:.1f}{ghost_str}")

        flag_val = 1.0 if phase_a_active else 0.0
        s3_obs = build_s3_obs(obs_30d, init_pose6, flag_val)
        action = get_s3_action(s3_obs, phase_a=phase_a_active)
        action_np = action[0].cpu().tolist()

        # Action delta (smoothness)
        action_delta_str = ""
        if prev_action is not None:
            delta = [action_np[i] - prev_action[i] for i in range(6)]
            max_delta = max(abs(d) for d in delta)
            if max_delta > 0.3:
                action_delta_str = f" Δarm_max={max_delta:.3f}"
        prev_action = action_np

        obs, _, ter, tru, _ = env.step(action)

        objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
        jp = env.robot.data.joint_pos[0]
        arm_jp = jp[env.arm_idx[:5]].cpu().tolist()
        grip_pos = jp[env.arm_idx[5]].item()
        src_dst = torch.norm(env.object_pos_w[0, :2] - env.dest_object_pos_w[0, :2]).item()
        arm_str = ",".join(f"{v:+.3f}" for v in arm_jp)
        act_arm_str = ",".join(f"{v:+.3f}" for v in action_np[:6])

        # Contact force
        has_contact = False
        cf_val = 0.0
        if env.contact_sensor is not None:
            cf_val = env._contact_force_per_env()[0].item()
            has_contact = cf_val > float(env.cfg.grasp_contact_threshold)

        # Tracking
        min_objZ = min(min_objZ, objZ)
        max_grip_open = max(max_grip_open, grip_pos)
        min_src_dst = min(min_src_dst, src_dst)
        if phase_a_active:
            phase_a_steps += 1
            # arm init_pose error
            init_np = init_pose6.cpu().tolist()
            arm_err = sum((arm_jp[i] - init_np[i])**2 for i in range(5)) ** 0.5
            arm_init_err_sum += arm_err
            arm_init_err_count += 1
            # Ghost carry detection (cf=0 연속)
            if not has_contact and s3_step > 5:
                phase_a_cf_zero_count += 1
                phase_a_cf_zero_max = max(phase_a_cf_zero_max, phase_a_cf_zero_count)
            else:
                phase_a_cf_zero_count = 0
        else:
            # Phase B arm trajectory tracking
            arm1_max_in_phase_b = max(arm1_max_in_phase_b, arm_jp[1])
            objZ_min_in_phase_b = min(objZ_min_in_phase_b, objZ)

        # Phase A: every 50 steps, Phase B: every 10 steps
        log_interval = 50 if phase_a_active else 10
        phase_b_elapsed = s3_step - phase_b_step_start if phase_b_step_start >= 0 else 0
        if s3_step % log_interval == 0:
            phase_str = "A" if phase_a_active else "B"
            contact_str = f"cf={cf_val:.1f}" if has_contact else "cf=0"
            act_base_str = ",".join(f"{v:+.3f}" for v in action_np[6:9])
            ghost_str = f" cf0_run={phase_a_cf_zero_count}" if phase_a_active and phase_a_cf_zero_count >= 3 else ""
            phb_str = f" phB_t={phase_b_elapsed}" if not phase_a_active else ""
            print(f"    [S3-{phase_str} t={s3_step:4d}] objZ={objZ:.3f} grip={grip_pos:.3f} "
                  f"base_dst={base_dst:.3f} src_dst={src_dst:.3f} {contact_str} "
                  f"arm=[{arm_str}] act_arm=[{act_arm_str}] act_base=[{act_base_str}]"
                  f"{action_delta_str}{ghost_str}{phb_str}")

        # Drop detection — Phase A: 0.04 (carrying height), Phase B: 0.029 (topple)
        drop_thresh = 0.04 if phase_a_active else 0.029
        if objZ < drop_thresh and s3_step > 10:
            phase_str_d = "A" if phase_a_active else "B"
            phase_b_dur = s3_step - phase_b_step_start if phase_b_step_start >= 0 else -1
            arm1_info = ""
            if arm1_at_phase_b_entry is not None:
                arm1_info = f" arm1: entry={arm1_at_phase_b_entry:+.3f}→now={arm_jp[1]:+.3f}"
            ghost_str = f" ghost_cf0_max={phase_a_cf_zero_max}" if phase_a_active else ""
            print(f"    [DROP] step={s3_step} phase={phase_str_d} phB_dur={phase_b_dur} | "
                  f"objZ={objZ:.3f} grip={grip_pos:.3f} cf={cf_val:.1f} | "
                  f"arm=[{arm_str}] act=[{act_arm_str}]{arm1_info}{ghost_str}")
            dropped = True
            break

        # Place detection — contact 기반 (데모 패턴: arm 내려놓으면 cf→0, 물체 서있음)
        if src_dst < S3_PLACE_RADIUS and not has_contact and objZ > 0.029 and objZ < 0.05 and s3_step > 50:
            phase_b_dur = s3_step - phase_b_step_start if phase_b_step_start >= 0 else -1
            arm1_delta = arm_jp[1] - arm1_at_phase_b_entry if arm1_at_phase_b_entry is not None else 0
            objZ_delta = objZ - objZ_at_phase_b_entry if objZ_at_phase_b_entry is not None else 0
            # Quality assessment
            is_real = phase_b_dur >= 30 and arm1_delta > 1.0
            quality = "REAL" if is_real else "SUSPECT"
            print(f"    [PLACE-{quality}] step={s3_step} phB_dur={phase_b_dur} | "
                  f"objZ={objZ:.3f} src_dst={src_dst:.3f} grip={grip_pos:.3f} cf={cf_val:.1f} | "
                  f"arm=[{arm_str}] | "
                  f"arm1: entry={arm1_at_phase_b_entry:+.3f}→now={arm_jp[1]:+.3f} (Δ={arm1_delta:+.3f}) | "
                  f"objZ: entry={objZ_at_phase_b_entry:.3f}→now={objZ:.3f} (Δ={objZ_delta:+.3f})")
            placed = True
            break

    status = "place" if placed else ("drop" if dropped else "timeout")
    avg_arm_err = arm_init_err_sum / max(arm_init_err_count, 1)
    phase_b_dur = s3_step - phase_b_step_start if phase_b_step_start >= 0 else -1
    results.append({"status": status, "s3_steps": s3_step, "phase_a_steps": phase_a_steps,
                     "phase_b_dur": phase_b_dur,
                     "min_objZ": min_objZ, "max_grip": max_grip_open, "min_src_dst": min_src_dst,
                     "arm_err": avg_arm_err,
                     "arm1_entry": arm1_at_phase_b_entry, "arm1_max_b": arm1_max_in_phase_b,
                     "objZ_entry": objZ_at_phase_b_entry, "objZ_min_b": objZ_min_in_phase_b,
                     "ghost_cf0": phase_a_cf_zero_max})
    # Phase B summary
    phb_info = ""
    if arm1_at_phase_b_entry is not None:
        phb_info = (f" | PhaseB: dur={phase_b_dur} arm1={arm1_at_phase_b_entry:+.3f}→max={arm1_max_in_phase_b:+.3f} "
                    f"objZ={objZ_at_phase_b_entry:.3f}→min={objZ_min_in_phase_b:.3f}")
    ghost_info = f" | ghost_cf0={phase_a_cf_zero_max}" if phase_a_cf_zero_max >= 5 else ""
    print(f"  Episode {ep+1}: {status} ({s3_step} steps) | "
          f"PhaseA={phase_a_steps} arm_err={avg_arm_err:.3f} | "
          f"min_objZ={min_objZ:.3f} max_grip={max_grip_open:.3f} min_src_dst={min_src_dst:.3f}"
          f"{phb_info}{ghost_info}\n")

# Summary
n = len(results)
n_place = sum(1 for r in results if r["status"] == "place")
n_drop = sum(1 for r in results if r["status"] == "drop")
n_drop_a = sum(1 for r in results if r["status"] == "drop" and r.get("phase_a_steps", 0) == r.get("s3_steps", 0))
n_drop_b = n_drop - n_drop_a
n_timeout = sum(1 for r in results if r["status"] == "timeout")
n_s2_fail = sum(1 for r in results if r["status"] == "s2_fail")

s3_results = [r for r in results if r["status"] != "s2_fail"]
avg_arm_err = sum(r.get("arm_err", 0) for r in s3_results) / max(len(s3_results), 1)
avg_min_src = sum(r.get("min_src_dst", 0) for r in s3_results) / max(len(s3_results), 1)
avg_max_grip = sum(r.get("max_grip", 0) for r in s3_results) / max(len(s3_results), 1)

place_results = [r for r in results if r["status"] == "place"]
n_real_place = sum(1 for r in place_results
                   if r.get("phase_b_dur", 0) >= 30
                   and r.get("arm1_max_b", -999) - (r.get("arm1_entry", 0) or 0) > 1.0)
n_suspect_place = n_place - n_real_place

print(f"\n{'='*60}")
print(f"  Results: {n} episodes")
print(f"  Place:      {n_place}/{n} ({n_place/max(n,1)*100:.0f}%) — REAL={n_real_place} SUSPECT={n_suspect_place}")
print(f"  Drop:       {n_drop}/{n} (PhaseA={n_drop_a}, PhaseB={n_drop_b})")
print(f"  Timeout:    {n_timeout}/{n}")
print(f"  S2 Fail:    {n_s2_fail}/{n}")
print(f"  ---")
print(f"  Avg PhaseA arm_err: {avg_arm_err:.3f}")
print(f"  Avg min src_dst:    {avg_min_src:.3f}")
print(f"  Avg max grip_open:  {avg_max_grip:.3f}")
# Phase B arm lowering stats
phb_results = [r for r in s3_results if r.get("arm1_entry") is not None]
if phb_results:
    arm1_deltas = [r["arm1_max_b"] - r["arm1_entry"] for r in phb_results]
    n_arm_moved = sum(1 for d in arm1_deltas if d > 0.5)
    n_arm_lowered = sum(1 for d in arm1_deltas if d > 1.5)
    n_arm_place_ready = sum(1 for d in arm1_deltas if d > 2.5)
    print(f"  ---")
    print(f"  Phase B arm lowering ({len(phb_results)} episodes):")
    print(f"    arm1 Δ>0.5 (started):    {n_arm_moved}/{len(phb_results)}")
    print(f"    arm1 Δ>1.5 (halfway):    {n_arm_lowered}/{len(phb_results)}")
    print(f"    arm1 Δ>2.5 (place-ready): {n_arm_place_ready}/{len(phb_results)}")
    avg_delta = sum(arm1_deltas) / len(arm1_deltas)
    max_delta = max(arm1_deltas)
    print(f"    avg arm1 Δ: {avg_delta:.3f}, max: {max_delta:.3f}")
# Ghost carry
ghost_eps = [r for r in s3_results if r.get("ghost_cf0", 0) >= 5]
if ghost_eps:
    print(f"  ---")
    print(f"  Ghost carry (cf0≥5): {len(ghost_eps)} episodes")
print(f"  ---")
print(f"  Per episode:")
for i, r in enumerate(results):
    phb_str = ""
    if r.get("arm1_entry") is not None:
        arm1_d = r["arm1_max_b"] - r["arm1_entry"]
        phb_str = f" phB={r['phase_b_dur']:4d} arm1Δ={arm1_d:+.2f}"
    ghost_str = f" ⚠ghost" if r.get("ghost_cf0", 0) >= 5 else ""
    print(f"    ep{i+1}: {r['status']:>7} | steps={r.get('s3_steps',0):4d} phA={r.get('phase_a_steps',0):3d}"
          f" arm_err={r.get('arm_err',0):.3f} min_src={r.get('min_src_dst',0):.3f}"
          f" max_grip={r.get('max_grip',0):.3f}{phb_str}{ghost_str}")
print(f"{'='*60}")

sim_app.close()
