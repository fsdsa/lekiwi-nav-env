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
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

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
parser.add_argument("--dest_object_scale", type=float, default=0.56)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--s2_max_steps", type=int, default=800)
parser.add_argument("--s2_lift_hold", type=int, default=200)
parser.add_argument("--s3_max_steps", type=int, default=3000)
parser.add_argument("--inference_steps", type=int, default=8)
parser.add_argument("--s3_phase_b_only", action="store_true")
parser.add_argument("--skip_s2", action="store_true", help="S2 없이 carry pose에서 바로 S3 시작")
parser.add_argument("--results_jsonl", type=str, default="")
parser.add_argument("--trace_jsonl", type=str, default="")
parser.add_argument("--s3_motion_release_xy", type=float, default=0.16)
parser.add_argument("--s3_motion_release_ee_z", type=float, default=0.09)
parser.add_argument("--s3_motion_retract_grip", type=float, default=0.90)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import torch, numpy as np, math
from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from skill3_bc_obs import (
    S3_BC_OBS_EE23_DIM,
    S3_BC_OBS_MOTION24_DIM,
    build_s3_bc_obs,
    build_s3_ee23_obs,
    build_s3_motion24_obs,
    ee_world_pos,
)

# ── Env ──
cfg = Skill2EnvCfg()
cfg.scene.num_envs = args.num_envs
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
cfg.dest_object_scale = args.dest_object_scale
cfg.grasp_success_height = 100.0
cfg.dest_spawn_dist_min = 0.6
cfg.dest_spawn_dist_max = 0.9

env = Skill2Env(cfg=cfg)
dev = env.device

S3_PHASE_B_DIST = 0.40
S3_PLACE_RADIUS = 0.18
S3_PLACE_MIN_GRIP = 0.45
S3_PLACE_MIN_UPRIGHT = 0.95
S3_REST_ARM_ERR = 0.45
S3_REST_GRIP_MAX = 0.15
S3_PLACE_OPEN_MIN_GRIP = 0.55
S3_MIN_PLACE_PHB_STEPS = 30
S3_MIN_PLACE_ARM1_DELTA = 1.0
S3_PLACE_DWELL_STEPS = 5
S3_PLACE_OPEN_DWELL_STEPS = 5
S3_REST_DWELL_STEPS = 5
S3_HANDOFF_MAX_RETRIES = 5
S3_HANDOFF_MIN_GRIP = 0.18
S3_HANDOFF_MIN_OBJZ = 0.12
S3_HANDOFF_RETRY_STEPS = 20

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
s3_is_motion24 = int(s3_cfg["obs_dim"]) == S3_BC_OBS_MOTION24_DIM
s3_is_ee23 = int(s3_cfg["obs_dim"]) == S3_BC_OBS_EE23_DIM
if s3_is_motion24 or s3_is_ee23:
    print(
        f"  S3 obs mode: {'motion24' if s3_is_motion24 else 'ee23'} "
        f"(release_xy={args.s3_motion_release_xy:.3f}, "
        f"release_ee_z={args.s3_motion_release_ee_z:.3f}, "
        f"retract_grip={args.s3_motion_retract_grip:.3f})"
    )
if s3_resip:
    print(f"  S3 ResiP: {args.s3_resip_checkpoint}")
print(f"  Episodes: {args.num_episodes} | Num envs: {args.num_envs}\n")
_bbox_off = getattr(env, "_object_bbox_center_local", None)
if _bbox_off is not None and _bbox_off.numel() == 3:
    print(f"  object_bbox_center_offset_z: {_bbox_off[2].item():.6f}\n")
if args.s3_phase_b_only:
    print("  S3 mode: Phase B only\n")

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

def build_s3_obs(
    obs_30d,
    init_pose6,
    phase_a_flag_val,
    place_open_flag_val=None,
    release_phase_flag_val=None,
    retract_started_flag_val=None,
):
    if int(s3_cfg["obs_dim"]) == S3_BC_OBS_EE23_DIM:
        return build_s3_ee23_obs(
            env,
            obs_30d,
            phase_a_flag_val,
            release_phase_flag=release_phase_flag_val,
            retract_started_flag=retract_started_flag_val,
            release_xy_thresh=float(args.s3_motion_release_xy),
            release_ee_z_thresh=float(args.s3_motion_release_ee_z),
            retract_grip_thresh=float(args.s3_motion_retract_grip),
        )
    if int(s3_cfg["obs_dim"]) == S3_BC_OBS_MOTION24_DIM:
        return build_s3_motion24_obs(
            env,
            obs_30d,
            phase_a_flag_val,
            release_phase_flag=release_phase_flag_val,
            retract_started_flag=retract_started_flag_val,
            release_xy_thresh=float(args.s3_motion_release_xy),
            release_ee_z_thresh=float(args.s3_motion_release_ee_z),
            retract_grip_thresh=float(args.s3_motion_retract_grip),
        )
    return build_s3_bc_obs(
        env,
        obs_30d,
        init_pose6,
        phase_a_flag_val,
        obs_dim=s3_cfg["obs_dim"],
        place_open_flag=place_open_flag_val,
        release_phase_flag=release_phase_flag_val,
        retract_started_flag=retract_started_flag_val,
        release_xy_thresh=float(args.s3_motion_release_xy),
        release_ee_z_thresh=float(args.s3_motion_release_ee_z),
        retract_grip_thresh=float(args.s3_motion_retract_grip),
    )


def update_motion24_latches(obs_30d, phase_flag, release_latch, retract_latch):
    if int(s3_cfg["obs_dim"]) == S3_BC_OBS_EE23_DIM:
        probe = build_s3_ee23_obs(
            env,
            obs_30d,
            phase_flag,
            release_phase_flag=None,
            retract_started_flag=None,
            release_xy_thresh=float(args.s3_motion_release_xy),
            release_ee_z_thresh=float(args.s3_motion_release_ee_z),
            retract_grip_thresh=float(args.s3_motion_retract_grip),
        )
        release_idx, retract_idx = 21, 22
    else:
        probe = build_s3_motion24_obs(
            env,
            obs_30d,
            phase_flag,
            release_phase_flag=None,
            retract_started_flag=None,
            release_xy_thresh=float(args.s3_motion_release_xy),
            release_ee_z_thresh=float(args.s3_motion_release_ee_z),
            retract_grip_thresh=float(args.s3_motion_retract_grip),
        )
        release_idx, retract_idx = 22, 23
    release_latch |= probe[:, release_idx] > 0.5
    retract_latch |= probe[:, retract_idx] > 0.5
    return release_latch, retract_latch

def source_uprightness():
    world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=dev)
    if getattr(env, "object_rigid", None) is not None:
        obj_quat = env.object_rigid.data.root_quat_w
        obj_up = quat_apply(obj_quat, world_up.expand(obj_quat.shape[0], -1))
        return obj_up[:, 2].clamp(0.0, 1.0)
    return torch.ones(1, device=dev)

# Phase-wise residual scale (must match train_resip.py main_combined)
s3_scale_a = torch.zeros(s3_cfg["act_dim"], device=dev)
s3_scale_a[0:5] = 0.0; s3_scale_a[5] = 0.0; s3_scale_a[6:9] = 0.10
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
            if torch.is_tensor(phase_a):
                phase_mask = phase_a.to(device=dev, dtype=torch.bool).unsqueeze(-1)
                s3_scale = torch.where(phase_mask, s3_scale_a.unsqueeze(0), s3_scale_b.unsqueeze(0))
            else:
                s3_scale = s3_scale_a if phase_a else s3_scale_b
            nact = base_nact + ra_mean * s3_scale
        else:
            nact = base_nact
        nact[:, 5] = torch.clamp(nact[:, 5], -0.45, 1.0)  # gripper clamp: grip_pos >= ~0.26 (끼임 방지)
        action = s3_dp.normalizer(nact, "action", forward=False)
    return action.clamp(-1, 1)

def respawn_dest_for_env_ids(env_ids: torch.Tensor):
    if env_ids.numel() == 0:
        return
    rpos = env.robot.data.root_pos_w[env_ids]
    rquat = env.robot.data.root_quat_w[env_ids]
    local_fwd = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=dev).expand(env_ids.numel(), -1)
    fwd = quat_apply(rquat, local_fwd)
    if args.s3_phase_b_only:
        dist = torch.full((env_ids.numel(),), S3_PHASE_B_DIST, dtype=torch.float32, device=dev)
        angle_noise = torch.zeros(env_ids.numel(), dtype=torch.float32, device=dev)
    else:
        dist = torch.rand(env_ids.numel(), device=dev) * (0.9 - 0.6) + 0.6
        angle_noise = (torch.rand(env_ids.numel(), device=dev) * 2 - 1) * 0.5
    cos_n = torch.cos(angle_noise)
    sin_n = torch.sin(angle_noise)
    fwd_x = fwd[:, 0] * cos_n - fwd[:, 1] * sin_n
    fwd_y = fwd[:, 0] * sin_n + fwd[:, 1] * cos_n
    dest_x = rpos[:, 0] + fwd_x * dist
    dest_y = rpos[:, 1] + fwd_y * dist
    dest_z = rpos[:, 2] - 0.03

    dest_rigid = env._dest_object_rigid
    if dest_rigid is None:
        return
    pose = dest_rigid.data.default_root_state[env_ids, :7].clone()
    pose[:, 0] = dest_x
    pose[:, 1] = dest_y
    pose[:, 2] = dest_z
    yaw = torch.rand(env_ids.numel(), device=dev) * 2 * math.pi - math.pi
    pose[:, 3] = torch.cos(yaw * 0.5)
    pose[:, 4] = 0.0
    pose[:, 5] = 0.0
    pose[:, 6] = torch.sin(yaw * 0.5)
    dest_rigid.write_root_pose_to_sim(pose, env_ids=env_ids)
    zero_vel = torch.zeros(env_ids.numel(), 6, dtype=torch.float32, device=dev)
    dest_rigid.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    env.dest_object_pos_w[env_ids, 0] = dest_x
    env.dest_object_pos_w[env_ids, 1] = dest_y
    env.dest_object_pos_w[env_ids, 2] = dest_z

def _results_path() -> str:
    if args.results_jsonl:
        return os.path.expanduser(args.results_jsonl)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.getcwd(), f"eval_s3_results_ne{args.num_envs}_ep{args.num_episodes}_{ts}.jsonl")

def _trace_path(results_path: str) -> str:
    if args.trace_jsonl:
        return os.path.expanduser(args.trace_jsonl)
    if results_path.endswith(".jsonl"):
        return results_path[:-6] + ".trace.jsonl"
    return results_path + ".trace.jsonl"

def write_results_jsonl(results, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_trace_line(trace_fh, payload: dict):
    if trace_fh is None:
        return
    trace_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

def partial_reset(env_ids: torch.Tensor):
    if env_ids.numel() == 0:
        return
    env._reset_idx(env_ids.to(dtype=torch.int64, device=dev))
    env.scene.write_data_to_sim()
    env.sim.forward()

def summarize_results(results, label: str = ""):
    n = len(results)
    n_place_rest = sum(1 for r in results if r["status"] == "place_rest")
    n_place_open = sum(1 for r in results if r["status"] == "place_open")
    n_place = sum(1 for r in results if r["status"] == "place")
    n_place_then_topple = sum(1 for r in results if r["status"] == "place_then_topple")
    n_drop = sum(1 for r in results if r["status"] == "drop")
    n_drop_a = sum(1 for r in results if r["status"] == "drop" and r.get("phase_a_steps", 0) == r.get("s3_steps", 0))
    n_drop_b = n_drop - n_drop_a
    n_timeout = sum(1 for r in results if r["status"] == "timeout")
    n_s2_fail = sum(1 for r in results if r["status"] == "s2_fail")
    n_handoff_invalid = sum(1 for r in results if r["status"] == "handoff_invalid")
    s3_results = [r for r in results if r["status"] not in ("s2_fail", "handoff_invalid")]
    avg_arm_err = sum(r.get("arm_err", 0) for r in s3_results) / max(len(s3_results), 1)
    avg_min_src = sum(r.get("min_src_dst", 0) for r in s3_results) / max(len(s3_results), 1)
    avg_max_grip = sum(r.get("max_grip", 0) for r in s3_results) / max(len(s3_results), 1)
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  Results: {n} episodes")
    print(f"  Place+rest: {n_place_rest}/{n}")
    print(f"  Place+open: {n_place_open}/{n}")
    print(f"  Place only: {n_place}/{n} ({n_place/max(n,1)*100:.0f}%)")
    print(f"  Place→topple: {n_place_then_topple}/{n}")
    print(f"  Drop:       {n_drop}/{n} (PhaseA={n_drop_a}, PhaseB={n_drop_b})")
    print(f"  Timeout:    {n_timeout}/{n}")
    print(f"  S2 Fail:    {n_s2_fail}/{n}")
    print(f"  Handoff invalid: {n_handoff_invalid}/{n}")
    print(f"  ---")
    print(f"  Avg PhaseA arm_err: {avg_arm_err:.3f}")
    print(f"  Avg min src_dst:    {avg_min_src:.3f}")
    print(f"  Avg max grip_open:  {avg_max_grip:.3f}")
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
    print(f"  ---")
    for i, r in enumerate(results):
        phb_str = ""
        if r.get("arm1_entry") is not None:
            arm1_d = r["arm1_max_b"] - r["arm1_entry"]
            phb_str = f" phB={r['phase_b_dur']:4d} arm1Δ={arm1_d:+.2f}"
        print(
            f"    ep{i+1}: {r['status']:>14} | steps={r.get('s3_steps',0):4d} phA={r.get('phase_a_steps',0):3d}"
            f" arm_err={r.get('arm_err',0):.3f} min_src={r.get('min_src_dst',0):.3f}"
            f" max_grip={r.get('max_grip',0):.3f}{phb_str}"
        )
    print(f"{'='*60}")

def run_batch_eval():
    if args.num_episodes > args.num_envs:
        raise ValueError(f"--num_episodes ({args.num_episodes}) must be <= --num_envs ({args.num_envs}) in batch eval.")
    n = args.num_envs
    env_ids_all = torch.arange(n, device=dev, dtype=torch.long)
    eval_ids = env_ids_all[: args.num_episodes]
    out_path = _results_path()
    trace_path = _trace_path(out_path)
    trace_fh = open(trace_path, "w", encoding="utf-8")
    print(f"  Batch trace: {trace_path}")

    obs, _ = env.reset()
    results = [None] * args.num_episodes
    attempts = torch.zeros(args.num_episodes, dtype=torch.long, device=dev)
    pending = torch.ones(args.num_episodes, dtype=torch.bool, device=dev)

    round_idx = 0
    while pending.any():
        round_idx += 1
        round_ids = eval_ids[pending]
        attempts[pending] += 1
        if round_idx > 1:
            partial_reset(round_ids)
            obs = env._get_observations()
        s2_dp.reset()
        print(f"\n  [Batch retry round {round_idx}] active episodes={int(pending.sum().item())}")

        round_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        round_mask[round_ids] = True

        s2_lift_counter = torch.zeros(n, dtype=torch.long, device=dev)
        s2_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        s2_failed = torch.zeros(n, dtype=torch.bool, device=dev)

        for s2_step in range(args.s2_max_steps):
            obs_30d = obs["policy"] if isinstance(obs, dict) else obs
            action = get_s2_action(obs_30d)
            action[~round_mask] = 0.0
            action[s2_failed] = 0.0
            obs, _, _, _, _ = env.step(action)

            grip_pos = env.robot.data.joint_pos[:, env.arm_idx[5]]
            grip_closed = grip_pos < float(env.cfg.grasp_gripper_threshold)
            cf = env._contact_force_per_env()
            has_contact = cf > float(env.cfg.grasp_contact_threshold)
            objZ = env.object_pos_w[:, 2] - env.scene.env_origins[:, 2]
            rootZ = env.object_rigid.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            base_dst = torch.norm(env.robot.data.root_pos_w[:, :2] - env.dest_object_pos_w[:, :2], dim=-1)

            good_hold = round_mask & (~s2_failed) & grip_closed & has_contact & (objZ > 0.05)
            s2_lift_counter = torch.where(good_hold, s2_lift_counter + 1, torch.zeros_like(s2_lift_counter))
            s2_lifted |= good_hold & (s2_lift_counter >= args.s2_lift_hold)

            fail_now = round_mask & (~s2_lifted) & (~s2_failed) & (s2_step > 20) & (objZ < 0.026)
            s2_failed |= fail_now

            if s2_step % 100 == 0 or s2_step == args.s2_max_steps - 1:
                print(
                    f"  [Batch S2 step={s2_step}] lifted={int((s2_lifted & round_mask).sum().item())}/{int(round_mask.sum().item())} "
                    f"failed={int((s2_failed & round_mask).sum().item())}"
                )
            if s2_step % 200 == 0 or s2_step == args.s2_max_steps - 1:
                for env_id in round_ids.tolist():
                    write_trace_line(trace_fh, {
                        "kind": "s2_trace",
                        "env_id": env_id,
                        "attempt": int(attempts[env_id].item()),
                        "step": s2_step,
                        "objZ": float(objZ[env_id].item()),
                        "rootZ": float(rootZ[env_id].item()),
                        "dZ": float((objZ[env_id] - rootZ[env_id]).item()),
                        "grip": float(grip_pos[env_id].item()),
                        "cf": float(cf[env_id].item()),
                        "base_dst": float(base_dst[env_id].item()),
                        "lift_counter": int(s2_lift_counter[env_id].item()),
                        "lifted": bool(s2_lifted[env_id].item()),
                        "failed": bool(s2_failed[env_id].item()),
                        "act_arm": [float(x) for x in action[env_id, :6].detach().cpu().tolist()],
                        "act_base": [float(x) for x in action[env_id, 6:9].detach().cpu().tolist()],
                    })
            if torch.all((~round_mask) | s2_lifted | s2_failed):
                break

        s2_timeout = round_mask & (~s2_lifted) & (~s2_failed)
        s2_failed |= s2_timeout
        retry_s2 = s2_failed.clone()

        valid_mask = round_mask & s2_lifted & (~s2_failed)
        valid_env_ids = valid_mask.nonzero(as_tuple=False).squeeze(-1)

        jp = env.robot.data.joint_pos
        init_pose6 = torch.cat([jp[:, env.arm_idx[:5]], jp[:, env.arm_idx[5:6]]], dim=-1).to(dev)
        respawn_dest_for_env_ids(valid_env_ids)

        s3_dp.reset()
        phase_a_active = valid_mask.clone()
        if args.s3_phase_b_only:
            phase_a_active.zero_()
        done = (~round_mask) | s2_failed
        placed = torch.zeros(n, dtype=torch.bool, device=dev)
        dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        toppled_after_place = torch.zeros(n, dtype=torch.bool, device=dev)
        handoff_invalid = torch.zeros(n, dtype=torch.bool, device=dev)

        phase_a_steps = torch.zeros(n, dtype=torch.long, device=dev)
        phase_b_step_start = torch.full((n,), -1, dtype=torch.long, device=dev)
        min_objZ = torch.full((n,), 1.0, dtype=torch.float32, device=dev)
        max_grip_open = torch.zeros(n, dtype=torch.float32, device=dev)
        min_src_dst = torch.full((n,), 10.0, dtype=torch.float32, device=dev)
        arm_err_sum = torch.zeros(n, dtype=torch.float32, device=dev)
        arm_err_count = torch.zeros(n, dtype=torch.long, device=dev)
        arm1_at_phase_b_entry = torch.full((n,), float("nan"), dtype=torch.float32, device=dev)
        arm1_max_in_phase_b = torch.full((n,), -999.0, dtype=torch.float32, device=dev)
        objZ_at_phase_b_entry = torch.full((n,), float("nan"), dtype=torch.float32, device=dev)
        objZ_min_in_phase_b = torch.full((n,), 999.0, dtype=torch.float32, device=dev)
        stable_place_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        place_open_confirmed = torch.zeros(n, dtype=torch.bool, device=dev)
        place_open_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        arm_rest_reached = torch.zeros(n, dtype=torch.bool, device=dev)
        rest_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        finish_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        place_dwell = torch.zeros(n, dtype=torch.long, device=dev)
        place_open_dwell = torch.zeros(n, dtype=torch.long, device=dev)
        rest_dwell = torch.zeros(n, dtype=torch.long, device=dev)
        release_phase_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        retract_started_latch = torch.zeros(n, dtype=torch.bool, device=dev)

        if args.s3_phase_b_only and valid_env_ids.numel() > 0:
            objZ = env.object_pos_w[:, 2] - env.scene.env_origins[:, 2]
            grip = env.robot.data.joint_pos[:, env.arm_idx[5]]
            cf = env._contact_force_per_env()
            arm1 = env.robot.data.joint_pos[:, env.arm_idx[1]]
            arm1_at_phase_b_entry[valid_env_ids] = arm1[valid_env_ids]
            objZ_at_phase_b_entry[valid_env_ids] = objZ[valid_env_ids]
            phase_b_step_start[valid_env_ids] = 0
            bad_start = valid_mask & (
                (grip < S3_HANDOFF_MIN_GRIP)
                | (cf <= float(env.cfg.grasp_contact_threshold))
                | (objZ < S3_HANDOFF_MIN_OBJZ)
            )
            for env_id in valid_env_ids.tolist():
                write_trace_line(trace_fh, {
                    "kind": "phase_b_start",
                    "env_id": env_id,
                    "attempt": int(attempts[env_id].item()),
                    "base_dst": float(torch.norm(env.robot.data.root_pos_w[env_id, :2] - env.dest_object_pos_w[env_id, :2]).item()),
                    "arm": [float(x) for x in env.robot.data.joint_pos[env_id, env.arm_idx[:5]].detach().cpu().tolist()],
                    "grip": float(grip[env_id].item()),
                    "objZ": float(objZ[env_id].item()),
                    "cf": float(cf[env_id].item()),
                    "upright": float(source_uprightness()[env_id].item()),
                })
            handoff_invalid |= bad_start
            done |= bad_start
            finish_step[bad_start] = 0

        for s3_step in range(args.s3_max_steps):
            active = round_mask & (~done)
            if not active.any():
                break

            obs_30d = obs["policy"] if isinstance(obs, dict) else obs
            base_dst = torch.norm(env.robot.data.root_pos_w[:, :2] - env.dest_object_pos_w[:, :2], dim=-1)

            to_phase_b = active & phase_a_active & (base_dst <= S3_PHASE_B_DIST)
            if to_phase_b.any():
                jp_now = env.robot.data.joint_pos
                objZ_now = env.object_pos_w[:, 2] - env.scene.env_origins[:, 2]
                arm1_at_phase_b_entry[to_phase_b] = jp_now[to_phase_b, env.arm_idx[1]]
                objZ_at_phase_b_entry[to_phase_b] = objZ_now[to_phase_b]
                phase_b_step_start[to_phase_b] = s3_step
                phase_a_active[to_phase_b] = False

            phase_flag = torch.zeros(n, dtype=torch.float32, device=dev) if args.s3_phase_b_only else phase_a_active.float()
            if s3_is_motion24:
                release_phase_latch, retract_started_latch = update_motion24_latches(
                    obs_30d, phase_flag, release_phase_latch, retract_started_latch
                )
            else:
                arm1_pre = env.robot.data.joint_pos[:, env.arm_idx[1]]
                grip_pre = env.robot.data.joint_pos[:, env.arm_idx[5]]
                src_dst_pre = torch.norm(env.object_pos_w[:, :2] - env.dest_object_pos_w[:, :2], dim=-1)
                phase_b_pre = active & (~phase_a_active)
                src_h_pre = env.object_pos_w[:, 2] - env.scene.env_origins[:, 2]
                cf_pre = env._contact_force_per_env()
                release_phase_latch |= phase_b_pre & (arm1_pre >= 2.0) & (src_dst_pre <= 0.18)
                retract_started_latch |= (
                    release_phase_latch
                    & (grip_pre >= 0.55)
                    & (src_dst_pre <= 0.18)
                    & (src_h_pre <= 0.055)
                    & (cf_pre <= float(env.cfg.grasp_contact_threshold))
                )
            s3_obs = build_s3_obs(
                obs_30d,
                init_pose6,
                phase_flag,
                place_open_flag_val=place_open_confirmed.float(),
                release_phase_flag_val=release_phase_latch.float(),
                retract_started_flag_val=retract_started_latch.float(),
            )
            action = get_s3_action(s3_obs, phase_a=phase_a_active)
            action[done] = 0.0
            obs, _, _, _, _ = env.step(action)

            objZ = env.object_pos_w[:, 2] - env.scene.env_origins[:, 2]
            rootZ = env.object_rigid.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            grip = env.robot.data.joint_pos[:, env.arm_idx[5]]
            cf = env._contact_force_per_env()
            has_contact = cf > float(env.cfg.grasp_contact_threshold)
            src_dst = torch.norm(env.object_pos_w[:, :2] - env.dest_object_pos_w[:, :2], dim=-1)
            upright = source_uprightness()
            base_dst = torch.norm(env.robot.data.root_pos_w[:, :2] - env.dest_object_pos_w[:, :2], dim=-1)
            arm1 = env.robot.data.joint_pos[:, env.arm_idx[1]]
            arm_now = env.robot.data.joint_pos[:, env.arm_idx[:5]]
            arm_rest = env.robot.data.default_joint_pos[:, env.arm_idx[:5]]
            arm_rest_err = torch.sqrt(torch.sum((arm_now - arm_rest) ** 2, dim=-1))

            min_objZ = torch.where(active, torch.minimum(min_objZ, objZ), min_objZ)
            max_grip_open = torch.where(active, torch.maximum(max_grip_open, grip), max_grip_open)
            min_src_dst = torch.where(active, torch.minimum(min_src_dst, src_dst), min_src_dst)

            phase_a_steps = phase_a_steps + (active & phase_a_active).long()
            if (active & phase_a_active).any():
                init_arm = init_pose6[:, :5]
                arm_now = env.robot.data.joint_pos[:, env.arm_idx[:5]]
                arm_err = torch.sqrt(torch.sum((arm_now - init_arm) ** 2, dim=-1))
                arm_err_sum += torch.where(active & phase_a_active, arm_err, torch.zeros_like(arm_err))
                arm_err_count += (active & phase_a_active).long()

            phase_b_mask = active & (~phase_a_active)
            arm1_max_in_phase_b = torch.where(phase_b_mask, torch.maximum(arm1_max_in_phase_b, arm1), arm1_max_in_phase_b)
            objZ_min_in_phase_b = torch.where(phase_b_mask, torch.minimum(objZ_min_in_phase_b, objZ), objZ_min_in_phase_b)

            stable_place_candidate = (
                active
                & (src_dst < S3_PLACE_RADIUS)
                & (~has_contact)
                & (objZ > 0.029)
                & (objZ < 0.05)
                & (upright > S3_PLACE_MIN_UPRIGHT)
                & (grip > S3_PLACE_MIN_GRIP)
                & (s3_step > 50)
                & (phase_b_step_start >= 0)
                & ((s3_step - phase_b_step_start) >= S3_MIN_PLACE_PHB_STEPS)
                & ((arm1 - arm1_at_phase_b_entry) > S3_MIN_PLACE_ARM1_DELTA)
            )
            place_dwell = torch.where(
                stable_place_candidate,
                place_dwell + 1,
                torch.zeros_like(place_dwell),
            )
            stable_place_now = stable_place_candidate & (place_dwell >= S3_PLACE_DWELL_STEPS)
            new_place = stable_place_now & (~placed)
            placed |= new_place
            stable_place_step[new_place] = s3_step

            place_open_candidate = (
                placed
                & stable_place_now
                & (~has_contact)
                & (upright > S3_PLACE_MIN_UPRIGHT)
                & (grip > S3_PLACE_OPEN_MIN_GRIP)
            )
            place_open_dwell = torch.where(
                place_open_candidate,
                place_open_dwell + 1,
                torch.zeros_like(place_open_dwell),
            )
            place_open_now = place_open_candidate & (place_open_dwell >= S3_PLACE_OPEN_DWELL_STEPS)
            new_place_open = place_open_now & (~place_open_confirmed)
            place_open_confirmed |= new_place_open
            place_open_step[new_place_open] = s3_step

            rest_candidate = (
                placed
                & place_open_confirmed
                & retract_started_latch
                & (~has_contact)
                & (arm_rest_err < S3_REST_ARM_ERR)
                & (grip < S3_REST_GRIP_MAX)
                & (upright > S3_PLACE_MIN_UPRIGHT)
            )
            rest_dwell = torch.where(
                rest_candidate,
                rest_dwell + 1,
                torch.zeros_like(rest_dwell),
            )
            rest_now = rest_candidate & (rest_dwell >= S3_REST_DWELL_STEPS)
            new_rest = rest_now & (~arm_rest_reached)
            arm_rest_reached |= new_rest
            rest_step[new_rest] = s3_step

            topple_now = active & placed & (~stable_place_now) & (stable_place_step >= 0) & (s3_step > stable_place_step) & (objZ < 0.029)
            new_topple = topple_now & (~done)
            toppled_after_place |= new_topple
            done |= new_topple
            finish_step[new_topple] = s3_step

            invalid_now = torch.zeros_like(active)
            if args.s3_phase_b_only and s3_step <= S3_HANDOFF_RETRY_STEPS:
                invalid_now = active & (~placed) & (~has_contact) & ((objZ < 0.05) | (grip < S3_HANDOFF_MIN_GRIP))
            if invalid_now.any():
                new_invalid = invalid_now & (~done)
                handoff_invalid |= new_invalid
                done |= new_invalid
                finish_step[new_invalid] = s3_step

            drop_thresh = torch.where(phase_a_active, 0.04, 0.029)
            drop_now = active & (~placed) & (objZ < drop_thresh) & (s3_step > 10)
            new_drop = drop_now & (~done)
            dropped |= new_drop
            done |= new_drop
            finish_step[new_drop] = s3_step

            if s3_step % 100 == 0 or s3_step == args.s3_max_steps - 1:
                print(
                    f"  [Batch S3 step={s3_step}] active={int(active.sum().item())} "
                    f"place={int((placed & round_mask).sum().item())}/{int(round_mask.sum().item())} "
                    f"open={int((place_open_confirmed & round_mask).sum().item())} "
                    f"rest={int((arm_rest_reached & round_mask).sum().item())} "
                    f"drop={int((dropped & round_mask).sum().item())} "
                    f"topple={int((toppled_after_place & round_mask).sum().item())} "
                    f"handoff_invalid={int((handoff_invalid & round_mask).sum().item())}"
                )
            for env_id in round_ids.tolist():
                if not active[env_id]:
                    continue
                interval = 50 if phase_a_active[env_id] else 10
                if s3_step % interval == 0:
                    write_trace_line(trace_fh, {
                        "kind": "s3_trace",
                        "env_id": env_id,
                        "attempt": int(attempts[env_id].item()),
                        "phase": "A" if phase_a_active[env_id].item() else "B",
                        "step": s3_step,
                        "phB_t": int(s3_step - phase_b_step_start[env_id].item()) if phase_b_step_start[env_id] >= 0 else 0,
                        "objZ": float(objZ[env_id].item()),
                        "rootZ": float(rootZ[env_id].item()),
                        "dZ": float((objZ[env_id] - rootZ[env_id]).item()),
                        "grip": float(grip[env_id].item()),
                        "upright": float(upright[env_id].item()),
                        "base_dst": float(base_dst[env_id].item()),
                        "src_dst": float(src_dst[env_id].item()),
                        "cf": float(cf[env_id].item()),
                        "arm": [float(x) for x in arm_now[env_id].detach().cpu().tolist()],
                        "act_arm": [float(x) for x in action[env_id, :6].detach().cpu().tolist()],
                        "act_base": [float(x) for x in action[env_id, 6:9].detach().cpu().tolist()],
                        "release_phase_flag": bool(release_phase_latch[env_id].item()),
                        "retract_started_flag": bool(retract_started_latch[env_id].item()),
                    })

            for env_id in new_place.nonzero(as_tuple=False).squeeze(-1).tolist():
                write_trace_line(trace_fh, {
                    "kind": "place",
                    "env_id": env_id,
                    "attempt": int(attempts[env_id].item()),
                    "step": s3_step,
                    "objZ": float(objZ[env_id].item()),
                    "rootZ": float(rootZ[env_id].item()),
                    "dZ": float((objZ[env_id] - rootZ[env_id]).item()),
                    "src_dst": float(src_dst[env_id].item()),
                    "grip": float(grip[env_id].item()),
                    "upright": float(upright[env_id].item()),
                    "cf": float(cf[env_id].item()),
                    "arm_rest_err": float(arm_rest_err[env_id].item()),
                    "release_phase_flag": bool(release_phase_latch[env_id].item()),
                    "retract_started_flag": bool(retract_started_latch[env_id].item()),
                })
            for env_id in new_rest.nonzero(as_tuple=False).squeeze(-1).tolist():
                write_trace_line(trace_fh, {
                    "kind": "place_rest",
                    "env_id": env_id,
                    "attempt": int(attempts[env_id].item()),
                    "step": s3_step,
                    "objZ": float(objZ[env_id].item()),
                    "src_dst": float(src_dst[env_id].item()),
                    "grip": float(grip[env_id].item()),
                    "upright": float(upright[env_id].item()),
                    "arm_rest_err": float(arm_rest_err[env_id].item()),
                    "release_phase_flag": bool(release_phase_latch[env_id].item()),
                    "retract_started_flag": bool(retract_started_latch[env_id].item()),
                })
            for env_id in new_place_open.nonzero(as_tuple=False).squeeze(-1).tolist():
                write_trace_line(trace_fh, {
                    "kind": "place_open",
                    "env_id": env_id,
                    "attempt": int(attempts[env_id].item()),
                    "step": s3_step,
                    "objZ": float(objZ[env_id].item()),
                    "src_dst": float(src_dst[env_id].item()),
                    "grip": float(grip[env_id].item()),
                    "upright": float(upright[env_id].item()),
                    "cf": float(cf[env_id].item()),
                    "release_phase_flag": bool(release_phase_latch[env_id].item()),
                    "retract_started_flag": bool(retract_started_latch[env_id].item()),
                })
            for env_id in new_topple.nonzero(as_tuple=False).squeeze(-1).tolist():
                write_trace_line(trace_fh, {
                    "kind": "place_then_topple",
                    "env_id": env_id,
                    "attempt": int(attempts[env_id].item()),
                    "step": s3_step,
                    "objZ": float(objZ[env_id].item()),
                    "src_dst": float(src_dst[env_id].item()),
                    "grip": float(grip[env_id].item()),
                    "upright": float(upright[env_id].item()),
                    "release_phase_flag": bool(release_phase_latch[env_id].item()),
                    "retract_started_flag": bool(retract_started_latch[env_id].item()),
                })
            for env_id in new_invalid.nonzero(as_tuple=False).squeeze(-1).tolist() if invalid_now.any() else []:
                write_trace_line(trace_fh, {
                    "kind": "handoff_invalid",
                    "env_id": env_id,
                    "attempt": int(attempts[env_id].item()),
                    "step": s3_step,
                    "objZ": float(objZ[env_id].item()),
                    "grip": float(grip[env_id].item()),
                    "cf": float(cf[env_id].item()),
                    "release_phase_flag": bool(release_phase_latch[env_id].item()),
                    "retract_started_flag": bool(retract_started_latch[env_id].item()),
                })
            for env_id in new_drop.nonzero(as_tuple=False).squeeze(-1).tolist():
                write_trace_line(trace_fh, {
                    "kind": "drop",
                    "env_id": env_id,
                    "attempt": int(attempts[env_id].item()),
                    "step": s3_step,
                    "phase": "A" if phase_a_active[env_id].item() else "B",
                    "objZ": float(objZ[env_id].item()),
                    "src_dst": float(src_dst[env_id].item()),
                    "grip": float(grip[env_id].item()),
                    "upright": float(upright[env_id].item()),
                    "cf": float(cf[env_id].item()),
                    "release_phase_flag": bool(release_phase_latch[env_id].item()),
                    "retract_started_flag": bool(retract_started_latch[env_id].item()),
                })

        unresolved = round_mask & (~done)
        finish_step[unresolved] = args.s3_max_steps - 1
        retry_handoff = handoff_invalid.clone()

        for env_id in round_ids.tolist():
            slot = env_id
            if retry_s2[env_id]:
                continue

            if retry_handoff[env_id]:
                continue

            if toppled_after_place[env_id]:
                status = "place_then_topple"
            elif arm_rest_reached[env_id]:
                status = "place_rest"
            elif place_open_confirmed[env_id]:
                status = "place_open"
            elif placed[env_id]:
                status = "place"
            elif dropped[env_id]:
                status = "drop"
            else:
                status = "timeout"

            arm_err = (arm_err_sum[env_id] / arm_err_count[env_id].clamp(min=1)).item()
            arm1_entry = None if torch.isnan(arm1_at_phase_b_entry[env_id]) else arm1_at_phase_b_entry[env_id].item()
            arm1_max_b = None if arm1_max_in_phase_b[env_id] < -100 else arm1_max_in_phase_b[env_id].item()
            objZ_entry = None if torch.isnan(objZ_at_phase_b_entry[env_id]) else objZ_at_phase_b_entry[env_id].item()
            objZ_min_b = None if objZ_min_in_phase_b[env_id] > 900 else objZ_min_in_phase_b[env_id].item()
            phase_b_dur = -1
            if phase_b_step_start[env_id] >= 0:
                phase_b_dur = int(finish_step[env_id].item() - phase_b_step_start[env_id].item())

            results[slot] = {
                "env_id": slot,
                "status": status,
                "attempts": int(attempts[slot].item()),
                "s3_steps": int(finish_step[env_id].item()),
                "phase_a_steps": int(phase_a_steps[env_id].item()),
                "phase_b_dur": phase_b_dur,
                "min_objZ": float(min_objZ[env_id].item()),
                "max_grip": float(max_grip_open[env_id].item()),
                "min_src_dst": float(min_src_dst[env_id].item()),
                "arm_err": float(arm_err),
                "arm1_entry": arm1_entry,
                "arm1_max_b": arm1_max_b,
                "objZ_entry": objZ_entry,
                "objZ_min_b": objZ_min_b,
                "place_open_confirmed": bool(place_open_confirmed[env_id].item()),
                "place_open_step": int(place_open_step[env_id].item()),
                "arm_rest_reached": bool(arm_rest_reached[env_id].item()),
                "rest_step": int(rest_step[env_id].item()),
                "release_phase_flag": bool(release_phase_latch[env_id].item()),
                "retract_started_flag": bool(retract_started_latch[env_id].item()),
            }
            write_trace_line(trace_fh, {
                "kind": "episode_done",
                "env_id": slot,
                "attempt": int(attempts[slot].item()),
                "status": status,
                "s3_steps": int(finish_step[env_id].item()),
                "phase_a_steps": int(phase_a_steps[env_id].item()),
                "phase_b_dur": phase_b_dur,
                "min_objZ": float(min_objZ[env_id].item()),
                "max_grip": float(max_grip_open[env_id].item()),
                "min_src_dst": float(min_src_dst[env_id].item()),
                "arm_err": float(arm_err),
                "place_open_confirmed": bool(place_open_confirmed[env_id].item()),
                "place_open_step": int(place_open_step[env_id].item()),
                "arm_rest_reached": bool(arm_rest_reached[env_id].item()),
                "rest_step": int(rest_step[env_id].item()),
                "release_phase_flag": bool(release_phase_latch[env_id].item()),
                "retract_started_flag": bool(retract_started_latch[env_id].item()),
            })
            pending[slot] = False

        print(
            f"  [Batch round {round_idx} done] completed={int((~pending).sum().item())}/{args.num_episodes} "
            f"remaining={int(pending.sum().item())}"
        )

    results = [r for r in results if r is not None]
    trace_fh.close()

    out_path = _results_path()
    write_results_jsonl(results, out_path)
    print(f"\n  Saved batch results: {out_path}")
    summarize_results(results, label=f"Batch eval | num_envs={args.num_envs}")
    sim_app.close()
    raise SystemExit(0)

# ── Main loop ──
if args.num_envs > 1:
    run_batch_eval()

results = []
for ep in range(args.num_episodes):
    episode_recorded = False
    for handoff_attempt in range(S3_HANDOFF_MAX_RETRIES):
        if handoff_attempt == 0:
            print(f"  [Episode {ep+1}/{args.num_episodes}]")
        else:
            print(f"  [Episode {ep+1}/{args.num_episodes} retry {handoff_attempt+1}/{S3_HANDOFF_MAX_RETRIES}]")

        # Phase 1: S2 expert lift (or skip)
        if args.skip_s2:
            # Skip S2 — joint position 직접 설정
            obs, _ = env.reset()
            carry_arm = torch.tensor([-0.07, -0.19, 0.28, -0.96, -0.03], device=dev)
            carry_grip = torch.tensor([0.30], device=dev)

            # 직접 joint position 설정
            jp = env.robot.data.joint_pos.clone()
            jp[0, env.arm_idx[:5]] = carry_arm
            jp[0, env.arm_idx[5]] = carry_grip[0]
            env.robot.write_joint_state_to_sim(jp, env.robot.data.joint_vel * 0)

            # 물체를 EE 위치에 이동
            ee_pos_val = ee_world_pos(env)
            obj_target = ee_pos_val[0].clone()
            obj_target[2] -= 0.03  # EE 아래 (그리퍼 안쪽)
            if hasattr(env, 'object_rigid') and env.object_rigid is not None:
                obj_state = env.object_rigid.data.default_root_state[0:1].clone()
                obj_state[0, :3] = obj_target
                env.object_rigid.write_root_pose_to_sim(obj_state[:, :7])
                env.object_pos_w[0] = obj_target

            # 물리 안정화 (50 step) — 매 step arm+grip 강제 유지
            for _ in range(50):
                action = torch.zeros(1, 9, device=dev)
                obs, _, _, _, _ = env.step(action)
                jp = env.robot.data.joint_pos.clone()
                jp[0, env.arm_idx[:5]] = carry_arm
                jp[0, env.arm_idx[5]] = carry_grip[0]
                env.robot.write_joint_state_to_sim(jp, env.robot.data.joint_vel * 0)

            lifted = True
            objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            cf = env._contact_force_per_env()[0].item() if env.contact_sensor is not None else 0
            print(f"    [SKIP S2] carry pose, objZ={objZ:.3f} cf={cf:.1f} grip={env.robot.data.joint_pos[0, env.arm_idx[5]].item():.3f}")
        else:
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
                episode_recorded = True
                break

        # Capture init_arm_pose
        jp = env.robot.data.joint_pos[0]
        init_pose6 = torch.cat([jp[env.arm_idx[:5]], jp[env.arm_idx[5:6]]]).to(dev)
        init_arm_np = jp[env.arm_idx[:5]].cpu().tolist()
        init_grip = jp[env.arm_idx[5]].item()
        print(f"    [S2→S3] arm3={init_arm_np[3]:+.3f} grip={init_grip:.3f}")

        # Dest respawn:
        # - default: robot front 0.6~0.9m
        # - phase-B-only eval: start directly from the Phase-B transition distance
        from isaaclab.utils.math import quat_apply as _qa
        rpos = env.robot.data.root_pos_w[0:1]
        rquat = env.robot.data.root_quat_w[0:1]
        fwd = _qa(rquat, torch.tensor([[0, 1, 0]], dtype=torch.float32, device=dev))
        if args.s3_phase_b_only:
            dist = torch.full((1,), S3_PHASE_B_DIST, dtype=torch.float32, device=dev)
        else:
            dist = torch.rand(1, device=dev) * (0.9 - 0.6) + 0.6
        if args.s3_phase_b_only:
            angle_noise = torch.zeros(1, device=dev)
        else:
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
        phase_a_active = not args.s3_phase_b_only
        dropped = False
        placed = False
        toppled_after_place = False
        place_open_confirmed = False
        rest_reached = False
        invalid_handoff = False
        invalid_reason = ""
        phase_a_steps = 0
        phase_b_step_start = 0 if args.s3_phase_b_only else -1
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
        stable_place_step = -1
        place_open_step = -1
        rest_step = -1
        place_dwell = 0
        place_open_dwell = 0
        rest_dwell = 0
        release_phase_latch = False
        retract_started_latch = False

        base_dst_start = torch.norm(env.robot.data.root_pos_w[0, :2] - env.dest_object_pos_w[0, :2]).item()

        if args.s3_phase_b_only:
            objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            rootZ = (env.object_rigid.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            objDz = objZ - rootZ
            jp_at_start = env.robot.data.joint_pos[0]
            arm_at_start = jp_at_start[env.arm_idx[:5]].cpu().tolist()
            grip_at_start = jp_at_start[env.arm_idx[5]].item()
            cf_val = 0.0
            if env.contact_sensor is not None:
                cf_val = env._contact_force_per_env()[0].item()
            upright_at_start = source_uprightness()[0].item()
            arm1_at_phase_b_entry = arm_at_start[1]
            objZ_at_phase_b_entry = objZ
            grip_at_phase_b_entry = grip_at_start
            arm_start_str = ",".join(f"{v:+.3f}" for v in arm_at_start)
            print(
                f"    [Phase B start] base_dst={base_dst_start:.3f} arm=[{arm_start_str}] "
                f"grip={grip_at_start:.3f} objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} "
                f"upright={upright_at_start:.3f} cf={cf_val:.1f}"
            )
            if (
                grip_at_start < S3_HANDOFF_MIN_GRIP
                or cf_val <= float(env.cfg.grasp_contact_threshold)
                or objZ < S3_HANDOFF_MIN_OBJZ
            ):
                invalid_handoff = True
                invalid_reason = (
                    f"start grip={grip_at_start:.3f} cf={cf_val:.1f} objZ={objZ:.3f}"
                )

        if invalid_handoff:
            print(f"    [S3 RETRY] invalid handoff — {invalid_reason}")
            if handoff_attempt == S3_HANDOFF_MAX_RETRIES - 1:
                results.append({"status": "handoff_invalid"})
                print(f"  Episode {ep+1}: handoff_invalid after {S3_HANDOFF_MAX_RETRIES} retries\n")
                episode_recorded = True
            continue

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
                rootZ_at_trans = (env.object_rigid.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                dz_at_trans = objZ - rootZ_at_trans
                upright_at_trans = source_uprightness()[0].item()
                print(f"    [Phase A→B] step={s3_step} base_dst={base_dst:.3f} arm=[{arm_trans_str}] "
                      f"grip={grip_at_trans:.3f} objZ={objZ:.3f} rootZ={rootZ_at_trans:.3f} dZ={dz_at_trans:.3f} "
                      f"upright={upright_at_trans:.3f} cf={cf_val:.1f}{ghost_str}")

            flag_val = 0.0 if args.s3_phase_b_only else (1.0 if phase_a_active else 0.0)
            if s3_is_motion24:
                _rel = torch.tensor([release_phase_latch], dtype=torch.bool, device=dev)
                _ret = torch.tensor([retract_started_latch], dtype=torch.bool, device=dev)
                _phase = torch.tensor([flag_val], dtype=torch.float32, device=dev)
                _rel, _ret = update_motion24_latches(obs_30d, _phase, _rel, _ret)
                release_phase_latch = bool(_rel[0].item())
                retract_started_latch = bool(_ret[0].item())
            else:
                arm1_pre = env.robot.data.joint_pos[0, env.arm_idx[1]].item()
                grip_pre = env.robot.data.joint_pos[0, env.arm_idx[5]].item()
                src_dst_pre = torch.norm(env.object_pos_w[0, :2] - env.dest_object_pos_w[0, :2]).item()
                src_h_pre = env.object_pos_w[0, 2].item() - env.scene.env_origins[0, 2].item()
                cf_pre = env._contact_force_per_env()[0].item()
                if (not phase_a_active) and arm1_pre >= 2.0 and src_dst_pre <= 0.18:
                    release_phase_latch = True
                if (
                    release_phase_latch
                    and grip_pre >= 0.55
                    and src_dst_pre <= 0.18
                    and src_h_pre <= 0.055
                    and cf_pre <= float(env.cfg.grasp_contact_threshold)
                ):
                    retract_started_latch = True
            s3_obs = build_s3_obs(
                obs_30d,
                init_pose6,
                flag_val,
                place_open_flag_val=float(place_open_confirmed),
                release_phase_flag_val=float(release_phase_latch),
                retract_started_flag_val=float(retract_started_latch),
            )
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

            rootZ = (env.object_rigid.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
            objDz = objZ - rootZ
            upright = source_uprightness()[0].item()
            jp = env.robot.data.joint_pos[0]
            arm_jp = jp[env.arm_idx[:5]].cpu().tolist()
            grip_pos = jp[env.arm_idx[5]].item()
            rest_arm_jp = env.robot.data.default_joint_pos[0, env.arm_idx[:5]].cpu().tolist()
            arm_rest_err = sum((arm_jp[i] - rest_arm_jp[i])**2 for i in range(5)) ** 0.5
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

            phase_b_dur_now = s3_step - phase_b_step_start if phase_b_step_start >= 0 else -1
            arm1_delta_now = arm_jp[1] - arm1_at_phase_b_entry if arm1_at_phase_b_entry is not None else 0.0
            stable_place_candidate = (
                src_dst < S3_PLACE_RADIUS
                and (not has_contact)
                and objZ > 0.029
                and objZ < 0.05
                and upright > S3_PLACE_MIN_UPRIGHT
                and grip_pos > S3_PLACE_MIN_GRIP
                and s3_step > 50
                and phase_b_dur_now >= S3_MIN_PLACE_PHB_STEPS
                and arm1_delta_now > S3_MIN_PLACE_ARM1_DELTA
            )
            place_dwell = place_dwell + 1 if stable_place_candidate else 0
            stable_place_now = stable_place_candidate and (place_dwell >= S3_PLACE_DWELL_STEPS)

            place_open_candidate = (
                placed
                and stable_place_now
                and (not has_contact)
                and upright > S3_PLACE_MIN_UPRIGHT
                and grip_pos > S3_PLACE_OPEN_MIN_GRIP
            )
            place_open_dwell = place_open_dwell + 1 if place_open_candidate else 0
            place_open_now = place_open_candidate and (place_open_dwell >= S3_PLACE_OPEN_DWELL_STEPS)

            rest_candidate = (
                placed
                and place_open_confirmed
                and retract_started_latch
                and (not has_contact)
                and arm_rest_err < S3_REST_ARM_ERR
                and grip_pos < S3_REST_GRIP_MAX
                and upright > S3_PLACE_MIN_UPRIGHT
            )
            rest_dwell = rest_dwell + 1 if rest_candidate else 0
            rest_now = rest_candidate and (rest_dwell >= S3_REST_DWELL_STEPS)

            if stable_place_now and not placed:
                print(
                    f"    [PLACE] step={s3_step} phB_dur={phase_b_dur_now} dwell={place_dwell} | "
                    f"objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} src_dst={src_dst:.3f} "
                    f"grip={grip_pos:.3f} upright={upright:.3f} cf={cf_val:.1f} | "
                    f"arm=[{arm_str}] | "
                    f"arm1: entry={arm1_at_phase_b_entry:+.3f}→now={arm_jp[1]:+.3f} (Δ={arm1_delta_now:+.3f}) | "
                    f"objZ: entry={objZ_at_phase_b_entry:.3f}→now={objZ:.3f} (Δ={objZ - objZ_at_phase_b_entry:+.3f})"
                )
                placed = True
                stable_place_step = s3_step

            if place_open_now and not place_open_confirmed:
                print(
                    f"    [PLACE+OPEN] step={s3_step} dwell={place_open_dwell} | "
                    f"objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} "
                    f"src_dst={src_dst:.3f} grip={grip_pos:.3f} arm_rest_err={arm_rest_err:.3f} upright={upright:.3f}"
                )
                place_open_confirmed = True
                place_open_step = s3_step

            if rest_now and not rest_reached:
                print(
                    f"    [PLACE+REST] step={s3_step} dwell={rest_dwell} | "
                    f"objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} "
                    f"src_dst={src_dst:.3f} grip={grip_pos:.3f} arm_rest_err={arm_rest_err:.3f} upright={upright:.3f}"
                )
                rest_reached = True
                rest_step = s3_step

            if placed and stable_place_step >= 0 and s3_step > stable_place_step and objZ < 0.029:
                print(
                    f"    [TOPPLED-AFTER-PLACE] step={s3_step} | "
                    f"objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} "
                    f"src_dst={src_dst:.3f} grip={grip_pos:.3f} upright={upright:.3f} cf={cf_val:.1f}"
                )
                toppled_after_place = True
                break

            if (
                args.s3_phase_b_only
                and s3_step <= S3_HANDOFF_RETRY_STEPS
                and (not placed)
                and (not has_contact)
                and (objZ < 0.05 or grip_pos < S3_HANDOFF_MIN_GRIP)
            ):
                print(
                    f"    [S3 HANDOFF INVALID] step={s3_step} "
                    f"objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} "
                    f"grip={grip_pos:.3f} upright={upright:.3f} cf={cf_val:.1f} src_dst={src_dst:.3f}"
                )
                invalid_handoff = True
                invalid_reason = (
                    f"early loss step={s3_step} grip={grip_pos:.3f} cf={cf_val:.1f} objZ={objZ:.3f}"
                )
                break

            # Drop detection — Phase A: 0.04 (carrying height), Phase B: 0.029 (topple)
            drop_thresh = 0.04 if phase_a_active else 0.029
            if not placed and objZ < drop_thresh and s3_step > 10:
                phase_str_d = "A" if phase_a_active else "B"
                arm1_info = ""
                if arm1_at_phase_b_entry is not None:
                    arm1_info = f" arm1: entry={arm1_at_phase_b_entry:+.3f}→now={arm_jp[1]:+.3f}"
                ghost_str = f" ghost_cf0_max={phase_a_cf_zero_max}" if phase_a_active else ""
                print(
                    f"    [DROP] step={s3_step} phase={phase_str_d} phB_dur={phase_b_dur_now} | "
                    f"objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} grip={grip_pos:.3f} "
                    f"upright={upright:.3f} cf={cf_val:.1f} | arm=[{arm_str}] act=[{act_arm_str}]"
                    f"{arm1_info}{ghost_str}"
                )
                dropped = True
                break

            # Phase A: every 50 steps, Phase B: every 10 steps
            log_interval = 50 if phase_a_active else 10
            phase_b_elapsed = s3_step - phase_b_step_start if phase_b_step_start >= 0 else 0
            if s3_step % log_interval == 0:
                phase_str = "A" if phase_a_active else "B"
                contact_str = f"cf={cf_val:.1f}" if has_contact else "cf=0"
                act_base_str = ",".join(f"{v:+.3f}" for v in action_np[6:9])
                ghost_str = f" cf0_run={phase_a_cf_zero_count}" if phase_a_active and phase_a_cf_zero_count >= 3 else ""
                phb_str = f" phB_t={phase_b_elapsed}" if not phase_a_active else ""
                print(f"    [S3-{phase_str} t={s3_step:4d}] objZ={objZ:.3f} rootZ={rootZ:.3f} dZ={objDz:.3f} "
                      f"grip={grip_pos:.3f} upright={upright:.3f} base_dst={base_dst:.3f} src_dst={src_dst:.3f} {contact_str} "
                      f"arm=[{arm_str}] act_arm=[{act_arm_str}] act_base=[{act_base_str}]"
                      f"{action_delta_str}{ghost_str}{phb_str}"
                      f" pd={place_dwell} od={place_open_dwell} rd={rest_dwell}"
                      f" rel={int(release_phase_latch)} ret={int(retract_started_latch)}")

        if invalid_handoff:
            print(f"    [S3 RETRY] invalid handoff — {invalid_reason}")
            if handoff_attempt == S3_HANDOFF_MAX_RETRIES - 1:
                results.append({"status": "handoff_invalid"})
                print(f"  Episode {ep+1}: handoff_invalid after {S3_HANDOFF_MAX_RETRIES} retries\n")
                episode_recorded = True
            continue

        status = (
            "place_then_topple" if toppled_after_place else
            ("place_rest" if rest_reached else
             ("place_open" if place_open_confirmed else
              ("place" if placed else ("drop" if dropped else "timeout"))))
        )
        avg_arm_err = arm_init_err_sum / max(arm_init_err_count, 1)
        phase_b_dur = s3_step - phase_b_step_start if phase_b_step_start >= 0 else -1
        results.append({"status": status, "s3_steps": s3_step, "phase_a_steps": phase_a_steps,
                         "phase_b_dur": phase_b_dur,
                         "min_objZ": min_objZ, "max_grip": max_grip_open, "min_src_dst": min_src_dst,
                         "arm_err": avg_arm_err,
                         "arm1_entry": arm1_at_phase_b_entry, "arm1_max_b": arm1_max_in_phase_b,
                         "objZ_entry": objZ_at_phase_b_entry, "objZ_min_b": objZ_min_in_phase_b,
                         "ghost_cf0": phase_a_cf_zero_max,
                         "place_open_confirmed": place_open_confirmed,
                         "place_open_step": place_open_step,
                         "arm_rest_reached": rest_reached,
                         "release_phase_flag": release_phase_latch,
                         "retract_started_flag": retract_started_latch,
                         "rest_step": rest_step})
        # Phase B summary
        phb_info = ""
        if arm1_at_phase_b_entry is not None:
            phb_info = (f" | PhaseB: dur={phase_b_dur} arm1={arm1_at_phase_b_entry:+.3f}→max={arm1_max_in_phase_b:+.3f} "
                        f"objZ={objZ_at_phase_b_entry:.3f}→min={objZ_min_in_phase_b:.3f}")
        ghost_info = f" | ghost_cf0={phase_a_cf_zero_max}" if phase_a_cf_zero_max >= 5 else ""
        open_info = f" | open_step={place_open_step}" if place_open_confirmed else ""
        rest_info = f" | rest_step={rest_step}" if rest_reached else ""
        print(f"  Episode {ep+1}: {status} ({s3_step} steps) | "
              f"PhaseA={phase_a_steps} arm_err={avg_arm_err:.3f} | "
              f"min_objZ={min_objZ:.3f} max_grip={max_grip_open:.3f} min_src_dst={min_src_dst:.3f}"
              f"{phb_info}{ghost_info}{open_info}{rest_info}\n")
        episode_recorded = True
        break

# Summary
n = len(results)
n_place_rest = sum(1 for r in results if r["status"] == "place_rest")
n_place_open = sum(1 for r in results if r["status"] == "place_open")
n_place = sum(1 for r in results if r["status"] == "place")
n_place_then_topple = sum(1 for r in results if r["status"] == "place_then_topple")
n_drop = sum(1 for r in results if r["status"] == "drop")
n_drop_a = sum(1 for r in results if r["status"] == "drop" and r.get("phase_a_steps", 0) == r.get("s3_steps", 0))
n_drop_b = n_drop - n_drop_a
n_timeout = sum(1 for r in results if r["status"] == "timeout")
n_s2_fail = sum(1 for r in results if r["status"] == "s2_fail")
n_handoff_invalid = sum(1 for r in results if r["status"] == "handoff_invalid")

s3_results = [r for r in results if r["status"] not in ("s2_fail", "handoff_invalid")]
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
print(f"  Place+rest: {n_place_rest}/{n}")
print(f"  Place+open: {n_place_open}/{n}")
print(f"  Place only: {n_place}/{n} ({n_place/max(n,1)*100:.0f}%) — REAL={n_real_place} SUSPECT={n_suspect_place}")
print(f"  Place→topple: {n_place_then_topple}/{n}")
print(f"  Drop:       {n_drop}/{n} (PhaseA={n_drop_a}, PhaseB={n_drop_b})")
print(f"  Timeout:    {n_timeout}/{n}")
print(f"  S2 Fail:    {n_s2_fail}/{n}")
print(f"  Handoff invalid: {n_handoff_invalid}/{n}")
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
