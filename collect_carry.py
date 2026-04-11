#!/usr/bin/env python3
"""Carry demo collection — multi-env, headless.

S2 expert가 lift → 전환 시점 arm pose 캡처 → carry (base 이동, arm 유지) 기록.
39D obs: Skill2Env 30D + dir_cmd(3D) + init_arm_pose(6D)
Action: [-1,1] normalized (arm_action_to_limits) + arm ±0.02 노이즈

Usage:
    python collect_carry.py \
        --s2_bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
        --s2_resip_checkpoint backup/appoachandlift/resip64%.pt \
        --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \
        --num_envs 8 --num_reps 20 --output demos/carry_120ep_39d.hdf5 --headless
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--s2_bc_checkpoint", type=str, required=True)
parser.add_argument("--s2_resip_checkpoint", type=str, default="")
parser.add_argument("--object_usd", type=str, required=True)
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--object_scale_phys", type=float, default=0.7)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_reps", type=int, default=20, help="per direction")
parser.add_argument("--carry_steps", type=int, default=600)
parser.add_argument("--s2_max_steps", type=int, default=800)
parser.add_argument("--s2_lift_hold", type=int, default=200)
parser.add_argument("--output", type=str, default="demos/carry_120ep_39d.hdf5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
launcher = AppLauncher(args)
sim_app = launcher.app

import h5py, math, torch, numpy as np
from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg

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
if args.dest_object_usd:
    cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
cfg.object_scale = args.object_scale_phys
cfg.grasp_success_height = 100.0  # task_success 비활성화

env = Skill2Env(cfg=cfg)
dev = env.device
N = args.num_envs

# ── S2 Expert 로드 ──
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
    for p in s2_resip.parameters():
        p.requires_grad = False
    # per_dim scale
    s2_per_dim = torch.zeros(s2_cfg["act_dim"], device=dev)
    s2_per_dim[0:5] = 0.20; s2_per_dim[5] = 0.25; s2_per_dim[6:9] = 0.35

# ── arm_action_to_limits mapping ──
override = getattr(env, "_arm_action_limits_override", None)
if override is not None:
    lim = override[0].detach().cpu().numpy()
else:
    lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx].detach().cpu().numpy()
arm_center = 0.5 * (lim[:, 0] + lim[:, 1])
arm_half_range = 0.5 * (lim[:, 1] - lim[:, 0])
arm_half_range = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)
arm_center_t = torch.tensor(arm_center, dtype=torch.float32, device=dev)
arm_half_t = torch.tensor(arm_half_range, dtype=torch.float32, device=dev)

# ── Directions (action space 직접값, navigate와 동일 방식) ──
# _apply_action: body_vel = action * max_vel
# max_lin_vel=0.5, max_ang_vel=3.0
# 0.7 * 0.5 = 0.35 m/s, -0.33 * 3.0 = -0.99 rad/s (CCW)
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

# ── Schedule: 6방향 × num_reps ──
schedule = []
for label in DIRECTIONS.keys():
    schedule.extend([label] * args.num_reps)
total_eps = len(schedule)
print(f"\nCollect carry: {total_eps} episodes ({len(DIRECTIONS)} dirs × {args.num_reps})")
print(f"  num_envs={N}, s2_max_steps={args.s2_max_steps}, s2_lift_hold={args.s2_lift_hold}")
print(f"  carry_steps={args.carry_steps}\n")

# ── Helpers ──
def get_s2_action(obs_30d):
    """S2 expert action (BC + optional ResiP)."""
    with torch.no_grad():
        base_nact = s2_dp.base_action_normalized(obs_30d)
        if s2_resip is not None:
            nobs = s2_dp.normalizer(obs_30d, "obs", forward=True).clamp(-3, 3)
            nobs = torch.nan_to_num(nobs, nan=0.0)
            ri = torch.cat([nobs, base_nact], dim=-1)
            ra, _, _, _, _ = s2_resip.get_action_and_value(ri)
            nact = base_nact + ra * s2_per_dim
        else:
            nact = base_nact
        action = s2_dp.normalizer(nact, "action", forward=False)
    return action.clamp(-1, 1)

def get_state_9d_batch():
    """(N, 9) robot state."""
    jp = env.robot.data.joint_pos
    arm = jp[:, env.arm_idx[:5]]
    grip = jp[:, env.arm_idx[5:6]]
    bv = env.robot.data.root_lin_vel_b[:, :2]
    wz = env.robot.data.root_ang_vel_b[:, 2:3]
    return torch.cat([arm, grip, bv, wz], dim=-1)

# ── Main collection loop ──
os.makedirs(os.path.dirname(args.output) or "demos", exist_ok=True)
hf = h5py.File(args.output, "w")
ep_saved = 0
t0 = time.time()

# Per-env state tracking
phase = torch.ones(N, dtype=torch.long, device=dev)  # 1=S2, 2=carry
s2_step = torch.zeros(N, dtype=torch.long, device=dev)
lift_counter = torch.zeros(N, dtype=torch.long, device=dev)
carry_step = torch.zeros(N, dtype=torch.long, device=dev)
init_arm_pose = torch.zeros(N, 6, device=dev)  # (N, arm5+grip1)
assigned_dir = [0] * N  # schedule index per env
carry_bufs = [{"obs": [], "act": [], "state": []} for _ in range(N)]
next_sched_idx = 0  # 글로벌 순차 할당 카운터

# Assign first batch of directions
for i in range(N):
    if next_sched_idx < total_eps:
        assigned_dir[i] = next_sched_idx
        next_sched_idx += 1

obs, _ = env.reset()
s2_dp.reset()

_dbg_step = 0
while ep_saved < total_eps and sim_app.is_running():
    obs_30d = obs["policy"] if isinstance(obs, dict) else obs
    _dbg_step += 1
    if _dbg_step % 100 == 0:
        _objZ0 = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
        _gp0 = env.robot.data.joint_pos[0, env.arm_idx[5]].item()
        _lc0 = lift_counter[0].item()
        _s2s0 = s2_step[0].item()
        _ph0 = phase[0].item()
        print(f"  [DBG] step={_dbg_step} phase={_ph0} s2_step={_s2s0} lift={_lc0} objZ={_objZ0:.3f} grip={_gp0:.3f}", flush=True)

    # ── S2 phase envs: run expert ──
    s2_mask = phase == 1
    if s2_mask.any():
        action_s2 = get_s2_action(obs_30d)
    else:
        action_s2 = torch.zeros(N, 9, device=dev)

    # ── Carry phase envs: arm hold + base command ──
    action_carry = torch.zeros(N, 9, device=dev)
    carry_mask = phase == 2
    if carry_mask.any():
        for i in range(N):
            if phase[i] != 2:
                continue
            sched_idx = assigned_dir[i]
            if sched_idx >= total_eps:
                continue
            label = schedule[sched_idx]
            bvx, bvy, bwz = DIRECTIONS[label]

            # arm action = init_arm_pose를 [-1,1]로 변환 + 노이즈
            arm_target = init_arm_pose[i].cpu().numpy()
            arm_norm = (arm_target - arm_center) / arm_half_range
            arm_norm = np.clip(arm_norm, -1, 1).astype(np.float32)
            arm_norm[:6] += np.random.randn(6).astype(np.float32) * 0.02

            action_carry[i, 0:6] = torch.tensor(arm_norm, device=dev)
            # navigate와 동일: action space 직접값 (나누기 없음)
            action_carry[i, 6] = bvx + np.random.normal(0, 0.005)
            action_carry[i, 7] = bvy + np.random.normal(0, 0.005)
            action_carry[i, 8] = bwz + np.random.normal(0, 0.005)

    # Combine actions
    action = torch.where(s2_mask.unsqueeze(-1), action_s2, action_carry)
    action = action.clamp(-1, 1)

    # Step
    obs, _, ter, tru, info = env.step(action)

    # ── Per-env logic ──
    for i in range(N):
        if phase[i] == 1:
            # S2 phase: check lift
            s2_step[i] += 1
            grip_pos = env.robot.data.joint_pos[i, env.arm_idx[5]].item()
            grip_closed = grip_pos < float(env.cfg.grasp_gripper_threshold)
            has_contact = False
            if env.contact_sensor is not None:
                cf = env._contact_force_per_env()[i].item()
                has_contact = cf > float(env.cfg.grasp_contact_threshold)
            objZ = (env.object_pos_w[i, 2] - env.scene.env_origins[i, 2]).item()

            eg = grip_closed and has_contact
            if eg and objZ > 0.05:
                lift_counter[i] += 1
            else:
                lift_counter[i] = 0

            # Fail: topple or timeout
            if (objZ < 0.026 and s2_step[i] > 20) or (s2_step[i] > args.s2_max_steps and objZ < 0.04):
                # Reset env + state
                phase[i] = 1; s2_step[i] = 0; lift_counter[i] = 0; carry_step[i] = 0
                obs, _ = env.reset()
                s2_dp.reset()
                break  # restart per-env loop after reset

            # Lift success → transition to carry (grip 범위 필터)
            if lift_counter[i] >= args.s2_lift_hold:
                _grip = env.robot.data.joint_pos[i, env.arm_idx[5]].item()
                if _grip < 0.10 or _grip > 0.45:
                    print(f"  [GRIP OOD] grip={_grip:.3f} not in [0.10, 0.45] — 리셋")
                    phase[i] = 1; s2_step[i] = 0; lift_counter[i] = 0; carry_step[i] = 0
                    obs, _ = env.reset()
                    s2_dp.reset()
                    break
                jp = env.robot.data.joint_pos[i]
                init_arm_pose[i, :5] = jp[env.arm_idx[:5]]
                init_arm_pose[i, 5] = jp[env.arm_idx[5]]
                phase[i] = 2
                carry_step[i] = 0
                carry_bufs[i] = {"obs": [], "act": [], "state": []}
                s2_dp.reset()

        elif phase[i] == 2:
            # Carry phase: record
            carry_step[i] += 1

            # 39D obs: 30D + dir_cmd(3D) + init_arm_pose(6D)
            obs_30 = obs["policy"][i].cpu().numpy() if isinstance(obs, dict) else obs[i].cpu().numpy()
            sched_idx = assigned_dir[i]
            label = schedule[sched_idx] if sched_idx < total_eps else "FORWARD"
            dir_cmd = np.array(DIR_CMD[label], dtype=np.float32)
            iap = init_arm_pose[i].cpu().numpy().astype(np.float32)
            obs_39 = np.concatenate([obs_30, dir_cmd, iap])

            act_np = action[i].cpu().numpy().astype(np.float32)
            state_np = get_state_9d_batch()[i].cpu().numpy().astype(np.float32)

            carry_bufs[i]["obs"].append(obs_39)
            carry_bufs[i]["act"].append(act_np)
            carry_bufs[i]["state"].append(state_np)

            # Drop detection
            objZ = (env.object_pos_w[i, 2] - env.scene.env_origins[i, 2]).item()
            if objZ < 0.05 and carry_step[i] > 10:
                # Drop → discard, reset env
                phase[i] = 1; s2_step[i] = 0; lift_counter[i] = 0; carry_step[i] = 0
                carry_bufs[i] = {"obs": [], "act": [], "state": []}
                obs, _ = env.reset()
                s2_dp.reset()
                break  # restart per-env loop after reset

            # Episode done
            if carry_step[i] >= args.carry_steps:
                buf = carry_bufs[i]
                if len(buf["obs"]) > 0:
                    # NaN check
                    state_arr = np.array(buf["state"], dtype=np.float32)
                    act_arr = np.array(buf["act"], dtype=np.float32)
                    if not np.any(np.isnan(state_arr)) and not np.any(np.isnan(act_arr)):
                        grp = hf.create_group(f"episode_{ep_saved}")
                        grp.create_dataset("obs", data=np.array(buf["obs"], dtype=np.float32))
                        grp.create_dataset("actions", data=act_arr)
                        grp.create_dataset("robot_state", data=state_arr)
                        grp.attrs["instruction"] = f"carry {label.lower()}"
                        grp.attrs["direction"] = label
                        grp.attrs["init_arm_pose"] = iap
                        grp.attrs["num_steps"] = len(buf["obs"])
                        hf.flush()
                        ep_saved += 1
                        elapsed = time.time() - t0
                        print(f"  [{ep_saved}/{total_eps}] {label} | {len(buf['obs'])} steps | {elapsed:.0f}s")
                    else:
                        print(f"  [REJECT] NaN in episode — skipped")

                # Reset for next direction
                phase[i] = 1; s2_step[i] = 0; lift_counter[i] = 0; carry_step[i] = 0
                carry_bufs[i] = {"obs": [], "act": [], "state": []}
                if next_sched_idx < total_eps:
                    assigned_dir[i] = next_sched_idx
                    next_sched_idx += 1
                s2_dp.reset()

    # Handle env resets (from env internal logic)
    done = ter | tru
    if done.any():
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.dim() == 0:
            done_ids = done_ids.unsqueeze(0)
        for idx in done_ids:
            i = idx.item()
            if phase[i] == 2:
                # env terminated during carry → discard
                carry_bufs[i] = {"obs": [], "act": [], "state": []}
            phase[i] = 1; s2_step[i] = 0; lift_counter[i] = 0; carry_step[i] = 0
            s2_dp.reset()

hf.close()
elapsed = time.time() - t0
print(f"\nDone: {ep_saved} episodes → {args.output} ({elapsed:.0f}s)")
sim_app.close()
