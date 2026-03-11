#!/usr/bin/env python3
"""
DPPO: Diffusion Policy Policy Optimization for LeKiwi Skill-3 (CarryAndPlace).

Fine-tunes BC Diffusion Policy UNet directly via PPO on denoising chain.
Adapted from train_dppo.py (Skill-2) for the carry-and-place task.

Initial state setup:
  Handoff buffer 방식은 friction grasp가 즉시 성립하지 않아 물체가 떨어짐.
  대신 eval_dp_bc.py와 동일한 방식을 N envs 배치로 수행:
    1) HDF5 데모에서 랜덤 에피소드 선택 -> 로봇/물체 초기상태 복원
    2) 240 step 점진 gripper closing + 매 step 물체를 EE에 텔레포트
    3) 120 step 자유 settle (friction grasp 성립)
    -> 모든 env에서 물체를 잡은 채로 시작

Usage:
    python train_dppo_skill3.py \
      --bc_checkpoint checkpoints/dp_bc_skill3_aug/dp_bc.pt \
      --demo demos_skill3/combined_skill3_20260227_091123.hdf5 \
      --object_usd /path/to/5_HTP/model_clean.usd \
      --dest_object_usd /path/to/ACE_Coffee_Mug/model_clean.usd \
      --num_envs 512 --headless
"""
from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description="DPPO for LeKiwi Skill-3 CarryAndPlace")
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--demo", type=str, required=True,
                    help="HDF5 demo file for initial state (carry_and_place demos)")
parser.add_argument("--skill", type=str, default="carry_and_place",
                    choices=["carry_and_place"])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
# DPPO
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
parser.add_argument("--n_steps", type=int, default=120)
parser.add_argument("--total_iters", type=int, default=1000)
parser.add_argument("--update_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=40000)
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
parser.add_argument("--reward_scale_running", type=lambda x: x.lower() == "true", default=True)
parser.add_argument("--r_carry_progress", type=float, default=20.0)
parser.add_argument("--r_ee_progress", type=float, default=30.0)
parser.add_argument("--r_heading", type=float, default=3.0)
parser.add_argument("--r_hold_bonus", type=float, default=0.5)
parser.add_argument("--r_place_prelim", type=float, default=200.0)
parser.add_argument("--r_place_final", type=float, default=300.0)
parser.add_argument("--r_drop_penalty", type=float, default=-200.0)
parser.add_argument("--r_time_penalty", type=float, default=-0.1)
# Place criteria
parser.add_argument("--place_obj_z_min", type=float, default=0.032)
parser.add_argument("--place_obj_z_max", type=float, default=0.034)
parser.add_argument("--place_radius", type=float, default=0.172)
# Grasp init
parser.add_argument("--grasp_close_steps", type=int, default=240)
parser.add_argument("--grasp_settle_steps", type=int, default=120)
parser.add_argument("--grasp_target_grip", type=float, default=0.45)
# Warmup
parser.add_argument("--warmup_steps", type=int, default=400)
parser.add_argument("--warmup_reset_interval", type=int, default=1800)
# Eval/save
parser.add_argument("--eval_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--save_dir", type=str, default="checkpoints/dppo_skill3")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import random, time
from pathlib import Path
import h5py, numpy as np, torch, torch.nn as nn, torch.optim as optim
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
from diffusion_policy import DiffusionPolicyAgent
from dppo_model import DPPODiffusion, RunningRewardScaler
from lekiwi_skill2_env import EE_LOCAL_OFFSET


def load_demo_init_states(path):
    entries = []
    with h5py.File(path, "r") as f:
        ep_keys = sorted([k for k in f.keys() if k.startswith("episode_")],
                         key=lambda x: int(x.split("_")[1]))
        for ek in ep_keys:
            grp = f[ek]; ea = dict(grp.attrs); obs0 = grp["obs"][0]
            entry = {"arm_joints_6": obs0[0:6].astype(np.float32)}
            if "robot_init_pos" in ea:
                entry["robot_init_pos"] = np.array(ea["robot_init_pos"], dtype=np.float32)
            if "robot_init_quat" in ea:
                entry["robot_init_quat"] = np.array(ea["robot_init_quat"], dtype=np.float32)
            entries.append(entry)
    print(f"  Loaded {len(entries)} demo init states from {path}")
    return entries


class LeKiwiEnvWrapper:
    def __init__(self, env):
        self.env = env; self.num_envs = env.num_envs; self.device = env.device
    def reset(self):
        od, _ = self.env.reset()
        return (od["policy"] if isinstance(od, dict) else od).to(self.device)
    def step(self, action):
        od, r, ter, tru, info = self.env.step(action)
        o = (od["policy"] if isinstance(od, dict) else od).to(self.device)
        return o, r.view(-1).to(self.device), ter.view(-1).to(self.device), tru.view(-1).to(self.device), info


def make_env(num_envs, args_):
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    cfg = Skill3EnvCfg(); cfg.scene.num_envs = num_envs; cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False; cfg.arm_limit_write_to_sim = False
    cfg.grasp_contact_threshold = 0.55; cfg.grasp_gripper_threshold = 0.65
    cfg.grasp_max_object_dist = 0.50; cfg.episode_length_s = 300.0
    cfg.place_obj_z_min = args_.place_obj_z_min; cfg.place_obj_z_max = args_.place_obj_z_max
    cfg.place_radius = args_.place_radius; cfg.place_grace_steps = 500
    cfg.handoff_buffer_path = ""  # no handoff — using demo grasp init
    if args_.object_usd: cfg.object_usd = os.path.expanduser(args_.object_usd)
    if args_.multi_object_json: cfg.multi_object_json = os.path.expanduser(args_.multi_object_json)
    if args_.dest_object_usd: cfg.dest_object_usd = os.path.expanduser(args_.dest_object_usd)
    cfg.gripper_contact_prim_path = args_.gripper_contact_prim_path
    if args_.arm_limit_json and os.path.isfile(args_.arm_limit_json): cfg.arm_limit_json = args_.arm_limit_json
    env = Skill3Env(cfg=cfg)
    print(f"  Env: carry_and_place, n={num_envs}, dev={env.device}")
    return LeKiwiEnvWrapper(env)


def load_bc_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False); cfg = ck["config"]
    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"], action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=cfg.get("inference_steps", 16),
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)
    sd = ck["model_state_dict"]
    agent.model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
    agent.normalizer.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("normalizer.")}, device=device)
    print(f"  BC: obs={cfg['obs_dim']} act={cfg['act_dim']} pred_h={cfg['pred_horizon']} act_h={cfg['action_horizon']}")
    return agent.model, agent.normalizer, cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Batched grasp init — eval_dp_bc.py _restore_init_state() logic for N envs
# ═══════════════════════════════════════════════════════════════════════════════

def batched_grasp_init(env_wrap, demo_entries, grasp_close_steps=240,
                       grasp_settle_steps=120, grasp_target_grip=0.45):
    """N envs: demo init state restore + gradual gripper close + object teleport."""
    raw = env_wrap.env; dev = raw.device; N = raw.num_envs
    all_ids = torch.arange(N, device=dev)
    n_demos = len(demo_entries)
    chosen = [demo_entries[random.randint(0, n_demos - 1)] for _ in range(N)]

    # 1. Robot base pos/quat
    root_states = raw.robot.data.default_root_state.clone()
    for i, e in enumerate(chosen):
        if "robot_init_pos" in e:
            root_states[i, 0:3] = torch.tensor(e["robot_init_pos"], device=dev)
        if "robot_init_quat" in e:
            root_states[i, 3:7] = torch.tensor(e["robot_init_quat"], device=dev)
    root_states[:, 7:] = 0.0
    raw.robot.write_root_state_to_sim(root_states, all_ids)
    if hasattr(raw, "home_pos_w"): raw.home_pos_w[:] = root_states[:, :3]

    # 2. Arm joints + gripper max open
    jp = raw.robot.data.default_joint_pos.clone()
    for i, e in enumerate(chosen):
        arm6 = torch.tensor(e["arm_joints_6"], device=dev)
        jp[i, raw.arm_idx[:5]] = arm6[:5]
    jp[:, raw.gripper_idx] = 1.4
    if hasattr(raw, "wheel_idx"): jp[:, raw.wheel_idx] = 0.0
    jv = torch.zeros_like(jp)
    raw.robot.write_joint_state_to_sim(jp, jv, env_ids=all_ids)
    raw.robot.set_joint_position_target(jp, env_ids=all_ids)
    raw.robot.set_joint_velocity_target(torch.zeros(N, raw.robot.num_joints, device=dev), env_ids=all_ids)
    for _ in range(10):
        raw.robot.write_data_to_sim(); raw.sim.step()
    raw.robot.update(raw.sim.cfg.dt)

    # 3. Object at EE
    fixed_jaw_idx = raw._fixed_jaw_body_idx
    ee_local = raw._ee_local_offset
    wrist_pos = raw.robot.data.body_pos_w[:, fixed_jaw_idx, :]
    wrist_quat = raw.robot.data.body_quat_w[:, fixed_jaw_idx, :]
    rot90 = torch.tensor([0.8192, -0.5736, 0.0, 0.0], dtype=torch.float32, device=dev).unsqueeze(0).expand(N, -1)
    obj_quat = quat_mul(wrist_quat, rot90)
    ee_pos = wrist_pos + quat_apply(wrist_quat, ee_local.expand(N, -1))
    bbox_center_local = torch.zeros(N, 3, device=dev)
    bbox_center_local[:, 2] = raw.object_bbox[:, 2] / 2.0
    bbox_center_world = quat_apply(obj_quat, bbox_center_local)
    obj_root_pos = ee_pos - bbox_center_world
    if raw.object_rigid is not None:
        obj_st = raw.object_rigid.data.root_state_w.clone()
        obj_st[:, 0:3] = obj_root_pos; obj_st[:, 3:7] = obj_quat; obj_st[:, 7:] = 0.0
        raw.object_rigid.write_root_state_to_sim(obj_st, all_ids)
    raw.object_pos_w[:] = ee_pos

    # Target grips from demo
    target_grips = torch.zeros(N, device=dev)
    for i, e in enumerate(chosen): target_grips[i] = e["arm_joints_6"][5]

    # 4. Gradual gripper close + object teleport (240 steps)
    for step_i in range(grasp_close_steps):
        t_frac = (step_i + 1) / grasp_close_steps
        grip_vals = target_grips + (grasp_target_grip - target_grips) * t_frac
        grip_jp = raw.robot.data.joint_pos_target.clone()
        grip_jp[:, raw.gripper_idx] = grip_vals
        raw.robot.set_joint_position_target(grip_jp, env_ids=all_ids)
        raw.robot.write_data_to_sim()
        # Teleport object to EE
        w_pos = raw.robot.data.body_pos_w[:, fixed_jaw_idx, :]
        w_quat = raw.robot.data.body_quat_w[:, fixed_jaw_idx, :]
        cur_ee = w_pos + quat_apply(w_quat, ee_local.expand(N, -1))
        cur_oq = quat_mul(w_quat, rot90)
        cur_bw = quat_apply(cur_oq, bbox_center_local)
        if raw.object_rigid is not None:
            os2 = raw.object_rigid.data.root_state_w.clone()
            os2[:, 0:3] = cur_ee - cur_bw; os2[:, 3:7] = cur_oq; os2[:, 7:] = 0.0
            raw.object_rigid.write_root_state_to_sim(os2, all_ids)
        raw.sim.step(); raw.robot.update(raw.sim.cfg.dt)
        if raw.object_rigid is not None: raw.object_rigid.update(raw.sim.cfg.dt)

    # 5. Free settle
    for _ in range(grasp_settle_steps):
        raw.robot.write_data_to_sim(); raw.sim.step()
    raw.robot.update(raw.sim.cfg.dt)
    if raw.object_rigid is not None: raw.object_rigid.update(raw.sim.cfg.dt)

    # 6. Internal state
    raw.object_grasped[:] = True; raw.just_dropped[:] = False
    if hasattr(raw, "intentional_placed"): raw.intentional_placed[:] = False
    raw._fallback_teleport_carry[:] = False
    if raw.object_rigid is not None: raw.object_pos_w[:] = raw.object_rigid.data.root_pos_w

    # 7. Dest object spawn
    raw._spawn_dest_object(all_ids)

    # 8. Reset tracking
    raw.task_success[:] = False; raw.just_grasped[:] = False
    raw.place_success_step[:] = 0; raw.preliminary_success[:] = False
    raw.prev_dest_dist[:] = 10.0; raw.prev_object_dist[:] = 10.0
    raw.episode_reward_sum[:] = 0.0; raw.actions[:] = 0.0; raw.prev_actions[:] = 0.0

    grip_sim = raw.robot.data.joint_pos[:, raw.gripper_idx].view(-1)
    grasped = raw.object_grasped.float().mean().item()
    print(f"    [grasp_init] grip={grip_sim.mean():.3f} grasped={grasped:.2%}")

    obs_dict = raw._get_observations()
    return (obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict).to(dev)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    demo_entries = load_demo_init_states(args.demo)
    assert len(demo_entries) > 0, f"No episodes in {args.demo}"

    env = make_env(args.num_envs, args); dev = env.device; N = env.num_envs
    jaw_idx, _ = env.env.robot.find_bodies(["Wrist_Roll_08c_v1"]); jaw_idx = jaw_idx[0]
    ee_off = torch.tensor(EE_LOCAL_OFFSET, device=dev).unsqueeze(0)

    unet, normalizer, bc_cfg = load_bc_checkpoint(args.bc_checkpoint, dev)
    OD, AD = bc_cfg["obs_dim"], bc_cfg["act_dim"]
    PRED_H, ACT_H = bc_cfg["pred_horizon"], bc_cfg["action_horizon"]
    assert OD == 29

    dppo = DPPODiffusion(
        unet_pretrained=unet, normalizer=normalizer, obs_dim=OD, act_dim=AD,
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
        eta=args.eta, norm_adv=args.norm_adv, device=str(dev),
    ).to(dev)

    actor_opt = optim.AdamW(dppo.actor_ft.parameters(), lr=args.actor_lr, weight_decay=0)
    critic_opt = optim.AdamW(dppo.critic.parameters(), lr=args.critic_lr, weight_decay=0)
    actor_sched = optim.lr_scheduler.CosineAnnealingLR(actor_opt, T_max=args.total_iters, eta_min=args.actor_lr*0.01)
    critic_sched = optim.lr_scheduler.CosineAnnealingLR(critic_opt, T_max=args.total_iters, eta_min=args.critic_lr*0.1)
    reward_scaler = RunningRewardScaler(N) if args.reward_scale_running else None

    start_itr = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        dppo.actor_ft.load_state_dict(ck["actor_ft_state_dict"])
        dppo.critic.load_state_dict(ck["critic_state_dict"])
        if "actor_opt" in ck: actor_opt.load_state_dict(ck["actor_opt"])
        if "critic_opt" in ck: critic_opt.load_state_dict(ck["critic_opt"])
        start_itr = ck.get("iteration", 0); print(f"  Resumed iter={start_itr}")

    S, K = args.n_steps, args.ft_denoising_steps
    WU, WU_CALLS = args.warmup_steps, args.warmup_steps // ACT_H
    WU_RESET = args.warmup_reset_interval
    PLACE_Z_MIN, PLACE_Z_MAX, PLACE_RAD = args.place_obj_z_min, args.place_obj_z_max, args.place_radius
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_sr, best_prelim, best_final = 0.0, 0, 0; t_start = time.time()

    ms_prelim = torch.zeros(N, dtype=torch.bool, device=dev)
    ms_final = torch.zeros(N, dtype=torch.bool, device=dev)
    prev_dd = torch.zeros(N, device=dev); prev_ee_dd = torch.zeros(N, device=dev)

    def ee_pos():
        wp = env.env.robot.data.body_pos_w[:, jaw_idx, :]; wq = env.env.robot.data.body_quat_w[:, jaw_idx, :]
        return wp + quat_apply(wq, ee_off.expand_as(wp))
    def base_dest_dist():
        dd = env.env.dest_object_pos_w - env.env.robot.data.root_pos_w
        db = quat_apply_inverse(env.env.robot.data.root_quat_w, dd)
        return torch.norm(db[:, :2], dim=-1).view(-1)
    def ee_dest_xy():
        return torch.nan_to_num(torch.norm(ee_pos()[:,:2] - env.env.dest_object_pos_w[:,:2], dim=-1).view(-1), nan=1.0)
    def heading_to_dest():
        dd = env.env.dest_object_pos_w - env.env.robot.data.root_pos_w
        db = quat_apply_inverse(env.env.robot.data.root_quat_w, dd)
        d = torch.norm(db[:,:2], dim=-1).clamp(min=1e-6)
        return (db[:,1] / d).view(-1)
    def obj_height():
        return (env.env.object_pos_w[:,2] - env.env.scene.env_origins[:,2]).view(-1)
    def obj_dest_xy():
        return torch.norm(env.env.object_pos_w[:,:2] - env.env.dest_object_pos_w[:,:2], dim=-1).view(-1)
    def check_place():
        oz = obj_height(); od = obj_dest_xy()
        return (oz >= PLACE_Z_MIN) & (oz <= PLACE_Z_MAX) & (od < PLACE_RAD)
    def reset_rew(mask):
        ms_prelim[mask] = False; ms_final[mask] = False
        prev_dd[mask] = base_dest_dist()[mask]; prev_ee_dd[mask] = ee_dest_xy()[mask]

    def reset_all():
        _ = env.reset()
        return batched_grasp_init(env, demo_entries,
            grasp_close_steps=args.grasp_close_steps,
            grasp_settle_steps=args.grasp_settle_steps,
            grasp_target_grip=args.grasp_target_grip)

    print(f"\n{'='*70}")
    print(f"  DPPO Skill-3 -- CarryAndPlace")
    print(f"  N={N} S={S} ACT_H={ACT_H} K={K}")
    print(f"  env_steps/iter = grasp_init {args.grasp_close_steps}+{args.grasp_settle_steps} + warmup {WU} + RL {S*ACT_H}")
    print(f"  demos: {len(demo_entries)} from {args.demo}")
    print(f"  R: carry={args.r_carry_progress} ee={args.r_ee_progress} head={args.r_heading} hold={args.r_hold_bonus}")
    print(f"     prelim={args.r_place_prelim} final={args.r_place_final} drop={args.r_drop_penalty} time={args.r_time_penalty}")
    print(f"{'='*70}\n")

    for itr in range(start_itr, args.total_iters):
        itr_t0 = time.time()
        ev = (itr % args.eval_interval == 0)
        if ev: dppo.eval()
        else: dppo.train(); dppo.actor.eval()

        obs = reset_all(); ms_prelim.zero_(); ms_final.zero_()

        # BC Warmup
        if WU_CALLS > 0:
            wu_ct = 0
            for wi in range(WU_CALLS):
                with torch.no_grad():
                    an, _ = dppo.sample_actions(obs, deterministic=True)
                    act = dppo.normalizer(an[:,:ACT_H].reshape(-1,AD),"action",forward=False).reshape(N,ACT_H,AD)
                for ai in range(ACT_H):
                    obs, _, ter, tru, _ = env.step(act[:,ai]); wu_ct += 1
                    if (ter|tru).any(): break
                if WU_RESET > 0 and wu_ct >= WU_RESET and wi < WU_CALLS-1:
                    obs = reset_all(); wu_ct = 0

        prev_dd[:] = base_dest_dist(); prev_ee_dd[:] = ee_dest_xy()
        dd0 = prev_dd.clone(); grip0 = env.env.robot.data.joint_pos[:, env.env.gripper_idx].view(-1)
        grasped0 = env.env.object_grasped.float().mean().item()
        print(f"\nIter {itr}/{args.total_iters} | {'EVAL' if ev else 'TRAIN'} | "
              f"Dest={dd0.mean():.3f}/{dd0.min():.3f} Grip={grip0.mean():.2f} Grasped={grasped0:.2%}")

        obs_b = torch.zeros(S,N,OD,device="cpu"); chain_b = torch.zeros(S,N,K+1,PRED_H,AD,device="cpu")
        rew_b = torch.zeros(S,N,device="cpu"); term_b = torch.zeros(S,N,device="cpu")
        first_b = torch.zeros(S+1,N,device="cpu"); first_b[0] = 1.0
        dp_, df_, dd_ = 0,0,0; prog_s, head_s, hold_s = 0.0, 0.0, 0

        for step in range(S):
            obs_b[step] = obs.cpu()
            with torch.no_grad():
                an, ch = dppo.sample_actions(obs, deterministic=ev)
                act = dppo.normalizer(an[:,:ACT_H].reshape(-1,AD),"action",forward=False).reshape(N,ACT_H,AD)
            chain_b[step] = ch.cpu()
            cr = torch.zeros(N,device=dev); ct = torch.zeros(N,dtype=torch.bool,device=dev)
            cd = torch.zeros(N,dtype=torch.bool,device=dev)

            for ai in range(ACT_H):
                obs_n,_,ter,tru,info = env.step(act[:,ai]); obs_n = torch.nan_to_num(obs_n,nan=0.0)
                alive = ~cd
                eg = info.get("object_grasped_mask", env.env.object_grasped).view(-1)
                jd = info.get("just_dropped_mask", torch.zeros(N,dtype=torch.bool,device=dev)).view(-1)
                cr += alive.float()*eg.float()*(args.r_hold_bonus/ACT_H); hold_s += (alive&eg).sum().item()
                dropped = jd.bool()&alive
                if dropped.any(): cr[dropped] += args.r_drop_penalty; ct |= dropped; dd_ += dropped.sum().item()
                pc = check_place()&(~eg)&alive
                np_ = pc&(~ms_prelim)
                if np_.any(): cr[np_] += args.r_place_prelim; ms_prelim |= np_; dp_ += np_.sum().item()
                fps_ = info.get("place_success_mask", env.env.task_success).view(-1)
                nf = fps_&(~ms_final)&alive
                if nf.any(): cr[nf] += args.r_place_final; ms_final |= nf; df_ += nf.sum().item()
                ct |= ter.bool(); cd |= (ter|tru).view(-1).bool(); obs = obs_n

            cbd = base_dest_dist(); gm = env.env.object_grasped.view(-1)&(~cd)
            prog = torch.clamp(prev_dd-cbd,-0.2,0.2); cr += gm.float()*prog*args.r_carry_progress
            prog_s += (prog*gm.float()).sum().item()
            ced = ee_dest_xy(); eep = torch.clamp(prev_ee_dd-ced,-0.2,0.2)
            cr += gm.float()*eep*args.r_ee_progress
            ch_ = heading_to_dest(); cr += gm.float()*ch_*(args.r_heading/ACT_H)
            head_s += (ch_*gm.float()).sum().item()
            prev_dd[:] = cbd; prev_ee_dd[:] = ced
            cr += (~cd).float()*args.r_time_penalty; cr[cd] = 0.0
            if cd.any(): reset_rew(cd)
            rew_b[step] = cr.cpu(); term_b[step] = ct.float().cpu(); first_b[step+1] = cd.float().cpu()

        with torch.no_grad():
            val_b = torch.zeros(S,N,device="cpu")
            for s in range(S): val_b[s] = dppo.get_value(obs_b[s].to(dev)).cpu()
            nv = dppo.get_value(obs).cpu()
            lp_b = torch.zeros(S,N,K,PRED_H,AD,device="cpu")
            for s in range(S): lp_b[s] = dppo.get_logprobs_all(obs_b[s].to(dev), chain_b[s].to(dev)).cpu()

        rn = rew_b.numpy()
        if reward_scaler is not None and not ev:
            fn = first_b[:-1].numpy()
            for s in range(S): rn[s] = reward_scaler(rn[s], fn[s])
            rew_b = torch.from_numpy(rn).float()

        adv_b = torch.zeros(S,N); lg = torch.zeros(N)
        for t in reversed(range(S)):
            nxt = nv.view(N) if t==S-1 else val_b[t+1]; nt = 1.0-term_b[t]
            d = rew_b[t]+args.gamma*nxt*nt-val_b[t]; lg = d+args.gamma*args.gae_lambda*nt*lg; adv_b[t] = lg
        ret_b = adv_b+val_b

        sr = df_/max(N,1); sr_p = dp_/max(N,1)
        fps = S*ACT_H*N/max(time.time()-itr_t0,1e-6)
        fgr = env.env.robot.data.joint_pos[:,env.env.gripper_idx].view(-1)
        fg = env.env.object_grasped.float().mean().item()
        print(f"  SR={sr:.2%}(prelim={sr_p:.2%}) | Place={dp_}/{df_} Drop={dd_} Hold={hold_s} | "
              f"Dest={base_dest_dist().min():.2f}/{base_dest_dist().mean():.2f} | "
              f"Grip={fgr.min():.2f}/{fgr.mean():.2f} | Grasped={fg:.2%} | R={rew_b.sum(0).mean():.1f} | FPS={fps:.0f}")

        if ev:
            if sr > best_sr:
                best_sr = sr
                torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
                    "critic_state_dict": dppo.critic.state_dict(),
                    "normalizer_state_dict": dppo.normalizer.state_dict(),
                    "bc_config": bc_cfg, "iteration": itr, "success_rate": sr, "args": vars(args),
                }, save_dir/"dppo_best.pt"); print(f"  * Best SR={sr:.2%}")
            if dp_ > best_prelim:
                best_prelim = dp_
                torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
                    "critic_state_dict": dppo.critic.state_dict(),
                    "normalizer_state_dict": dppo.normalizer.state_dict(),
                    "bc_config": bc_cfg, "iteration": itr, "prelim_places": dp_, "args": vars(args),
                }, save_dir/"dppo_best_prelim.pt"); print(f"  * Best Prelim={dp_}")
            if df_ > best_final: best_final = df_
            print(f"  Best: SR={best_sr:.2%} Prelim={best_prelim} Final={best_final}"); continue

        of = obs_b.reshape(S*N,OD); cf = chain_b.reshape(S*N,K+1,PRED_H,AD)
        rf = ret_b.reshape(S*N); vf = val_b.reshape(S*N); af = adv_b.reshape(S*N)
        lpf = lp_b.reshape(S*N,K,PRED_H,AD)
        tot = S*N*K; bs = min(args.batch_size, tot); cfs = []
        for ep in range(args.update_epochs):
            brk = False; perm = torch.randperm(tot,device="cpu"); nb = max(1,tot//bs)
            for bi in range(nb):
                i0,i1 = bi*bs, min((bi+1)*bs, tot); idx = perm[i0:i1]
                bi_ = idx//K; di_ = idx%K
                o,cp,cn = of[bi_].to(dev), cf[bi_,di_].to(dev), cf[bi_,di_+1].to(dev)
                r_,v_,a_,lp = rf[bi_].to(dev), vf[bi_].to(dev), af[bi_].to(dev), lpf[bi_,di_].to(dev)
                pg,vl,kl,clf = dppo.loss_ppo(o,cp,cn,di_.to(dev),r_,v_,a_,lp,reward_horizon=ACT_H)
                loss = pg+vl*args.vf_coef; cfs.append(clf)
                actor_opt.zero_grad(); critic_opt.zero_grad(); loss.backward()
                if itr >= args.n_critic_warmup_itr:
                    if args.max_grad_norm: nn.utils.clip_grad_norm_(dppo.actor_ft.parameters(), args.max_grad_norm)
                    actor_opt.step()
                critic_opt.step()
                if args.target_kl and kl > args.target_kl:
                    print(f"    KL stop ep{ep}: {kl:.4f}>{args.target_kl}"); brk=True; break
            if brk: break
        actor_sched.step(); critic_sched.step()
        vy = np.var(rf.numpy()); evr = np.nan if vy==0 else 1-np.var(rf.numpy()-vf.numpy())/vy
        sps = int(S*ACT_H*N*(itr-start_itr+1)/max(time.time()-t_start,1))
        print(f"  pg={pg.item():.4f} v={vl.item():.4f} kl={kl:.4f} clip={np.mean(cfs):.3f} ev={evr:.3f} SPS={sps}")
        if (itr+1)%args.save_interval==0 or itr==args.total_iters-1:
            torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
                "critic_state_dict": dppo.critic.state_dict(),
                "normalizer_state_dict": dppo.normalizer.state_dict(),
                "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                "bc_config": bc_cfg, "iteration": itr+1, "args": vars(args),
            }, save_dir/f"dppo_iter{itr+1}.pt"); print(f"  Saved iter {itr+1}")

    print(f"\nDone in {time.time()-t_start:.0f}s | Best SR={best_sr:.2%} Prelim={best_prelim} Final={best_final}")
    torch.save({"actor_ft_state_dict": dppo.actor_ft.state_dict(),
        "critic_state_dict": dppo.critic.state_dict(),
        "normalizer_state_dict": dppo.normalizer.state_dict(),
        "bc_config": bc_cfg, "iteration": args.total_iters,
        "best_success_rate": best_sr, "args": vars(args),
    }, save_dir/"dppo_final.pt")
    env.env.close(); simulation_app.close()

if __name__ == "__main__":
    main()
