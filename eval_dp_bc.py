#!/usr/bin/env python3
"""
Diffusion Policy BC 체크포인트를 Isaac Sim에서 GUI로 돌려보는 평가 스크립트.

Usage:
    python eval_dp_bc.py --skill carry_and_place \
        --dp_checkpoint checkpoints/dp_bc_skill3/dp_bc_epoch150.pt \
        --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
        --demo demos_skill3/combined_skill3_20260227_091123.hdf5 \
        --num_episodes 20
"""
from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description="Diffusion Policy BC Eval in Isaac Sim (GUI)")
parser.add_argument("--skill", type=str, required=True,
                    choices=["approach_and_grasp", "carry_and_place"])
parser.add_argument("--dp_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--demo", type=str, default="")
parser.add_argument("--inference_steps", type=int, default=16)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--arm_limit_json", type=str,
                    default="calibration/arm_limits_measured.json")
parser.add_argument("--handoff_buffer", type=str, default="")
parser.add_argument("--arm_action_scale", type=float, default=1.0)
parser.add_argument("--obs_override", action="store_true",
                    help="Override mismatched obs dims with demo values (grip_force, bbox, base_vel)")
parser.add_argument("--open_loop_demo", action="store_true",
                    help="Feed actual demo obs to model instead of env obs (sanity check)")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 카메라 초기 뷰 설정 ──
import sys
import h5py
import torch
import numpy as np

from diffusion_policy import DiffusionPolicyAgent


def load_dp_checkpoint(ckpt_path, device, inference_steps=16):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"],
        action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=inference_steps,
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)
    state_dict = ckpt["model_state_dict"]
    model_state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    norm_state = {k[len("normalizer."):]: v for k, v in state_dict.items() if k.startswith("normalizer.")}
    agent.model.load_state_dict(model_state)
    agent.normalizer.load_state_dict(norm_state)
    agent.eval()
    print(f"Loaded DP: {ckpt_path}")
    print(f"  obs_dim={cfg['obs_dim']}, act_dim={cfg['act_dim']}, "
          f"pred_horizon={cfg['pred_horizon']}, action_horizon={cfg['action_horizon']}")
    print(f"  down_dims={cfg.get('down_dims', [256, 512, 1024])}")
    print(f"  inference_steps={inference_steps}")
    print(f"  epoch={ckpt.get('epoch', '?')}, train_loss={ckpt.get('train_loss', '?')}")
    return agent, cfg


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
            "obs": grp["obs"][:],
            "actions": grp["actions"][:],
            "ep_attrs": dict(grp.attrs),
            "object_pos_w": grp["object_pos_w"][:] if "object_pos_w" in grp else None,
            "object_quat_w": grp["object_quat_w"][:] if "object_quat_w" in grp else None,
        })
    print(f"\n  Demo loaded: {args.demo} ({len(demo_episodes)} episodes)")
    if args.num_episodes > len(demo_episodes):
        args.num_episodes = len(demo_episodes)
        print(f"  num_episodes clamped to {args.num_episodes}")

# ── Env 생성 ──
if args.skill == "approach_and_grasp":
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = 1
elif args.skill == "carry_and_place":
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    env_cfg = Skill3EnvCfg()
    env_cfg.scene.num_envs = 1
    if args.handoff_buffer:
        env_cfg.handoff_buffer_path = args.handoff_buffer

env_cfg.enable_domain_randomization = False
env_cfg.arm_limit_write_to_sim = False
env_cfg.dr_action_delay_steps = 0
env_cfg.grasp_contact_threshold = 0.1
env_cfg.grasp_max_object_dist = 0.50

if demo_file is not None:
    hdf5_attrs = dict(demo_file.attrs)
    env_cfg.object_mass = float(hdf5_attrs.get("object_mass", env_cfg.object_mass))
    env_cfg.object_scale = float(hdf5_attrs.get("object_scale_phys", env_cfg.object_scale))
    env_cfg.arm_action_scale = float(hdf5_attrs.get("arm_action_scale", env_cfg.arm_action_scale))
    env_cfg.max_lin_vel = float(hdf5_attrs.get("max_lin_vel", env_cfg.max_lin_vel))
    env_cfg.max_ang_vel = float(hdf5_attrs.get("max_ang_vel", env_cfg.max_ang_vel))
    if not args.object_usd and "object_usd" in hdf5_attrs:
        env_cfg.object_usd = str(hdf5_attrs["object_usd"])
    print(f"  [Config from HDF5] mass={env_cfg.object_mass}, "
          f"scale={env_cfg.object_scale}, "
          f"arm_action_scale={env_cfg.arm_action_scale}")
else:
    env_cfg.spawn_heading_noise_std = 0.1
    env_cfg.spawn_heading_max_rad = 0.26

# 데모 max step=1535 + grace 500 = 2035 steps. step_dt=0.04 → 81.4초. 여유 포함
env_cfg.episode_length_s = 100.0
env_cfg.dest_spawn_dist_min = 0.6
env_cfg.dest_spawn_dist_max = 0.7
env_cfg.dest_spawn_min_separation = 0.3

if args.object_usd:
    env_cfg.object_usd = os.path.expanduser(args.object_usd)
if args.multi_object_json:
    env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
if args.dest_object_usd:
    env_cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
    env_cfg.arm_limit_json = args.arm_limit_json

if args.skill == "approach_and_grasp":
    env = Skill2Env(cfg=env_cfg)
else:
    env = Skill3Env(cfg=env_cfg)

# ── 모델 로드 ──
dp_agent, dp_cfg = load_dp_checkpoint(args.dp_checkpoint, env.device, args.inference_steps)
action_horizon = dp_cfg["action_horizon"]
obs_dim = dp_cfg["obs_dim"]
action_dim = dp_cfg["act_dim"]
n_params = dp_agent.get_num_params()

mode_str = "HDF5 초기 상태 복원" if demo_episodes else "랜덤 리셋"
print(f"\n{'='*60}")
print(f"  Diffusion Policy BC Eval — {args.skill} ({mode_str})")
print(f"  Checkpoint: {args.dp_checkpoint}")
print(f"  Model: {n_params:,} params, obs_dim={obs_dim}, action_dim={action_dim}")
print(f"  Exec: predict {dp_cfg['pred_horizon']} steps, execute {action_horizon} steps")
print(f"  Episodes: {args.num_episodes}")
print(f"  arm_action_scale: {args.arm_action_scale}")
print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
#  초기 상태 복원
# ═══════════════════════════════════════════════════════════════

def _restore_init_state(ep_data):
    """HDF5 에피소드의 초기 상태로 env를 복원.

    carry_and_place:
      1. robot base를 HDF5 위치로
      2. arm S3 시작 자세 + gripper target_grip(열린 상태)
      3. 물체를 EE에 배치
      4. gripper를 0.35까지 닫으면서 매 스텝 물체를 EE에 텔레포트
      5. 자유 settle (마찰만으로 유지되는지 확인)
    """
    device = env.device
    env_id = torch.tensor([0], device=device)
    ea = ep_data["ep_attrs"]

    # 1. 로봇 base 위치+방향
    if "robot_init_pos" in ea and "robot_init_quat" in ea:
        rs = env.robot.data.root_state_w.clone()
        rs[0, 0:3] = torch.tensor(ea["robot_init_pos"], dtype=torch.float32, device=device)
        rs[0, 3:7] = torch.tensor(ea["robot_init_quat"], dtype=torch.float32, device=device)
        rs[0, 7:] = 0.0
        env.robot.write_root_state_to_sim(rs, env_id)
        env.home_pos_w[0] = rs[0, :3]

    init_joints = torch.tensor(ep_data["obs"][0, 0:6], dtype=torch.float32, device=device)
    target_grip = init_joints[5].item()  # ~0.58

    if args.skill == "carry_and_place":
        from isaaclab.utils.math import quat_apply, quat_mul

        # ── 2. arm S3 자세 + gripper 열린 상태 ──
        jp = env.robot.data.default_joint_pos[0:1].clone()
        jp[0, env.arm_idx[:5]] = init_joints[:5]
        jp[0, env.gripper_idx] = 1.4  # gripper max open (~1.489 limit)
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

        # ── 디버그: wrist 로컬 축 방향 확인 ──
        wrist_pos = env.robot.data.body_pos_w[0, env._fixed_jaw_body_idx, :]
        wrist_quat = env.robot.data.body_quat_w[0, env._fixed_jaw_body_idx, :]

        local_x = quat_apply(wrist_quat.unsqueeze(0), torch.tensor([[1,0,0]], dtype=torch.float32, device=device))[0]
        local_y = quat_apply(wrist_quat.unsqueeze(0), torch.tensor([[0,1,0]], dtype=torch.float32, device=device))[0]
        local_z = quat_apply(wrist_quat.unsqueeze(0), torch.tensor([[0,0,1]], dtype=torch.float32, device=device))[0]

        print(f"    [DEBUG] wrist_pos: {wrist_pos.cpu().tolist()}")
        print(f"    [DEBUG] local_x (world): {local_x.cpu().tolist()}")
        print(f"    [DEBUG] local_y (world): {local_y.cpu().tolist()}")
        print(f"    [DEBUG] local_z (world): {local_z.cpu().tolist()}")
        print(f"    [DEBUG] ee_local_offset: {env._ee_local_offset.cpu().tolist()}")

        ee_default = wrist_pos + quat_apply(wrist_quat.unsqueeze(0), env._ee_local_offset)[0]
        print(f"    [DEBUG] ee_default (world): {ee_default.cpu().tolist()}")

        # ── 3. EE 위치 계산 + 물체 배치 ──
        rot90_local = torch.tensor([0.8192, -0.5736, 0.0, 0.0], dtype=torch.float32, device=device)  # 70deg
        obj_quat = quat_mul(wrist_quat.unsqueeze(0), rot90_local.unsqueeze(0))[0]

        # 순수 EE 위치 (오프셋 수정 없음)
        ee_pos = wrist_pos + quat_apply(wrist_quat.unsqueeze(0), env._ee_local_offset)[0]

        # 물체 bbox center 보정: root→bbox center 차이만큼 빼서 bbox center가 EE에 오도록
        obj_bbox = env.object_bbox[0]  # [sx, sy, sz]
        bbox_center_local = torch.tensor([0.0, 0.0, obj_bbox[2].item() / 2.0],
                                         dtype=torch.float32, device=device)
        bbox_center_world = quat_apply(obj_quat.unsqueeze(0), bbox_center_local.unsqueeze(0))[0]
        obj_root_pos = ee_pos - bbox_center_world

        print(f"    [DEBUG-grasp] wrist_pos:  {[f'{v:.4f}' for v in wrist_pos.cpu().tolist()]}")
        print(f"    [DEBUG-grasp] ee_pos:     {[f'{v:.4f}' for v in ee_pos.cpu().tolist()]}")
        print(f"    [DEBUG-grasp] obj_bbox:   {[f'{v:.4f}' for v in obj_bbox.cpu().tolist()]}")
        print(f"    [DEBUG-grasp] bbox_ctr_l: {[f'{v:.4f}' for v in bbox_center_local.cpu().tolist()]}")
        print(f"    [DEBUG-grasp] bbox_ctr_w: {[f'{v:.4f}' for v in bbox_center_world.cpu().tolist()]}")
        print(f"    [DEBUG-grasp] obj_root:   {[f'{v:.4f}' for v in obj_root_pos.cpu().tolist()]}")

        obj_state = env.object_rigid.data.root_state_w.clone()
        obj_state[0, 0:3] = obj_root_pos
        obj_state[0, 3:7] = obj_quat
        obj_state[0, 7:] = 0.0
        env.object_rigid.write_root_state_to_sim(obj_state, env_id)
        env.object_pos_w[0] = ee_pos

        # ── 4. gripper를 0.35까지 닫으면서 매 스텝 물체를 EE에 텔레포트 ──
        grasp_grip = 0.45
        n_close = 240
        for i in range(n_close):
            t_frac = (i + 1) / n_close
            grip_val = target_grip + (grasp_grip - target_grip) * t_frac
            grip_jp = env.robot.data.joint_pos_target[0:1].clone()
            grip_jp[0, env.gripper_idx] = grip_val
            env.robot.set_joint_position_target(grip_jp, env_ids=env_id)
            env.robot.write_data_to_sim()

            # 물체를 현재 EE에 고정 (매 스텝) — bbox center가 EE에 오도록 보정
            w_pos = env.robot.data.body_pos_w[0, env._fixed_jaw_body_idx, :]
            w_quat = env.robot.data.body_quat_w[0, env._fixed_jaw_body_idx, :]
            cur_ee = w_pos + quat_apply(w_quat.unsqueeze(0), env._ee_local_offset)[0]
            cur_obj_quat = quat_mul(w_quat.unsqueeze(0), rot90_local.unsqueeze(0))[0]
            cur_bbox_w = quat_apply(cur_obj_quat.unsqueeze(0), bbox_center_local.unsqueeze(0))[0]
            obj_st = env.object_rigid.data.root_state_w.clone()
            obj_st[0, 0:3] = cur_ee - cur_bbox_w
            obj_st[0, 3:7] = cur_obj_quat
            obj_st[0, 7:] = 0.0
            env.object_rigid.write_root_state_to_sim(obj_st, env_id)

            env.sim.step()
            env.robot.update(env.sim.cfg.dt)
            env.object_rigid.update(env.sim.cfg.dt)

        # ── 5. 자유 settle (마찰만으로 유지되는지 확인) ──
        for _ in range(120):
            env.robot.write_data_to_sim()
            env.sim.step()
        env.robot.update(env.sim.cfg.dt)
        env.object_rigid.update(env.sim.cfg.dt)

        obj_z = env.object_rigid.data.root_pos_w[0, 2].item()
        grip_sim = env.robot.data.joint_pos[0, env.gripper_idx].item()
        print(f"    [restore] grip={grip_sim:.3f} grasp_grip={grasp_grip} obj_z={obj_z:.3f}")

        # 내부 상태
        env.object_grasped[0] = True
        env.just_dropped[0] = False
        env.intentional_placed[0] = False
        env._fallback_teleport_carry[0] = False
        env.object_pos_w[0] = env.object_rigid.data.root_pos_w[0]

        # dest object 랜덤 스폰 (로봇 전방 heading 기준)
        env._spawn_dest_object(env_id)

    else:
        # Skill-2
        jp = env.robot.data.default_joint_pos[0:1].clone()
        jp[0, env.arm_idx] = init_joints
        jv = torch.zeros_like(jp)
        env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_id)
        env.robot.set_joint_position_target(jp, env_ids=env_id)

        for _ in range(3):
            env.robot.write_data_to_sim()
            env.sim.step()
        env.robot.update(env.sim.cfg.dt)

        obj_state = env.object_rigid.data.root_state_w.clone()
        if ep_data.get("object_pos_w") is not None:
            obj_state[0, 0:3] = torch.tensor(ep_data["object_pos_w"][0],
                                              dtype=torch.float32, device=device)
        elif "object_init_pos" in ea:
            obj_state[0, 0:3] = torch.tensor(ea["object_init_pos"],
                                              dtype=torch.float32, device=device)
        if ep_data.get("object_quat_w") is not None:
            obj_state[0, 3:7] = torch.tensor(ep_data["object_quat_w"][0],
                                              dtype=torch.float32, device=device)
        elif "object_init_quat" in ea:
            obj_state[0, 3:7] = torch.tensor(ea["object_init_quat"],
                                              dtype=torch.float32, device=device)
        obj_state[0, 7:] = 0.0
        env.object_rigid.write_root_state_to_sim(obj_state, env_id)
        env.object_pos_w[0] = obj_state[0, :3]

        for _ in range(3):
            env.robot.write_data_to_sim()
            env.sim.step()
        env.robot.update(env.sim.cfg.dt)
        env.object_rigid.update(env.sim.cfg.dt)


# ═══════════════════════════════════════════════════════════════
#  실행 루프
# ═══════════════════════════════════════════════════════════════

episode = 0
successes = 0
step_count = 0
obs, _ = env.reset()

if demo_episodes:
    _restore_init_state(demo_episodes[0])
    obs = env._get_observations()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)

obs_t = obs["policy"] if isinstance(obs, dict) else obs
print(f"  [DEBUG] obs shape={obs_t.shape}", flush=True)
print(f"  [DEBUG] arm+grip: {obs_t[0,:6].cpu().tolist()}", flush=True)
print(f"  [DEBUG] target_rel: {obs_t[0,21:24].cpu().tolist()}", flush=True)
if obs_t.shape[1] == 30:
    print(f"  [DEBUG] bbox+cat: {obs_t[0,26:30].cpu().tolist()}", flush=True)
elif obs_t.shape[1] == 29:
    print(f"  [DEBUG] grip_force+bbox+cat: {obs_t[0,24:29].cpu().tolist()}", flush=True)
sys.stdout.flush()

dp_agent.reset()

# ── 현재 에피소드의 데모 obs/action (비교용) ──
cur_demo_obs = demo_episodes[min(episode, len(demo_episodes)-1)]["obs"] if demo_episodes else None
cur_demo_act = demo_episodes[min(episode, len(demo_episodes)-1)]["actions"] if demo_episodes else None

while episode < args.num_episodes and simulation_app.is_running():
    obs_t = obs["policy"].to(env.device) if isinstance(obs, dict) else obs.to(env.device)

    # ── open_loop_demo: 환경 obs 대신 데모 obs 직접 사용 ──
    if args.open_loop_demo and cur_demo_obs is not None and step_count < len(cur_demo_obs):
        obs_t = torch.tensor(cur_demo_obs[step_count:step_count+1],
                             dtype=torch.float32, device=env.device)

    # ── obs_override: mismatched dims를 데모 값으로 교체 ──
    elif args.obs_override and cur_demo_obs is not None and step_count < len(cur_demo_obs):
        obs_t = obs_t.clone()
        demo_t = cur_demo_obs[min(step_count, len(cur_demo_obs)-1)]
        # grip_force (dim 24): demo 21-82N, eval ~0
        obs_t[0, 24] = demo_t[24]
        # bbox_norm (dims 25:28): scale mismatch
        obs_t[0, 25:28] = torch.tensor(demo_t[25:28], dtype=torch.float32, device=env.device)
        # cat_norm (dim 28)
        obs_t[0, 28] = demo_t[28]

    with torch.no_grad():
        # eval_resip.py와 동일: base_action_normalized → denormalize
        base_naction = dp_agent.base_action_normalized(obs_t)
        naction = base_naction
        action = dp_agent.normalizer(naction, "action", forward=False)

    if args.arm_action_scale != 1.0:
        action = action.clone()
        action[:, :5] = (action[:, :5] * args.arm_action_scale).clamp(-1.0, 1.0)

    if step_count % 50 == 0:
        o_grip = obs_t[0, 5].item()
        o_dest = obs_t[0, 21:24].cpu().tolist()
        a = action[0].cpu().tolist()
        obj_z = env.object_rigid.data.root_pos_w[0, 2].item()
        env_z = env.scene.env_origins[0, 2].item() if hasattr(env.scene, "env_origins") else 0.0
        obj_h = obj_z - env_z
        dest_xy = (o_dest[0]**2 + o_dest[1]**2)**0.5
        grip_f = obs_t[0, 24].item() if obs_t.shape[1] > 24 else 0.0
        grasped = env.object_grasped[0].item() if hasattr(env, "object_grasped") else False
        # EE-object distance (drop 판정 기준)
        if hasattr(env, '_fixed_jaw_body_idx') and env._fixed_jaw_body_idx >= 0:
            from isaaclab.utils.math import quat_apply as _qa
            _wp = env.robot.data.body_pos_w[0, env._fixed_jaw_body_idx]
            _wq = env.robot.data.body_quat_w[0, env._fixed_jaw_body_idx]
            _ee = _wp + _qa(_wq.unsqueeze(0), env._ee_local_offset)[0]
            ee_obj_d = torch.norm(_ee - env.object_rigid.data.root_pos_w[0]).item()
        else:
            ee_obj_d = -1.0
        base_act = a[6:9]
        print(f"  [t={step_count}] grip={o_grip:.3f} obj_h={obj_h:.3f} dest_d={dest_xy:.3f} "
              f"grip_f={grip_f:.1f} ee_obj={ee_obj_d:.3f} grasped={grasped} "
              f"base=[{base_act[0]:.2f},{base_act[1]:.2f},{base_act[2]:.2f}] "
              f"arm=[{a[0]:.2f},{a[1]:.2f},{a[2]:.2f},{a[3]:.2f},{a[4]:.2f}] grip_act={a[5]:.2f}",
              flush=True)
        # 데모 obs vs eval obs 비교 (첫 3회만 상세)
        if cur_demo_obs is not None and step_count < len(cur_demo_obs) and step_count < 150:
            dt = cur_demo_obs[step_count]
            et = obs_t[0].cpu().numpy()
            print(f"         demo_obs vs eval_obs (abs diff > 0.1):", flush=True)
            dim_names = ["arm0","arm1","arm2","arm3","arm4","grip",
                         "bvx","bvy","bwz","lvx","lvy","lvz","avx","avy","avz",
                         "armv0","armv1","armv2","armv3","armv4","gripv",
                         "dest_x","dest_y","dest_z","grip_f",
                         "bbox0","bbox1","bbox2","cat"]
            for d in range(min(29, len(dt))):
                diff = abs(dt[d] - et[d])
                if diff > 0.1:
                    nm = dim_names[d] if d < len(dim_names) else f"d{d}"
                    print(f"           [{d:2d}] {nm:8s}: demo={dt[d]:+.4f}  eval={et[d]:+.4f}  Δ={diff:.4f}", flush=True)

    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    done = terminated.any() or truncated.any()
    if done:
        episode += 1
        _ps = info.get("place_success_mask", info.get("task_success", torch.zeros(1)))
        success = _ps.any().item() if hasattr(_ps, 'any') else bool(_ps)
        if success:
            successes += 1
        status = "SUCCESS" if success else "FAIL"
        # 종료 원인 분석
        reasons = []
        if success:
            reasons.append("PLACE_SUCCESS")
        else:
            reasons.append("TIMEOUT")
        reason_str = "+".join(reasons) if reasons else "UNKNOWN"
        # 종료 시점 상태
        fin_grip = env.robot.data.joint_pos[0, env.gripper_idx].item()
        fin_obj_h = env.object_rigid.data.root_pos_w[0, 2].item() - (env.scene.env_origins[0, 2].item() if hasattr(env.scene, "env_origins") else 0.0)
        fin_dest_d = torch.norm(env.dest_object_pos_w[0, :2] - env.robot.data.root_pos_w[0, :2]).item() if hasattr(env, 'dest_object_pos_w') else -1.0
        print(f"  Episode {episode}/{args.num_episodes}: {status} ({reason_str}) "
              f"| {step_count} steps | grip={fin_grip:.3f} obj_h={fin_obj_h:.3f} dest_d={fin_dest_d:.3f} "
              f"| cumulative: {successes}/{episode} = {successes/episode*100:.0f}%",
              flush=True)
        step_count = 0
        dp_agent.reset()

        if episode < args.num_episodes:
            obs, _ = env.reset()
            if demo_episodes and episode < len(demo_episodes):
                _restore_init_state(demo_episodes[episode])
                obs = env._get_observations()
                env.robot.update(env.sim.cfg.dt)
                env.object_rigid.update(env.sim.cfg.dt)

print(f"\n  === 결과: {successes}/{args.num_episodes} 성공 "
      f"({successes/max(episode,1)*100:.0f}%) ===\n")

if demo_file is not None:
    demo_file.close()
env.close()
simulation_app.close()
