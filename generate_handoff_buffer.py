#!/usr/bin/env python3
"""
Skill-2 Expert 실행 -> 성공 에피소드 종료 상태를 Handoff Buffer로 저장.

Usage:
    python generate_handoff_buffer.py \
      --checkpoint logs/ppo_skill2/best_agent.pt \
      --num_entries 500 --num_envs 64 \
      --output handoff_buffer.pkl \
      --multi_object_json object_catalog.json \
      --gripper_contact_prim_path "..." \
      --dynamics_json calibration/tuned_dynamics.json \
      --arm_limit_json calibration/arm_limits_real2sim.json \
      --headless
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_entries", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--output", type=str, default="handoff_buffer.pkl")
parser.add_argument("--noise_arm_std", type=float, default=0.05,
                    help="Arm joint noise std (rad). VLA의 부정확한 grasp 상태 모사")
parser.add_argument("--noise_obj_xy_std", type=float, default=0.02,
                    help="Object position noise std (m)")
parser.add_argument("--noise_base_xy_std", type=float, default=0.03,
                    help="Base position noise std (m)")
parser.add_argument("--noise_base_yaw_std", type=float, default=0.1,
                    help="Base orientation noise std (rad)")
parser.add_argument("--dynamics_json", type=str, default=None)
parser.add_argument("--calibration_json", type=str, default=None)
parser.add_argument("--arm_limit_json", type=str, default=None)
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np
import torch
from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
from models import PolicyNet, ValueNet
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler


def main():
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.calibration_json:
        env_cfg.calibration_json = os.path.expanduser(args.calibration_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
    if args.multi_object_json:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    # Curriculum을 최대로 열어서 다양한 거리 커버
    env_cfg.object_dist_min = 0.5
    env_cfg.curriculum_current_max_dist = env_cfg.object_dist_max

    env = Skill2Env(cfg=env_cfg)
    wrapped = wrap_env(env, wrapper="isaaclab")
    device = wrapped.device

    models = {
        "policy": PolicyNet(wrapped.observation_space, wrapped.action_space, device),
        "value": ValueNet(wrapped.observation_space, wrapped.action_space, device),
    }
    memory = RandomMemory(memory_size=24, num_envs=args.num_envs, device=device)
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": wrapped.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    agent = PPO(models=models, memory=memory, cfg=cfg_ppo,
                observation_space=wrapped.observation_space,
                action_space=wrapped.action_space, device=device)
    agent.load(args.checkpoint)
    agent.set_running_mode("eval")

    entries = []
    obs, _ = env.reset()
    print(f"\n  Collecting {args.num_entries} handoff entries...")

    while len(entries) < args.num_entries:
        with torch.no_grad():
            action = agent.act({"states": obs["policy"]}, timestep=0, timesteps=1)[0]
        obs, _, terminated, truncated, _ = env.step(action)

        success = env.task_success
        if success.any():
            sids = success.nonzero(as_tuple=False).squeeze(-1)
            for sid in sids:
                i = sid.item()
                # Active object의 orientation 읽기
                oi = int(env.active_object_idx[i].item())
                if env._multi_object and oi < len(env.object_rigids):
                    obj_quat = env.object_rigids[oi].data.root_quat_w[i].cpu().tolist()
                elif env.object_rigid is not None:
                    obj_quat = env.object_rigid.data.root_quat_w[i].cpu().tolist()
                else:
                    obj_quat = [1.0, 0.0, 0.0, 0.0]  # identity

                entry = {
                    "base_pos": env.robot.data.root_pos_w[i].cpu().tolist(),
                    "base_ori": env.robot.data.root_quat_w[i].cpu().tolist(),
                    "arm_joints": env.robot.data.joint_pos[i, env.arm_idx[:5]].cpu().tolist(),
                    "gripper_state": env.robot.data.joint_pos[i, env.arm_idx[5]].item(),
                    "object_pos": env.object_pos_w[i].cpu().tolist(),
                    "object_ori": obj_quat,
                    "object_type_idx": env.active_object_idx[i].item(),
                }

                # Noise injection — VLA의 부정확한 Skill-2 출력을 모사
                if args.noise_arm_std > 0:
                    entry["arm_joints"] = [
                        v + np.random.normal(0, args.noise_arm_std) for v in entry["arm_joints"]
                    ]
                if args.noise_obj_xy_std > 0:
                    entry["object_pos"][0] += np.random.normal(0, args.noise_obj_xy_std)
                    entry["object_pos"][1] += np.random.normal(0, args.noise_obj_xy_std)
                if args.noise_base_xy_std > 0:
                    entry["base_pos"][0] += np.random.normal(0, args.noise_base_xy_std)
                    entry["base_pos"][1] += np.random.normal(0, args.noise_base_xy_std)
                if args.noise_base_yaw_std > 0:
                    w, x, y, z = entry["base_ori"]
                    cur_yaw = 2.0 * np.arctan2(z, w)
                    new_yaw = cur_yaw + np.random.normal(0, args.noise_base_yaw_std)
                    entry["base_ori"] = [float(np.cos(new_yaw/2)), 0.0, 0.0, float(np.sin(new_yaw/2))]

                entries.append(entry)

        if len(entries) % 50 == 0 and len(entries) > 0:
            print(f"    {len(entries)}/{args.num_entries}")

    entries = entries[:args.num_entries]
    with open(args.output, "wb") as f:
        pickle.dump(entries, f)
    print(f"\n  Saved {len(entries)} entries to {args.output}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
