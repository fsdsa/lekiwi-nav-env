#!/usr/bin/env python3
"""Convert existing Skill-3 36D demos into motion-prior 24D demos.

Input expectation:
  - obs: 36D = policy29 + init_pose6 + phase_a_flag
  - actions: 9D
  - robot_state: 9D (arm/grip pos + base vel)
  - robot_pos_w / robot_quat_w saved per step

This script reconstructs EE pose from saved robot root pose + arm joints inside
the simulator, then builds the new 24D motion-prior observation:
  arm/grip pos 6
  arm/grip vel 6
  base vel 3
  ee->dest body XY 2
  dest_rel_body XYZ 3
  ee_z 1
  phase_a_flag 1
  release_phase_flag 1
  retract_started_flag 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dest_object_usd", type=str, required=True)
    parser.add_argument("--object_usd", type=str, default="")
    parser.add_argument("--object_scale_phys", type=float, default=1.0)
    parser.add_argument(
        "--gripper_contact_prim_path",
        type=str,
        default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1",
    )
    parser.add_argument("--release_xy", type=float, default=0.12)
    parser.add_argument("--release_ee_z", type=float, default=0.10)
    parser.add_argument("--retract_grip", type=float, default=0.55)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.num_envs = 1
    return args


args = parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

from isaaclab.utils.math import quat_apply
from lekiwi_skill2_env import Skill2EnvCfg, Skill2Env
from skill3_bc_obs import build_s3_motion24_obs, S3_BC_OBS_MOTION24_DIM, S3_BC_MOTION24_NAMES


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _copy_group_with_new_obs(src_grp, dst_grp, new_obs: np.ndarray) -> None:
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v
    for key in src_grp.keys():
        if key == "obs":
            dst_grp.create_dataset("obs", data=new_obs.astype(np.float32), compression="gzip")
        else:
            dst_grp.create_dataset(key, data=src_grp[key][:], compression="gzip")


def _reconstruct_dest_world(robot_pos_w: np.ndarray, robot_quat_w: np.ndarray, dest_rel_body: np.ndarray) -> np.ndarray:
    robot_pos = torch.from_numpy(robot_pos_w).float()
    robot_quat = torch.from_numpy(robot_quat_w).float()
    dest_rel = torch.from_numpy(dest_rel_body).float()
    dest_world = robot_pos + quat_apply(robot_quat, dest_rel)
    return dest_world.cpu().numpy()


def main() -> None:
    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(in_path, "r") as f_in:
        object_usd = args.object_usd or str(f_in.attrs.get("object_usd", "")).strip()
        if not object_usd:
            raise ValueError("--object_usd is required when input HDF5 has no object_usd attr")

    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.object_usd = os.path.expanduser(object_usd)
    cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    cfg.object_scale = float(args.object_scale_phys)
    cfg.gripper_contact_prim_path = str(args.gripper_contact_prim_path)
    cfg.grasp_success_height = 100.0
    env = Skill2Env(cfg=cfg)
    env.reset()
    device = env.device
    env_id = torch.tensor([0], device=device, dtype=torch.long)

    with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        _copy_attrs(f_in, f_out)
        f_out.attrs["obs_dim"] = S3_BC_OBS_MOTION24_DIM
        f_out.attrs["obs_version"] = "skill3_motion24_v1"
        f_out.attrs["obs_feature_names"] = ",".join(S3_BC_MOTION24_NAMES)
        f_out.attrs["source_hdf5"] = str(in_path)
        f_out.attrs["release_xy"] = float(args.release_xy)
        f_out.attrs["release_ee_z"] = float(args.release_ee_z)
        f_out.attrs["retract_grip"] = float(args.retract_grip)
        f_out.attrs["dest_object_usd"] = os.path.expanduser(args.dest_object_usd)

        ep_names = sorted([k for k in f_in.keys() if k.startswith("episode_")], key=lambda x: int(x.split("_")[1]))
        for ep_name in ep_names:
            src = f_in[ep_name]
            obs36 = src["obs"][:].astype(np.float32)
            actions = src["actions"][:].astype(np.float32)
            robot_state = src["robot_state"][:].astype(np.float32)
            robot_pos_w = src["robot_pos_w"][:].astype(np.float32)
            robot_quat_w = src["robot_quat_w"][:].astype(np.float32)

            if obs36.shape[1] != 36:
                raise ValueError(f"{ep_name}: expected 36D obs, got {obs36.shape[1]}")

            policy29 = obs36[:, :29]
            phase_a_flag = obs36[:, 35]
            dest_rel_body = policy29[:, 21:24]
            dest_world = _reconstruct_dest_world(robot_pos_w, robot_quat_w, dest_rel_body)

            new_obs = np.zeros((obs36.shape[0], S3_BC_OBS_MOTION24_DIM), dtype=np.float32)
            release_latch = False
            retract_latch = False

            root_state = env.robot.data.root_state_w.clone()
            joint_pos = env.robot.data.default_joint_pos[0:1].clone()
            joint_vel = torch.zeros_like(joint_pos)

            for t in range(obs36.shape[0]):
                root_state[0, 0:3] = torch.tensor(robot_pos_w[t], dtype=torch.float32, device=device)
                root_state[0, 3:7] = torch.tensor(robot_quat_w[t], dtype=torch.float32, device=device)
                root_state[0, 7:] = 0.0
                env.robot.write_root_state_to_sim(root_state, env_id)
                env.home_pos_w[0] = root_state[0, 0:3]

                joint_pos[:] = env.robot.data.default_joint_pos[0:1]
                joint_pos[0, env.arm_idx] = torch.tensor(robot_state[t, :6], dtype=torch.float32, device=device)
                joint_pos[0, env.wheel_idx] = 0.0
                joint_vel.zero_()
                env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_id)
                env.robot.set_joint_position_target(joint_pos, env_ids=env_id)

                env.dest_object_pos_w[0] = torch.tensor(dest_world[t], dtype=torch.float32, device=device)
                env.sim.forward()
                env.robot.update(env.sim.cfg.dt)

                motion_obs = build_s3_motion24_obs(
                    env,
                    torch.tensor(policy29[t:t+1], dtype=torch.float32, device=device),
                    float(phase_a_flag[t]),
                    release_phase_flag=None,
                    retract_started_flag=None,
                    release_xy_thresh=float(args.release_xy),
                    release_ee_z_thresh=float(args.release_ee_z),
                    retract_grip_thresh=float(args.retract_grip),
                )[0].detach().cpu().numpy().astype(np.float32)

                release_latch = release_latch or bool(motion_obs[22] > 0.5)
                retract_latch = retract_latch or bool(motion_obs[23] > 0.5)
                motion_obs[22] = 1.0 if release_latch else 0.0
                motion_obs[23] = 1.0 if retract_latch else 0.0
                new_obs[t] = motion_obs

            dst = f_out.create_group(ep_name)
            _copy_group_with_new_obs(src, dst, new_obs)

    print(f"input:  {in_path}")
    print(f"output: {out_path}")
    print(f"obs_dim: {S3_BC_OBS_MOTION24_DIM}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sim_app.close()
