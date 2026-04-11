#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


EE23_NAMES = (
    "arm0",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "grip",
    "armvel0",
    "armvel1",
    "armvel2",
    "armvel3",
    "armvel4",
    "grip_vel",
    "base_vx",
    "base_vy",
    "base_wz",
    "ee_to_dest_body_x",
    "ee_to_dest_body_y",
    "ee_to_dest_body_z",
    "dest_rel_body_x",
    "dest_rel_body_y",
    "phase_a_flag",
    "release_phase_flag",
    "retract_started_flag",
)


def quat_apply_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector v into local frame defined by q (wxyz)."""
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    w = q[..., 0:1]
    xyz = -q[..., 1:4]  # conjugate
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def infer_default_output(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}_ee23.hdf5"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Skill-3 phased53 HDF5 to ee23 HDF5")
    parser.add_argument("--input", required=True, help="input phased53 HDF5")
    parser.add_argument("--output", default=None, help="output ee23 HDF5")
    parser.add_argument("--release_xy", type=float, default=0.12)
    parser.add_argument("--release_ee_z", type=float, default=0.10)
    parser.add_argument("--retract_grip", type=float, default=0.90)
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output) if args.output else infer_default_output(input_path)

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value
        f_out.attrs["obs_dim"] = 23
        f_out.attrs["obs_version"] = "skill3_ee23_v1"
        f_out.attrs["obs_feature_names"] = ",".join(EE23_NAMES)
        f_out.attrs["s3_obs_mode"] = "ee23"
        f_out.attrs["s3_motion_release_xy"] = float(args.release_xy)
        f_out.attrs["s3_motion_release_ee_z"] = float(args.release_ee_z)
        f_out.attrs["s3_motion_retract_grip"] = float(args.retract_grip)
        f_out.attrs["source_hdf5"] = input_path

        ep_names = sorted(k for k in f_in.keys() if k.startswith("episode_"))
        for ep in ep_names:
            g_in = f_in[ep]
            obs53 = g_in["obs"][:].astype(np.float32)
            robot_quat = g_in["robot_quat_w"][:].astype(np.float32)
            dest_pos = g_in["dest_pos_w"][:].astype(np.float32)
            ee_pos = g_in["ee_pos_w"][:].astype(np.float32)

            arm_grip_pos = obs53[:, 0:6]
            arm_grip_vel = obs53[:, 15:21]
            base_vel = obs53[:, 6:9]
            dest_rel_body_xy = obs53[:, 21:23]
            phase_a_flag = obs53[:, 35:36]
            grip_pos = obs53[:, 5:6]

            ee_to_dest_world = dest_pos - ee_pos
            ee_to_dest_body = quat_apply_inverse_wxyz(robot_quat, ee_to_dest_world)
            ee_xy = np.linalg.norm(ee_to_dest_body[:, :2], axis=1, keepdims=True)
            ee_z = ee_pos[:, 2:3]
            phase_b = phase_a_flag < 0.5

            release_now = phase_b & (ee_xy <= float(args.release_xy)) & (ee_z <= float(args.release_ee_z))
            release_flag = np.zeros((len(obs53), 1), dtype=np.float32)
            retract_flag = np.zeros((len(obs53), 1), dtype=np.float32)
            release_active = False
            retract_active = False
            for i in range(len(obs53)):
                if bool(release_now[i, 0]):
                    release_active = True
                if release_active and float(grip_pos[i, 0]) >= float(args.retract_grip):
                    retract_active = True
                release_flag[i, 0] = 1.0 if release_active else 0.0
                retract_flag[i, 0] = 1.0 if retract_active else 0.0

            obs23 = np.concatenate(
                [
                    arm_grip_pos,
                    arm_grip_vel,
                    base_vel,
                    ee_to_dest_body,
                    dest_rel_body_xy,
                    phase_a_flag,
                    release_flag,
                    retract_flag,
                ],
                axis=1,
            ).astype(np.float32)

            if obs23.shape[1] != 23:
                raise RuntimeError(f"{ep}: expected 23 dims, got {obs23.shape[1]}")

            g_out = f_out.create_group(ep)
            for name in g_in.keys():
                if name == "obs":
                    continue
                g_out.create_dataset(name, data=g_in[name][:])
            g_out.create_dataset("obs", data=obs23)
            for key, value in g_in.attrs.items():
                g_out.attrs[key] = value

    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
