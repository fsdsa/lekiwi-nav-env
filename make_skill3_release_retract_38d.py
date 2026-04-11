#!/usr/bin/env python3
"""Build 38D release/retract demos from smoothed_v2.

Adds two sub-phase signals to the original 36D Skill-3 BC observation:
  - release_retract_flag
  - release_open_started_flag

Creates:
1. full 38D demos: keep the full episode, append both flags.
2. short 38D demos: crop from a "already lowered + near target + just before open"
   start to the end, keep retract/rest, and append both flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _ep_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _quat_apply_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w = quat_wxyz[:, 0:1]
    xyz = quat_wxyz[:, 1:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + w * t + np.cross(xyz, t)


def _reconstruct_src_dst_xy(obs: np.ndarray, robot_pos_w: np.ndarray, robot_quat_w: np.ndarray, object_pos_w: np.ndarray) -> np.ndarray:
    dest_rel_body = obs[:, 21:24]
    dest_pos_w = robot_pos_w + _quat_apply_np(robot_quat_w, dest_rel_body)
    return np.linalg.norm(object_pos_w[:, :2] - dest_pos_w[:, :2], axis=-1)


def _find_crop_start(
    obs: np.ndarray,
    actions: np.ndarray,
    src_dst_xy: np.ndarray,
    obj_center_z: np.ndarray,
    arm1_ready: float,
    src_dst_max: float,
    objz_min: float,
    objz_max: float,
    grip_open_thresh: float,
    lookback: int,
) -> tuple[int, int]:
    arm1 = obs[:, 1]
    grip_act = actions[:, 5]

    open_idx = np.flatnonzero(grip_act > grip_open_thresh)
    if len(open_idx) == 0:
        open_start = int(np.argmax(grip_act))
    else:
        open_start = int(open_idx[0])

    lo = max(0, open_start - lookback)
    hi = open_start + 1
    idx = np.arange(len(obs))
    ready_mask = (
        (idx >= lo)
        & (idx < hi)
        & (arm1 >= arm1_ready)
        & (src_dst_xy <= src_dst_max)
        & (obj_center_z >= objz_min)
        & (obj_center_z <= objz_max)
    )
    ready = np.flatnonzero(ready_mask)
    if len(ready) > 0:
        start = int(ready[0])
        return start, open_start

    arm_only = np.flatnonzero((idx >= lo) & (idx < hi) & (arm1 >= arm1_ready))
    if len(arm_only) > 0:
        start = int(arm_only[0])
        return start, open_start

    start = max(0, open_start - min(lookback, 80))
    return start, open_start


def _append_flags(obs: np.ndarray, start_idx: int, open_start_idx: int) -> np.ndarray:
    release_flag = np.zeros((len(obs), 1), dtype=np.float32)
    open_flag = np.zeros((len(obs), 1), dtype=np.float32)
    release_flag[start_idx:, 0] = 1.0
    open_flag[open_start_idx:, 0] = 1.0
    return np.concatenate([obs, release_flag, open_flag], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("demos_skill3/smoothed_v2.hdf5"))
    parser.add_argument("--full_output", type=Path, default=Path("demos_skill3/smoothed_v2_release_retract_flag38.hdf5"))
    parser.add_argument("--short_output", type=Path, default=Path("demos_skill3/smoothed_v2_release_retract_short_38d.hdf5"))
    parser.add_argument("--arm1_ready", type=float, default=2.2)
    parser.add_argument("--src_dst_max", type=float, default=0.18)
    parser.add_argument("--objz_min", type=float, default=0.029)
    parser.add_argument("--objz_max", type=float, default=0.050)
    parser.add_argument("--bbox_center_offset_z", type=float, default=0.03108454868197441)
    parser.add_argument("--grip_open_thresh", type=float, default=0.55)
    parser.add_argument("--lookback", type=int, default=80)
    args = parser.parse_args()

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    args.short_output.parent.mkdir(parents=True, exist_ok=True)

    starts = []
    opens = []
    srcs = []
    arm1s = []
    lengths = []

    with h5py.File(args.input, "r") as f_in, \
            h5py.File(args.full_output, "w") as f_full, \
            h5py.File(args.short_output, "w") as f_short:
        for dst in (f_full, f_short):
            _copy_attrs(f_in, dst)
            dst.attrs["source_hdf5"] = str(args.input)
            dst.attrs["obs_dim"] = 38
            dst.attrs["release_retract_flag_added"] = True
            dst.attrs["release_open_started_flag_added"] = True
            dst.attrs["arm1_ready"] = args.arm1_ready
            dst.attrs["src_dst_max"] = args.src_dst_max
            dst.attrs["objz_min"] = args.objz_min
            dst.attrs["objz_max"] = args.objz_max
            dst.attrs["bbox_center_offset_z"] = args.bbox_center_offset_z
            dst.attrs["grip_open_thresh"] = args.grip_open_thresh
            dst.attrs["lookback"] = args.lookback

        for ep in sorted(f_in.keys(), key=_ep_sort_key):
            src = f_in[ep]
            arrays = {k: src[k][:] for k in src.keys()}
            obs = arrays["obs"]
            actions = arrays["actions"]
            robot_pos_w = arrays["robot_pos_w"]
            robot_quat_w = arrays["robot_quat_w"]
            object_pos_w = arrays["object_pos_w"]

            src_dst_xy = _reconstruct_src_dst_xy(obs, robot_pos_w, robot_quat_w, object_pos_w)
            obj_center_z = object_pos_w[:, 2] + args.bbox_center_offset_z
            start_idx, open_start_idx = _find_crop_start(
                obs=obs,
                actions=actions,
                src_dst_xy=src_dst_xy,
                obj_center_z=obj_center_z,
                arm1_ready=args.arm1_ready,
                src_dst_max=args.src_dst_max,
                objz_min=args.objz_min,
                objz_max=args.objz_max,
                grip_open_thresh=args.grip_open_thresh,
                lookback=args.lookback,
            )

            starts.append(start_idx)
            opens.append(open_start_idx)
            srcs.append(float(src_dst_xy[start_idx]))
            arm1s.append(float(obs[start_idx, 1]))
            lengths.append(len(obs) - start_idx)

            full_grp = f_full.create_group(ep)
            _copy_attrs(src, full_grp)
            full_grp.attrs["release_retract_start_idx"] = int(start_idx)
            full_grp.attrs["release_open_start_idx"] = int(open_start_idx)
            full_grp.attrs["release_retract_src_dst_xy"] = float(src_dst_xy[start_idx])
            full_grp.attrs["release_retract_arm1"] = float(obs[start_idx, 1])
            full_grp.attrs["release_retract_obj_center_z"] = float(obj_center_z[start_idx])
            for key, value in arrays.items():
                if key == "obs":
                    full_grp.create_dataset(key, data=_append_flags(value, start_idx, open_start_idx), compression="gzip")
                else:
                    full_grp.create_dataset(key, data=value, compression="gzip")

            short_grp = f_short.create_group(ep)
            _copy_attrs(src, short_grp)
            short_grp.attrs["release_retract_start_idx"] = int(start_idx)
            short_grp.attrs["release_open_start_idx"] = int(open_start_idx)
            short_grp.attrs["release_retract_src_dst_xy"] = float(src_dst_xy[start_idx])
            short_grp.attrs["release_retract_arm1"] = float(obs[start_idx, 1])
            short_grp.attrs["release_retract_obj_center_z"] = float(obj_center_z[start_idx])

            short_obs = obs[start_idx:].copy()
            short_release_flag = np.ones((len(short_obs), 1), dtype=np.float32)
            open_rel = max(0, open_start_idx - start_idx)
            short_open_flag = np.zeros((len(short_obs), 1), dtype=np.float32)
            short_open_flag[open_rel:, 0] = 1.0
            short_obs_38 = np.concatenate([short_obs, short_release_flag, short_open_flag], axis=-1)
            short_grp.create_dataset("obs", data=short_obs_38, compression="gzip")
            for key, value in arrays.items():
                if key == "obs":
                    continue
                short_grp.create_dataset(key, data=value[start_idx:], compression="gzip")

    print(f"input:  {args.input}")
    print(f"full:   {args.full_output}")
    print(f"short:  {args.short_output}")
    print(f"episodes: {len(starts)}")
    print(
        "release_start mean/min/max: "
        f"{np.mean(starts):.1f} / {np.min(starts)} / {np.max(starts)}"
    )
    print(
        "open_start mean/min/max: "
        f"{np.mean(opens):.1f} / {np.min(opens)} / {np.max(opens)}"
    )
    print(
        "crop src_dst mean/min/max: "
        f"{np.mean(srcs):.4f} / {np.min(srcs):.4f} / {np.max(srcs):.4f}"
    )
    print(
        "crop arm1 mean/min/max: "
        f"{np.mean(arm1s):.4f} / {np.min(arm1s):.4f} / {np.max(arm1s):.4f}"
    )
    print(
        "short length mean/min/max: "
        f"{np.mean(lengths):.1f} / {np.min(lengths)} / {np.max(lengths)}"
    )


if __name__ == "__main__":
    main()
