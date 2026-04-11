#!/usr/bin/env python3
"""Build 38D Skill-3 demos with explicit release/retract phase signals.

Observation layout:
  36D original obs
  + release_phase_flag
  + retract_started_flag

Two outputs are created from smoothed_v2:
1. full 38D: keep the full episode and append both flags.
2. short 38D: crop from an already-lowered, near-target pre-open state
   through release and retract/rest, then append both flags.
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


def _reconstruct_src_dst_xy(
    obs: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    object_pos_w: np.ndarray,
) -> np.ndarray:
    dest_rel_body = obs[:, 21:24]
    dest_pos_w = robot_pos_w + _quat_apply_np(robot_quat_w, dest_rel_body)
    return np.linalg.norm(object_pos_w[:, :2] - dest_pos_w[:, :2], axis=-1)


def _find_release_start(
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
        return int(ready[0]), open_start

    arm_only = np.flatnonzero((idx >= lo) & (idx < hi) & (arm1 >= arm1_ready))
    if len(arm_only) > 0:
        return int(arm_only[0]), open_start

    return max(0, open_start - min(lookback, 100)), open_start


def _find_retract_start(
    obs: np.ndarray,
    actions: np.ndarray,
    release_start: int,
    open_start: int,
    open_hold_min: int,
    retract_delay: int,
) -> int:
    arm1 = obs[:, 1]
    grip_act = actions[:, 5]
    if open_start >= len(obs) - 1:
        return len(obs) - 1

    peak_rel = int(np.argmax(arm1[open_start:]))
    peak_idx = open_start + peak_rel
    base_start = max(open_start + open_hold_min, peak_idx + retract_delay)
    base_start = min(base_start, len(obs) - 1)

    # Prefer the first step after the hold where arm1 is genuinely descending
    # while the gripper action is no longer strongly opening.
    for idx in range(base_start, len(obs) - 3):
        a0, a1, a2, a3 = arm1[idx : idx + 4]
        g0 = grip_act[idx]
        if (a1 <= a0 and a2 <= a1 and a3 <= a2) and (g0 <= 0.15):
            return idx
    return base_start


def _append_flags(obs: np.ndarray, release_start: int, retract_start: int) -> np.ndarray:
    release_flag = np.zeros((len(obs), 1), dtype=np.float32)
    retract_flag = np.zeros((len(obs), 1), dtype=np.float32)
    release_flag[release_start:, 0] = 1.0
    retract_flag[retract_start:, 0] = 1.0
    return np.concatenate([obs, release_flag, retract_flag], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("demos_skill3/smoothed_v2.hdf5"))
    parser.add_argument(
        "--full_output",
        type=Path,
        default=Path("demos_skill3/smoothed_v2_release_retract_v21_full38.hdf5"),
    )
    parser.add_argument(
        "--short_output",
        type=Path,
        default=Path("demos_skill3/smoothed_v2_release_retract_v21_short38.hdf5"),
    )
    parser.add_argument("--arm1_ready", type=float, default=2.0)
    parser.add_argument("--src_dst_max", type=float, default=0.18)
    parser.add_argument("--objz_min", type=float, default=0.029)
    parser.add_argument("--objz_max", type=float, default=0.055)
    parser.add_argument("--bbox_center_offset_z", type=float, default=0.03108454868197441)
    parser.add_argument("--grip_open_thresh", type=float, default=0.05)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--open_hold_min", type=int, default=25)
    parser.add_argument("--retract_delay", type=int, default=10)
    args = parser.parse_args()

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    args.short_output.parent.mkdir(parents=True, exist_ok=True)

    release_starts = []
    open_starts = []
    retract_starts = []
    crop_srcs = []
    crop_arm1s = []
    short_lengths = []

    with h5py.File(args.input, "r") as f_in, \
            h5py.File(args.full_output, "w") as f_full, \
            h5py.File(args.short_output, "w") as f_short:
        for dst in (f_full, f_short):
            _copy_attrs(f_in, dst)
            dst.attrs["source_hdf5"] = str(args.input)
            dst.attrs["obs_dim"] = 38
            dst.attrs["release_phase_flag_added"] = True
            dst.attrs["retract_started_flag_added"] = True
            dst.attrs["arm1_ready"] = args.arm1_ready
            dst.attrs["src_dst_max"] = args.src_dst_max
            dst.attrs["objz_min"] = args.objz_min
            dst.attrs["objz_max"] = args.objz_max
            dst.attrs["bbox_center_offset_z"] = args.bbox_center_offset_z
            dst.attrs["grip_open_thresh"] = args.grip_open_thresh
            dst.attrs["lookback"] = args.lookback
            dst.attrs["open_hold_min"] = args.open_hold_min
            dst.attrs["retract_delay"] = args.retract_delay

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
            release_start, open_start = _find_release_start(
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
            retract_start = _find_retract_start(
                obs=obs,
                actions=actions,
                release_start=release_start,
                open_start=open_start,
                open_hold_min=args.open_hold_min,
                retract_delay=args.retract_delay,
            )

            release_starts.append(release_start)
            open_starts.append(open_start)
            retract_starts.append(retract_start)
            crop_srcs.append(float(src_dst_xy[release_start]))
            crop_arm1s.append(float(obs[release_start, 1]))
            short_lengths.append(len(obs) - release_start)

            full_grp = f_full.create_group(ep)
            _copy_attrs(src, full_grp)
            full_grp.attrs["release_phase_start_idx"] = int(release_start)
            full_grp.attrs["open_start_idx"] = int(open_start)
            full_grp.attrs["retract_start_idx"] = int(retract_start)
            full_grp.attrs["release_src_dst_xy"] = float(src_dst_xy[release_start])
            full_grp.attrs["release_arm1"] = float(obs[release_start, 1])
            full_grp.attrs["release_obj_center_z"] = float(obj_center_z[release_start])
            for key, value in arrays.items():
                if key == "obs":
                    full_grp.create_dataset(
                        key,
                        data=_append_flags(value, release_start, retract_start),
                        compression="gzip",
                    )
                else:
                    full_grp.create_dataset(key, data=value, compression="gzip")

            short_grp = f_short.create_group(ep)
            _copy_attrs(src, short_grp)
            short_grp.attrs["release_phase_start_idx"] = int(release_start)
            short_grp.attrs["open_start_idx"] = int(open_start)
            short_grp.attrs["retract_start_idx"] = int(retract_start)
            short_grp.attrs["release_src_dst_xy"] = float(src_dst_xy[release_start])
            short_grp.attrs["release_arm1"] = float(obs[release_start, 1])
            short_grp.attrs["release_obj_center_z"] = float(obj_center_z[release_start])

            short_obs = obs[release_start:].copy()
            short_release_flag = np.ones((len(short_obs), 1), dtype=np.float32)
            short_retract_flag = np.zeros((len(short_obs), 1), dtype=np.float32)
            retract_rel = max(0, retract_start - release_start)
            short_retract_flag[retract_rel:, 0] = 1.0
            short_obs_38 = np.concatenate([short_obs, short_release_flag, short_retract_flag], axis=-1)
            short_grp.create_dataset("obs", data=short_obs_38, compression="gzip")
            for key, value in arrays.items():
                if key == "obs":
                    continue
                short_grp.create_dataset(key, data=value[release_start:], compression="gzip")

    print(f"input:  {args.input}")
    print(f"full:   {args.full_output}")
    print(f"short:  {args.short_output}")
    print(f"episodes: {len(release_starts)}")
    print(
        "release_start mean/min/max: "
        f"{np.mean(release_starts):.1f} / {np.min(release_starts)} / {np.max(release_starts)}"
    )
    print(
        "open_start mean/min/max: "
        f"{np.mean(open_starts):.1f} / {np.min(open_starts)} / {np.max(open_starts)}"
    )
    print(
        "retract_start mean/min/max: "
        f"{np.mean(retract_starts):.1f} / {np.min(retract_starts)} / {np.max(retract_starts)}"
    )
    print(
        "crop src_dst mean/min/max: "
        f"{np.mean(crop_srcs):.4f} / {np.min(crop_srcs):.4f} / {np.max(crop_srcs):.4f}"
    )
    print(
        "crop arm1 mean/min/max: "
        f"{np.mean(crop_arm1s):.4f} / {np.min(crop_arm1s):.4f} / {np.max(crop_arm1s):.4f}"
    )
    print(
        "short length mean/min/max: "
        f"{np.mean(short_lengths):.1f} / {np.min(short_lengths)} / {np.max(short_lengths)}"
    )


if __name__ == "__main__":
    main()
