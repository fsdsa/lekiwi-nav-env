#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def quat_apply_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.T
    qvec = np.stack([x, y, z], axis=1)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (w[:, None] * uv + uuv)


def longest_run(indices: np.ndarray) -> tuple[int, int] | None:
    if len(indices) == 0:
        return None
    start = prev = int(indices[0])
    best = (start, start)
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        if prev - start > best[1] - best[0]:
            best = (start, prev)
        start = prev = idx
    if prev - start > best[1] - best[0]:
        best = (start, prev)
    return best


def pick_anchor(obs: np.ndarray, actions: np.ndarray, arm1: np.ndarray, grip: np.ndarray, run: tuple[int, int]) -> int:
    a, b = run
    ids = np.arange(a, b + 1)
    local_vel = np.linalg.norm(obs[ids, 15:21], axis=1)
    d_arm1 = np.abs(np.diff(arm1[ids], prepend=arm1[ids[0]]))
    d_grip = np.abs(np.diff(grip[ids], prepend=grip[ids[0]]))
    grip_act = np.abs(actions[ids, 5])
    # Favor low-motion held states near the bottom plateau.
    score = 5.0 * d_arm1 + 2.0 * d_grip + 0.15 * local_vel + 0.25 * grip_act
    return int(ids[np.argmin(score)])


def copy_with_tail_boost(
    src_path: Path,
    dst_path: Path,
    arm_margin: float,
    min_grip_hold: float,
    upright_thresh: float,
    min_arm1_max: float,
    min_run_len: int,
    segment_len: int,
    repeat_count: int,
) -> None:
    boosted = []
    total_steps_before = 0
    total_steps_after = 0

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for ep_name in sorted(src.keys()):
            src_grp = src[ep_name]
            obs = src_grp["obs"][:]
            actions = src_grp["actions"][:]
            quat = src_grp["object_quat_w"][:]
            total_steps_before += len(obs)

            world_up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(quat), 1))
            upright = quat_apply_wxyz(quat.astype(np.float32), world_up)[:, 2]

            phase_b = obs[:, -1] < 0.5
            arm1 = obs[:, 1]
            grip = obs[:, 5]
            arm1_max = float(arm1.max())

            candidate = (
                phase_b
                & (arm1 > arm1_max - arm_margin)
                & (grip > min_grip_hold)
                & (upright > upright_thresh)
            )
            run = longest_run(np.where(candidate)[0])

            extra_slice = None
            anchor = None
            if run is not None:
                run_len = run[1] - run[0] + 1
                if arm1_max >= min_arm1_max and run_len >= min_run_len:
                    anchor = pick_anchor(obs, actions, arm1, grip, run)
                    half = segment_len // 2
                    seg_start = max(run[0], anchor - half)
                    seg_end = min(run[1] + 1, seg_start + segment_len)
                    seg_start = max(run[0], seg_end - segment_len)
                    extra_slice = slice(seg_start, seg_end)

            dst_grp = dst.create_group(ep_name)
            for key, ds in src_grp.items():
                data = ds[:]
                if extra_slice is not None:
                    tail = data[extra_slice]
                    if tail.ndim == 0:
                        boosted_data = data
                    else:
                        boosted_data = np.concatenate([data] + [tail] * repeat_count, axis=0)
                else:
                    boosted_data = data
                dst_grp.create_dataset(key, data=boosted_data, compression="gzip")

            new_len = len(dst_grp["obs"])
            total_steps_after += new_len
            if extra_slice is not None and anchor is not None:
                seg_obs = obs[extra_slice]
                seg_up = upright[extra_slice]
                boosted.append(
                    (
                        ep_name,
                        extra_slice.start,
                        extra_slice.stop,
                        anchor,
                        len(seg_obs),
                        float(seg_obs[:, 1].mean()),
                        float(seg_obs[:, 5].mean()),
                        float(seg_up.mean()),
                    )
                )

    print(f"source: {src_path}")
    print(f"output: {dst_path}")
    print(f"episodes: {len(boosted)} boosted / {len(h5py.File(dst_path, 'r').keys())} total")
    print(f"steps: {total_steps_before} -> {total_steps_after} (+{total_steps_after - total_steps_before})")
    for row in boosted:
        ep_name, s, e, anchor, seg_len_used, arm1_mean, grip_mean, up_mean = row
        print(
            f"{ep_name}: seg={s}:{e} anchor={anchor} len={seg_len_used} "
            f"arm1={arm1_mean:.3f} grip={grip_mean:.3f} upright={up_mean:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d.hdf5",
    )
    parser.add_argument(
        "--dst",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d_contactholdboost.hdf5",
    )
    parser.add_argument("--arm-margin", type=float, default=0.08)
    parser.add_argument("--min-grip-hold", type=float, default=0.20)
    parser.add_argument("--upright-thresh", type=float, default=0.97)
    parser.add_argument("--min-arm1-max", type=float, default=2.80)
    parser.add_argument("--min-run-len", type=int, default=40)
    parser.add_argument("--segment-len", type=int, default=24)
    parser.add_argument("--repeat-count", type=int, default=2)
    args = parser.parse_args()

    copy_with_tail_boost(
        src_path=Path(args.src),
        dst_path=Path(args.dst),
        arm_margin=args.arm_margin,
        min_grip_hold=args.min_grip_hold,
        upright_thresh=args.upright_thresh,
        min_arm1_max=args.min_arm1_max,
        min_run_len=args.min_run_len,
        segment_len=args.segment_len,
        repeat_count=args.repeat_count,
    )


if __name__ == "__main__":
    main()
