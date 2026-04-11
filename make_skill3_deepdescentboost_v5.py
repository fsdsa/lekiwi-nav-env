#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def find_window(obs: np.ndarray, arm_drop_thresh: float, pre_steps: int, post_steps: int) -> tuple[int, int, int, int]:
    phase_b = obs[:, -1] < 0.5
    phase_b_ids = np.where(phase_b)[0]
    if len(phase_b_ids) == 0:
        return 0, len(obs), -1, -1

    arm1 = obs[:, 1]
    grip = obs[:, 5]

    t_arm_peak = int(phase_b_ids[np.argmax(arm1[phase_b_ids])])
    arm_peak = float(arm1[t_arm_peak])

    prelift_end = int(phase_b_ids[-1])
    for t in phase_b_ids[phase_b_ids >= t_arm_peak]:
        if arm1[t] < arm_peak - arm_drop_thresh:
            prelift_end = int(t)
            break

    window = np.arange(t_arm_peak, max(t_arm_peak + 1, prelift_end))
    if len(window) == 0:
        window = np.array([t_arm_peak], dtype=np.int64)
    t_grip_peak = int(window[np.argmax(grip[window])])

    start = max(int(phase_b_ids[0]), t_arm_peak - pre_steps)
    end = min(len(obs), t_grip_peak + post_steps + 1)
    return start, end, t_arm_peak, t_grip_peak


def build_boosted_dataset(
    src_path: Path,
    dst_path: Path,
    min_arm_peak: float,
    arm_drop_thresh: float,
    pre_steps: int,
    post_steps: int,
) -> None:
    copied = 0
    added = 0
    total_before = 0
    total_after = 0
    clip_rows: list[tuple[str, int, int, float, float, float]] = []

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        episode_names = sorted(src.keys())

        for ep_name in episode_names:
            src_grp = src[ep_name]
            dst_grp = dst.create_group(ep_name)
            for key, ds in src_grp.items():
                dst_grp.create_dataset(key, data=ds[:], compression="gzip")
            copied += 1
            total_before += len(src_grp["obs"])
            total_after += len(src_grp["obs"])

        next_idx = copied
        for ep_name in episode_names:
            src_grp = src[ep_name]
            obs = src_grp["obs"][:]
            arm1 = obs[:, 1]
            phase_b = obs[:, -1] < 0.5
            phase_b_ids = np.where(phase_b)[0]
            if len(phase_b_ids) == 0:
                continue

            arm_peak = float(arm1[phase_b_ids].max())
            if arm_peak < min_arm_peak:
                continue

            start, end, t_arm_peak, t_grip_peak = find_window(obs, arm_drop_thresh, pre_steps, post_steps)
            if end - start < 40:
                continue

            new_name = f"episode_{next_idx}"
            next_idx += 1
            dst_grp = dst.create_group(new_name)
            for key, ds in src_grp.items():
                dst_grp.create_dataset(key, data=ds[start:end], compression="gzip")
            added += 1
            total_after += end - start
            clip_rows.append(
                (
                    ep_name,
                    start,
                    end,
                    arm_peak,
                    float(obs[t_grip_peak, 5]),
                    float(obs[t_grip_peak, 1]),
                )
            )

    print(f"source: {src_path}")
    print(f"output: {dst_path}")
    print(f"episodes: {copied} base + {added} clips = {copied + added}")
    print(f"steps: {total_before} -> {total_after} (+{total_after - total_before})")
    for ep_name, start, end, arm_peak, grip_peak, arm_at_grip in clip_rows:
        print(
            f"{ep_name}: clip={start}:{end} len={end-start} "
            f"arm_peak={arm_peak:.3f} grip_peak={grip_peak:.3f} arm1@grip={arm_at_grip:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d_trimmed.hdf5",
    )
    parser.add_argument(
        "--dst",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d_trimmed_deepdescentboost.hdf5",
    )
    parser.add_argument("--min-arm-peak", type=float, default=2.8)
    parser.add_argument("--arm-drop-thresh", type=float, default=0.15)
    parser.add_argument("--pre-steps", type=int, default=20)
    parser.add_argument("--post-steps", type=int, default=20)
    args = parser.parse_args()

    build_boosted_dataset(
        src_path=Path(args.src),
        dst_path=Path(args.dst),
        min_arm_peak=args.min_arm_peak,
        arm_drop_thresh=args.arm_drop_thresh,
        pre_steps=args.pre_steps,
        post_steps=args.post_steps,
    )


if __name__ == "__main__":
    main()
