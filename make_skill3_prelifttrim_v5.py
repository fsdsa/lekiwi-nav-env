#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def find_prelift_grip_max(obs: np.ndarray, arm_drop_thresh: float) -> tuple[int, int, int]:
    phase_b = obs[:, -1] < 0.5
    phase_b_ids = np.where(phase_b)[0]
    if len(phase_b_ids) == 0:
        return len(obs) - 1, -1, -1

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
    return t_grip_peak, t_arm_peak, prelift_end


def trim_demo(
    src_path: Path,
    dst_path: Path,
    extra_steps: int,
    arm_drop_thresh: float,
    min_len: int,
) -> None:
    total_before = 0
    total_after = 0
    rows: list[tuple[str, int, int, int, int, float, float, float]] = []

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for ep_name in sorted(src.keys()):
            src_grp = src[ep_name]
            obs = src_grp["obs"][:]
            total_before += len(obs)

            t_grip_peak, t_arm_peak, prelift_end = find_prelift_grip_max(obs, arm_drop_thresh)
            cut_end = min(len(obs), max(min_len, t_grip_peak + extra_steps))
            cut_end = max(cut_end, t_grip_peak + 1)

            dst_grp = dst.create_group(ep_name)
            for key, ds in src_grp.items():
                data = ds[:]
                trimmed = data[:cut_end]
                dst_grp.create_dataset(key, data=trimmed, compression="gzip")

            total_after += cut_end
            phase_b = obs[:, -1] < 0.5
            arm1 = obs[:, 1]
            grip = obs[:, 5]
            rows.append(
                (
                    ep_name,
                    len(obs),
                    cut_end,
                    t_arm_peak,
                    t_grip_peak,
                    float(arm1[t_arm_peak]) if t_arm_peak >= 0 else float("nan"),
                    float(grip[t_grip_peak]),
                    float(arm1[t_grip_peak]),
                )
            )

    print(f"source: {src_path}")
    print(f"output: {dst_path}")
    print(f"steps: {total_before} -> {total_after} ({total_after - total_before:+d})")
    for ep_name, old_len, new_len, t_arm_peak, t_grip_peak, arm_peak, grip_peak, arm_at_grip in rows:
        print(
            f"{ep_name}: old={old_len} new={new_len} "
            f"arm_peak_t={t_arm_peak} arm1={arm_peak:.3f} "
            f"grip_peak_t={t_grip_peak} grip={grip_peak:.3f} arm1@grip={arm_at_grip:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d.hdf5",
    )
    parser.add_argument(
        "--dst",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d_prelifttrim100.hdf5",
    )
    parser.add_argument("--extra-steps", type=int, default=100)
    parser.add_argument("--arm-drop-thresh", type=float, default=0.15)
    parser.add_argument("--min-len", type=int, default=200)
    args = parser.parse_args()

    trim_demo(
        src_path=Path(args.src),
        dst_path=Path(args.dst),
        extra_steps=args.extra_steps,
        arm_drop_thresh=args.arm_drop_thresh,
        min_len=args.min_len,
    )


if __name__ == "__main__":
    main()
