#!/usr/bin/env python3
"""Create release+retract sub-phase demos from smoothed_v2.

Outputs:
1. A full dataset with one extra flag dim appended to obs:
   release_retract_flag = 0 before the short sub-phase, 1 inside it.
2. A short-only dataset cut from the release/retract start to episode end,
   with release_retract_flag = 1 for all steps.
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


def _find_release_start(
    obs: np.ndarray,
    actions: np.ndarray,
    arm1_ready: float,
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
    ready = np.flatnonzero((arm1 >= arm1_ready) & (np.arange(len(arm1)) <= open_start) & (np.arange(len(arm1)) >= lo))
    if len(ready) > 0:
        start = int(ready[0])
    else:
        all_ready = np.flatnonzero((arm1 >= arm1_ready) & (np.arange(len(arm1)) <= open_start))
        if len(all_ready) > 0:
            start = int(all_ready[0])
        else:
            start = max(0, open_start - min(lookback, 40))
    return start, open_start


def _append_flag(obs: np.ndarray, start_idx: int) -> np.ndarray:
    flag = np.zeros((len(obs), 1), dtype=np.float32)
    flag[start_idx:, 0] = 1.0
    return np.concatenate([obs, flag], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("demos_skill3/smoothed_v2.hdf5"))
    parser.add_argument("--full_output", type=Path, default=Path("demos_skill3/smoothed_v2_release_retract_flag37.hdf5"))
    parser.add_argument("--short_output", type=Path, default=Path("demos_skill3/smoothed_v2_release_retract_short_37d.hdf5"))
    parser.add_argument("--arm1_ready", type=float, default=2.2)
    parser.add_argument("--grip_open_thresh", type=float, default=0.55)
    parser.add_argument("--lookback", type=int, default=80)
    args = parser.parse_args()

    args.full_output.parent.mkdir(parents=True, exist_ok=True)
    args.short_output.parent.mkdir(parents=True, exist_ok=True)

    release_starts = []
    open_starts = []
    short_lengths = []

    with h5py.File(args.input, "r") as f_in, \
            h5py.File(args.full_output, "w") as f_full, \
            h5py.File(args.short_output, "w") as f_short:
        _copy_attrs(f_in, f_full)
        _copy_attrs(f_in, f_short)

        for dst in (f_full, f_short):
            dst.attrs["source_hdf5"] = str(args.input)
            dst.attrs["release_retract_flag_added"] = True
            dst.attrs["obs_dim"] = 37
            dst.attrs["release_retract_arm1_ready"] = args.arm1_ready
            dst.attrs["release_retract_grip_open_thresh"] = args.grip_open_thresh
            dst.attrs["release_retract_lookback"] = args.lookback

        for ep in sorted(f_in.keys(), key=_ep_sort_key):
            src = f_in[ep]
            arrays = {k: src[k][:] for k in src.keys()}
            obs = arrays["obs"]
            actions = arrays["actions"]

            start_idx, open_start = _find_release_start(obs, actions, args.arm1_ready, args.grip_open_thresh, args.lookback)
            release_starts.append(start_idx)
            open_starts.append(open_start)
            short_lengths.append(len(obs) - start_idx)

            # full dataset: append extra flag dim
            full_grp = f_full.create_group(ep)
            _copy_attrs(src, full_grp)
            full_grp.attrs["release_retract_start_idx"] = int(start_idx)
            full_grp.attrs["release_open_start_idx"] = int(open_start)
            for key, value in arrays.items():
                if key == "obs":
                    full_grp.create_dataset(key, data=_append_flag(value, start_idx), compression="gzip")
                else:
                    full_grp.create_dataset(key, data=value, compression="gzip")

            # short dataset: cut from start_idx and keep flag=1
            short_grp = f_short.create_group(ep)
            _copy_attrs(src, short_grp)
            short_grp.attrs["release_retract_start_idx"] = int(start_idx)
            short_grp.attrs["release_open_start_idx"] = int(open_start)
            short_obs = obs[start_idx:].copy()
            short_flag = np.ones((len(short_obs), 1), dtype=np.float32)
            short_obs = np.concatenate([short_obs, short_flag], axis=-1)
            short_grp.create_dataset("obs", data=short_obs, compression="gzip")
            for key, value in arrays.items():
                if key == "obs":
                    continue
                short_grp.create_dataset(key, data=value[start_idx:], compression="gzip")

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
        "short length mean/min/max: "
        f"{np.mean(short_lengths):.1f} / {np.min(short_lengths)} / {np.max(short_lengths)}"
    )


if __name__ == "__main__":
    main()
