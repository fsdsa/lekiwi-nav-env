#!/usr/bin/env python3
"""Append retract/rest tails from the original 20-episode demo onto the 40-episode release-open demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _ep_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _best_orig_match(orig_inits: np.ndarray, init_pose6: np.ndarray) -> int:
    d = np.linalg.norm(orig_inits - init_pose6[None, :], axis=1)
    return int(np.argmin(d))


def _best_tail_start(orig_obs: np.ndarray, src_last_obs6: np.ndarray, min_append: int) -> int:
    arm1 = orig_obs[:, 1]
    peak = int(np.argmax(arm1))
    hi = max(peak + 1, len(orig_obs) - min_append)
    cand = np.arange(peak, hi, dtype=np.int64)
    if len(cand) == 0:
        cand = np.arange(peak, len(orig_obs) - 1, dtype=np.int64)
    if len(cand) == 0:
        return len(orig_obs) - 1

    orig_obs6 = orig_obs[cand, :6]
    diff = orig_obs6 - src_last_obs6[None, :]
    # Arm posture dominates; grip is a secondary match.
    score = np.linalg.norm(diff[:, :5], axis=1) + 0.5 * np.abs(diff[:, 5])
    return int(cand[int(np.argmin(score))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("demos_skill3/combined_skill3_grip_s07_v5_36d_final_phaseAcarrysplice_releaseopen.hdf5"),
    )
    parser.add_argument(
        "--orig",
        type=Path,
        default=Path("demos_skill3/combined_skill3_grip_s07_v5_36d.hdf5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demos_skill3/combined_skill3_grip_s07_v5_36d_final_phaseAcarrysplice_releaseopen_retractrest.hdf5"),
    )
    parser.add_argument("--min_append", type=int, default=80)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.orig, "r") as f_orig, h5py.File(args.input, "r") as f_in, h5py.File(args.output, "w") as f_out:
        _copy_attrs(f_in, f_out)
        f_out.attrs["source_hdf5"] = str(args.input)
        f_out.attrs["orig_tail_hdf5"] = str(args.orig)
        f_out.attrs["retract_rest_appended"] = True
        f_out.attrs["retract_rest_min_append"] = args.min_append

        orig_eps = sorted(f_orig.keys(), key=_ep_sort_key)
        orig_inits = np.stack([f_orig[ep]["obs"][0, 29:35] for ep in orig_eps]).astype(np.float32)

        added_lengths = []
        start_indices = []
        map_indices = []

        for ep in sorted(f_in.keys(), key=_ep_sort_key):
            src_grp = f_in[ep]
            dst_grp = f_out.create_group(ep)
            _copy_attrs(src_grp, dst_grp)

            src_arrays = {k: src_grp[k][:] for k in src_grp.keys()}
            src_obs = src_arrays["obs"]
            src_last_obs6 = src_obs[-1, :6].astype(np.float32)
            src_init = src_obs[0, 29:35].astype(np.float32)

            orig_idx = _best_orig_match(orig_inits, src_init)
            orig_ep = orig_eps[orig_idx]
            orig_arrays = {k: f_orig[orig_ep][k][:] for k in f_orig[orig_ep].keys()}
            orig_obs = orig_arrays["obs"]

            tail_start = _best_tail_start(orig_obs, src_last_obs6, args.min_append)

            for key, src_value in src_arrays.items():
                tail_value = orig_arrays[key][tail_start + 1 :]
                merged = np.concatenate([src_value, tail_value], axis=0)
                if key == "obs":
                    merged = merged.copy()
                    merged[:, -1] = 0.0
                dst_grp.create_dataset(key, data=merged, compression="gzip")

            dst_grp.attrs["orig_tail_episode"] = orig_ep
            dst_grp.attrs["orig_tail_start_idx"] = int(tail_start + 1)
            dst_grp.attrs["orig_tail_added_len"] = int(len(orig_obs) - (tail_start + 1))
            map_indices.append(orig_idx)
            start_indices.append(int(tail_start + 1))
            added_lengths.append(int(len(orig_obs) - (tail_start + 1)))

    added_np = np.asarray(added_lengths)
    starts_np = np.asarray(start_indices)
    maps_np = np.asarray(map_indices)
    print(f"input:  {args.input}")
    print(f"orig:   {args.orig}")
    print(f"output: {args.output}")
    print(f"episodes: {len(added_lengths)}")
    print(
        "mapped orig episode ids unique/count: "
        f"{len(np.unique(maps_np))} / {len(maps_np)}"
    )
    print(
        "orig tail start mean/min/max: "
        f"{starts_np.mean():.1f} / {starts_np.min()} / {starts_np.max()}"
    )
    print(
        "appended len mean/min/max: "
        f"{added_np.mean():.1f} / {added_np.min()} / {added_np.max()}"
    )


if __name__ == "__main__":
    main()
