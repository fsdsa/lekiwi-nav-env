#!/usr/bin/env python3
"""Trim a skill3 HDF5 so each episode starts from its first Phase-B frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def sorted_episode_keys(handle: h5py.File) -> list[str]:
    return sorted(handle.keys(), key=lambda x: int(x.split("_")[-1]))


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d_final_phaseAcarrysplice.hdf5",
    )
    parser.add_argument(
        "--output",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d_final_phaseAcarrysplice_phaseBonly.hdf5",
    )
    args = parser.parse_args()

    src_path = Path(args.input)
    dst_path = Path(args.output)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    starts: list[int] = []
    before_lengths: list[int] = []
    after_lengths: list[int] = []

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_attrs(src, dst)
        dst.attrs["source_file"] = str(src_path)
        dst.attrs["phase_a_removed"] = 1

        for ep_key in sorted_episode_keys(src):
            src_ep = src[ep_key]
            obs = src_ep["obs"][:]
            phase = obs[:, -1]
            phase_b_idx = np.where(phase < 0.5)[0]
            start = int(phase_b_idx[0]) if len(phase_b_idx) else 0

            dst_ep = dst.create_group(ep_key)
            copy_attrs(src_ep, dst_ep)
            dst_ep.attrs["phase_b_start_index"] = start

            for ds_key in src_ep.keys():
                data = src_ep[ds_key][:]
                trimmed = data[start:]
                if ds_key == "obs":
                    trimmed = trimmed.copy()
                    trimmed[:, -1] = 0.0
                dst_ep.create_dataset(ds_key, data=trimmed, compression="gzip")

            starts.append(start)
            before_lengths.append(len(obs))
            after_lengths.append(len(obs) - start)

    starts_np = np.asarray(starts)
    before_np = np.asarray(before_lengths)
    after_np = np.asarray(after_lengths)
    print(f"input:  {src_path}")
    print(f"output: {dst_path}")
    print(f"episodes: {len(starts)}")
    print(
        "phase_b_start mean/min/max: "
        f"{starts_np.mean():.1f} / {starts_np.min()} / {starts_np.max()}"
    )
    print(
        "length before mean/min/max: "
        f"{before_np.mean():.1f} / {before_np.min()} / {before_np.max()}"
    )
    print(
        "length after  mean/min/max: "
        f"{after_np.mean():.1f} / {after_np.min()} / {after_np.max()}"
    )


if __name__ == "__main__":
    main()
