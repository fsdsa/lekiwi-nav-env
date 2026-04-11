#!/usr/bin/env python3
"""Combine v21 full/short 38D demos into a weighted mixed training set."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def _ep_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _copy_group(src_grp, dst_grp) -> None:
    _copy_attrs(src_grp, dst_grp)
    for key in src_grp.keys():
        dst_grp.create_dataset(key, data=src_grp[key][:], compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full_input",
        type=Path,
        default=Path("demos_skill3/smoothed_v2_release_retract_v21_full38.hdf5"),
    )
    parser.add_argument(
        "--short_input",
        type=Path,
        default=Path("demos_skill3/smoothed_v2_release_retract_v21_short38.hdf5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demos_skill3/smoothed_v2_release_retract_v21_mix38.hdf5"),
    )
    parser.add_argument("--short_repeats", type=int, default=2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.full_input, "r") as f_full, \
            h5py.File(args.short_input, "r") as f_short, \
            h5py.File(args.output, "w") as f_out:
        _copy_attrs(f_full, f_out)
        f_out.attrs["source_full_hdf5"] = str(args.full_input)
        f_out.attrs["source_short_hdf5"] = str(args.short_input)
        f_out.attrs["obs_dim"] = 38
        f_out.attrs["mixed_dataset"] = True
        f_out.attrs["num_full_episodes"] = len(f_full.keys())
        f_out.attrs["num_short_episodes"] = len(f_short.keys())
        f_out.attrs["short_repeats"] = args.short_repeats

        ep_idx = 0
        for ep in sorted(f_full.keys(), key=_ep_sort_key):
            out_grp = f_out.create_group(f"episode_{ep_idx}")
            _copy_group(f_full[ep], out_grp)
            out_grp.attrs["source_split"] = "full"
            out_grp.attrs["source_episode_name"] = ep
            ep_idx += 1

        for rep in range(args.short_repeats):
            for ep in sorted(f_short.keys(), key=_ep_sort_key):
                out_grp = f_out.create_group(f"episode_{ep_idx}")
                _copy_group(f_short[ep], out_grp)
                out_grp.attrs["source_split"] = "short"
                out_grp.attrs["source_episode_name"] = ep
                out_grp.attrs["short_repeat_idx"] = rep
                ep_idx += 1

    print(f"full:   {args.full_input}")
    print(f"short:  {args.short_input}")
    print(f"output: {args.output}")
    print(f"short_repeats: {args.short_repeats}")
    print(f"episodes: {ep_idx}")


if __name__ == "__main__":
    main()
