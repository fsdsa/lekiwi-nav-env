import argparse
from pathlib import Path

import h5py
import numpy as np


def _copy_dataset(src_group, dst_group, name, step_slice):
    data = src_group[name]
    if data.ndim >= 1 and data.shape[0] >= step_slice.stop:
        dst_group.create_dataset(name, data=data[step_slice])
    else:
        dst_group.create_dataset(name, data=data[()])


def main():
    parser = argparse.ArgumentParser(description="Create Phase-A-oversampled motion24 Skill-3 HDF5")
    parser.add_argument("--src", required=True, help="Source motion24 HDF5")
    parser.add_argument("--dst", required=True, help="Destination HDF5")
    parser.add_argument(
        "--phase_a_duplicates",
        type=int,
        default=4,
        help="How many extra Phase-A-only copies to add per episode (4 -> total 5x Phase A)",
    )
    parser.add_argument(
        "--phase_a_flag_idx",
        type=int,
        default=21,
        help="Obs index for phase_a_flag in motion24 dataset",
    )
    args = parser.parse_args()

    src_path = Path(args.src).expanduser().resolve()
    dst_path = Path(args.dst).expanduser().resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["source_hdf5"] = str(src_path)
        dst.attrs["phase_a_short_duplicates"] = int(args.phase_a_duplicates)
        dst.attrs["phase_a_short_source"] = "motion24_phase_a_slice"

        ep_keys = sorted(
            [k for k in src.keys() if k.startswith("episode_")],
            key=lambda x: int(x.split("_")[1]),
        )

        out_idx = 0
        total_steps = 0
        total_phase_a_steps = 0
        total_phase_b_steps = 0
        added_short = 0

        for ep_key in ep_keys:
            src_ep = src[ep_key]
            obs = src_ep["obs"][:]
            phase_a_mask = obs[:, args.phase_a_flag_idx] > 0.5
            phase_a_idx = np.where(phase_a_mask)[0]
            if len(phase_a_idx) == 0:
                continue

            # 1) original full episode
            dst_ep = dst.create_group(f"episode_{out_idx}")
            out_idx += 1
            for k, v in src_ep.attrs.items():
                dst_ep.attrs[k] = v
            dst_ep.attrs["source_episode"] = ep_key
            dst_ep.attrs["phase_a_short"] = False
            for name in src_ep.keys():
                _copy_dataset(src_ep, dst_ep, name, slice(0, len(obs)))
            total_steps += len(obs)
            total_phase_a_steps += int(phase_a_mask.sum())
            total_phase_b_steps += int((~phase_a_mask).sum())

            # 2) duplicated short Phase-A-only episodes
            start = int(phase_a_idx[0])
            stop = int(phase_a_idx[-1]) + 1
            step_slice = slice(start, stop)
            short_len = stop - start
            for dup in range(args.phase_a_duplicates):
                short_ep = dst.create_group(f"episode_{out_idx}")
                out_idx += 1
                for k, v in src_ep.attrs.items():
                    short_ep.attrs[k] = v
                short_ep.attrs["source_episode"] = ep_key
                short_ep.attrs["phase_a_short"] = True
                short_ep.attrs["phase_a_short_dup_idx"] = dup
                short_ep.attrs["num_steps"] = short_len
                short_ep.attrs["num_active_steps"] = short_len
                for name in src_ep.keys():
                    _copy_dataset(src_ep, short_ep, name, step_slice)
                total_steps += short_len
                total_phase_a_steps += short_len
                added_short += 1

        print(f"Source: {src_path}")
        print(f"Output: {dst_path}")
        print(f"Episodes written: {out_idx}")
        print(f"Added short Phase-A episodes: {added_short}")
        print(f"Total steps: {total_steps}")
        print(f"Phase A steps: {total_phase_a_steps} ({total_phase_a_steps / max(total_steps, 1):.3f})")
        print(f"Phase B steps: {total_phase_b_steps} ({total_phase_b_steps / max(total_steps, 1):.3f})")


if __name__ == "__main__":
    main()
