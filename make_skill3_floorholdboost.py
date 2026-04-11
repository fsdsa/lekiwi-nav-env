#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by quaternion(s). quat is (N,4) in wxyz."""
    w = quat[:, 0:1]
    qvec = quat[:, 1:4]
    vec = np.broadcast_to(vec.reshape(1, 3), qvec.shape)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (w * uv + uuv)


def upright_from_quat(quat: np.ndarray) -> np.ndarray:
    """World-z of local +Z axis."""
    return 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)


def longest_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask.tolist()):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def load_bbox_center_offset_z(object_usd: str, scale: float) -> float:
    catalog_path = Path("object_catalog_all.json")
    if catalog_path.exists():
        with catalog_path.open() as f:
            catalog = json.load(f)
        for item in catalog:
            if item.get("usd") == object_usd:
                return float(item["bbox"][2]) * float(scale) * 0.5
    # 5_HTP @ scale 0.7 fallback used in training logs.
    return 0.03108454868197441


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Create floor-hold bootstrap clips for Skill3 BC.")
    parser.add_argument(
        "--base_demo",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d.hdf5",
        help="Base 36D demo to copy and append boosted clips to.",
    )
    parser.add_argument(
        "--donor_demo",
        default="demos/combined_skill3_20260323_221300_36d.hdf5",
        help="Donor 36D demo with reliable object/dest world metadata.",
    )
    parser.add_argument(
        "--output_demo",
        default="demos_skill3/combined_skill3_grip_s07_v5_36d_floorholdboost.hdf5",
        help="Output HDF5 path.",
    )
    parser.add_argument("--max_clips", type=int, default=10)
    parser.add_argument("--repeat_stable", type=int, default=3)
    parser.add_argument("--pre_context", type=int, default=8)
    parser.add_argument("--post_context", type=int, default=8)
    parser.add_argument("--min_run", type=int, default=8)
    parser.add_argument("--objz_min", type=float, default=0.031)
    parser.add_argument("--objz_max", type=float, default=0.036)
    parser.add_argument("--src_dst_min", type=float, default=0.10)
    parser.add_argument("--src_dst_max", type=float, default=0.14)
    parser.add_argument("--upright_min", type=float, default=0.95)
    parser.add_argument("--grip_pos_min", type=float, default=0.25)
    parser.add_argument("--grip_act_max", type=float, default=-0.20)
    args = parser.parse_args()

    base_path = Path(args.base_demo)
    donor_path = Path(args.donor_demo)
    output_path = Path(args.output_demo)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(donor_path, "r") as donor:
        object_usd = str(donor.attrs.get("object_usd", ""))
        scale = float(donor.attrs.get("object_scale_phys", 1.0))
        bbox_offset_z = load_bbox_center_offset_z(object_usd, scale)
        bbox_offset = np.array([0.0, 0.0, bbox_offset_z], dtype=np.float32)

        clip_specs: list[dict] = []
        for ep_name in sorted(k for k in donor.keys() if k.startswith("episode_")):
            ep = donor[ep_name]
            obs = ep["obs"][:]
            actions = ep["actions"][:]
            root_pos = ep["object_pos_w"][:]
            obj_quat = ep["object_quat_w"][:]
            dest_pos = ep["dest_pos_w"][:]

            corrected_objz = root_pos[:, 2] + quat_apply(obj_quat, bbox_offset)[:, 2] - dest_pos[:, 2]
            src_dst = np.linalg.norm(root_pos[:, :2] - dest_pos[:, :2], axis=1)
            upright = upright_from_quat(obj_quat)
            phase_b = obs[:, -1] < 0.5
            grip_pos = obs[:, 5]
            grip_act = actions[:, 5]

            stable_mask = (
                phase_b
                & (corrected_objz >= args.objz_min)
                & (corrected_objz <= args.objz_max)
                & (src_dst >= args.src_dst_min)
                & (src_dst <= args.src_dst_max)
                & (upright >= args.upright_min)
                & (grip_pos >= args.grip_pos_min)
                & (grip_act <= args.grip_act_max)
            )

            for run_start, run_end in longest_true_runs(stable_mask):
                run_len = run_end - run_start
                if run_len < args.min_run:
                    continue
                clip_specs.append(
                    {
                        "ep_name": ep_name,
                        "run_start": run_start,
                        "run_end": run_end,
                        "run_len": run_len,
                        "objz_mean": float(corrected_objz[run_start:run_end].mean()),
                        "src_dst_mean": float(src_dst[run_start:run_end].mean()),
                        "upright_mean": float(upright[run_start:run_end].mean()),
                    }
                )

        clip_specs.sort(key=lambda x: x["run_len"], reverse=True)
        chosen = clip_specs[: args.max_clips]

        with h5py.File(base_path, "r") as base, h5py.File(output_path, "w") as out:
            copy_attrs(base.attrs, out.attrs)

            next_idx = 0
            for ep_name in sorted(k for k in base.keys() if k.startswith("episode_")):
                src = base[ep_name]
                dst = out.create_group(f"episode_{next_idx}")
                copy_attrs(src.attrs, dst.attrs)
                for key, ds in src.items():
                    dst.create_dataset(key, data=ds[:])
                next_idx += 1

            for spec in chosen:
                ep = donor[spec["ep_name"]]
                total_len = ep["obs"].shape[0]
                prefix_start = max(0, spec["run_start"] - args.pre_context)
                suffix_end = min(total_len, spec["run_end"] + args.post_context)

                prefix = np.arange(prefix_start, spec["run_start"], dtype=np.int64)
                stable = np.arange(spec["run_start"], spec["run_end"], dtype=np.int64)
                suffix = np.arange(spec["run_end"], suffix_end, dtype=np.int64)
                idx = np.concatenate([prefix] + [stable] * args.repeat_stable + [suffix])

                dst = out.create_group(f"episode_{next_idx}")
                copy_attrs(ep.attrs, dst.attrs)
                dst.attrs["bootstrap_source_episode"] = spec["ep_name"]
                dst.attrs["bootstrap_run_start"] = spec["run_start"]
                dst.attrs["bootstrap_run_end"] = spec["run_end"]
                dst.attrs["bootstrap_repeat_stable"] = args.repeat_stable
                dst.attrs["bootstrap_corrected_objz_mean"] = spec["objz_mean"]
                dst.attrs["bootstrap_src_dst_mean"] = spec["src_dst_mean"]
                dst.attrs["bootstrap_upright_mean"] = spec["upright_mean"]

                for key, ds in ep.items():
                    dst.create_dataset(key, data=ds[:][idx])
                next_idx += 1

    print(f"[floorholdboost] wrote: {output_path}")
    print(f"[floorholdboost] selected clips: {len(chosen)}")
    for spec in chosen:
        print(
            f"  {spec['ep_name']} run={spec['run_start']}:{spec['run_end']} len={spec['run_len']} "
            f"objZ={spec['objz_mean']:.4f} src_dst={spec['src_dst_mean']:.4f} upright={spec['upright_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
