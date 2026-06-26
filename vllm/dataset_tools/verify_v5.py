"""
Post-build integrity check for a LeRobot v3.0 dataset.
Runs a battery of checks and exits non-zero on any failure.

Checks:
  1. info.json / episodes/ meta / data parquet row counts consistent
  2. No NaN / inf in state/action
  3. Every episode's global index range contiguous; no dup / gap
  4. frame_index starts at 0 within each episode, contiguous
  5. Task strings match tasks.parquet entries
  6. stats.json q01/q99/mean/std match actual data within tolerance
  7. Video files exist; random-sample frames decode to expected shape
  8. Random-sample LeRobotDataset __getitem__ returns consistent frames
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def fail(msg: str) -> None:
    print(f"  ✗ FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="dataset root path")
    ap.add_argument("--tol-stats", type=float, default=1e-3,
                    help="tolerance for stats q01/q99 match (default 1e-3)")
    ap.add_argument("--sample-frames", type=int, default=20,
                    help="random frames to decode-check (default 20)")
    args = ap.parse_args()

    root = Path(args.path)
    errors = 0

    print(f"═══ VERIFY: {root} ═══")

    # 1. info.json
    info = json.load(open(root / "meta" / "info.json"))
    print("\n[1] info.json")
    print(f"  total_episodes: {info['total_episodes']}")
    print(f"  total_frames:   {info['total_frames']}")
    print(f"  fps:            {info['fps']}")
    print(f"  codebase:       {info['codebase_version']}")

    # 2. data parquet
    print("\n[2] data parquet consistency")
    df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    if len(df) != info["total_frames"]:
        fail(f"data parquet has {len(df)} rows, info says {info['total_frames']}")
        errors += 1
    else:
        ok(f"data parquet rows == info.total_frames ({len(df)})")

    n_ep = df["episode_index"].nunique()
    if n_ep != info["total_episodes"]:
        fail(f"unique episode_index = {n_ep}, info says {info['total_episodes']}")
        errors += 1
    else:
        ok(f"unique episodes == info.total_episodes ({n_ep})")

    if df["index"].max() != info["total_frames"] - 1:
        fail(f"max global index = {df['index'].max()}, expected {info['total_frames']-1}")
        errors += 1
    elif df["index"].min() != 0:
        fail(f"min global index = {df['index'].min()}, expected 0")
        errors += 1
    elif len(df["index"].unique()) != len(df):
        fail("duplicate global index detected")
        errors += 1
    else:
        ok("global index is contiguous 0..N-1 with no duplicates")

    # 3. per-episode integrity
    print("\n[3] per-episode integrity")
    bad_ep = 0
    for ep_idx, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        fi = g["frame_index"].values
        if fi[0] != 0 or (np.diff(fi) != 1).any():
            fail(f"episode {ep_idx}: frame_index not contiguous from 0")
            bad_ep += 1
            if bad_ep > 5:
                fail("... more episodes with bad frame_index (truncated)")
                break
    if bad_ep == 0:
        ok(f"all {n_ep} episodes have contiguous frame_index 0..len-1")
    else:
        errors += 1

    # 4. NaN / inf in state / action
    print("\n[4] state / action NaN / inf check")
    for field in ("observation.state", "action"):
        arr = np.stack([np.asarray(r, dtype=np.float64) for r in df[field]])
        if np.isnan(arr).any() or np.isinf(arr).any():
            fail(f"{field}: contains NaN or inf")
            errors += 1
        else:
            ok(f"{field}: finite everywhere (shape={arr.shape})")

    # 5. tasks
    print("\n[5] tasks.parquet")
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        ok(f"tasks.parquet has {len(tasks_df)} labels")
        used = sorted(df["task_index"].unique().tolist())
        known = sorted(tasks_df.reset_index()["task_index"].tolist()
                       if "task_index" in tasks_df.reset_index().columns
                       else tasks_df["task_index"].tolist()
                       if "task_index" in tasks_df.columns
                       else list(range(len(tasks_df))))
        missing = [t for t in used if t not in known]
        if missing:
            fail(f"data has task_index {missing} but tasks.parquet does not declare them")
            errors += 1
        else:
            ok(f"all {len(used)} used task_indices declared in tasks.parquet")
    else:
        fail("tasks.parquet missing")
        errors += 1

    # 6. stats.json vs actual
    print("\n[6] stats.json vs actual (tol={})".format(args.tol_stats))
    stats = json.load(open(root / "meta" / "stats.json"))
    for field in ("observation.state", "action"):
        arr = np.stack([np.asarray(r, dtype=np.float64) for r in df[field]])
        s = stats[field]
        for stat_name, computer in [
            ("mean", lambda a: a.mean(0)),
            ("std",  lambda a: a.std(0)),
            ("min",  lambda a: a.min(0)),
            ("max",  lambda a: a.max(0)),
            ("q01",  lambda a: np.percentile(a, 1, axis=0)),
            ("q50",  lambda a: np.percentile(a, 50, axis=0)),
            ("q99",  lambda a: np.percentile(a, 99, axis=0)),
        ]:
            expected = computer(arr)
            got = np.asarray(s[stat_name])
            if expected.shape != got.shape:
                fail(f"{field}.{stat_name} shape {got.shape} ≠ expected {expected.shape}")
                errors += 1
                continue
            diff = np.abs(expected - got).max()
            if diff > args.tol_stats:
                fail(f"{field}.{stat_name} diff max={diff:.6f} > tol={args.tol_stats}")
                errors += 1
            else:
                ok(f"{field}.{stat_name} within tolerance (max diff {diff:.2e})")

    if stats["observation.state"]["count"][0] != info["total_frames"]:
        fail(f"stats.count={stats['observation.state']['count'][0]} ≠ info.total_frames={info['total_frames']}")
        errors += 1
    else:
        ok("stats.count == info.total_frames")

    # 7. video files exist
    print("\n[7] video files")
    ep_meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for key in ("observation.images.front", "observation.images.wrist"):
        file_indices = set(ep_meta[f"videos/{key}/file_index"].unique().tolist())
        video_dir = root / "videos" / key / "chunk-000"
        for fi in file_indices:
            f = video_dir / f"file-{int(fi):03d}.mp4"
            if not f.exists():
                fail(f"missing video: {f}")
                errors += 1
        if all((video_dir / f"file-{int(fi):03d}.mp4").exists() for fi in file_indices):
            ok(f"{key}: all {len(file_indices)} video files present")

    # 8. random sample via LeRobotDataset
    print(f"\n[8] LeRobotDataset sample decode ({args.sample_frames} frames)")
    ds = LeRobotDataset(root.name, root=str(root))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(ds), size=min(args.sample_frames, len(ds)), replace=False)
    decode_ok = 0
    for idx in sample_idx:
        f = ds[int(idx)]
        good = True
        for key in ("observation.images.front", "observation.images.wrist"):
            if key not in f:
                fail(f"idx {idx}: missing {key}")
                good = False
                break
            t = f[key]
            if t.shape[-2:] != (400, 640):
                fail(f"idx {idx}: {key} shape {tuple(t.shape)} bad")
                good = False
                break
        if good:
            decode_ok += 1
    if decode_ok == len(sample_idx):
        ok(f"all {decode_ok}/{len(sample_idx)} sampled frames decoded cleanly")
    else:
        fail(f"only {decode_ok}/{len(sample_idx)} sampled frames decoded cleanly")
        errors += 1

    # summary
    print("\n" + "═" * 70)
    if errors == 0:
        print(f"✓ ALL CHECKS PASS ({root})")
        sys.exit(0)
    else:
        print(f"✗ {errors} FAILURE(S) — do NOT train on this dataset until fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
