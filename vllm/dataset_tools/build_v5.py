"""
Build lekiwi_viva_v5: merged dataset from filtered v4 + new approach recordings.

Pipeline
--------
1. Audit each source dataset (schema, fps, features, episode counts).
2. Filter source 1 (v4): drop episodes by index (old approach_and_lift 0..99).
3. Append source 2 (local 50 approach_and_lift).
4. Append source 3 (server 50 approach_and_lift).
5. Re-use LeRobotDataset.create + add_frame + save_episode so lerobot itself
   handles chunking, video encoding, stats baking into per-episode meta, etc.
6. Post-build: load v5 and verify episode/frame counts, stats sanity.

Typical usage
-------------
    python build_v5.py \
        --v4       /home/jovyan/lerobot_data/lekiwi_viva_v4 \
        --local50  /path/to/local_collected_50 \
        --server50 /path/to/server_collected_50 \
        --out      /home/jovyan/lerobot_data/lekiwi_viva_v5 \
        --drop-v4-episodes 0-99 \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ───────────────────────────── expected schema ─────────────────────────────
# Derived from lekiwi_viva_v4 info.json. All sources MUST match.
EXPECTED_FPS = 25
EXPECTED_STATE_DIM = 9
EXPECTED_ACTION_DIM = 9
EXPECTED_IMG_HW = (400, 640)  # height, width
EXPECTED_IMG_KEYS = ("observation.images.front", "observation.images.wrist")
EXPECTED_ROBOT_TYPE = "lekiwi_client"

# task string → intended task_index (must match tasks.parquet of v4)
# Used only for sanity check / error message; actual task_index is assigned
# by the new dataset based on insertion order. To keep indices stable with
# v4 we force the same task order during the first-seen loop below.
V4_TASK_ORDER = [
    "look around to find the target object",          # 0
    "approach the target object",                      # 1
    "pick up the target object",                       # 2
    "return to the starting position",                 # 3
    "find the target and bring it back",               # 4
    "approach and lift the medicine bottle",           # 5
    "navigate forward",                                # 6
    "navigate backward",                               # 7
    "navigate turn left",                              # 8
    "navigate turn right",                             # 9
    "navigate strafe left",                            # 10
    "navigate strafe right",                           # 11
    "carry forward",                                   # 12
    "carry backward",                                  # 13
    "carry left",                                      # 14
    "carry right",                                     # 15
    "carry turn left",                                 # 16
    "carry turn right",                                # 17
]

# ───────────────────────────── helpers ─────────────────────────────────────


def parse_index_range(spec: str) -> set[int]:
    """'0-99,150,200-210' → {0..99, 150, 200..210}"""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def audit(path: str, label: str) -> dict:
    """Load a LeRobot dataset and return key facts. Raises on obvious mismatch."""
    print(f"\n─── audit: {label} ({path}) ───")
    if not Path(path).exists():
        raise FileNotFoundError(path)
    ds = LeRobotDataset(Path(path).name, root=path)
    info: dict = {
        "path": path,
        "label": label,
        "num_episodes": ds.num_episodes,
        "num_frames": len(ds),
        "fps": ds.fps,
        "features": list(ds.features.keys()),
    }
    print(f"  num_episodes: {info['num_episodes']}")
    print(f"  num_frames:   {info['num_frames']}")
    print(f"  fps:          {info['fps']}")

    if ds.fps != EXPECTED_FPS:
        raise ValueError(f"{label}: fps={ds.fps}, expected {EXPECTED_FPS}")

    # Schema checks
    f = ds[0]
    for key in EXPECTED_IMG_KEYS:
        if key not in f:
            raise ValueError(f"{label}: missing image key {key}")
        h, w = f[key].shape[-2], f[key].shape[-1]
        if (h, w) != EXPECTED_IMG_HW:
            raise ValueError(f"{label}: {key} shape={h}x{w}, expected {EXPECTED_IMG_HW}")
    if f["observation.state"].shape[0] != EXPECTED_STATE_DIM:
        raise ValueError(f"{label}: state dim {f['observation.state'].shape[0]} ≠ 9")
    if f["action"].shape[0] != EXPECTED_ACTION_DIM:
        raise ValueError(f"{label}: action dim {f['action'].shape[0]} ≠ 9")

    # Episode index range / task distribution
    import pandas as pd
    df = pd.read_parquet(Path(path) / "data" / "chunk-000" / "file-000.parquet")
    ep_range = (int(df["episode_index"].min()), int(df["episode_index"].max()))
    task_counts = df["task_index"].value_counts().sort_index().to_dict()
    info["episode_index_range"] = ep_range
    info["task_index_counts"] = {int(k): int(v) for k, v in task_counts.items()}

    # Task label → task_index map (read tasks.parquet)
    tasks_parquet = Path(path) / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        info["task_labels"] = tasks_df.reset_index().to_dict("records")
        print(f"  tasks: {len(tasks_df)} labels")

    print(f"  episode_index range: {ep_range}")
    print(f"  task_index counts:   {info['task_index_counts']}")

    del ds
    return info


def canonical_image_for_add_frame(t: torch.Tensor) -> np.ndarray:
    """
    __getitem__ returns CHW float32 in [0, 1] for image features.
    add_frame expects HWC uint8 [0, 255] (standard lerobot recording format).
    """
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    if t.dtype == torch.uint8:
        # already HWC uint8?
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            t = t.permute(1, 2, 0)
        return t.cpu().numpy()
    # float path
    arr = t.clamp(0, 1).mul(255).round().to(torch.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.permute(1, 2, 0)
    return arr.cpu().numpy()


def copy_episode_frames(
    src: LeRobotDataset,
    src_episode_index: int,
    dst: LeRobotDataset,
) -> tuple[int, str]:
    """Copy one episode from src to dst. Returns (n_frames, task_string)."""
    ep_info = src.meta.episodes[src_episode_index]
    from_idx = int(ep_info["dataset_from_index"])
    to_idx = int(ep_info["dataset_to_index"])
    n_frames = to_idx - from_idx

    task_str = None
    for i in range(from_idx, to_idx):
        f = src[i]
        frame = {
            "observation.state": f["observation.state"].cpu().numpy().astype(np.float32),
            "action": f["action"].cpu().numpy().astype(np.float32),
            "observation.images.front": canonical_image_for_add_frame(f["observation.images.front"]),
            "observation.images.wrist": canonical_image_for_add_frame(f["observation.images.wrist"]),
            "task": f["task"],
        }
        if task_str is None:
            task_str = f["task"]
        dst.add_frame(frame)

    dst.save_episode()
    return n_frames, task_str


# ───────────────────────────── main ────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--v4", required=True, help="lekiwi_viva_v4 path")
    ap.add_argument("--new-sources", nargs="+", required=True,
                    help="one or more new dataset paths to append after filtered v4")
    ap.add_argument("--out", required=True, help="output dataset path (e.g., lekiwi_viva_v5)")
    ap.add_argument("--drop-v4-episodes", default="0-99",
                    help="episode_index spec to drop from v4 (default: 0-99)")
    ap.add_argument("--dry-run", action="store_true",
                    help="audit + plan only, do not build")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing --out if it exists")
    ap.add_argument("--video-backend", default="pyav",
                    help="video backend for output dataset (pyav recommended)")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--skip-stats-recompute", action="store_true",
                    help="skip post-build exact-quantile stats recompute")
    args = ap.parse_args()

    t_start = time.time()

    # 1. Audit all sources
    print("═" * 70)
    print("PHASE 1 / AUDIT sources")
    print("═" * 70)
    info_v4 = audit(args.v4, "v4")
    info_new = [audit(p, f"new[{i}]") for i, p in enumerate(args.new_sources)]

    drop_v4_set = parse_index_range(args.drop_v4_episodes)
    keep_v4 = info_v4["num_episodes"] - len(drop_v4_set)
    total_new_ep = keep_v4 + sum(s["num_episodes"] for s in info_new)

    print("\n─── merge plan ───")
    print(f"  v4:       keep {keep_v4} / {info_v4['num_episodes']} "
          f"(drop {len(drop_v4_set)} episode indices)")
    for i, s in enumerate(info_new):
        print(f"  new[{i}]:  +{s['num_episodes']}  ({s['path']})")
    print(f"  total new dataset: {total_new_ep} episodes")

    if args.dry_run:
        print("\n[dry-run] stopping before build.")
        return

    # 2. Prepare output directory
    out_path = Path(args.out)
    if out_path.exists():
        if not args.force:
            print(f"ERROR: {out_path} exists. Use --force to overwrite.", file=sys.stderr)
            sys.exit(2)
        print(f"  removing existing {out_path} ...")
        shutil.rmtree(out_path)

    # 3. Build new empty dataset with v4's features
    print("\n═" * 70)
    print("PHASE 2 / CREATE new dataset")
    print("═" * 70)

    # Load v4 to get feature spec
    src_v4 = LeRobotDataset(Path(args.v4).name, root=args.v4)
    features = {
        k: v for k, v in src_v4.features.items()
        # lerobot auto-adds frame_index/episode_index/index/task_index/timestamp;
        # don't pass them to create
        if k not in ("frame_index", "episode_index", "index", "task_index", "timestamp")
    }

    dst = LeRobotDataset.create(
        repo_id=out_path.name,
        fps=EXPECTED_FPS,
        features=features,
        root=str(out_path),
        robot_type=EXPECTED_ROBOT_TYPE,
        use_videos=True,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
    )
    print(f"  created: {out_path}")

    # 4. Copy filtered v4 episodes first — preserves task_index alignment if
    #    the very first episodes include all 14 unique tasks. Navigate/carry
    #    span episode_index 100..977, new approach in v4 starts at 978.
    print("\n═" * 70)
    print("PHASE 3 / COPY filtered v4")
    print("═" * 70)
    t_v4 = time.time()
    v4_kept_indices = [i for i in range(info_v4["num_episodes"]) if i not in drop_v4_set]
    n_kept_frames = 0
    for ep_idx in tqdm(v4_kept_indices, desc="v4 episodes"):
        n, _ = copy_episode_frames(src_v4, ep_idx, dst)
        n_kept_frames += n
    print(f"  v4 → copied {len(v4_kept_indices)} ep, {n_kept_frames} frames "
          f"in {time.time()-t_v4:.1f}s")
    del src_v4

    # 5. Append new sources in order
    for i, source_path in enumerate(args.new_sources):
        src_label = f"new[{i}]"
        print("\n═" * 70)
        print(f"PHASE 4.{i+1} / APPEND {src_label} ({source_path})")
        print("═" * 70)
        t_s = time.time()
        src = LeRobotDataset(Path(source_path).name, root=source_path)
        n_src_frames = 0
        for ep_idx in tqdm(range(src.num_episodes), desc=f"{src_label} episodes"):
            n, _ = copy_episode_frames(src, ep_idx, dst)
            n_src_frames += n
        print(f"  {src_label} → copied {src.num_episodes} ep, {n_src_frames} frames "
              f"in {time.time()-t_s:.1f}s")
        del src

    # 6. Finalize
    print("\n═" * 70)
    print("PHASE 5 / FINALIZE")
    print("═" * 70)
    dst.finalize()
    print("  finalize() done.")

    # 7. Recompute EXACT quantile stats — lerobot's incremental stats uses
    #    approximate quantiles. For training we want exact stats.json matching
    #    actual data distribution (verified by verify_v5.py with tol 1e-5).
    if not args.skip_stats_recompute:
        print("\n═" * 70)
        print("PHASE 6 / RECOMPUTE EXACT STATS")
        print("═" * 70)
        recompute_stats_exact(out_path)

    # 8. Post-build sanity print
    print("\n═" * 70)
    print("PHASE 7 / SANITY CHECK")
    print("═" * 70)
    check = LeRobotDataset(out_path.name, root=str(out_path))
    print(f"  num_episodes: {check.num_episodes} (expected {total_new_ep})")
    print(f"  num_frames:   {len(check)}")
    assert check.num_episodes == total_new_ep, "episode count mismatch"

    import pandas as pd
    df_new = pd.read_parquet(out_path / "data" / "chunk-000" / "file-000.parquet")
    act = np.stack([np.array(r) for r in df_new["action"]])
    print(f"  action arm[1] mean={act[:,1].mean():.4f} std={act[:,1].std():.4f}")
    print(f"  action arm[1] q01={np.percentile(act[:,1],1):.4f}  q99={np.percentile(act[:,1],99):.4f}")

    stats_path = out_path / "meta" / "stats.json"
    s = json.load(open(stats_path))
    q01 = s["action"]["q01"][1]
    q99 = s["action"]["q99"][1]
    print(f"  stats.json action arm[1]: q01={q01:.4f}  q99={q99:.4f}")

    elapsed = time.time() - t_start
    print(f"\n✓ DONE. Total {elapsed/60:.1f} min")
    print(f"  Dataset: {out_path}")
    print(f"  Verify:  python verify_v5.py --path {out_path}")


def recompute_stats_exact(ds_root: Path) -> None:
    """Recompute exact percentile stats on final merged parquet data.
    lerobot's incremental save uses approximate quantiles; we need exact."""
    import pandas as pd
    stats_path = ds_root / "meta" / "stats.json"
    s = json.load(open(stats_path))
    df = pd.read_parquet(ds_root / "data" / "chunk-000" / "file-000.parquet")
    print(f"  Loaded {len(df)} frames for stats recompute")

    # Vector fields (state, action): shape (N, 9)
    for field in ("observation.state", "action"):
        arr = np.stack([np.asarray(r, dtype=np.float64) for r in df[field]])
        s[field]["min"]  = arr.min(axis=0).tolist()
        s[field]["max"]  = arr.max(axis=0).tolist()
        s[field]["mean"] = arr.mean(axis=0).tolist()
        s[field]["std"]  = arr.std(axis=0).tolist()
        s[field]["count"] = [int(arr.shape[0])]
        for qlabel, qval in [("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)]:
            s[field][qlabel] = np.percentile(arr, qval, axis=0).tolist()

    # Scalar metadata fields
    for field in ("episode_index", "frame_index", "index", "task_index", "timestamp"):
        if field not in s:
            continue
        arr = df[field].to_numpy().astype(np.float64)
        s[field]["min"]  = [float(arr.min())]
        s[field]["max"]  = [float(arr.max())]
        s[field]["mean"] = [float(arr.mean())]
        s[field]["std"]  = [float(arr.std())]
        s[field]["count"] = [len(arr)]
        for qlabel, qval in [("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)]:
            s[field][qlabel] = [float(np.percentile(arr, qval))]

    with open(stats_path, "w") as f:
        json.dump(s, f, indent=2)
    print(f"  ✓ stats.json recomputed with exact quantiles")


if __name__ == "__main__":
    main()
