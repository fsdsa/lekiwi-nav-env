"""
Convert expert teleop hdf5 (scene_expert_*.hdf5) into a LeRobot v3.0 dataset.

Expected hdf5 schema (per episode):
    episode_{N}/
        actions       : (T, 9) float32
        robot_state   : (T, 9) float32
        images/
            base_rgb  : (T, 400, 640, 3) uint8
            wrist_rgb : (T, 400, 640, 3) uint8

Output matches lekiwi_viva_v4 schema:
    observation.state        (9D float32)
    action                   (9D float32)
    observation.images.front (H=400, W=640, RGB, av1 via pyav)
    observation.images.wrist (same)

The resulting dataset can be merged with the v4 nav+carry subset via
build_v5.py (or a simplified merge that drops all task=5 from v4).
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),
        "names": [
            "arm_shoulder_pan.pos", "arm_shoulder_lift.pos",
            "arm_elbow_flex.pos", "arm_wrist_flex.pos",
            "arm_wrist_roll.pos", "arm_gripper.pos",
            "x.vel", "y.vel", "theta.vel",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (9,),
        "names": [
            "arm_shoulder_pan.pos", "arm_shoulder_lift.pos",
            "arm_elbow_flex.pos", "arm_wrist_flex.pos",
            "arm_wrist_roll.pos", "arm_gripper.pos",
            "x.vel", "y.vel", "theta.vel",
        ],
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": (400, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (400, 640, 3),
        "names": ["height", "width", "channels"],
    },
}

TASK_LABEL = "approach and lift the medicine bottle"


def sorted_episode_keys(hf: h5py.File) -> list[str]:
    # episode keys like "episode_0", "episode_1", ..., "episode_99"
    def nk(k: str) -> int:
        return int(k.split("_", 1)[1])
    return sorted(hf.keys(), key=nk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="input hdf5 file path")
    ap.add_argument("--out", required=True, help="output dataset root path")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--task", default=TASK_LABEL)
    ap.add_argument("--force", action="store_true", help="overwrite existing --out")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="convert only first N episodes (for testing)")
    args = ap.parse_args()

    hdf5_path = Path(args.hdf5)
    out_path = Path(args.out)
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    if out_path.exists():
        if not args.force:
            raise FileExistsError(f"{out_path} exists (use --force)")
        shutil.rmtree(out_path)

    t_start = time.time()

    # ── Pre-scan: episode count + frame totals
    with h5py.File(hdf5_path, "r") as hf:
        ep_keys = sorted_episode_keys(hf)
        if args.max_episodes:
            ep_keys = ep_keys[: args.max_episodes]
        total_frames = sum(int(hf[k]["actions"].shape[0]) for k in ep_keys)
        print(f"Input hdf5: {hdf5_path}")
        print(f"  episodes: {len(ep_keys)}")
        print(f"  total frames: {total_frames}")

    # ── Create LeRobot dataset
    ds = LeRobotDataset.create(
        repo_id=out_path.name,
        fps=args.fps,
        features=FEATURES,
        root=str(out_path),
        robot_type="lekiwi_client",
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=4,
    )
    print(f"\nCreated dataset at: {out_path}")

    # ── Iterate episodes → add_frame → save_episode
    with h5py.File(hdf5_path, "r") as hf:
        for ep_key in tqdm(ep_keys, desc="episodes"):
            ep = hf[ep_key]
            actions = np.asarray(ep["actions"], dtype=np.float32)       # (T, 9)
            state = np.asarray(ep["robot_state"], dtype=np.float32)     # (T, 9)
            base_rgb = np.asarray(ep["images"]["base_rgb"], dtype=np.uint8)    # (T, H, W, 3)
            wrist_rgb = np.asarray(ep["images"]["wrist_rgb"], dtype=np.uint8)

            T = actions.shape[0]
            assert state.shape[0] == T
            assert base_rgb.shape[0] == T and wrist_rgb.shape[0] == T

            for i in range(T):
                frame = {
                    "observation.state":        state[i],
                    "action":                   actions[i],
                    "observation.images.front": base_rgb[i],
                    "observation.images.wrist": wrist_rgb[i],
                    "task":                     args.task,
                }
                ds.add_frame(frame)

            ds.save_episode()

    ds.finalize()

    elapsed = time.time() - t_start
    print(f"\n✓ Conversion done in {elapsed/60:.1f} min")
    print(f"  Output: {out_path}")
    print(f"  Verify with: python verify_v5.py --path {out_path}")


if __name__ == "__main__":
    main()
