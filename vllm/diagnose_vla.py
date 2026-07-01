"""
Diagnostics for 210K ckpt.

Tests
-----
A. Repeat-sampling variance: same observation → VLA N times → output std.
   If flow-matching sampling noise is dominant, std >> training-data std.

B. Forensic replay: one full episode → predict at every frame, compare to GT.
   Surfaces where in trajectory the policy diverges (early / mid / late).

C. num_steps sensitivity: re-run A with num_steps ∈ {10, 20, 30}.
   If variance drops meaningfully at 20/30, we know integration error is a factor.

Prereq: VLA server must be running at http://localhost:8002.
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


VLA_URL = "http://localhost:8002"
# LEROBOT_DATASET_ROOT 환경변수로 재정의 가능 (기본 ~/lerobot_data/lekiwi_viva_v4)
DATASET_ROOT = os.environ.get("LEROBOT_DATASET_ROOT", os.path.expanduser("~/lerobot_data/lekiwi_viva_v4"))


def encode_img(t: torch.Tensor) -> str:
    """CHW float32 [0,1] → HWC uint8 → PNG base64."""
    arr = (t.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_vla(frame: dict, num_steps: int | None = None,
              noise_seed: int | None = None) -> np.ndarray:
    """Returns (chunk_size, 9) action chunk."""
    payload = {
        "base_image_b64": encode_img(frame["observation.images.front"]),
        "wrist_image_b64": encode_img(frame["observation.images.wrist"]),
        "state": frame["observation.state"].tolist(),
        "instruction": frame["task"],
    }
    if num_steps is not None:
        payload["num_steps"] = num_steps
    if noise_seed is not None:
        payload["noise_seed"] = noise_seed
    r = requests.post(f"{VLA_URL}/act", json=payload, timeout=30)
    r.raise_for_status()
    return np.array(r.json()["actions"])


def find_frame_with_task(ds: LeRobotDataset, task_idx: int, offset: int = 50) -> int:
    """Return global frame index inside the first episode matching task_idx,
    offset frames into the episode (to skip transient start frames)."""
    # Find first episode starting with this task
    for ep in range(ds.num_episodes):
        ep_info = ds.meta.episodes[ep]
        f_idx = int(ep_info["dataset_from_index"])
        t_idx = int(ds[f_idx]["task_index"])
        if t_idx == task_idx:
            ep_len = int(ep_info["dataset_to_index"]) - f_idx
            return f_idx + min(offset, max(0, ep_len // 4))
    raise ValueError(f"task_index {task_idx} not found")


# ── Test A ──────────────────────────────────────────────────────────────

def test_a_variance(ds: LeRobotDataset, n_samples: int = 10,
                    num_steps: int | None = None, label_prefix: str = "") -> dict:
    print(f"\n═══ TEST A{label_prefix}: Repeated sampling variance ═══")
    if num_steps is not None:
        print(f"   num_steps = {num_steps}")
    results = {}
    # task 8 = navigate turn left (state nearly constant → action should be constant)
    # task 5 = approach (state variable → action variable but conditional-sharp)
    probes = [
        ("navigate_turn_left", 8),
        ("navigate_forward",    6),
        ("carry_forward",      12),
        ("approach_lift_mid",   5),
    ]
    for label, task_idx in probes:
        try:
            f_idx = find_frame_with_task(ds, task_idx)
        except ValueError as e:
            print(f"  {label}: skip ({e})")
            continue
        frame = ds[f_idx]

        # Sample N times with fresh noise each call (no noise_seed)
        t0 = time.time()
        samples = np.stack([query_vla(frame, num_steps=num_steps) for _ in range(n_samples)])
        dt = time.time() - t0
        # samples: (N, chunk=50, 9)
        first_action = samples[:, 0, :]  # (N, 9) — first action of each chunk
        gt_action = frame["action"].cpu().numpy()

        print(f"\n  [{label}] frame={f_idx}  task='{frame['task'][:50]}'  "
              f"({n_samples} samples in {dt:.1f}s)")
        print(f"    sample mean (arm[0:6]): {[f'{m:+.3f}' for m in first_action.mean(0)[:6]]}")
        print(f"    sample std  (arm[0:6]): {[f'{s:.4f}' for s in first_action.std(0)[:6]]}")
        print(f"    GT action  (arm[0:6]):  {[f'{g:+.3f}' for g in gt_action[:6]]}")
        print(f"    sample std  (base[6:9]): {[f'{s:.4f}' for s in first_action.std(0)[6:9]]}")
        results[label] = {
            "frame_idx": f_idx,
            "sample_mean_first": first_action.mean(0).tolist(),
            "sample_std_first": first_action.std(0).tolist(),
            "gt_action": gt_action.tolist(),
        }
    return results


# ── Test B ──────────────────────────────────────────────────────────────

def test_b_forensic(ds: LeRobotDataset, episode_index: int | None = None,
                    stride: int = 10) -> dict:
    """Replay one episode, compute per-frame prediction error vs GT action."""
    print(f"\n═══ TEST B: Forensic replay ═══")
    if episode_index is None:
        # Pick first episode from the "new v3" approach set (episode_index 978+)
        for ep in range(ds.num_episodes):
            ep_info = ds.meta.episodes[ep]
            f_idx = int(ep_info["dataset_from_index"])
            if int(ds[f_idx]["task_index"]) == 5 and ep >= 978:
                episode_index = ep
                break
    ep_info = ds.meta.episodes[episode_index]
    from_i = int(ep_info["dataset_from_index"])
    to_i = int(ep_info["dataset_to_index"])
    ep_len = to_i - from_i
    print(f"  Episode {episode_index} (task=approach_and_lift), length={ep_len}")
    print(f"  Sampling every {stride} frames: {len(range(from_i, to_i, stride))} predictions")

    preds, gts, frame_idxs = [], [], []
    t0 = time.time()
    for i in range(from_i, to_i, stride):
        f = ds[i]
        actions = query_vla(f, num_steps=None)  # default num_steps
        preds.append(actions[0])  # first action of chunk
        gts.append(f["action"].cpu().numpy())
        frame_idxs.append(i - from_i)  # within-episode frame_index
    preds = np.stack(preds)
    gts = np.stack(gts)
    errs = preds - gts
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n  Per-dim MAE (over whole episode):")
    mae = np.abs(errs).mean(0)
    dim_names = ["arm_pan", "arm_lift", "arm_elbow", "arm_wristF", "arm_wristR",
                 "gripper", "vx", "vy", "wz"]
    for i, n in enumerate(dim_names):
        print(f"    {n:11s}: MAE={mae[i]:.3f}  std_pred={preds[:,i].std():.3f}  std_gt={gts[:,i].std():.3f}")

    print(f"\n  Phase-wise arm[1] (shoulder lift) MAE:")
    phases = [
        ("early (fr 0-20)",   lambda fi: fi < 20),
        ("fr 20-100",         lambda fi: 20 <= fi < 100),
        ("fr 100-300",        lambda fi: 100 <= fi < 300),
        ("late (fr 300+)",    lambda fi: fi >= 300),
    ]
    fi_arr = np.array(frame_idxs)
    for label, cond in phases:
        mask = np.array([cond(fi) for fi in fi_arr])
        if mask.sum() < 2: continue
        arm1_err = np.abs(errs[mask, 1]).mean()
        gripper_err = np.abs(errs[mask, 5]).mean()
        arm1_pred = preds[mask, 1].mean()
        arm1_gt = gts[mask, 1].mean()
        print(f"    {label:20s} (n={mask.sum():3d}): "
              f"arm[1] MAE={arm1_err:.3f} (pred={arm1_pred:+.2f} gt={arm1_gt:+.2f}) | "
              f"grip MAE={gripper_err:.3f}")

    return {
        "episode_index": int(episode_index),
        "stride": stride,
        "mae_per_dim": mae.tolist(),
        "per_frame_pred": preds.tolist(),
        "per_frame_gt": gts.tolist(),
        "frame_indices": [int(x) for x in frame_idxs],
    }


# ── Test C ──────────────────────────────────────────────────────────────

def test_c_num_steps(ds: LeRobotDataset) -> dict:
    print(f"\n═══ TEST C: num_steps sensitivity ═══")
    results = {}
    for ns in [10, 20, 30]:
        r = test_a_variance(ds, n_samples=10, num_steps=ns,
                            label_prefix=f" (num_steps={ns})")
        results[f"num_steps_{ns}"] = r
    return results


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="a,b,c", help="comma-sep subset of {a,b,c}")
    ap.add_argument("--output", default="/tmp/diagnose_vla.json")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--episode", type=int, default=None,
                    help="episode index for test B (default: first new-v3 approach)")
    ap.add_argument("--stride", type=int, default=10,
                    help="frame stride for test B")
    args = ap.parse_args()

    # Server sanity
    try:
        h = requests.get(f"{VLA_URL}/health", timeout=3).json()
        print(f"VLA health: {h}")
    except Exception as e:
        print(f"VLA server not reachable at {VLA_URL}: {e}", file=sys.stderr)
        sys.exit(2)

    ds = LeRobotDataset(Path(DATASET_ROOT).name, root=DATASET_ROOT)
    print(f"Dataset: {len(ds)} frames, {ds.num_episodes} episodes\n")

    out: dict = {}
    tests = {t.strip().lower() for t in args.tests.split(",")}
    if "a" in tests:
        out["A"] = test_a_variance(ds, n_samples=args.n_samples)
    if "b" in tests:
        out["B"] = test_b_forensic(ds, episode_index=args.episode, stride=args.stride)
    if "c" in tests:
        out["C"] = test_c_num_steps(ds)

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n✓ Results written to {args.output}")


if __name__ == "__main__":
    main()
