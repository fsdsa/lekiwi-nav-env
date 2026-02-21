#!/usr/bin/env python3
"""
Convert LeKiwi RL expert HDF5 files to LeRobot Dataset v3 format.

Input (from collect_demos.py, v6):
  episode_k/
    actions       (T, 9)
    robot_state   (T, 9)              [optional]
    subtask_ids   (T,)                [optional]
    obs           (T, 37 or 33 or 24) [fallback for robot_state/subtask_ids]
    images/
      base_rgb    (T, H, W, 3)        [optional]
      wrist_rgb   (T, H, W, 3)        [optional]

Output (LeRobot v3 local dataset):
  <output_root>/
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/subtasks.parquet             [if --include_subtask_index]
    data/chunk-xxx/file-xxx.parquet
    videos/observation.images.front/chunk-xxx/file-xxx.mp4
    videos/observation.images.wrist/chunk-xxx/file-xxx.mp4
    meta/tasks.jsonl                  [compatibility helper]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_SUBTASK_ID_TO_TEXT = {
    0: "look around to find the target object",
    1: "approach the target object",
    2: "pick up the target object",
    3: "return to the starting position",
}
DEFAULT_FULL_TASK_TEXT = "find the target and bring it back"


@dataclass
class EpisodeInfo:
    h5_path: Path
    episode_key: str
    steps: int
    has_base_rgb: bool
    has_wrist_rgb: bool
    has_robot_state: bool
    has_subtask_ids: bool
    obs_dim: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeKiwi HDF5 -> LeRobot v3 converter")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="HDF5 file path(s) or glob(s), e.g. outputs/rl_demos/*.hdf5",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output directory for LeRobot v3 dataset (must not exist unless --overwrite)",
    )
    parser.add_argument("--repo_id", type=str, default="local/lekiwi_fetch_v6")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--robot_type", type=str, default="lekiwi_client")
    parser.add_argument("--task_source", type=str, choices=["subtask", "full_task"], default="subtask")
    parser.add_argument("--full_task_text", type=str, default=DEFAULT_FULL_TASK_TEXT)
    parser.add_argument("--no_videos", action="store_true", help="Do not store camera streams as videos")
    parser.add_argument(
        "--include_subtask_index",
        action="store_true",
        help="Store subtask_index feature and write meta/subtasks.parquet",
    )
    parser.add_argument(
        "--skip_episodes_without_images",
        action="store_true",
        help="When videos are enabled, skip episodes missing required image streams",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec used by LeRobotDataset.create",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Inspect input episodes and print conversion plan without writing output",
    )
    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(Path(p) for p in expanded if Path(p).is_file())
        elif Path(pattern).is_file():
            paths.append(Path(pattern))
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError(f"No input HDF5 files found for: {list(patterns)}")
    return unique


def sorted_episode_keys(h5f: h5py.File) -> list[str]:
    keys = [k for k in h5f.keys() if k.startswith("episode_")]
    keys.sort(key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)
    return keys


def _parse_subtask_mapping(raw: object) -> dict[int, str]:
    if raw is None:
        return {}

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}

    out: dict[int, str] = {}
    for k, v in parsed.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    return out


def read_subtask_mapping_from_attrs(attrs, default: dict[int, str] | None = None) -> dict[int, str]:
    """Read subtask mapping from an attrs container (file or episode)."""
    mapping = _parse_subtask_mapping(attrs.get("subtask_id_to_text_json"))
    if mapping:
        return mapping
    if default is not None:
        return dict(default)
    return {}


def read_subtask_mapping(h5f: h5py.File) -> dict[int, str]:
    """Read file-level mapping with default fallback."""
    return read_subtask_mapping_from_attrs(h5f.attrs, default=DEFAULT_SUBTASK_ID_TO_TEXT)


def read_episode_full_task_text(
    grp: h5py.Group,
    episode_subtask_map: dict[int, str],
    default_full_task_text: str,
) -> str:
    """Read per-episode full-task instruction with fallback order."""
    raw = grp.attrs.get("instruction")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if 10 in episode_subtask_map and episode_subtask_map[10].strip():
        return episode_subtask_map[10]
    return default_full_task_text


def inspect_episode(h5_path: Path, h5f: h5py.File, ep_key: str) -> EpisodeInfo:
    grp = h5f[ep_key]
    has_actions = "actions" in grp
    steps = int(grp["actions"].shape[0]) if has_actions else 0

    has_obs = "obs" in grp
    obs_dim = int(grp["obs"].shape[1]) if has_obs and grp["obs"].ndim == 2 else None

    has_robot_state = "robot_state" in grp
    has_subtask_ids = "subtask_ids" in grp

    has_base_rgb = False
    has_wrist_rgb = False
    if "images" in grp:
        img_grp = grp["images"]
        has_base_rgb = "base_rgb" in img_grp
        has_wrist_rgb = "wrist_rgb" in img_grp

    return EpisodeInfo(
        h5_path=h5_path,
        episode_key=ep_key,
        steps=steps,
        has_base_rgb=has_base_rgb,
        has_wrist_rgb=has_wrist_rgb,
        has_robot_state=has_robot_state,
        has_subtask_ids=has_subtask_ids,
        obs_dim=obs_dim,
    )


def convert_to_vla_units(data_9d: np.ndarray) -> np.ndarray:
    """
    v3.0: sim velocity(m/s, rad/s) = real velocity(m/s, rad/s) — 단위 변환 불필요.
    이전 displacement 방식의 m→mm 변환 제거됨.
    이 함수는 하위 호환성을 위해 유지하되, 데이터를 그대로 반환한다.
    """
    return data_9d.copy()


def infer_robot_state_from_obs(obs: np.ndarray) -> np.ndarray:
    """Fallback extraction of robot_state(9D) from RL obs."""
    if obs.ndim != 2:
        raise ValueError(f"obs must be 2D, got shape {obs.shape}")
    dim = obs.shape[1]
    if dim == 20:
        # Skill-1 Navigate obs: arm_pos(0:5) + gripper(5:6) + base_body_vel(6:9)
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1).astype(np.float32)
    if dim == 30:
        # Skill-2 obs: arm_pos(0:5) + gripper(5:6) + base_body_vel(6:9)
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1).astype(np.float32)
    if dim == 29:
        # Skill-3 obs: arm_pos(0:5) + gripper(5:6) + base_body_vel(6:9)
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1).astype(np.float32)
    if dim == 37:
        # legacy privileged obs: arm_joint_pos(18:24) + wheel_vel(30:33)
        return np.concatenate([obs[:, 18:24], obs[:, 30:33]], axis=1).astype(np.float32)
    if dim == 33:
        # legacy: arm_joint_pos(18:24) + wheel_vel(30:33)
        return np.concatenate([obs[:, 18:24], obs[:, 30:33]], axis=1).astype(np.float32)
    if dim == 24:
        # legacy nav env: arm_joint_pos(9:15) + wheel_vel(21:24)
        return np.concatenate([obs[:, 9:15], obs[:, 21:24]], axis=1).astype(np.float32)
    raise ValueError(f"Cannot infer robot_state from obs dim={dim}")


def infer_subtask_ids_from_obs(obs: np.ndarray) -> np.ndarray:
    """Fallback extraction of subtask IDs from phase_onehot in obs (37D/33D/24D)."""
    if obs.ndim != 2 or obs.shape[1] < 10:
        return np.zeros((obs.shape[0],), dtype=np.int64)
    phase_onehot = obs[:, 6:10]
    return np.argmax(phase_onehot, axis=1).astype(np.int64)


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.input)
    output_root = Path(args.output_root).expanduser().resolve()

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace.")
    if output_root.exists() and args.overwrite:
        # avoid destructive shell calls; use Python stdlib
        import shutil

        shutil.rmtree(output_root)

    episodes: list[EpisodeInfo] = []
    # Global bootstrap mapping. Per-episode mapping is applied during conversion.
    subtask_id_to_text = dict(DEFAULT_SUBTASK_ID_TO_TEXT)
    for h5_path in input_paths:
        with h5py.File(h5_path, "r") as h5f:
            file_subtask_map = read_subtask_mapping(h5f)
            subtask_id_to_text.update(file_subtask_map)
            for ep_key in sorted_episode_keys(h5f):
                ep = inspect_episode(h5_path, h5f, ep_key)
                if ep.steps > 0:
                    episodes.append(ep)

    if not episodes:
        raise RuntimeError("No valid episodes found in input HDF5 files.")

    # Determine available camera streams from episodes
    any_base = any(ep.has_base_rgb for ep in episodes)
    any_wrist = any(ep.has_wrist_rgb for ep in episodes)
    use_videos = (not args.no_videos) and (any_base or any_wrist)

    if args.dry_run:
        print("=" * 72)
        print("HDF5 -> LeRobot v3 dry-run")
        print(f"inputs: {len(input_paths)} file(s)")
        print(f"episodes: {len(episodes)}")
        print(f"videos enabled: {use_videos}")
        print(f"base_rgb present in any episode: {any_base}")
        print(f"wrist_rgb present in any episode: {any_wrist}")
        print(f"include_subtask_index: {args.include_subtask_index}")
        print(f"task_source: {args.task_source}")
        print("=" * 72)
        return

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "lerobot is not installed in this environment.\n"
            "Run this converter in an environment with lerobot>=0.4.0 installed "
            "(e.g. your A100 lerobotpi0 env)."
        ) from exc

    # Build features
    # yubinnn11/lekiwi3 (LeRobot v3.0) 채널 이름
    state_names = [
        "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
        "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
        "x.vel", "y.vel", "theta.vel",
    ]
    action_names = [
        "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
        "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
        "x.vel", "y.vel", "theta.vel",
    ]

    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (9,), "names": state_names},
        "action": {"dtype": "float32", "shape": (9,), "names": action_names},
    }
    if args.include_subtask_index:
        features["subtask_index"] = {"dtype": "int64", "shape": (1,), "names": None}

    base_shape = None
    wrist_shape = None
    if use_videos:
        # infer camera resolution from first episode containing each stream
        for ep in episodes:
            if (not ep.has_base_rgb) and (not ep.has_wrist_rgb):
                continue
            with h5py.File(ep.h5_path, "r") as h5f:
                grp = h5f[ep.episode_key]
                if ep.has_base_rgb and base_shape is None:
                    base_shape = tuple(grp["images"]["base_rgb"].shape[1:4])
                if ep.has_wrist_rgb and wrist_shape is None:
                    wrist_shape = tuple(grp["images"]["wrist_rgb"].shape[1:4])
            if (any_base and base_shape is not None) and (any_wrist and wrist_shape is not None):
                break

        if base_shape is not None:
            features["observation.images.front"] = {
                "dtype": "video",
                "shape": base_shape,
                "names": ["height", "width", "channels"],
            }
        if wrist_shape is not None:
            features["observation.images.wrist"] = {
                "dtype": "video",
                "shape": wrist_shape,
                "names": ["height", "width", "channels"],
            }

    print("=" * 72)
    print("Creating LeRobot v3 dataset")
    print(f"repo_id: {args.repo_id}")
    print(f"output_root: {output_root}")
    print(f"episodes to process: {len(episodes)}")
    print(f"videos enabled: {use_videos}")
    print(f"features: {list(features.keys())}")
    print("=" * 72)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=output_root,
        robot_type=args.robot_type,
        use_videos=use_videos,
        vcodec=args.vcodec,
    )

    # Pre-register known task strings for stable ordering when possible
    ordered_subtask_ids = sorted(subtask_id_to_text.keys())
    ordered_subtasks = [subtask_id_to_text[sid] for sid in ordered_subtask_ids]
    try:
        dataset.meta.save_episode_tasks(ordered_subtasks + [args.full_task_text])
    except Exception:
        # Non-fatal; tasks are also registered lazily while saving episodes
        pass

    converted = 0
    skipped = 0

    for ep in episodes:
        with h5py.File(ep.h5_path, "r") as h5f:
            grp = h5f[ep.episode_key]
            file_subtask_map = read_subtask_mapping(h5f)
            # Episode attrs override file attrs to preserve object-specific instructions.
            episode_subtask_map = dict(file_subtask_map)
            episode_subtask_map.update(read_subtask_mapping_from_attrs(grp.attrs))
            episode_full_task_text = read_episode_full_task_text(
                grp=grp,
                episode_subtask_map=episode_subtask_map,
                default_full_task_text=args.full_task_text,
            )

            actions_raw = np.asarray(grp["actions"], dtype=np.float32)
            actions = convert_to_vla_units(actions_raw)
            n_steps = actions.shape[0]

            # robot_state
            if "robot_state" in grp:
                robot_state = convert_to_vla_units(np.asarray(grp["robot_state"], dtype=np.float32))
            elif "obs" in grp:
                robot_state = convert_to_vla_units(infer_robot_state_from_obs(np.asarray(grp["obs"])))
            else:
                print(f"[SKIP] {ep.h5_path.name}:{ep.episode_key} - no robot_state/obs")
                skipped += 1
                continue

            if robot_state.shape[0] != n_steps:
                print(f"[SKIP] {ep.h5_path.name}:{ep.episode_key} - length mismatch robot_state/actions")
                skipped += 1
                continue

            # subtask ids
            if "subtask_ids" in grp:
                subtask_ids = np.asarray(grp["subtask_ids"], dtype=np.int64).reshape(-1)
            elif "obs" in grp:
                subtask_ids = infer_subtask_ids_from_obs(np.asarray(grp["obs"]))
            else:
                subtask_ids = np.zeros((n_steps,), dtype=np.int64)

            if subtask_ids.shape[0] != n_steps:
                subtask_ids = np.resize(subtask_ids, (n_steps,))

            base_rgb = None
            wrist_rgb = None
            if use_videos and "images" in grp:
                img_grp = grp["images"]
                if "observation.images.front" in features:
                    if "base_rgb" in img_grp:
                        base_rgb = img_grp["base_rgb"]
                    elif args.skip_episodes_without_images:
                        print(f"[SKIP] {ep.h5_path.name}:{ep.episode_key} - missing base_rgb")
                        skipped += 1
                        continue
                if "observation.images.wrist" in features:
                    if "wrist_rgb" in img_grp:
                        wrist_rgb = img_grp["wrist_rgb"]
                    elif args.skip_episodes_without_images:
                        print(f"[SKIP] {ep.h5_path.name}:{ep.episode_key} - missing wrist_rgb")
                        skipped += 1
                        continue
            elif use_videos and args.skip_episodes_without_images:
                print(f"[SKIP] {ep.h5_path.name}:{ep.episode_key} - missing images group")
                skipped += 1
                continue

            for t in range(n_steps):
                sid = int(subtask_ids[t])
                subtask_text = episode_subtask_map.get(sid, subtask_id_to_text.get(sid, f"subtask_{sid}"))
                task_text = subtask_text if args.task_source == "subtask" else episode_full_task_text

                frame: dict = {
                    "task": task_text,
                    "observation.state": np.asarray(robot_state[t], dtype=np.float32),
                    "action": np.asarray(actions[t], dtype=np.float32),
                }

                if args.include_subtask_index:
                    frame["subtask_index"] = np.array([sid], dtype=np.int64)

                if "observation.images.front" in features:
                    if base_rgb is not None:
                        frame["observation.images.front"] = np.asarray(base_rgb[t], dtype=np.uint8)
                    elif args.skip_episodes_without_images:
                        raise RuntimeError("Unexpected: base_rgb missing after episode-level check.")
                    else:
                        # fallback black frame
                        h, w, c = features["observation.images.front"]["shape"]
                        frame["observation.images.front"] = np.zeros((h, w, c), dtype=np.uint8)

                if "observation.images.wrist" in features:
                    if wrist_rgb is not None:
                        frame["observation.images.wrist"] = np.asarray(wrist_rgb[t], dtype=np.uint8)
                    elif args.skip_episodes_without_images:
                        raise RuntimeError("Unexpected: wrist_rgb missing after episode-level check.")
                    else:
                        h, w, c = features["observation.images.wrist"]["shape"]
                        frame["observation.images.wrist"] = np.zeros((h, w, c), dtype=np.uint8)

                dataset.add_frame(frame)

            dataset.save_episode()
            converted += 1
            print(
                f"[OK] episode={converted:04d} "
                f"src={ep.h5_path.name}:{ep.episode_key} "
                f"steps={n_steps}"
            )

    dataset.finalize()

    # Write subtasks parquet (optional, used by LeRobot subtask API)
    if args.include_subtask_index:
        import pandas as pd

        subtask_texts = [subtask_id_to_text[sid] for sid in sorted(subtask_id_to_text)]
        subtask_indices = list(sorted(subtask_id_to_text))
        subtasks_df = pd.DataFrame({"subtask_index": subtask_indices}, index=subtask_texts)
        subtasks_path = output_root / "meta" / "subtasks.parquet"
        subtasks_path.parent.mkdir(parents=True, exist_ok=True)
        subtasks_df.to_parquet(subtasks_path)

    # Write compatibility tasks.jsonl (legacy helper for external tooling)
    tasks_jsonl_path = output_root / "meta" / "tasks.jsonl"
    tasks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df = dataset.meta.tasks.sort_values("task_index")
    with open(tasks_jsonl_path, "w", encoding="utf-8") as f:
        for _, row in tasks_df.iterrows():
            record = {"task_index": int(row["task_index"]), "task": str(row["task"])}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 72)
    print("Conversion done")
    print(f"converted episodes: {converted}")
    print(f"skipped episodes:   {skipped}")
    print(f"output: {output_root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
