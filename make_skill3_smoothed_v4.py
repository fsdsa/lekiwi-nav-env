#!/usr/bin/env python3
"""Build a Phase-B-only demo that preserves v16 entry states but repairs weak tails.

Strategy:
- start from smoothed_v2 (so Phase-B entry distribution stays intact)
- identify weak-lowering episodes
- splice in matched strong donor tails after the recipient's peak-lowering point
- then enforce longer open plateau and smooth retract transition
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _ep_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _find_retract_start(arm1: np.ndarray, open_start: int, sustain: int = 6) -> int:
    peak = int(np.argmax(arm1))
    start = max(open_start + 4, peak)
    for i in range(start, len(arm1) - sustain):
        if np.mean(np.diff(arm1[i : i + sustain + 1])) < -0.008:
            return i
    return len(arm1) - 1


def _ensure_plateau(
    obs: np.ndarray,
    actions: np.ndarray,
    robot_state: np.ndarray,
    open_start: int,
    retract_start: int,
    min_open_steps: int,
    obs_open: float,
    act_open: float,
    rs_open: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    current_open_steps = max(0, retract_start - open_start)
    extra = max(0, min_open_steps - current_open_steps)
    plateau_end_idx = max(open_start, retract_start - 1)

    obs[open_start:retract_start, 5] = np.maximum(obs[open_start:retract_start, 5], obs_open)
    actions[open_start:retract_start, 5] = np.maximum(actions[open_start:retract_start, 5], act_open)
    robot_state[open_start:retract_start, 5] = np.maximum(robot_state[open_start:retract_start, 5], rs_open)

    if extra <= 0:
        return obs, actions, robot_state, retract_start

    obs_plateau = np.repeat(obs[plateau_end_idx : plateau_end_idx + 1], extra, axis=0)
    act_plateau = np.repeat(actions[plateau_end_idx : plateau_end_idx + 1], extra, axis=0)
    rs_plateau = np.repeat(robot_state[plateau_end_idx : plateau_end_idx + 1], extra, axis=0)

    obs_plateau[:, 5] = obs_open
    act_plateau[:, 5] = act_open
    rs_plateau[:, 5] = rs_open
    if obs.shape[1] >= 36:
        obs_plateau[:, -1] = 0.0

    obs = np.concatenate([obs[:retract_start], obs_plateau, obs[retract_start:]], axis=0)
    actions = np.concatenate([actions[:retract_start], act_plateau, actions[retract_start:]], axis=0)
    robot_state = np.concatenate([robot_state[:retract_start], rs_plateau, robot_state[retract_start:]], axis=0)
    return obs, actions, robot_state, retract_start + extra


def _blend_retract(
    obs: np.ndarray,
    actions: np.ndarray,
    robot_state: np.ndarray,
    retract_start: int,
    blend_steps: int,
    obs_open: float,
    act_open: float,
    rs_open: float,
) -> None:
    if retract_start >= len(obs) - 2:
        return
    end = min(len(obs), retract_start + blend_steps)
    if end - retract_start <= 1:
        return

    tgt_obs = obs[retract_start:end, 5].copy()
    tgt_act = actions[retract_start:end, 5].copy()
    tgt_rs = robot_state[retract_start:end, 5].copy()

    alpha = np.linspace(0.0, 1.0, end - retract_start, dtype=np.float32)
    obs[retract_start:end, 5] = (1.0 - alpha) * obs_open + alpha * tgt_obs
    actions[retract_start:end, 5] = (1.0 - alpha) * act_open + alpha * tgt_act
    robot_state[retract_start:end, 5] = (1.0 - alpha) * rs_open + alpha * tgt_rs


def _smooth_large_action_drops(
    obs: np.ndarray,
    actions: np.ndarray,
    robot_state: np.ndarray,
    max_step_drop: float,
    smooth_steps: int,
) -> None:
    if len(actions) < 3:
        return
    i = 0
    while i < len(actions) - 1:
        drop = float(actions[i + 1, 5] - actions[i, 5])
        if drop >= -max_step_drop:
            i += 1
            continue
        end = min(len(actions) - 1, i + smooth_steps)
        a0 = float(actions[i, 5])
        a1 = float(actions[end, 5])
        o0 = float(obs[i, 5])
        o1 = float(obs[end, 5])
        r0 = float(robot_state[i, 5])
        r1 = float(robot_state[end, 5])
        alpha = np.linspace(0.0, 1.0, end - i + 1, dtype=np.float32)
        actions[i : end + 1, 5] = (1.0 - alpha) * a0 + alpha * a1
        obs[i : end + 1, 5] = (1.0 - alpha) * o0 + alpha * o1
        robot_state[i : end + 1, 5] = (1.0 - alpha) * r0 + alpha * r1
        i = end


def _choose_donor_idx(
    all_obs: list[np.ndarray],
    strong_ids: list[int],
    recipient_obs: np.ndarray,
    recipient_ref_idx: int,
    recipient_ref_val: float,
) -> tuple[int, int]:
    target = recipient_obs[recipient_ref_idx]
    best = None
    for donor_id in strong_ids:
        donor_obs = all_obs[donor_id]
        donor_arm1 = donor_obs[:, 1]
        cand = np.where(donor_arm1 >= max(1.8, recipient_ref_val - 0.05))[0]
        if len(cand) == 0:
            cand = np.arange(np.argmax(donor_arm1), len(donor_obs), dtype=np.int64)
        if len(cand) == 0:
            continue
        donor_slice = donor_obs[cand]
        # Match arm/grip posture and local destination relation.
        d_arm = np.linalg.norm(donor_slice[:, :6] - target[:6], axis=1)
        d_dest = np.linalg.norm(donor_slice[:, 21:24] - target[21:24], axis=1)
        score = d_arm + 0.5 * d_dest
        k = int(np.argmin(score))
        score_v = float(score[k])
        donor_idx = int(cand[k])
        if best is None or score_v < best[0]:
            best = (score_v, donor_id, donor_idx)
    if best is None:
        raise RuntimeError("No donor found for weak-lowering episode")
    return int(best[1]), int(best[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("demos_skill3/smoothed_v2.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("demos_skill3/smoothed_v4.hdf5"))
    parser.add_argument("--min_arm1_peak", type=float, default=2.8)
    parser.add_argument("--open_thresh", type=float, default=0.55)
    parser.add_argument("--min_open_steps", type=int, default=180)
    parser.add_argument("--blend_steps", type=int, default=28)
    parser.add_argument("--max_step_drop", type=float, default=0.12)
    parser.add_argument("--smooth_drop_steps", type=int, default=24)
    parser.add_argument("--obs_open", type=float, default=1.36)
    parser.add_argument("--act_open", type=float, default=0.70)
    parser.add_argument("--rs_open", type=float, default=1.48)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as f_in:
        eps = sorted(f_in.keys(), key=_ep_sort_key)
        all_groups = [{k: f_in[ep][k][:] for k in f_in[ep].keys()} for ep in eps]
        all_obs = [g["obs"] for g in all_groups]
        peaks = [float(g["obs"][:, 1].max()) for g in all_groups]
        strong_ids = [i for i, peak in enumerate(peaks) if peak >= args.min_arm1_peak]
        weak_ids = [i for i, peak in enumerate(peaks) if peak < args.min_arm1_peak]

        repaired = 0
        inserted_steps = []
        jump_before = []
        jump_after = []
        open_before = []
        open_after = []

        with h5py.File(args.output, "w") as f_out:
            _copy_attrs(f_in, f_out)
            f_out.attrs["source_hdf5"] = str(args.input)
            f_out.attrs["smoothed_v4"] = True
            f_out.attrs["min_arm1_peak"] = args.min_arm1_peak
            f_out.attrs["weak_episode_count"] = len(weak_ids)

            for out_idx, ep in enumerate(eps):
                src = all_groups[out_idx]
                obs = src["obs"].copy()
                actions = src["actions"].copy()
                robot_state = src["robot_state"].copy()
                other = {k: v.copy() for k, v in src.items() if k not in {"obs", "actions", "robot_state"}}

                arm1 = obs[:, 1]
                peak_idx = int(np.argmax(arm1))
                peak_val = float(arm1[peak_idx])

                donor_ep = None
                donor_idx = None
                replaced_from = None
                if peak_val < args.min_arm1_peak:
                    rise_idx = np.where(obs[:, 1] >= max(1.6, peak_val - 0.4))[0]
                    ref_idx = int(rise_idx[0]) if len(rise_idx) else peak_idx
                    donor_ep, donor_idx = _choose_donor_idx(all_obs, strong_ids, obs, ref_idx, float(obs[ref_idx, 1]))
                    donor = all_groups[donor_ep]
                    replaced_from = ref_idx
                    obs = np.concatenate([obs[: replaced_from], donor["obs"][donor_idx + 1 :]], axis=0)
                    actions = np.concatenate([actions[: replaced_from], donor["actions"][donor_idx + 1 :]], axis=0)
                    robot_state = np.concatenate([robot_state[: replaced_from], donor["robot_state"][donor_idx + 1 :]], axis=0)
                    for key in other.keys():
                        other[key] = np.concatenate([other[key][: replaced_from], donor[key][donor_idx + 1 :]], axis=0)
                    repaired += 1

                open_idx = np.flatnonzero(actions[:, 5] > args.open_thresh)
                if len(open_idx) == 0:
                    open_start = int(np.argmax(obs[:, 1]))
                else:
                    open_start = int(open_idx[0])
                retract_start = _find_retract_start(obs[:, 1], open_start)

                jump_before.append(float(np.min(np.diff(actions[:, 5]))))
                open_before.append(float(np.mean(actions[:, 5] > 0.5)))

                obs2, act2, rs2, retract_start2 = _ensure_plateau(
                    obs.copy(),
                    actions.copy(),
                    robot_state.copy(),
                    open_start,
                    retract_start,
                    args.min_open_steps,
                    args.obs_open,
                    args.act_open,
                    args.rs_open,
                )
                _blend_retract(obs2, act2, rs2, retract_start2, args.blend_steps, args.obs_open, args.act_open, args.rs_open)
                _smooth_large_action_drops(obs2, act2, rs2, args.max_step_drop, args.smooth_drop_steps)

                inserted_steps.append(int(retract_start2 - retract_start))
                jump_after.append(float(np.min(np.diff(act2[:, 5]))))
                open_after.append(float(np.mean(act2[:, 5] > 0.5)))

                dst = f_out.create_group(f"episode_{out_idx}")
                _copy_attrs(f_in[ep], dst)
                dst.attrs["source_episode"] = ep
                dst.attrs["arm1_peak"] = peak_val
                dst.attrs["repaired_tail"] = donor_ep is not None
                if donor_ep is not None:
                    dst.attrs["donor_episode"] = eps[donor_ep]
                    dst.attrs["donor_start_idx"] = int(donor_idx + 1)
                    dst.attrs["replace_from_idx"] = int(replaced_from)
                dst.attrs["open_start"] = int(open_start)
                dst.attrs["retract_start_orig"] = int(retract_start)
                dst.attrs["retract_start_new"] = int(retract_start2)
                dst.attrs["inserted_steps"] = int(retract_start2 - retract_start)

                dst.create_dataset("obs", data=obs2, compression="gzip")
                dst.create_dataset("actions", data=act2, compression="gzip")
                dst.create_dataset("robot_state", data=rs2, compression="gzip")
                for key, value in other.items():
                    dst.create_dataset(key, data=value, compression="gzip")

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"episodes: {len(eps)} strong={len(strong_ids)} weak={len(weak_ids)} repaired={repaired}")
    if inserted_steps:
        print(f"inserted_steps mean/min/max: {np.mean(inserted_steps):.1f} / {np.min(inserted_steps)} / {np.max(inserted_steps)}")
        print(f"open_ratio (>0.5 act) before -> after: {np.mean(open_before):.4f} -> {np.mean(open_after):.4f}")
        print(f"min one-step grip-action delta before -> after: {np.mean(jump_before):.4f} -> {np.mean(jump_after):.4f}")


if __name__ == "__main__":
    main()
