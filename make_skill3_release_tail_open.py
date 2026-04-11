from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _ep_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def _copy_group(src: h5py.Group, dst_parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    dst = dst_parent.create_group(name)
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    return dst


def _upright_z_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    x = quat[:, 1]
    y = quat[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def _find_retract_start(arm1_tail: np.ndarray, peak_local: int, drop_thresh: float, sustain: int) -> int:
    peak_val = float(arm1_tail[peak_local])
    for i in range(peak_local + 1, len(arm1_tail) - sustain + 1):
        if np.all(arm1_tail[i : i + sustain] < peak_val - drop_thresh):
            return i
    return len(arm1_tail)


def _make_monotonic_tail(
    old_tail: np.ndarray,
    start_min: float,
    target_min: float,
    max_value: float,
    hold_len: int,
) -> np.ndarray:
    ramp_len = max(1, len(old_tail) - hold_len)
    start_value = max(float(old_tail[0]), start_min)
    end_value = min(max_value, max(float(old_tail.max()), target_min))
    ramp = np.linspace(start_value, end_value, ramp_len, dtype=np.float32)
    hold = np.full((hold_len,), end_value, dtype=np.float32)
    new_tail = np.concatenate([ramp, hold], axis=0)
    if len(new_tail) < len(old_tail):
        pad = np.full((len(old_tail) - len(new_tail),), end_value, dtype=np.float32)
        new_tail = np.concatenate([new_tail, pad], axis=0)
    new_tail = new_tail[: len(old_tail)]
    new_tail = np.maximum(new_tail, old_tail)
    new_tail = np.maximum.accumulate(new_tail)
    return np.clip(new_tail, None, max_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel Skill-3 release tails with monotonic gripper-open ramps.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/demos_skill3/combined_skill3_grip_s07_v5_36d_final_phaseAcarrysplice.hdf5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/demos_skill3/combined_skill3_grip_s07_v5_36d_final_phaseAcarrysplice_releaseopen.hdf5"),
    )
    parser.add_argument("--arm1_min", type=float, default=2.2)
    parser.add_argument("--upright_min", type=float, default=0.9)
    parser.add_argument("--peak_window", type=int, default=120)
    parser.add_argument("--min_tail_len", type=int, default=24)
    parser.add_argument("--hold_len", type=int, default=15)
    parser.add_argument("--peak_drop_thresh", type=float, default=0.08)
    parser.add_argument("--peak_drop_sustain", type=int, default=3)
    parser.add_argument("--obs_start_open", type=float, default=1.00)
    parser.add_argument("--obs_target_open", type=float, default=1.36)
    parser.add_argument("--obs_max_open", type=float, default=1.42)
    parser.add_argument("--act_start_open", type=float, default=0.35)
    parser.add_argument("--act_target_open", type=float, default=0.70)
    parser.add_argument("--act_max_open", type=float, default=0.75)
    parser.add_argument("--rs_start_open", type=float, default=1.18)
    parser.add_argument("--rs_target_open", type=float, default=1.48)
    parser.add_argument("--rs_max_open", type=float, default=1.52)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f_in:
        eps = sorted(f_in.keys(), key=_ep_sort_key)

        with h5py.File(args.output, "w") as f_out:
            for k, v in f_in.attrs.items():
                f_out.attrs[k] = v
            f_out.attrs["source_hdf5"] = str(args.input)
            f_out.attrs["release_tail_relabel"] = True
            f_out.attrs["release_tail_note"] = (
                "Phase-B tail relabel with direct monotonic open ramps for obs/actions/robot_state."
            )

            modified_eps = 0
            skipped_eps = 0
            before_obs_max = []
            after_obs_max = []
            before_act_max = []
            after_act_max = []
            tail_lengths = []

            for ep in eps:
                src_grp = f_in[ep]
                dst_grp = _copy_group(src_grp, f_out, ep)
                arrays: dict[str, np.ndarray] = {key: src_grp[key][:] for key in src_grp.keys()}

                obs = arrays["obs"]
                actions = arrays["actions"]
                robot_state = arrays["robot_state"]
                quat = arrays["object_quat_w"]

                phase_b = np.flatnonzero(obs[:, -1] < 0.5)
                if len(phase_b) == 0:
                    skipped_eps += 1
                else:
                    arm1_b = obs[phase_b, 1]
                    grip_b = obs[phase_b, 5]
                    upright_b = _upright_z_from_quat_wxyz(quat[phase_b])
                    peak_local = int(np.argmax(arm1_b))
                    peak_global = int(phase_b[peak_local])
                    peak_val = float(arm1_b[peak_local])

                    cond_idx = np.flatnonzero((arm1_b >= args.arm1_min) & (upright_b >= args.upright_min))
                    start_global = None
                    if len(cond_idx) > 0:
                        peak_window_start = max(0, peak_local - args.peak_window)
                        tail_candidates = cond_idx[cond_idx >= peak_window_start]
                        if len(tail_candidates) > 0:
                            start_global = int(phase_b[int(tail_candidates[0])])

                    retract_local = _find_retract_start(
                        arm1_b, peak_local, args.peak_drop_thresh, args.peak_drop_sustain
                    )
                    end_global = int(phase_b[retract_local - 1]) if retract_local > 0 else peak_global
                    if retract_local == len(arm1_b):
                        end_global = int(phase_b[-1])

                    if start_global is None or end_global <= start_global:
                        skipped_eps += 1
                    else:
                        tail_idx = np.arange(start_global, end_global + 1, dtype=np.int64)
                        if len(tail_idx) < args.min_tail_len or peak_val < args.arm1_min:
                            skipped_eps += 1
                        else:
                            hold_len = min(args.hold_len, max(10, len(tail_idx) // 4))
                            grip_tail_old = obs[tail_idx, 5].astype(np.float32)
                            act_tail_old = actions[tail_idx, 5].astype(np.float32)
                            rs_tail_old = robot_state[tail_idx, 5].astype(np.float32)

                            new_obs_tail = _make_monotonic_tail(
                                grip_tail_old,
                                args.obs_start_open,
                                args.obs_target_open,
                                args.obs_max_open,
                                hold_len,
                            )
                            new_act_tail = _make_monotonic_tail(
                                act_tail_old,
                                args.act_start_open,
                                args.act_target_open,
                                args.act_max_open,
                                hold_len,
                            )
                            new_rs_tail = _make_monotonic_tail(
                                rs_tail_old,
                                args.rs_start_open,
                                args.rs_target_open,
                                args.rs_max_open,
                                hold_len,
                            )

                            before_obs_max.append(float(grip_tail_old.max()))
                            after_obs_max.append(float(new_obs_tail.max()))
                            before_act_max.append(float(act_tail_old.max()))
                            after_act_max.append(float(new_act_tail.max()))
                            tail_lengths.append(int(len(tail_idx)))

                            obs[tail_idx, 5] = new_obs_tail
                            actions[tail_idx, 5] = new_act_tail
                            robot_state[tail_idx, 5] = new_rs_tail

                            arrays["obs"] = obs
                            arrays["actions"] = actions
                            arrays["robot_state"] = robot_state
                            dst_grp.attrs["release_tail_relabel"] = True
                            dst_grp.attrs["release_tail_start_idx"] = int(start_global)
                            dst_grp.attrs["release_tail_end_idx"] = int(end_global)
                            dst_grp.attrs["release_tail_len"] = int(len(tail_idx))
                            dst_grp.attrs["release_tail_peak_arm1"] = peak_val
                            modified_eps += 1

                for key, value in arrays.items():
                    dst_grp.create_dataset(key, data=value)

            f_out.attrs["release_tail_modified_episode_count"] = modified_eps
            f_out.attrs["release_tail_skipped_episode_count"] = skipped_eps
            if before_obs_max:
                f_out.attrs["release_tail_obs_grip_max_before_mean"] = float(np.mean(before_obs_max))
                f_out.attrs["release_tail_obs_grip_max_after_mean"] = float(np.mean(after_obs_max))
                f_out.attrs["release_tail_act_grip_max_before_mean"] = float(np.mean(before_act_max))
                f_out.attrs["release_tail_act_grip_max_after_mean"] = float(np.mean(after_act_max))
                f_out.attrs["release_tail_len_mean"] = float(np.mean(tail_lengths))

    print(f"Modified episodes: {modified_eps}, skipped: {skipped_eps}")
    if before_obs_max:
        print(f"Tail obs grip max mean: {np.mean(before_obs_max):.4f} -> {np.mean(after_obs_max):.4f}")
        print(f"Tail act grip max mean: {np.mean(before_act_max):.4f} -> {np.mean(after_act_max):.4f}")
        print(f"Tail length mean: {np.mean(tail_lengths):.1f}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
