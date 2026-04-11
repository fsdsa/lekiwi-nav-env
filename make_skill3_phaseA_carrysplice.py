from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _ep_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def _init_key(init6: np.ndarray) -> tuple[float, ...]:
    # These pose values were recorded as float32 and exact matches exist across
    # the two datasets. Rounding keeps dictionary keys stable.
    return tuple(np.round(np.asarray(init6, dtype=np.float32), 6).tolist())


def _copy_group(src: h5py.Group, dst_parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    dst = dst_parent.create_group(name)
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Create BC-only skill3 HDF5 with Phase-A arm/grip spliced from carry demos.")
    parser.add_argument(
        "--skill3",
        type=Path,
        default=Path("/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/demos_skill3/combined_skill3_grip_s07_v5_36d_final.hdf5"),
    )
    parser.add_argument(
        "--carry",
        type=Path,
        default=Path("/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/demos/carry_120ep_39d.hdf5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/demos_skill3/combined_skill3_grip_s07_v5_36d_final_phaseAcarrysplice.hdf5"),
    )
    args = parser.parse_args()

    with h5py.File(args.carry, "r") as f_carry:
        carry_eps = sorted(f_carry.keys(), key=_ep_sort_key)
        carry_index: dict[tuple[float, ...], list[str]] = {}
        carry_cache: dict[str, dict[str, np.ndarray]] = {}

        for ep in carry_eps:
            obs = f_carry[ep]["obs"][:]
            actions = f_carry[ep]["actions"][:]
            robot_state = f_carry[ep]["robot_state"][:]
            init6 = obs[0, -6:].astype(np.float32)
            key = _init_key(init6)
            carry_index.setdefault(key, []).append(ep)
            carry_cache[ep] = {
                "obs": obs,
                "actions": actions,
                "robot_state": robot_state,
                "init6": init6,
            }

        with h5py.File(args.skill3, "r") as f_skill, h5py.File(args.output, "w") as f_out:
            for k, v in f_skill.attrs.items():
                f_out.attrs[k] = v
            f_out.attrs["source_skill3_hdf5"] = str(args.skill3)
            f_out.attrs["source_carry_hdf5"] = str(args.carry)
            f_out.attrs["synthetic_bc_only"] = True
            f_out.attrs["phase_a_splice"] = "carry arm/grip obs+actions+robot_state"

            skill_eps = sorted(f_skill.keys(), key=_ep_sort_key)

            matched = 0
            unchanged = 0
            before_arm_l2 = []
            after_arm_l2 = []
            before_grip_abs = []
            after_grip_abs = []

            for ep in skill_eps:
                src_grp = f_skill[ep]
                dst_grp = _copy_group(src_grp, f_out, ep)

                arrays: dict[str, np.ndarray] = {
                    key: src_grp[key][:] for key in src_grp.keys()
                }

                obs = arrays["obs"]
                actions = arrays["actions"]
                robot_state = arrays["robot_state"]

                phase_a_mask = obs[:, -1] > 0.5
                phase_a_idx = np.flatnonzero(phase_a_mask)

                if len(phase_a_idx) == 0:
                    unchanged += 1
                else:
                    init6 = obs[0, -7:-1].astype(np.float32)
                    key = _init_key(init6)
                    candidates = carry_index.get(key, [])
                    if not candidates:
                        raise RuntimeError(f"No carry demo matches init pose for {ep}")

                    L = len(phase_a_idx)
                    best_ep = None
                    best_score = None
                    best_pack = None
                    for cep in candidates:
                        pack = carry_cache[cep]
                        if len(pack["obs"]) < L:
                            continue
                        rs = pack["robot_state"][:L, :6]
                        arm_l2 = np.linalg.norm(rs[:, :5] - init6[:5], axis=1).mean()
                        grip_abs = np.abs(rs[:, 5] - init6[5]).mean()
                        score = float(arm_l2 + 2.0 * grip_abs)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_ep = cep
                            best_pack = pack

                    if best_pack is None or best_ep is None:
                        raise RuntimeError(f"No long-enough carry demo for {ep} (phaseA={L})")

                    # Pre-splice stats
                    rs_old = robot_state[phase_a_idx, :6]
                    before_arm_l2.append(float(np.linalg.norm(rs_old[:, :5] - init6[:5], axis=1).mean()))
                    before_grip_abs.append(float(np.abs(rs_old[:, 5] - init6[5]).mean()))

                    # Splice only the arm/grip related channels so task/world context
                    # from skill3 stays intact while carry behavior replaces Phase A.
                    c_obs = best_pack["obs"][:L]
                    c_act = best_pack["actions"][:L]
                    c_rs = best_pack["robot_state"][:L]

                    # obs: arm/grip position [0:6], arm/grip velocity [15:21]
                    obs[phase_a_idx, 0:6] = c_obs[:, 0:6]
                    obs[phase_a_idx, 15:21] = c_obs[:, 15:21]
                    # actions: arm/grip target [0:6]
                    actions[phase_a_idx, 0:6] = c_act[:, 0:6]
                    # robot_state: arm/grip [0:6], keep base body vel [6:9] from skill3
                    robot_state[phase_a_idx, 0:6] = c_rs[:, 0:6]

                    arrays["obs"] = obs
                    arrays["actions"] = actions
                    arrays["robot_state"] = robot_state
                    dst_grp.attrs["phase_a_spliced_from_carry"] = best_ep
                    dst_grp.attrs["phase_a_splice_steps"] = int(L)
                    dst_grp.attrs["phase_a_splice_score"] = float(best_score)
                    matched += 1

                    rs_new = robot_state[phase_a_idx, :6]
                    after_arm_l2.append(float(np.linalg.norm(rs_new[:, :5] - init6[:5], axis=1).mean()))
                    after_grip_abs.append(float(np.abs(rs_new[:, 5] - init6[5]).mean()))

                for key, value in arrays.items():
                    dst_grp.create_dataset(key, data=value)

            if before_arm_l2:
                f_out.attrs["phase_a_spliced_episode_count"] = matched
                f_out.attrs["phase_a_unchanged_episode_count"] = unchanged
                f_out.attrs["phase_a_arm_l2_before_mean"] = float(np.mean(before_arm_l2))
                f_out.attrs["phase_a_arm_l2_after_mean"] = float(np.mean(after_arm_l2))
                f_out.attrs["phase_a_grip_abs_before_mean"] = float(np.mean(before_grip_abs))
                f_out.attrs["phase_a_grip_abs_after_mean"] = float(np.mean(after_grip_abs))

                print(f"Spliced episodes: {matched}, unchanged: {unchanged}")
                print(
                    f"Phase-A arm L2 mean: {np.mean(before_arm_l2):.4f} -> {np.mean(after_arm_l2):.4f}"
                )
                print(
                    f"Phase-A grip abs mean: {np.mean(before_grip_abs):.4f} -> {np.mean(after_grip_abs):.4f}"
                )
            else:
                print("No Phase-A episodes were found.")

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
