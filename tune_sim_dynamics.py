#!/usr/bin/env python3
"""
Tune sim dynamics to match real command->encoder responses.

This script performs a simple random-search SysID loop:
  real command + encoder log -> sim command replay -> error -> parameter update

Input expected from calibrate_real_robot.py:
  - wheel_radius.encoder_log + wheel_radius.command
  - base_radius.encoder_log  + base_radius.command
  - arm_sysid.tests          + {cmd,pos} trajectories

Usage:
  python tune_sim_dynamics.py \
      --calibration calibration/calibration_latest.json \
      --cmd_transform_mode real_to_sim \
      --cmd_linear_map identity \
      --cmd_lin_scale 1.0166 --cmd_ang_scale 1.2360 --cmd_wz_sign -1.0 \
      --iterations 60 --output calibration/tuned_dynamics.json --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LeKiwi sim dynamics tuner (command-based)")
parser.add_argument("--calibration", type=str, required=True)
parser.add_argument("--output", type=str, default="calibration/tuned_dynamics.json")
parser.add_argument("--iterations", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--encoder_unit", type=str, default="auto", choices=["auto", "rad", "deg", "m100"])
parser.add_argument("--fallback_vx", type=float, default=0.15)
parser.add_argument("--fallback_wz", type=float, default=1.0)
parser.add_argument(
    "--geometry_source",
    type=str,
    default="config",
    choices=["config", "calibration"],
    help="wheel/base radius source when --wheel_radius/--base_radius are not set",
)
parser.add_argument("--wheel_radius", type=float, default=None, help="override wheel radius [m]")
parser.add_argument("--base_radius", type=float, default=None, help="override base radius [m]")

parser.add_argument("--damping_min", type=float, default=0.4)
parser.add_argument("--damping_max", type=float, default=2.5)
parser.add_argument("--friction_min", type=float, default=0.0)
parser.add_argument("--friction_max", type=float, default=1.5)
parser.add_argument("--dyn_friction_min", type=float, default=0.0)
parser.add_argument("--dyn_friction_max", type=float, default=1.0)
parser.add_argument("--viscous_friction_min", type=float, default=0.0)
parser.add_argument("--viscous_friction_max", type=float, default=1.0)
parser.add_argument("--armature_min", type=float, default=0.5)
parser.add_argument("--armature_max", type=float, default=2.0)
parser.add_argument("--lin_scale_min", type=float, default=0.7)
parser.add_argument("--lin_scale_max", type=float, default=1.3)
parser.add_argument("--ang_scale_min", type=float, default=0.7)
parser.add_argument("--ang_scale_max", type=float, default=1.3)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--arm_weight", type=float, default=1.0, help="weight of arm RMSE term in objective")

parser.add_argument(
    "--cmd_transform_mode",
    type=str,
    default="none",
    choices=["none", "real_to_sim"],
    help="optional base command transform before replay",
)
parser.add_argument(
    "--cmd_linear_map",
    type=str,
    default="identity",
    choices=["identity", "flip_180", "rot_cw_90", "rot_ccw_90"],
    help="2D linear map used when --cmd_transform_mode real_to_sim",
)
parser.add_argument(
    "--cmd_lin_scale",
    type=float,
    default=1.0,
    help="linear scale used when --cmd_transform_mode real_to_sim",
)
parser.add_argument(
    "--cmd_ang_scale",
    type=float,
    default=1.0,
    help="angular scale used when --cmd_transform_mode real_to_sim",
)
parser.add_argument(
    "--cmd_wz_sign",
    type=float,
    default=1.0,
    help="wz sign used when --cmd_transform_mode real_to_sim",
)

parser.add_argument("--arm_damping_min", type=float, default=0.4)
parser.add_argument("--arm_damping_max", type=float, default=2.5)
parser.add_argument("--arm_stiffness_min", type=float, default=0.5)
parser.add_argument("--arm_stiffness_max", type=float, default=2.0)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

from lekiwi_robot_cfg import (
    ARM_JOINT_NAMES,
    BASE_RADIUS,
    LEKIWI_CFG,
    WHEEL_ANGLES_RAD,
    WHEEL_JOINT_NAMES,
    WHEEL_RADIUS,
)

# Canonical mapping (left: sim joint name, right: real motor ID)
SIM_JOINT_TO_REAL_MOTOR_ID = {
    "axle_0_joint": 8,
    "axle_1_joint": 9,
    "axle_2_joint": 7,
    "STS3215_03a_v1_4_Revolute_57": 6,
    "STS3215_03a_Wrist_Roll_v1_Revolute_55": 5,
    "STS3215_03a_v1_3_Revolute_53": 4,
    "STS3215_03a_v1_2_Revolute_51": 3,
    "STS3215_03a_v1_1_Revolute_49": 2,
    "STS3215_03a_v1_Revolute_45": 1,
}
REAL_WHEEL_ID_TO_SIM_JOINT = {
    motor_id: sim_joint
    for sim_joint, motor_id in SIM_JOINT_TO_REAL_MOTOR_ID.items()
    if sim_joint.startswith("axle_")
}
REAL_ARM_ID_TO_SIM_JOINT = {
    motor_id: sim_joint
    for sim_joint, motor_id in SIM_JOINT_TO_REAL_MOTOR_ID.items()
    if sim_joint.startswith("STS3215_")
}

LINEAR_MAPS = {
    "identity": ((1.0, 0.0), (0.0, 1.0)),
    "flip_180": ((-1.0, 0.0), (0.0, -1.0)),
    "rot_cw_90": ((0.0, 1.0), (-1.0, 0.0)),
    "rot_ccw_90": ((0.0, -1.0), (1.0, 0.0)),
}


def load_json(path: str) -> dict:
    p = Path(path).expanduser()
    matches = sorted(p.parent.glob(p.name)) if any(ch in p.name for ch in "*?[]") else [p]
    if not matches:
        raise FileNotFoundError(f"calibration file not found: {path}")
    with open(matches[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_resolved_path"] = str(matches[-1])
    return data


def _pick_positive_finite(*vals: object) -> float | None:
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 1e-8:
            return f
    return None


def resolve_motion_geometry(cal: dict) -> tuple[float, float, dict]:
    wr_cfg = float(WHEEL_RADIUS)
    br_cfg = float(BASE_RADIUS)
    wr_cal = _pick_positive_finite(cal.get("wheel_radius", {}).get("wheel_radius_m"))
    br_cal = _pick_positive_finite(cal.get("base_radius", {}).get("base_radius_m"))

    wr_arg = _pick_positive_finite(args.wheel_radius)
    br_arg = _pick_positive_finite(args.base_radius)

    if wr_arg is not None:
        wr = wr_arg
        wr_src = "arg"
    elif args.geometry_source == "calibration" and wr_cal is not None:
        wr = wr_cal
        wr_src = "calibration"
    else:
        wr = wr_cfg
        wr_src = "config"

    if br_arg is not None:
        br = br_arg
        br_src = "arg"
    elif args.geometry_source == "calibration" and br_cal is not None:
        br = br_cal
        br_src = "calibration"
    else:
        br = br_cfg
        br_src = "config"

    meta = {
        "geometry_source_arg": str(args.geometry_source),
        "wheel_radius_config": wr_cfg,
        "base_radius_config": br_cfg,
        "wheel_radius_calibration": wr_cal,
        "base_radius_calibration": br_cal,
        "wheel_radius_source": wr_src,
        "base_radius_source": br_src,
    }
    return float(wr), float(br), meta


def normalize_key(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def extract_motor_id_candidates(key: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", key)]


def infer_sim_wheel_from_real_key(key: str) -> str | None:
    nk = normalize_key(key)
    if "axle2" in nk or "frontleft" in nk or "baseleftwheel" in nk or "leftwheel" in nk:
        return "axle_2_joint"
    if "axle1" in nk or "frontright" in nk or "baserightwheel" in nk or "rightwheel" in nk:
        return "axle_1_joint"
    if "axle0" in nk or "back" in nk or "rear" in nk or "basebackwheel" in nk:
        return "axle_0_joint"
    for mid in reversed(extract_motor_id_candidates(key)):
        if mid in REAL_WHEEL_ID_TO_SIM_JOINT:
            return REAL_WHEEL_ID_TO_SIM_JOINT[mid]
    return None


def infer_sim_arm_from_real_key(key: str) -> str | None:
    nk = normalize_key(key)
    token_to_joint = {
        "armshoulderpan": "STS3215_03a_v1_Revolute_45",
        "armshoulderlift": "STS3215_03a_v1_1_Revolute_49",
        "armelbowflex": "STS3215_03a_v1_2_Revolute_51",
        "armwristflex": "STS3215_03a_v1_3_Revolute_53",
        "armwristroll": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
        "armgripper": "STS3215_03a_v1_4_Revolute_57",
        "shoulderpan": "STS3215_03a_v1_Revolute_45",
        "shoulderlift": "STS3215_03a_v1_1_Revolute_49",
        "elbow": "STS3215_03a_v1_2_Revolute_51",
        "wristflex": "STS3215_03a_v1_3_Revolute_53",
        "wristroll": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
        "gripper": "STS3215_03a_v1_4_Revolute_57",
    }
    for tok, joint in token_to_joint.items():
        if tok in nk:
            return joint
    # weak fallback for generic ordering tags
    if "joint0" in nk or "arm0" in nk:
        return "STS3215_03a_v1_Revolute_45"
    if "joint1" in nk or "arm1" in nk:
        return "STS3215_03a_v1_1_Revolute_49"
    if "joint2" in nk or "arm2" in nk:
        return "STS3215_03a_v1_2_Revolute_51"
    if "joint3" in nk or "arm3" in nk:
        return "STS3215_03a_v1_3_Revolute_53"
    if "joint4" in nk or "arm4" in nk:
        return "STS3215_03a_Wrist_Roll_v1_Revolute_55"
    if "joint5" in nk or "arm5" in nk:
        return "STS3215_03a_v1_4_Revolute_57"
    for mid in reversed(extract_motor_id_candidates(key)):
        if mid in REAL_ARM_ID_TO_SIM_JOINT:
            return REAL_ARM_ID_TO_SIM_JOINT[mid]
    return None


def _arm_sort_key(real_key: str) -> tuple[int, str]:
    ids = extract_motor_id_candidates(real_key)
    tail = ids[-1] if ids else 10**9
    return (tail, real_key)


def build_arm_order_mapping(joint_keys: list[str]) -> dict[str, str]:
    """Fallback mapping: sort real keys and map in-order to ARM_JOINT_NAMES."""
    unique = sorted({k for k in joint_keys if k}, key=_arm_sort_key)
    if len(unique) != len(ARM_JOINT_NAMES):
        return {}
    return {real_key: ARM_JOINT_NAMES[i] for i, real_key in enumerate(unique)}


def infer_encoder_unit(series: dict[str, np.ndarray], user_choice: str) -> str:
    if user_choice in ("rad", "deg", "m100"):
        return user_choice
    vals_all = []
    for vals in series.values():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals_all.append(arr)
    if not vals_all:
        return "rad"
    all_v = np.concatenate(vals_all)
    if _has_m100_wrap_signature(series):
        return "m100"
    p95_abs = float(np.percentile(np.abs(all_v), 95))
    if 60.0 <= p95_abs <= 110.0:
        return "m100"
    if p95_abs > 20.0:
        return "deg"
    return "rad"


def to_rad_array(v: np.ndarray, unit: str) -> np.ndarray:
    if unit == "deg":
        return np.deg2rad(v)
    if unit == "m100":
        return v * (np.pi / 100.0)
    return v


def to_unwrapped_rad_array(v: np.ndarray, unit: str) -> np.ndarray:
    return np.unwrap(to_rad_array(v.astype(np.float64), unit))


def resolve_encoder_unit(declared_unit: str, series: dict[str, np.ndarray], user_choice: str) -> str:
    if user_choice in ("rad", "deg", "m100"):
        return user_choice

    unit = str(declared_unit or "auto").strip().lower()
    if unit not in ("rad", "deg", "m100"):
        unit = "auto"

    inferred = infer_encoder_unit(series, "auto")
    if unit == "auto":
        return inferred

    vals_all = []
    for vals in series.values():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            vals_all.append(arr)
    if not vals_all:
        return unit
    p95_abs = float(np.percentile(np.abs(np.concatenate(vals_all)), 95))
    has_m100_wrap = _has_m100_wrap_signature(series)

    if unit == "rad" and p95_abs > 20.0:
        return inferred
    if unit == "deg" and p95_abs < 5.0:
        return inferred
    if unit == "deg" and has_m100_wrap:
        return "m100"
    if unit == "m100" and not (60.0 <= p95_abs <= 110.0):
        return inferred
    return unit


def _has_m100_wrap_signature(series: dict[str, np.ndarray]) -> bool:
    jumps = []
    p95_abs = []
    for vals in series.values():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            continue
        p95_abs.append(float(np.percentile(np.abs(arr), 95)))
        jumps.append(float(np.percentile(np.abs(np.diff(arr)), 95)))
    if not jumps or not p95_abs:
        return False
    max_p95 = float(np.max(p95_abs))
    max_jump = float(np.max(jumps))
    return (60.0 <= max_p95 <= 110.0) and (120.0 <= max_jump <= 260.0)


def _fill_nan(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    out = values.copy()
    valid = np.isfinite(out)
    if not valid.any():
        return np.zeros_like(out)
    first = np.argmax(valid)
    out[:first] = out[first]
    for i in range(first + 1, len(out)):
        if not np.isfinite(out[i]):
            out[i] = out[i - 1]
    return out


def _wz_to_rad_per_s(cmd: dict, default_wz: float) -> float:
    """Command dict의 wz 단위를 명시값 우선으로 rad/s로 정규화."""
    wz_raw = float(cmd.get("wz", default_wz))
    unit = str(cmd.get("wz_unit", "")).strip().lower()

    if unit in ("deg_per_s", "deg/s", "degps", "degree_per_s", "degrees_per_second"):
        return float(math.radians(wz_raw))
    if unit in ("rad_per_s", "rad/s", "radps", "radian_per_s", "radians_per_second"):
        return float(wz_raw)

    # Backward compatibility for legacy logs without wz_unit.
    return float(math.radians(wz_raw) if abs(wz_raw) > 8.0 else wz_raw)


def extract_command_sequences(cal: dict, fallback_vx: float, fallback_wz: float) -> list[dict]:
    sequences: list[dict] = []

    for block_name in ("wheel_radius", "base_radius"):
        block = cal.get(block_name)
        if not isinstance(block, dict):
            continue
        log = block.get("encoder_log")
        if not isinstance(log, list) or not log:
            continue

        first_enc = log[0].get("encoders", {}) if isinstance(log[0], dict) else {}
        wheel_keys = block.get("wheel_keys")
        if not isinstance(wheel_keys, list) or not wheel_keys:
            wheel_keys = sorted(first_enc.keys())
        if not wheel_keys:
            continue

        t = []
        wheel_series = {k: [] for k in wheel_keys}
        for i, entry in enumerate(log):
            if not isinstance(entry, dict):
                continue
            tt = float(entry.get("t", i * 0.02))
            enc = entry.get("encoders", {}) if isinstance(entry.get("encoders", {}), dict) else {}
            t.append(tt)
            for k in wheel_keys:
                wheel_series[k].append(float(enc.get(k, np.nan)))

        if len(t) < 3:
            continue

        t_arr = np.asarray(t, dtype=np.float64)
        order = np.argsort(t_arr)
        t_arr = t_arr[order]

        ws = {}
        for k, vals in wheel_series.items():
            arr = np.asarray(vals, dtype=np.float64)[order]
            ws[k] = _fill_nan(arr)

        cmd = block.get("command", {}) if isinstance(block.get("command", {}), dict) else {}
        if block_name == "wheel_radius":
            default_cmd = {"vx": fallback_vx, "vy": 0.0, "wz": 0.0}
        else:
            default_cmd = {"vx": 0.0, "vy": 0.0, "wz": fallback_wz}

        command = {
            "vx": float(cmd.get("vx", default_cmd["vx"])),
            "vy": float(cmd.get("vy", default_cmd["vy"])),
            "wz": float(cmd.get("wz", default_cmd["wz"])),
        }

        unit = resolve_encoder_unit(str(block.get("encoder_unit", "auto")), ws, args.encoder_unit)

        command["wz"] = _wz_to_rad_per_s(cmd, default_cmd["wz"])

        pairs = []
        for key in wheel_keys:
            sim_joint = infer_sim_wheel_from_real_key(key)
            if sim_joint not in WHEEL_JOINT_NAMES:
                continue
            pairs.append(
                {
                    "real_key": key,
                    "sim_joint": sim_joint,
                    "real_pos_rad": to_unwrapped_rad_array(ws[key], unit),
                }
            )

        if not pairs:
            continue

        sequences.append(
            {
                "name": block_name,
                "time_s": t_arr,
                "command": command,
                "pairs": pairs,
                "encoder_unit": unit,
            }
        )

    return sequences


def extract_arm_sysid_tests(cal: dict, fallback_unit: str = "auto") -> list[dict]:
    block = cal.get("arm_sysid")
    if not isinstance(block, dict):
        return []

    unit = str(block.get("unit", fallback_unit)).strip().lower()
    if unit == "auto":
        unit = "rad"
    if unit not in ("rad", "deg", "m100"):
        unit = "rad"

    tests_raw = block.get("tests", [])
    if not isinstance(tests_raw, list):
        return []

    raw_joint_keys = [
        str(item.get("joint_key", ""))
        for item in tests_raw
        if isinstance(item, dict) and str(item.get("joint_key", ""))
    ]
    order_map = build_arm_order_mapping(raw_joint_keys)

    tests: list[dict] = []
    for test_idx, item in enumerate(tests_raw):
        if not isinstance(item, dict):
            continue
        joint_key = str(item.get("joint_key", ""))
        sim_joint = infer_sim_arm_from_real_key(joint_key)
        if sim_joint not in ARM_JOINT_NAMES:
            sim_joint = order_map.get(joint_key)
        if sim_joint not in ARM_JOINT_NAMES:
            continue

        t = np.asarray(item.get("t", []), dtype=np.float64)
        cmd = np.asarray(item.get("cmd", []), dtype=np.float64)
        pos = np.asarray(item.get("pos", []), dtype=np.float64)
        if t.size < 5 or cmd.size != t.size or pos.size != t.size:
            continue

        unit_local = unit

        order = np.argsort(t)
        t = t[order]
        cmd = cmd[order]
        pos = pos[order]

        tests.append(
            {
                "name": f"arm_{item.get('type', 'unknown')}_{joint_key}_{test_idx:02d}",
                "joint_key": joint_key,
                "sim_joint": sim_joint,
                "type": str(item.get("type", "unknown")),
                "time_s": t,
                "cmd_rad": to_unwrapped_rad_array(cmd, unit_local),
                "pos_rad": to_unwrapped_rad_array(pos, unit_local),
                "unit": unit_local,
            }
        )

    return tests


def kiwi_ik(vx: float, vy: float, wz: float, wheel_radius: float, base_radius: float) -> np.ndarray:
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in WHEEL_ANGLES_RAD], dtype=np.float64)
    return m.dot(np.array([vx, vy, wz], dtype=np.float64)) / max(wheel_radius, 1e-6)


def _command_transform_meta() -> dict:
    return {
        "mode": str(args.cmd_transform_mode),
        "linear_map": str(args.cmd_linear_map),
        "lin_scale": float(args.cmd_lin_scale),
        "ang_scale": float(args.cmd_ang_scale),
        "wz_sign": float(args.cmd_wz_sign),
    }


def apply_command_transform(vx: float, vy: float, wz: float) -> tuple[float, float, float]:
    if args.cmd_transform_mode != "real_to_sim":
        return float(vx), float(vy), float(wz)

    m = LINEAR_MAPS[str(args.cmd_linear_map)]
    vx_m = m[0][0] * float(vx) + m[0][1] * float(vy)
    vy_m = m[1][0] * float(vx) + m[1][1] * float(vy)
    vx_m *= float(args.cmd_lin_scale)
    vy_m *= float(args.cmd_lin_scale)
    wz_m = float(wz) * float(args.cmd_wz_sign) * float(args.cmd_ang_scale)
    return vx_m, vy_m, wz_m


def align_and_compare(real_t: np.ndarray, real_series: np.ndarray, sim_t: np.ndarray, sim_series: np.ndarray) -> dict:
    t_end = min(float(real_t[-1]), float(sim_t[-1]))
    n = max(5, min(len(real_t), len(sim_t)))
    t_common = np.linspace(0.0, max(t_end, 1e-6), n)

    rr = np.interp(t_common, real_t, real_series)
    ss = np.interp(t_common, sim_t, sim_series)

    rr = np.unwrap(rr)
    ss = np.unwrap(ss)

    rr = rr - rr[0]
    ss = ss - ss[0]

    err = ss - rr
    return {
        "mae_rad": float(np.mean(np.abs(err))),
        "rmse_rad": float(np.sqrt(np.mean(err**2))),
        "max_err_rad": float(np.max(np.abs(err))),
        "time_s": t_common,
        "real_delta_rad": rr,
        "sim_delta_rad": ss,
    }


def align_and_compare_best_polarity(
    real_t: np.ndarray, real_series: np.ndarray, sim_t: np.ndarray, sim_series: np.ndarray
) -> dict:
    aligned = align_and_compare(real_t=real_t, real_series=real_series, sim_t=sim_t, sim_series=sim_series)
    aligned["sim_polarity"] = 1
    flipped = align_and_compare(real_t=real_t, real_series=real_series, sim_t=sim_t, sim_series=-sim_series)
    flipped["sim_polarity"] = -1
    return flipped if flipped["rmse_rad"] < aligned["rmse_rad"] else aligned


class CommandRunner:
    def __init__(self, wheel_radius: float, base_radius: float):
        self.wheel_radius = float(wheel_radius)
        self.base_radius = float(base_radius)

        sim_cfg = SimulationCfg(
            dt=0.02,
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            device=args.device,
        )
        self.sim = SimulationContext(sim_cfg)

        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
        )

        robot_cfg = LEKIWI_CFG.replace(prim_path="/World/Robot")
        self.robot = Articulation(robot_cfg)
        self.sim.reset()
        self.dt = float(self.sim.get_physics_dt())
        self.robot.update(self.dt)

        self.arm_idx = torch.tensor(self.robot.find_joints(ARM_JOINT_NAMES)[0], dtype=torch.long, device=self.robot.device)
        self.wheel_idx = torch.tensor(self.robot.find_joints(WHEEL_JOINT_NAMES)[0], dtype=torch.long, device=self.robot.device)
        self.arm_name_to_idx = {self.robot.joint_names[i]: i for i in self.arm_idx.tolist()}
        self.wheel_name_to_idx = {self.robot.joint_names[i]: i for i in self.wheel_idx.tolist()}

        self.default_root_state = self.robot.data.default_root_state.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()

        self.base_wheel_stiffness = self.robot.data.joint_stiffness[:, self.wheel_idx].clone()
        self.base_wheel_damping = self.robot.data.joint_damping[:, self.wheel_idx].clone()
        self.base_wheel_armature = self.robot.data.joint_armature[:, self.wheel_idx].clone()
        self.base_arm_stiffness = self.robot.data.joint_stiffness[:, self.arm_idx].clone()
        self.base_arm_damping = self.robot.data.joint_damping[:, self.arm_idx].clone()
        self.base_arm_armature = self.robot.data.joint_armature[:, self.arm_idx].clone()

    def close(self):
        self.sim.clear_all_callbacks()

    def reset_robot(self):
        self.robot.write_root_state_to_sim(self.default_root_state)
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.reset()
        for _ in range(10):
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.dt)

    def apply_dynamics(self, params: dict):
        wheel_ids = self.wheel_idx.tolist()
        arm_ids = self.arm_idx.tolist()

        wheel_stiff = float(params.get("wheel_stiffness_scale", 1.0))
        wheel_damp = float(params.get("wheel_damping_scale", 1.0))
        wheel_arm = float(params.get("wheel_armature_scale", 1.0))
        wheel_fc = float(params.get("wheel_friction_coeff", 0.0))
        wheel_dfc = float(params.get("wheel_dynamic_friction_coeff", 0.0))
        wheel_vfc = float(params.get("wheel_viscous_friction_coeff", 0.0))
        arm_stiff = float(params.get("arm_stiffness_scale", 1.0))
        arm_damp = float(params.get("arm_damping_scale", 1.0))
        arm_arm = float(params.get("arm_armature_scale", 1.0))

        self.robot.write_joint_stiffness_to_sim(
            self.base_wheel_stiffness * wheel_stiff,
            joint_ids=wheel_ids,
        )
        self.robot.write_joint_damping_to_sim(
            self.base_wheel_damping * wheel_damp,
            joint_ids=wheel_ids,
        )
        self.robot.write_joint_armature_to_sim(
            self.base_wheel_armature * wheel_arm,
            joint_ids=wheel_ids,
        )
        self.robot.write_joint_friction_coefficient_to_sim(
            torch.full_like(self.base_wheel_damping, wheel_fc),
            joint_ids=wheel_ids,
        )
        self.robot.write_joint_dynamic_friction_coefficient_to_sim(
            torch.full_like(self.base_wheel_damping, wheel_dfc),
            joint_ids=wheel_ids,
        )
        self.robot.write_joint_viscous_friction_coefficient_to_sim(
            torch.full_like(self.base_wheel_damping, wheel_vfc),
            joint_ids=wheel_ids,
        )
        self.robot.write_joint_stiffness_to_sim(
            self.base_arm_stiffness * arm_stiff,
            joint_ids=arm_ids,
        )
        self.robot.write_joint_damping_to_sim(
            self.base_arm_damping * arm_damp,
            joint_ids=arm_ids,
        )
        # Armature typically smaller impact for position loops, but keep parity with wheel tuning.
        self.robot.write_joint_armature_to_sim(
            self.base_arm_armature * arm_arm,
            joint_ids=arm_ids,
        )

    def run_sequence(self, command: dict, num_steps: int, lin_scale: float, ang_scale: float) -> dict:
        self.reset_robot()

        vx_base, vy_base, wz_base = apply_command_transform(
            float(command["vx"]),
            float(command["vy"]),
            float(command["wz"]),
        )
        vx = float(vx_base) * lin_scale
        vy = float(vy_base) * lin_scale
        wz = float(wz_base) * ang_scale

        arm_target = self.default_joint_pos.clone()

        t = []
        wheel_pos = {name: [] for name in self.wheel_name_to_idx}

        for _ in range(num_steps):
            wheel_radps = kiwi_ik(vx, vy, wz, self.wheel_radius, self.base_radius)

            vel_target = torch.zeros((1, self.robot.num_joints), device=self.robot.device)
            vel_target[:, self.wheel_idx] = torch.tensor(wheel_radps, dtype=torch.float32, device=self.robot.device).unsqueeze(0)
            self.robot.set_joint_velocity_target(vel_target)
            self.robot.set_joint_position_target(arm_target)

            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.dt)

            t.append((len(t) + 1) * self.dt)
            jp = self.robot.data.joint_pos[0].detach().cpu().numpy()
            for name, idx in self.wheel_name_to_idx.items():
                wheel_pos[name].append(float(jp[idx]))

        return {
            "time_s": np.asarray(t, dtype=np.float64),
            "wheel_pos_rad": {k: np.asarray(v, dtype=np.float64) for k, v in wheel_pos.items()},
        }

    def run_arm_test(self, sim_joint: str, cmd_rad: np.ndarray, time_s: np.ndarray) -> dict:
        self.reset_robot()
        idx = self.arm_name_to_idx.get(sim_joint)
        if idx is None:
            raise RuntimeError(f"Unknown sim arm joint: {sim_joint}")

        wheel_zero = torch.zeros((1, self.robot.num_joints), device=self.robot.device)
        t_out = []
        pos_out = []

        cmd0 = float(cmd_rad[0]) if len(cmd_rad) > 0 else 0.0
        sim_base = float(self.default_joint_pos[0, idx].item())

        num_steps = min(len(cmd_rad), len(time_s))
        for i in range(num_steps):
            arm_target = self.default_joint_pos.clone()
            arm_target[:, idx] = sim_base + float(cmd_rad[i] - cmd0)

            self.robot.set_joint_velocity_target(wheel_zero)
            self.robot.set_joint_position_target(arm_target)

            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.dt)

            t_out.append((i + 1) * self.dt)
            pos_out.append(float(self.robot.data.joint_pos[0, idx].item()))

        return {
            "time_s": np.asarray(t_out, dtype=np.float64),
            "pos_rad": np.asarray(pos_out, dtype=np.float64),
        }


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "wheel_stiffness_scale": 1.0,
        "wheel_damping_scale": float(rng.uniform(args.damping_min, args.damping_max)),
        "wheel_armature_scale": float(rng.uniform(args.armature_min, args.armature_max)),
        "wheel_friction_coeff": float(rng.uniform(args.friction_min, args.friction_max)),
        "wheel_dynamic_friction_coeff": float(rng.uniform(args.dyn_friction_min, args.dyn_friction_max)),
        "wheel_viscous_friction_coeff": float(rng.uniform(args.viscous_friction_min, args.viscous_friction_max)),
        "lin_cmd_scale": float(rng.uniform(args.lin_scale_min, args.lin_scale_max)),
        "ang_cmd_scale": float(rng.uniform(args.ang_scale_min, args.ang_scale_max)),
        "arm_stiffness_scale": float(rng.uniform(args.arm_stiffness_min, args.arm_stiffness_max)),
        "arm_damping_scale": float(rng.uniform(args.arm_damping_min, args.arm_damping_max)),
        "arm_armature_scale": float(rng.uniform(args.armature_min, args.armature_max)),
    }


def evaluate_candidate(runner: CommandRunner, sequences: list[dict], arm_tests: list[dict], params: dict) -> dict:
    runner.apply_dynamics(params)

    seq_reports = []
    arm_reports = []
    mae_list = []
    rmse_sq = []

    for seq in sequences:
        sim_trace = runner.run_sequence(
            command=seq["command"],
            num_steps=len(seq["time_s"]),
            lin_scale=params["lin_cmd_scale"],
            ang_scale=params["ang_cmd_scale"],
        )

        pair_reports = []
        for pair in seq["pairs"]:
            sim_joint = pair["sim_joint"]
            if sim_joint not in sim_trace["wheel_pos_rad"]:
                continue

            aligned = align_and_compare_best_polarity(
                real_t=seq["time_s"],
                real_series=np.asarray(pair["real_pos_rad"], dtype=np.float64),
                sim_t=sim_trace["time_s"],
                sim_series=np.asarray(sim_trace["wheel_pos_rad"][sim_joint], dtype=np.float64),
            )

            pair_reports.append(
                {
                    "real_key": pair["real_key"],
                    "sim_joint": sim_joint,
                    "mae_rad": aligned["mae_rad"],
                    "rmse_rad": aligned["rmse_rad"],
                    "max_err_rad": aligned["max_err_rad"],
                    "sim_polarity": int(aligned["sim_polarity"]),
                    "time_s": aligned["time_s"],
                    "real_delta_rad": aligned["real_delta_rad"],
                    "sim_delta_rad": aligned["sim_delta_rad"],
                }
            )
            mae_list.append(aligned["mae_rad"])
            rmse_sq.append(aligned["rmse_rad"] ** 2)

        seq_reports.append({"name": seq["name"], "pairs": pair_reports})

    arm_mae_list = []
    arm_rmse_sq = []
    for test in arm_tests:
        sim_trace = runner.run_arm_test(
            sim_joint=test["sim_joint"],
            cmd_rad=np.asarray(test["cmd_rad"], dtype=np.float64),
            time_s=np.asarray(test["time_s"], dtype=np.float64),
        )

        aligned = align_and_compare(
            real_t=np.asarray(test["time_s"], dtype=np.float64),
            real_series=np.asarray(test["pos_rad"], dtype=np.float64),
            sim_t=np.asarray(sim_trace["time_s"], dtype=np.float64),
            sim_series=np.asarray(sim_trace["pos_rad"], dtype=np.float64),
        )
        arm_reports.append(
            {
                "name": test["name"],
                "joint_key": test["joint_key"],
                "sim_joint": test["sim_joint"],
                "type": test["type"],
                "mae_rad": aligned["mae_rad"],
                "rmse_rad": aligned["rmse_rad"],
                "max_err_rad": aligned["max_err_rad"],
                "time_s": aligned["time_s"],
                "real_delta_rad": aligned["real_delta_rad"],
                "sim_delta_rad": aligned["sim_delta_rad"],
            }
        )
        arm_mae_list.append(aligned["mae_rad"])
        arm_rmse_sq.append(aligned["rmse_rad"] ** 2)

    mean_mae = float(np.mean(mae_list)) if mae_list else float("inf")
    mean_rmse = float(np.sqrt(np.mean(rmse_sq))) if rmse_sq else float("inf")
    arm_mean_mae = float(np.mean(arm_mae_list)) if arm_mae_list else float("inf")
    arm_mean_rmse = float(np.sqrt(np.mean(arm_rmse_sq))) if arm_rmse_sq else float("inf")

    terms = []
    if np.isfinite(mean_rmse):
        terms.append(mean_rmse)
    if np.isfinite(arm_mean_rmse):
        terms.append(arm_mean_rmse * float(args.arm_weight))
    score = float(np.mean(terms)) if terms else float("inf")

    return {
        "score": score,
        "mean_mae_rad": mean_mae,
        "mean_rmse_rad": mean_rmse,
        "arm_mean_mae_rad": arm_mean_mae,
        "arm_mean_rmse_rad": arm_mean_rmse,
        "sequences": seq_reports,
        "arm_tests": arm_reports,
    }


def main():
    cal = load_json(args.calibration)
    wr, br, geom_meta = resolve_motion_geometry(cal)

    sequences = extract_command_sequences(cal, fallback_vx=args.fallback_vx, fallback_wz=args.fallback_wz)
    arm_tests = extract_arm_sysid_tests(cal, fallback_unit=args.encoder_unit)
    if not sequences and not arm_tests:
        raise RuntimeError(
            "No usable wheel/arm sysid data found. "
            "Run calibrate_real_robot.py --mode wheel_radius/base_radius and/or --mode arm_sysid first."
        )

    print("=" * 72)
    print("LeKiwi sim dynamics tuning")
    print(f"calibration: {cal.get('_resolved_path', args.calibration)}")
    print(f"sequences: {[s['name'] for s in sequences]}")
    print(f"arm_tests: {len(arm_tests)}")
    print(f"iterations: {args.iterations}")
    print(
        f"wheel_radius={wr:.6f} ({geom_meta['wheel_radius_source']}, "
        f"cfg={geom_meta['wheel_radius_config']:.6f}, cal={geom_meta['wheel_radius_calibration']}), "
        f"base_radius={br:.6f} ({geom_meta['base_radius_source']}, "
        f"cfg={geom_meta['base_radius_config']:.6f}, cal={geom_meta['base_radius_calibration']})"
    )
    print(f"command_transform={_command_transform_meta()}")
    print("=" * 72)

    runner = CommandRunner(wheel_radius=wr, base_radius=br)
    rng = np.random.default_rng(args.seed)

    tried = []

    baseline_params = {
        "wheel_stiffness_scale": 1.0,
        "wheel_damping_scale": 1.0,
        "wheel_armature_scale": 1.0,
        "wheel_friction_coeff": 0.0,
        "wheel_dynamic_friction_coeff": 0.0,
        "wheel_viscous_friction_coeff": 0.0,
        "lin_cmd_scale": 1.0,
        "ang_cmd_scale": 1.0,
        "arm_stiffness_scale": 1.0,
        "arm_damping_scale": 1.0,
        "arm_armature_scale": 1.0,
    }

    baseline_eval = evaluate_candidate(runner, sequences, arm_tests, baseline_params)
    tried.append({"iter": -1, "params": baseline_params, "eval": baseline_eval})
    print(
        f"iter -1 | score={baseline_eval['score']:.6f} "
        f"wheel_rmse={baseline_eval['mean_rmse_rad']:.6f} "
        f"arm_rmse={baseline_eval['arm_mean_rmse_rad']:.6f} (baseline)"
    )

    for i in range(args.iterations):
        params = sample_params(rng)
        eva = evaluate_candidate(runner, sequences, arm_tests, params)
        tried.append({"iter": i, "params": params, "eval": eva})
        print(
            f"iter {i:03d} | score={eva['score']:.6f} "
            f"wheel_rmse={eva['mean_rmse_rad']:.6f} arm_rmse={eva['arm_mean_rmse_rad']:.6f}"
        )

    tried.sort(key=lambda x: x["eval"]["score"])
    best = tried[0]
    top_k = tried[: max(1, args.top_k)]

    out = {
        "calibration_path": cal.get("_resolved_path", args.calibration),
        "seed": args.seed,
        "iterations": args.iterations,
        "arm_weight": float(args.arm_weight),
        "geometry": geom_meta,
        "command_transform": _command_transform_meta(),
        "wheel_radius_used": float(wr),
        "base_radius_used": float(br),
        "best_score": float(best["eval"]["score"]),
        "best_params": best["params"],
        "best_eval": {
            "mean_mae_rad": float(best["eval"]["mean_mae_rad"]),
            "mean_rmse_rad": float(best["eval"]["mean_rmse_rad"]),
            "arm_mean_mae_rad": float(best["eval"]["arm_mean_mae_rad"]),
            "arm_mean_rmse_rad": float(best["eval"]["arm_mean_rmse_rad"]),
            "sequences": [
                {
                    "name": s["name"],
                    "pairs": [
                        {
                            "real_key": p["real_key"],
                            "sim_joint": p["sim_joint"],
                            "mae_rad": float(p["mae_rad"]),
                            "rmse_rad": float(p["rmse_rad"]),
                            "max_err_rad": float(p["max_err_rad"]),
                            "time_s": p["time_s"].tolist(),
                            "real_delta_rad": p["real_delta_rad"].tolist(),
                            "sim_delta_rad": p["sim_delta_rad"].tolist(),
                        }
                        for p in s["pairs"]
                    ],
                }
                for s in best["eval"]["sequences"]
            ],
            "arm_tests": [
                {
                    "name": t["name"],
                    "joint_key": t["joint_key"],
                    "sim_joint": t["sim_joint"],
                    "type": t["type"],
                    "mae_rad": float(t["mae_rad"]),
                    "rmse_rad": float(t["rmse_rad"]),
                    "max_err_rad": float(t["max_err_rad"]),
                    "time_s": t["time_s"].tolist(),
                    "real_delta_rad": t["real_delta_rad"].tolist(),
                    "sim_delta_rad": t["sim_delta_rad"].tolist(),
                }
                for t in best["eval"]["arm_tests"]
            ],
        },
        "top_k": [
            {
                "iter": item["iter"],
                "score": float(item["eval"]["score"]),
                "mean_mae_rad": float(item["eval"]["mean_mae_rad"]),
                "mean_rmse_rad": float(item["eval"]["mean_rmse_rad"]),
                "arm_mean_mae_rad": float(item["eval"]["arm_mean_mae_rad"]),
                "arm_mean_rmse_rad": float(item["eval"]["arm_mean_rmse_rad"]),
                "params": item["params"],
            }
            for item in top_k
        ],
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Tuning done")
    print(f"best score: {out['best_score']:.6f}")
    print(f"best params: {out['best_params']}")
    print(f"output: {out_path}")
    print("=" * 72)

    runner.close()
    sim_app.close()


if __name__ == "__main__":
    main()
