#!/usr/bin/env python3
"""
Isaac Sim Replay for LeKiwi calibration.

Modes:
  1) position (legacy): replay recorded joint positions directly
  2) command (new): replay recorded base commands into sim, compare encoder response

Usage:
  python replay_in_sim.py --calibration calibration/calibration_latest.json --mode position
  python replay_in_sim.py --calibration calibration/calibration_latest.json --mode command --headless
  python replay_in_sim.py --calibration calibration/calibration_latest.json --mode arm_command --headless
  python replay_in_sim.py --calibration calibration/calibration_latest.json --mode command \
      --dynamics_json calibration/tuned_dynamics.json --series_path calibration/replay_series.json \
      --cmd_transform_mode auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LeKiwi Sim Replay (calibration verification)")
parser.add_argument("--calibration", type=str, required=True, help="calibration JSON path")
parser.add_argument("--mode", type=str, default="position", choices=["position", "command", "arm_command"])
parser.add_argument("--mapping_json", type=str, default=None, help="real_key -> sim_joint_name mapping JSON")
parser.add_argument("--joint_unit", type=str, default="auto", choices=["auto", "rad", "deg"])
parser.add_argument("--encoder_unit", type=str, default="auto", choices=["auto", "rad", "deg", "m100"])
parser.add_argument("--report_path", type=str, default=None)
parser.add_argument("--series_path", type=str, default=None, help="save aligned real/sim time-series JSON")
parser.add_argument("--wheel_radius", type=float, default=None)
parser.add_argument("--base_radius", type=float, default=None)
parser.add_argument(
    "--geometry_source",
    type=str,
    default="config",
    choices=["config", "calibration"],
    help="wheel/base radius source when --wheel_radius/--base_radius are not set",
)
parser.add_argument("--fallback_vx", type=float, default=0.15, help="used if command metadata is missing")
parser.add_argument("--fallback_wz", type=float, default=1.0, help="used if command metadata is missing")
parser.add_argument("--lin_cmd_scale", type=float, default=None, help="override linear cmd scale (default: from dynamics_json)")
parser.add_argument("--ang_cmd_scale", type=float, default=None, help="override angular cmd scale (default: from dynamics_json)")
parser.add_argument("--dynamics_json", type=str, default=None, help="optional tuned dynamics JSON")
parser.add_argument("--arm_limit_json", type=str, default=None, help="optional arm joint limit JSON (real2sim calibration)")
parser.add_argument("--arm_limit_margin_rad", type=float, default=0.0, help="margin added to arm limits from --arm_limit_json")
parser.add_argument(
    "--cmd_transform_mode",
    type=str,
    default="auto",
    choices=["auto", "none", "real_to_sim"],
    help="base command transform mode before replay",
)
parser.add_argument(
    "--cmd_linear_map",
    type=str,
    default="auto",
    choices=["auto", "identity", "flip_180", "rot_cw_90", "rot_ccw_90"],
    help="2D linear map for --cmd_transform_mode real_to_sim",
)
parser.add_argument("--cmd_lin_scale", type=float, default=None, help="linear scale for command transform")
parser.add_argument("--cmd_ang_scale", type=float, default=None, help="angular scale for command transform")
parser.add_argument("--cmd_wz_sign", type=float, default=None, help="wz sign for command transform")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg
from lekiwi_robot_cfg import (
    ARM_JOINT_NAMES,
    BASE_RADIUS,
    LEKIWI_CFG,
    WHEEL_ANGLES_RAD,
    WHEEL_JOINT_NAMES,
    WHEEL_RADIUS,
)


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
    target = matches[-1]
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_resolved_path"] = str(target)
    return data


def _pick_positive_finite(*vals: object) -> float | None:
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f) and f > 1e-8:
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


def _to_float_or_default(v: object, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return float(f)


def _default_command_transform() -> dict:
    return {
        "mode": "none",
        "linear_map": "identity",
        "lin_scale": 1.0,
        "ang_scale": 1.0,
        "wz_sign": 1.0,
    }


def _normalize_command_transform(raw: dict | None) -> dict:
    cfg = _default_command_transform()
    if not isinstance(raw, dict):
        return cfg

    mode = str(raw.get("mode", cfg["mode"])).strip().lower()
    if mode not in ("none", "real_to_sim"):
        mode = cfg["mode"]

    linear_map = str(raw.get("linear_map", cfg["linear_map"])).strip().lower()
    if linear_map not in LINEAR_MAPS:
        linear_map = cfg["linear_map"]

    cfg["mode"] = mode
    cfg["linear_map"] = linear_map
    cfg["lin_scale"] = _to_float_or_default(raw.get("lin_scale"), cfg["lin_scale"])
    cfg["ang_scale"] = _to_float_or_default(raw.get("ang_scale"), cfg["ang_scale"])
    cfg["wz_sign"] = _to_float_or_default(raw.get("wz_sign"), cfg["wz_sign"])
    return cfg


def resolve_command_transform(dynamics_payload: dict | None) -> dict:
    dyn_cfg = _default_command_transform()
    if isinstance(dynamics_payload, dict):
        dyn_cfg = _normalize_command_transform(dynamics_payload.get("command_transform"))

    mode_arg = str(args.cmd_transform_mode).strip().lower()
    if mode_arg == "auto":
        mode = str(dyn_cfg["mode"])
    else:
        mode = mode_arg
    if mode not in ("none", "real_to_sim"):
        mode = "none"

    linear_map_arg = str(args.cmd_linear_map).strip().lower()
    linear_map = dyn_cfg["linear_map"] if linear_map_arg == "auto" else linear_map_arg
    if linear_map not in LINEAR_MAPS:
        linear_map = "identity"

    lin_scale = dyn_cfg["lin_scale"] if args.cmd_lin_scale is None else _to_float_or_default(args.cmd_lin_scale, 1.0)
    ang_scale = dyn_cfg["ang_scale"] if args.cmd_ang_scale is None else _to_float_or_default(args.cmd_ang_scale, 1.0)
    wz_sign = dyn_cfg["wz_sign"] if args.cmd_wz_sign is None else _to_float_or_default(args.cmd_wz_sign, 1.0)

    return {
        "mode": mode,
        "linear_map": linear_map,
        "lin_scale": float(lin_scale),
        "ang_scale": float(ang_scale),
        "wz_sign": float(wz_sign),
    }


def apply_command_transform(vx: float, vy: float, wz: float, cfg: dict) -> tuple[float, float, float]:
    if str(cfg.get("mode", "none")) != "real_to_sim":
        return float(vx), float(vy), float(wz)

    linear_map = str(cfg.get("linear_map", "identity"))
    if linear_map not in LINEAR_MAPS:
        linear_map = "identity"
    m = LINEAR_MAPS[linear_map]

    vx_m = m[0][0] * float(vx) + m[0][1] * float(vy)
    vy_m = m[1][0] * float(vx) + m[1][1] * float(vy)
    vx_m *= _to_float_or_default(cfg.get("lin_scale"), 1.0)
    vy_m *= _to_float_or_default(cfg.get("lin_scale"), 1.0)
    wz_m = float(wz) * _to_float_or_default(cfg.get("wz_sign"), 1.0) * _to_float_or_default(cfg.get("ang_scale"), 1.0)
    return float(vx_m), float(vy_m), float(wz_m)


def _extract_arm_limits_payload(payload: dict) -> dict[str, tuple[float, float]]:
    if not isinstance(payload, dict):
        return {}
    block = payload.get("joint_limits_rad")
    if block is None:
        block = payload.get("arm_joint_limits_rad", payload)
    if not isinstance(block, dict):
        return {}

    out: dict[str, tuple[float, float]] = {}
    for sim_joint, val in block.items():
        lo, hi = None, None
        if isinstance(val, dict) and "min" in val and "max" in val:
            lo = val.get("min")
            hi = val.get("max")
        elif isinstance(val, (list, tuple)) and len(val) >= 2:
            lo, hi = val[0], val[1]
        if lo is None or hi is None:
            continue
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lo_f) or not np.isfinite(hi_f):
            continue
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        if abs(hi_f - lo_f) < 1e-8:
            continue
        out[str(sim_joint)] = (lo_f, hi_f)
    return out


def _load_arm_limits(path: str | None) -> dict[str, tuple[float, float]]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"arm_limit_json not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    limits = _extract_arm_limits_payload(payload)
    if not limits:
        raise ValueError(f"arm_limit_json has no valid limits: {p}")
    return limits


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


def build_auto_mapping(real_keys: list[str], sim_joint_names: list[str]) -> dict[str, str]:
    sim_norm = {name: normalize_key(name) for name in sim_joint_names}
    inv_norm = {v: k for k, v in sim_norm.items()}
    mapping: dict[str, str] = {}

    for rk in real_keys:
        if rk in sim_joint_names:
            mapping[rk] = rk
            continue
        nk = normalize_key(rk)
        if nk in inv_norm:
            mapping[rk] = inv_norm[nk]
            continue
        sj = infer_sim_wheel_from_real_key(rk)
        if sj is not None:
            mapping[rk] = sj

    token_to_sim = {
        "shoulderpan": "STS3215_03a_v1_Revolute_45",
        "shoulderlift": "STS3215_03a_v1_1_Revolute_49",
        "elbow": "STS3215_03a_v1_2_Revolute_51",
        "wristflex": "STS3215_03a_v1_3_Revolute_53",
        "wristroll": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
        "gripper": "STS3215_03a_v1_4_Revolute_57",
    }
    for rk in real_keys:
        if rk in mapping:
            continue
        nk = normalize_key(rk)
        for tok, sj in token_to_sim.items():
            if tok in nk:
                mapping[rk] = sj
                break

    return mapping


def load_mapping_override(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("mapping_json must be a dict(real_key->sim_joint_name)")
    return {str(k): str(v) for k, v in raw.items()}


def infer_joint_unit(frames: list[dict], mapped_real_keys: list[str], user_choice: str) -> str:
    if user_choice in ("rad", "deg", "m100"):
        return user_choice
    deltas = []
    for i in range(1, len(frames)):
        prev = frames[i - 1].get("positions", {})
        cur = frames[i].get("positions", {})
        for k in mapped_real_keys:
            if k in prev and k in cur:
                d = abs(float(cur[k]) - float(prev[k]))
                if np.isfinite(d):
                    deltas.append(d)
    if not deltas:
        return "rad"
    p95 = float(np.percentile(deltas, 95))
    if 2.0 <= p95 <= 15.0:
        return "m100"
    return "deg" if p95 > 10.0 else "rad"


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
    # Wrapped encoder streams that live in [-100, 100] typically use m100.
    if _has_m100_wrap_signature(series):
        return "m100"
    p95_abs = float(np.percentile(np.abs(all_v), 95))
    if 60.0 <= p95_abs <= 110.0:
        return "m100"
    if p95_abs > 20.0:
        return "deg"
    return "rad"


def to_rad(v: float, unit: str) -> float:
    if unit == "deg":
        return float(np.deg2rad(v))
    if unit == "m100":
        return float(v * (np.pi / 100.0))
    return float(v)


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
    # Some historical files labeled wrapped [-100, 100] streams as deg.
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
    # m100 wrapped streams: values close to +/-100 with occasional ~200 jumps.
    return (60.0 <= max_p95 <= 110.0) and (120.0 <= max_jump <= 260.0)


def kiwi_ik(vx: float, vy: float, wz: float, wheel_radius: float, base_radius: float) -> np.ndarray:
    m = np.array([[np.cos(a), np.sin(a), base_radius] for a in WHEEL_ANGLES_RAD], dtype=np.float64)
    return m.dot(np.array([vx, vy, wz], dtype=np.float64)) / max(wheel_radius, 1e-6)


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
            "duration_s": float(cmd.get("duration_s", t_arr[-1])),
            "sample_hz": float(cmd.get("sample_hz", 1.0 / max(np.median(np.diff(t_arr)), 1e-6))),
        }

        command["wz"] = _wz_to_rad_per_s(cmd, default_cmd["wz"])

        sequences.append(
            {
                "name": block_name,
                "time_s": t_arr,
                "wheel_series": ws,
                "wheel_keys": wheel_keys,
                "encoder_unit": resolve_encoder_unit(str(block.get("encoder_unit", "auto")), ws, args.encoder_unit),
                "command": command,
            }
        )

    return sequences


def extract_arm_sysid_tests(cal: dict, fallback_unit: str = "auto") -> list[dict]:
    block = cal.get("arm_sysid")
    if not isinstance(block, dict):
        return []

    unit = str(block.get("unit", fallback_unit))
    if unit == "auto":
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


class ArticulationCommandRunner:
    def __init__(
        self,
        wheel_radius: float,
        base_radius: float,
        arm_limits: dict[str, tuple[float, float]] | None = None,
        arm_limit_margin_rad: float = 0.0,
    ):
        self.wheel_radius = float(wheel_radius)
        self.base_radius = float(base_radius)

        sim_cfg = SimulationCfg(
            dt=0.02,
            render_interval=1,
            gravity=(0.0, 0.0, -9.81),
            device=args.device,
        )
        self.sim = SimulationContext(sim_cfg)
        if not args.headless:
            self.sim.set_camera_view(eye=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.3))

        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=1800.0, color=(0.9, 0.9, 0.9)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=1800.0, color=(0.9, 0.9, 0.9))
        )

        robot_cfg = LEKIWI_CFG.replace(prim_path="/World/Robot")
        self.robot = Articulation(robot_cfg)

        self.sim.reset()
        self.dt = float(self.sim.get_physics_dt())
        self.robot.update(self.dt)

        self.arm_idx = torch.tensor(self.robot.find_joints(ARM_JOINT_NAMES)[0], dtype=torch.long, device=self.robot.device)
        self.wheel_idx = torch.tensor(self.robot.find_joints(WHEEL_JOINT_NAMES)[0], dtype=torch.long, device=self.robot.device)
        self.arm_names = [self.robot.joint_names[i] for i in self.arm_idx.tolist()]
        self.arm_name_to_idx = {self.robot.joint_names[i]: i for i in self.arm_idx.tolist()}
        self.wheel_names = [self.robot.joint_names[i] for i in self.wheel_idx.tolist()]

        if arm_limits:
            self._apply_arm_limits(arm_limits, margin_rad=float(arm_limit_margin_rad))

        self.default_root_state = self.robot.data.default_root_state.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()

        self.base_wheel_stiffness = self.robot.data.joint_stiffness[:, self.wheel_idx].clone()
        self.base_wheel_damping = self.robot.data.joint_damping[:, self.wheel_idx].clone()
        self.base_wheel_armature = self.robot.data.joint_armature[:, self.wheel_idx].clone()
        self.base_arm_stiffness = self.robot.data.joint_stiffness[:, self.arm_idx].clone()
        self.base_arm_damping = self.robot.data.joint_damping[:, self.arm_idx].clone()
        self.base_arm_armature = self.robot.data.joint_armature[:, self.arm_idx].clone()

    def _apply_arm_limits(self, arm_limits: dict[str, tuple[float, float]], margin_rad: float):
        joint_ids = []
        limits_rows = []
        min_limit = -2.0 * np.pi + 1e-6
        max_limit = 2.0 * np.pi - 1e-6
        for sim_joint, (lo, hi) in arm_limits.items():
            idx = self.arm_name_to_idx.get(sim_joint)
            if idx is None:
                continue
            l = float(min(lo, hi) - margin_rad)
            h = float(max(lo, hi) + margin_rad)
            l = max(l, min_limit)
            h = min(h, max_limit)
            if (not np.isfinite(l)) or (not np.isfinite(h)) or abs(h - l) < 1e-8:
                continue
            joint_ids.append(int(idx))
            limits_rows.append([l, h])

        if not joint_ids:
            return
        limits = torch.tensor(limits_rows, dtype=torch.float32, device=self.robot.device).unsqueeze(0)
        limits = limits.repeat(self.robot.num_instances, 1, 1)
        self.robot.write_joint_position_limit_to_sim(limits, joint_ids=joint_ids, warn_limit_violation=False)
        print(f"  [ArmLimits] applied to runner: {len(joint_ids)} joints (margin={margin_rad:.4f} rad)")

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

    def apply_dynamics(self, params: dict | None):
        if not params:
            return

        joint_ids = self.wheel_idx.tolist()
        arm_ids = self.arm_idx.tolist()

        if "wheel_stiffness_scale" in params:
            scale = float(params["wheel_stiffness_scale"])
            self.robot.write_joint_stiffness_to_sim(self.base_wheel_stiffness * scale, joint_ids=joint_ids)

        if "wheel_damping_scale" in params:
            scale = float(params["wheel_damping_scale"])
            self.robot.write_joint_damping_to_sim(self.base_wheel_damping * scale, joint_ids=joint_ids)

        if "wheel_armature_scale" in params:
            scale = float(params["wheel_armature_scale"])
            self.robot.write_joint_armature_to_sim(self.base_wheel_armature * scale, joint_ids=joint_ids)

        if "wheel_friction_coeff" in params:
            val = float(params["wheel_friction_coeff"])
            self.robot.write_joint_friction_coefficient_to_sim(
                torch.full_like(self.base_wheel_damping, val), joint_ids=joint_ids
            )

        if "wheel_dynamic_friction_coeff" in params:
            val = float(params["wheel_dynamic_friction_coeff"])
            self.robot.write_joint_dynamic_friction_coefficient_to_sim(
                torch.full_like(self.base_wheel_damping, val), joint_ids=joint_ids
            )

        if "wheel_viscous_friction_coeff" in params:
            val = float(params["wheel_viscous_friction_coeff"])
            self.robot.write_joint_viscous_friction_coefficient_to_sim(
                torch.full_like(self.base_wheel_damping, val), joint_ids=joint_ids
            )

        if "arm_stiffness_scale" in params:
            scale = float(params["arm_stiffness_scale"])
            self.robot.write_joint_stiffness_to_sim(self.base_arm_stiffness * scale, joint_ids=arm_ids)

        if "arm_damping_scale" in params:
            scale = float(params["arm_damping_scale"])
            self.robot.write_joint_damping_to_sim(self.base_arm_damping * scale, joint_ids=arm_ids)

        if "arm_armature_scale" in params:
            scale = float(params["arm_armature_scale"])
            self.robot.write_joint_armature_to_sim(self.base_arm_armature * scale, joint_ids=arm_ids)

    def run_sequence(self, seq: dict, lin_cmd_scale: float, ang_cmd_scale: float, cmd_transform: dict) -> dict:
        self.reset_robot()

        steps = len(seq["time_s"])
        if steps < 2:
            raise RuntimeError(f"sequence '{seq['name']}' is too short")

        cmd = seq["command"]
        vx_base, vy_base, wz_base = apply_command_transform(
            float(cmd["vx"]),
            float(cmd["vy"]),
            float(cmd["wz"]),
            cfg=cmd_transform,
        )
        vx = float(vx_base) * lin_cmd_scale
        vy = float(vy_base) * lin_cmd_scale
        wz = float(wz_base) * ang_cmd_scale

        arm_target = self.default_joint_pos.clone()

        t = []
        wheel_pos = {name: [] for name in self.wheel_names}

        for _ in range(steps):
            wheel_radps = kiwi_ik(vx, vy, wz, self.wheel_radius, self.base_radius)

            vel_target = torch.zeros((1, self.robot.num_joints), device=self.robot.device)
            vel_target[:, self.wheel_idx] = torch.tensor(wheel_radps, dtype=torch.float32, device=self.robot.device).unsqueeze(0)
            self.robot.set_joint_velocity_target(vel_target)

            self.robot.set_joint_position_target(arm_target)

            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.dt)

            t.append((len(t) + 1) * self.dt)
            jp = self.robot.data.joint_pos[0, self.wheel_idx].detach().cpu().numpy()
            for i, name in enumerate(self.wheel_names):
                wheel_pos[name].append(float(jp[i]))

        return {
            "time_s": np.asarray(t, dtype=np.float64),
            "wheel_pos_rad": {k: np.asarray(v, dtype=np.float64) for k, v in wheel_pos.items()},
            "command": {"vx": vx, "vy": vy, "wz": wz},
            "command_real": {
                "vx": float(cmd["vx"]),
                "vy": float(cmd["vy"]),
                "wz": float(cmd["wz"]),
            },
            "command_transformed": {"vx": vx_base, "vy": vy_base, "wz": wz_base},
        }

    def run_arm_sequence(self, test: dict) -> dict:
        self.reset_robot()

        sim_joint = str(test["sim_joint"])
        idx = self.arm_name_to_idx.get(sim_joint)
        if idx is None:
            raise RuntimeError(f"unknown sim arm joint: {sim_joint}")

        cmd = np.asarray(test["cmd_rad"], dtype=np.float64)
        t_in = np.asarray(test["time_s"], dtype=np.float64)
        steps = min(len(cmd), len(t_in))
        if steps < 2:
            raise RuntimeError(f"arm test '{test['name']}' is too short")

        # Real arm sysid commands are recorded around the real robot's rest pose.
        # Replay in sim should apply the same *delta* around sim default joint pose.
        cmd0 = float(cmd[0])
        sim_base = float(self.default_joint_pos[0, idx].item())

        t = []
        pos = []
        wheel_zero = torch.zeros((1, self.robot.num_joints), device=self.robot.device)

        for i in range(steps):
            arm_target = self.default_joint_pos.clone()
            arm_target[:, idx] = sim_base + float(cmd[i] - cmd0)
            self.robot.set_joint_velocity_target(wheel_zero)
            self.robot.set_joint_position_target(arm_target)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.dt)

            t.append((i + 1) * self.dt)
            pos.append(float(self.robot.data.joint_pos[0, idx].item()))

        return {
            "time_s": np.asarray(t, dtype=np.float64),
            "pos_rad": np.asarray(pos, dtype=np.float64),
            "command": {"sim_joint": sim_joint},
        }


def align_and_compare(
    real_t: np.ndarray,
    real_series: np.ndarray,
    sim_t: np.ndarray,
    sim_series: np.ndarray,
) -> dict:
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
        "time_s": t_common,
        "real_delta_rad": rr,
        "sim_delta_rad": ss,
        "mae_rad": float(np.mean(np.abs(err))),
        "rmse_rad": float(np.sqrt(np.mean(err**2))),
        "max_err_rad": float(np.max(np.abs(err))),
    }


def align_and_compare_best_polarity(
    real_t: np.ndarray,
    real_series: np.ndarray,
    sim_t: np.ndarray,
    sim_series: np.ndarray,
) -> dict:
    aligned = align_and_compare(real_t=real_t, real_series=real_series, sim_t=sim_t, sim_series=sim_series)
    aligned["sim_polarity"] = 1
    flipped = align_and_compare(real_t=real_t, real_series=real_series, sim_t=sim_t, sim_series=-sim_series)
    flipped["sim_polarity"] = -1
    return flipped if flipped["rmse_rad"] < aligned["rmse_rad"] else aligned


def run_position_mode(cal: dict, wr: float, br: float) -> tuple[dict, dict | None]:
    print("\n[Mode] position replay")

    traj_block = cal.get("trajectory")
    if not isinstance(traj_block, dict) or not traj_block.get("trajectory"):
        print("  trajectory가 없습니다. (calibrate_real_robot.py --mode record_trajectory)")
        return {"mode": "position", "error": "trajectory_missing"}, None

    frames: list[dict] = traj_block["trajectory"]
    fps = float(traj_block.get("fps", 25.0))
    print(f"  trajectory frames: {len(frames)} @ {fps:.2f} fps")

    real_keys = sorted(frames[0].get("positions", {}).keys())
    sim_joint_names = ARM_JOINT_NAMES + WHEEL_JOINT_NAMES

    mapping = build_auto_mapping(real_keys, sim_joint_names)
    mapping.update(load_mapping_override(args.mapping_json))
    valid_mapping = {rk: sj for rk, sj in mapping.items() if sj in sim_joint_names}

    if not valid_mapping:
        print("  유효한 관절 매핑이 없습니다. --mapping_json을 제공하세요.")
        return {"mode": "position", "error": "no_valid_mapping"}, None

    joint_unit = infer_joint_unit(frames, list(valid_mapping.keys()), args.joint_unit)
    print(f"  inferred joint unit: {joint_unit}")

    cfg = LeKiwiNavEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(cfg.episode_length_s, 10_000.0)
    if args.arm_limit_json:
        cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
        cfg.arm_limit_margin_rad = float(args.arm_limit_margin_rad)
    env = LeKiwiNavEnv(cfg=cfg)
    env.reset()

    sim_indices: dict[str, int] = {}
    for sj in sim_joint_names:
        idxs, _ = env.robot.find_joints([sj])
        if len(idxs) == 1:
            sim_indices[sj] = int(idxs[0])

    zero_action = torch.zeros(1, env.cfg.action_space, device=env.device)
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    err_acc: dict[str, list[float]] = defaultdict(list)

    for i, fr in enumerate(frames):
        pos = fr.get("positions", {})
        cur_joint_pos = env.robot.data.joint_pos[0].clone()
        cur_joint_vel = torch.zeros_like(cur_joint_pos)

        for real_key, sim_name in valid_mapping.items():
            if real_key not in pos:
                continue
            idx = sim_indices.get(sim_name)
            if idx is None:
                continue
            cur_joint_pos[idx] = to_rad(float(pos[real_key]), joint_unit)

        env.robot.write_joint_state_to_sim(cur_joint_pos.unsqueeze(0), cur_joint_vel.unsqueeze(0), env_ids=env_ids)
        env.step(zero_action)

        sim_pos = env.robot.data.joint_pos[0]
        for real_key, sim_name in valid_mapping.items():
            if real_key not in pos:
                continue
            idx = sim_indices.get(sim_name)
            if idx is None:
                continue
            target = to_rad(float(pos[real_key]), joint_unit)
            err_acc[sim_name].append(abs(float(sim_pos[idx].item()) - target))

        if (i + 1) % max(1, int(fps)) == 0:
            print(f"    frame {i + 1:5d}/{len(frames)}")

    report = {
        "mode": "position",
        "calibration_path": cal.get("_resolved_path", args.calibration),
        "num_frames": len(frames),
        "joint_unit": joint_unit,
        "mapped_joints": valid_mapping,
        "measured_wheel_radius": wr,
        "measured_base_radius": br,
        "config_wheel_radius": WHEEL_RADIUS,
        "config_base_radius": BASE_RADIUS,
        "mae_rad": {},
        "max_err_rad": {},
    }

    print("\n  replay error summary")
    for sim_name in sorted(err_acc.keys()):
        vals = np.asarray(err_acc[sim_name], dtype=np.float64)
        mae = float(np.mean(vals))
        mx = float(np.max(vals))
        report["mae_rad"][sim_name] = mae
        report["max_err_rad"][sim_name] = mx
        print(f"    {sim_name:<40} mae={mae:.6f} rad  max={mx:.6f} rad")

    env.close()
    return report, None


def run_command_mode(
    cal: dict,
    wr: float,
    br: float,
    dynamics_params: dict | None,
    lin_cmd_scale: float,
    ang_cmd_scale: float,
    cmd_transform: dict,
) -> tuple[dict, dict]:
    print("\n[Mode] command replay")

    sequences = extract_command_sequences(cal, fallback_vx=args.fallback_vx, fallback_wz=args.fallback_wz)
    if not sequences:
        print("  command+encoder 로그가 없습니다. wheel_radius/base_radius 측정을 먼저 수행하세요.")
        return {"mode": "command", "error": "no_command_sequences"}, {"mode": "command", "sequences": []}

    print(f"  sequences: {[s['name'] for s in sequences]}")

    arm_limits = _load_arm_limits(args.arm_limit_json)
    runner = ArticulationCommandRunner(
        wheel_radius=wr,
        base_radius=br,
        arm_limits=arm_limits,
        arm_limit_margin_rad=args.arm_limit_margin_rad,
    )
    runner.apply_dynamics(dynamics_params)

    report = {
        "mode": "command",
        "calibration_path": cal.get("_resolved_path", args.calibration),
        "measured_wheel_radius": wr,
        "measured_base_radius": br,
        "config_wheel_radius": WHEEL_RADIUS,
        "config_base_radius": BASE_RADIUS,
        "lin_cmd_scale": float(lin_cmd_scale),
        "ang_cmd_scale": float(ang_cmd_scale),
        "command_transform": dict(cmd_transform),
        "dynamics_params": dynamics_params or {},
        "sequences": [],
        "global_mae_rad": None,
        "global_rmse_rad": None,
    }

    series_out = {
        "mode": "command",
        "calibration_path": cal.get("_resolved_path", args.calibration),
        "lin_cmd_scale": float(lin_cmd_scale),
        "ang_cmd_scale": float(ang_cmd_scale),
        "command_transform": dict(cmd_transform),
        "sequences": [],
    }

    global_abs_err = []
    global_sq_err = []

    for seq in sequences:
        sim_trace = runner.run_sequence(
            seq,
            lin_cmd_scale=float(lin_cmd_scale),
            ang_cmd_scale=float(ang_cmd_scale),
            cmd_transform=cmd_transform,
        )

        mapping = build_auto_mapping(seq["wheel_keys"], WHEEL_JOINT_NAMES)
        mapping.update(load_mapping_override(args.mapping_json))
        valid_mapping = {rk: sj for rk, sj in mapping.items() if sj in WHEEL_JOINT_NAMES}

        enc_unit = resolve_encoder_unit(seq.get("encoder_unit", "auto"), seq["wheel_series"], args.encoder_unit)

        seq_pairs = []
        seq_abs_err = []
        seq_sq_err = []

        for real_key, sim_joint in sorted(valid_mapping.items()):
            real_vals = to_unwrapped_rad_array(np.asarray(seq["wheel_series"][real_key], dtype=np.float64), enc_unit)
            sim_vals = np.unwrap(np.asarray(sim_trace["wheel_pos_rad"][sim_joint], dtype=np.float64))

            aligned = align_and_compare_best_polarity(
                real_t=np.asarray(seq["time_s"], dtype=np.float64),
                real_series=real_vals,
                sim_t=np.asarray(sim_trace["time_s"], dtype=np.float64),
                sim_series=sim_vals,
            )

            seq_pairs.append(
                {
                    "real_key": real_key,
                    "sim_joint": sim_joint,
                    "mae_rad": aligned["mae_rad"],
                    "rmse_rad": aligned["rmse_rad"],
                    "max_err_rad": aligned["max_err_rad"],
                    "sim_polarity": int(aligned["sim_polarity"]),
                }
            )

            seq_abs_err.append(aligned["mae_rad"])
            seq_sq_err.append(aligned["rmse_rad"] ** 2)
            global_abs_err.append(aligned["mae_rad"])
            global_sq_err.append(aligned["rmse_rad"] ** 2)

            series_out["sequences"].append(
                {
                    "sequence": seq["name"],
                    "real_key": real_key,
                    "sim_joint": sim_joint,
                    "encoder_unit": enc_unit,
                    "command": sim_trace["command"],
                    "command_real": sim_trace["command_real"],
                    "command_transformed": sim_trace["command_transformed"],
                    "time_s": aligned["time_s"].tolist(),
                    "real_delta_rad": aligned["real_delta_rad"].tolist(),
                    "sim_delta_rad": aligned["sim_delta_rad"].tolist(),
                    "mae_rad": aligned["mae_rad"],
                    "rmse_rad": aligned["rmse_rad"],
                    "max_err_rad": aligned["max_err_rad"],
                    "sim_polarity": int(aligned["sim_polarity"]),
                }
            )

        seq_summary = {
            "name": seq["name"],
            "encoder_unit": enc_unit,
            "command": sim_trace["command"],
            "command_real": sim_trace["command_real"],
            "command_transformed": sim_trace["command_transformed"],
            "mapped_wheels": seq_pairs,
            "mean_mae_rad": float(np.mean(seq_abs_err)) if seq_abs_err else None,
            "mean_rmse_rad": float(np.sqrt(np.mean(seq_sq_err))) if seq_sq_err else None,
        }
        report["sequences"].append(seq_summary)

        print(
            f"  {seq['name']}: mean_mae={seq_summary['mean_mae_rad']:.6f} rad, "
            f"mean_rmse={seq_summary['mean_rmse_rad']:.6f} rad"
        )

    if global_abs_err:
        report["global_mae_rad"] = float(np.mean(global_abs_err))
        report["global_rmse_rad"] = float(np.sqrt(np.mean(global_sq_err)))

    runner.close()
    return report, series_out


def run_arm_command_mode(cal: dict, wr: float, br: float, dynamics_params: dict | None) -> tuple[dict, dict]:
    print("\n[Mode] arm_command replay")

    tests = extract_arm_sysid_tests(cal, fallback_unit=args.encoder_unit)
    if not tests:
        print("  arm_sysid 데이터가 없습니다. calibrate_real_robot.py --mode arm_sysid를 먼저 수행하세요.")
        return {"mode": "arm_command", "error": "no_arm_sysid"}, {"mode": "arm_command", "sequences": []}

    print(f"  arm tests: {len(tests)}")

    arm_limits = _load_arm_limits(args.arm_limit_json)
    runner = ArticulationCommandRunner(
        wheel_radius=wr,
        base_radius=br,
        arm_limits=arm_limits,
        arm_limit_margin_rad=args.arm_limit_margin_rad,
    )
    runner.apply_dynamics(dynamics_params)

    report = {
        "mode": "arm_command",
        "calibration_path": cal.get("_resolved_path", args.calibration),
        "dynamics_params": dynamics_params or {},
        "tests": [],
        "global_mae_rad": None,
        "global_rmse_rad": None,
    }
    series_out = {
        "mode": "arm_command",
        "calibration_path": cal.get("_resolved_path", args.calibration),
        "sequences": [],
    }

    global_abs_err = []
    global_sq_err = []

    for test in tests:
        sim_trace = runner.run_arm_sequence(test)
        aligned = align_and_compare(
            real_t=np.asarray(test["time_s"], dtype=np.float64),
            real_series=np.asarray(test["pos_rad"], dtype=np.float64),
            sim_t=np.asarray(sim_trace["time_s"], dtype=np.float64),
            sim_series=np.asarray(sim_trace["pos_rad"], dtype=np.float64),
        )

        row = {
            "name": test["name"],
            "joint_key": test["joint_key"],
            "sim_joint": test["sim_joint"],
            "type": test["type"],
            "mae_rad": aligned["mae_rad"],
            "rmse_rad": aligned["rmse_rad"],
            "max_err_rad": aligned["max_err_rad"],
        }
        report["tests"].append(row)
        global_abs_err.append(aligned["mae_rad"])
        global_sq_err.append(aligned["rmse_rad"] ** 2)
        series_out["sequences"].append(
            {
                "sequence": test["name"],
                "real_key": test["joint_key"],
                "sim_joint": test["sim_joint"],
                "encoder_unit": "rad",
                "command": sim_trace["command"],
                "time_s": aligned["time_s"].tolist(),
                "real_delta_rad": aligned["real_delta_rad"].tolist(),
                "sim_delta_rad": aligned["sim_delta_rad"].tolist(),
                "mae_rad": aligned["mae_rad"],
                "rmse_rad": aligned["rmse_rad"],
                "max_err_rad": aligned["max_err_rad"],
            }
        )
        print(
            f"  {test['name']}: mae={aligned['mae_rad']:.6f} rad, "
            f"rmse={aligned['rmse_rad']:.6f} rad"
        )

    if global_abs_err:
        report["global_mae_rad"] = float(np.mean(global_abs_err))
        report["global_rmse_rad"] = float(np.sqrt(np.mean(global_sq_err)))

    runner.close()
    return report, series_out


def main():
    cal = load_json(args.calibration)
    print(f"\n  calibration: {cal.get('_resolved_path', args.calibration)}")
    print(f"  timestamp:   {cal.get('timestamp', 'unknown')}")

    wr, br, geom_meta = resolve_motion_geometry(cal)

    print("\n  parameter compare")
    print(
        f"    wheel_radius used  = {wr:.6f} m "
        f"(source={geom_meta['wheel_radius_source']}, "
        f"config={geom_meta['wheel_radius_config']:.6f}, "
        f"cal={geom_meta['wheel_radius_calibration']})"
    )
    print(
        f"    base_radius used   = {br:.6f} m "
        f"(source={geom_meta['base_radius_source']}, "
        f"config={geom_meta['base_radius_config']:.6f}, "
        f"cal={geom_meta['base_radius_calibration']})"
    )
    if args.arm_limit_json:
        print(
            f"    arm_limit_json       = {Path(args.arm_limit_json).expanduser()} "
            f"(margin={args.arm_limit_margin_rad:.4f} rad)"
        )

    dynamics_params = None
    dynamics_payload = None
    if args.dynamics_json:
        with open(Path(args.dynamics_json).expanduser(), "r", encoding="utf-8") as f:
            dyn = json.load(f)
        if isinstance(dyn, dict):
            dynamics_payload = dyn
        if isinstance(dyn, dict) and "best_params" in dyn and isinstance(dyn["best_params"], dict):
            dynamics_params = dyn["best_params"]
        elif isinstance(dyn, dict):
            dynamics_params = dyn

    lin_cmd_scale = args.lin_cmd_scale
    if lin_cmd_scale is None:
        lin_cmd_scale = _to_float_or_default((dynamics_params or {}).get("lin_cmd_scale"), 1.0)
    else:
        lin_cmd_scale = _to_float_or_default(lin_cmd_scale, 1.0)

    ang_cmd_scale = args.ang_cmd_scale
    if ang_cmd_scale is None:
        ang_cmd_scale = _to_float_or_default((dynamics_params or {}).get("ang_cmd_scale"), 1.0)
    else:
        ang_cmd_scale = _to_float_or_default(ang_cmd_scale, 1.0)

    cmd_transform = resolve_command_transform(dynamics_payload)
    print(f"    lin_cmd_scale        = {lin_cmd_scale:.6f}")
    print(f"    ang_cmd_scale        = {ang_cmd_scale:.6f}")
    print(f"    command_transform    = {cmd_transform}")

    if args.mode == "position":
        report, series = run_position_mode(cal, wr=wr, br=br)
    elif args.mode == "command":
        report, series = run_command_mode(
            cal,
            wr=wr,
            br=br,
            dynamics_params=dynamics_params,
            lin_cmd_scale=lin_cmd_scale,
            ang_cmd_scale=ang_cmd_scale,
            cmd_transform=cmd_transform,
        )
    else:
        report, series = run_arm_command_mode(cal, wr=wr, br=br, dynamics_params=dynamics_params)

    if isinstance(report, dict):
        report["geometry"] = geom_meta

    if args.report_path:
        rp = Path(args.report_path).expanduser()
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  report saved: {rp}")

    if args.series_path and series is not None:
        sp = Path(args.series_path).expanduser()
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(series, f, indent=2, ensure_ascii=False)
        print(f"  series saved: {sp}")

    if not args.headless:
        print("\n  GUI keep-alive... Ctrl+C")
        try:
            while sim_app.is_running():
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass

    sim_app.close()


if __name__ == "__main__":
    main()
