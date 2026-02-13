#!/usr/bin/env python3
"""
Isaac Sim live action receiver/logger (Desktop side).

Receives teleop packets over TCP from laptop `teleop_dual_logger.py`,
applies the same commands to sim LeKiwi, and logs sim response.

This script is designed for the user's topology:
  real robot <-> laptop (leader + keyboard) <-> desktop (Isaac Sim)

Usage (Desktop):
  conda activate env_isaaclab
  source ~/isaacsim/setup_conda_env.sh
  python sim_action_receiver_logger.py \
    --listen_host 0.0.0.0 --listen_port 16000 \
    --calibration_json calibration/calibration_latest.json \
    --encoder_calibration_json ~/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_lekiwi.json \
    --arm_mapping joint_range \
    --dynamics_json calibration/tuned_dynamics.json \
    --headless
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="LeKiwi live sim receiver/logger")
parser.add_argument("--listen_host", type=str, default="0.0.0.0")
parser.add_argument("--listen_port", type=int, default=16000)
parser.add_argument("--sim_dt", type=float, default=1.0 / 30.0)
parser.add_argument("--max_packets", type=int, default=0, help="0 means unlimited")
parser.add_argument("--log_dir", type=str, default="logs/dual_teleop")
parser.add_argument("--series_path", type=str, default=None, help="output for compare_real_sim.py")
parser.add_argument("--series_autosave_packets", type=int, default=300, help="autosave interval for --series_path (0 to disable)")
parser.add_argument("--dynamics_json", type=str, default=None, help="tuned_dynamics.json path")
parser.add_argument("--calibration_json", type=str, default=None, help="optional calibration_latest.json for real joint ranges")
parser.add_argument("--arm_input_unit", type=str, default="m100_100", choices=["m100_100", "deg", "rad", "auto"])
parser.add_argument("--arm_mapping", type=str, default="joint_range", choices=["joint_range", "scale"])
parser.add_argument("--arm_action_scale", type=float, default=1.5, help="fallback scale for arm_input_unit=m100_100")
parser.add_argument("--apply_cmd_scale", action="store_true", help="apply lin/ang cmd scales from dynamics json")
parser.add_argument(
    "--encoder_calibration_json",
    type=str,
    default=None,
    help="optional LeRobot motor calibration json (e.g. ~/.cache/huggingface/lerobot/calibration/robots/lekiwi/<id>.json)",
)
parser.add_argument(
    "--encoder_robot_id",
    type=str,
    default="my_lekiwi",
    help="LeRobot robot id for auto-resolving encoder calibration path when --encoder_calibration_json is omitted",
)
parser.add_argument(
    "--apply_mapper_limits",
    dest="apply_mapper_limits",
    action="store_true",
    default=True,
    help="apply mapper-derived arm joint limits to Isaac Sim (default: enabled)",
)
parser.add_argument(
    "--no_apply_mapper_limits",
    dest="apply_mapper_limits",
    action="store_false",
    help="do not overwrite Isaac Sim arm joint limits from mapper",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from lekiwi_robot_cfg import (
    ARM_JOINT_NAMES,
    BASE_RADIUS,
    LEKIWI_CFG,
    WHEEL_ANGLES_RAD,
    WHEEL_JOINT_NAMES,
    WHEEL_RADIUS,
)


REAL_ARM_KEY_TO_SIM_JOINT = {
    "arm_shoulder_pan.pos": "STS3215_03a_v1_Revolute_45",
    "arm_shoulder_lift.pos": "STS3215_03a_v1_1_Revolute_49",
    "arm_elbow_flex.pos": "STS3215_03a_v1_2_Revolute_51",
    "arm_wrist_flex.pos": "STS3215_03a_v1_3_Revolute_53",
    "arm_wrist_roll.pos": "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "arm_gripper.pos": "STS3215_03a_v1_4_Revolute_57",
}

SIM_JOINT_TO_REAL_MOTOR_ID = {
    "STS3215_03a_v1_Revolute_45": 1,
    "STS3215_03a_v1_1_Revolute_49": 2,
    "STS3215_03a_v1_2_Revolute_51": 3,
    "STS3215_03a_v1_3_Revolute_53": 4,
    "STS3215_03a_Wrist_Roll_v1_Revolute_55": 5,
    "STS3215_03a_v1_4_Revolute_57": 6,
}
REAL_ARM_ID_TO_SIM_JOINT = {mid: name for name, mid in SIM_JOINT_TO_REAL_MOTOR_ID.items()}
SIM_JOINT_TO_REAL_ARM_KEY = {sim_joint: real_key for real_key, sim_joint in REAL_ARM_KEY_TO_SIM_JOINT.items()}
REAL_ARM_KEY_TO_MOTOR_NAME = {
    "arm_shoulder_pan.pos": "arm_shoulder_pan",
    "arm_shoulder_lift.pos": "arm_shoulder_lift",
    "arm_elbow_flex.pos": "arm_elbow_flex",
    "arm_wrist_flex.pos": "arm_wrist_flex",
    "arm_wrist_roll.pos": "arm_wrist_roll",
    "arm_gripper.pos": "arm_gripper",
}


def _safe_float(payload: dict, key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _is_gripper_real_key(real_key: str) -> bool:
    return "gripper" in _normalize_key(real_key)


class EncoderCalibrationMapper:
    """Convert LeRobot normalized STS3215 commands (m100_100/0_100) into radians."""

    def __init__(self, by_motor_name: dict[str, dict[str, float]], ticks_per_turn: int = 4096):
        self.by_motor_name = by_motor_name
        self.by_motor_id: dict[int, dict[str, float]] = {}
        for cfg in by_motor_name.values():
            mid = int(cfg.get("id", -1))
            if mid > 0:
                self.by_motor_id[mid] = cfg
        self.ticks_per_turn = int(ticks_per_turn)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EncoderCalibrationMapper | None:
        if not isinstance(payload, dict):
            return None
        by_motor_name: dict[str, dict[str, float]] = {}
        for motor_name, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            try:
                motor_id = int(raw.get("id"))
                drive_mode = int(raw.get("drive_mode", 0))
                range_min = float(raw.get("range_min"))
                range_max = float(raw.get("range_max"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(range_min) or not np.isfinite(range_max):
                continue
            if abs(range_max - range_min) < 1e-9:
                continue
            by_motor_name[str(motor_name)] = {
                "id": float(motor_id),
                "drive_mode": float(drive_mode),
                "range_min": float(range_min),
                "range_max": float(range_max),
            }
        if not by_motor_name:
            return None
        return cls(by_motor_name=by_motor_name)

    def _get_cfg_for_real_key(self, real_key: str) -> dict[str, float] | None:
        motor_name = REAL_ARM_KEY_TO_MOTOR_NAME.get(real_key)
        if motor_name and motor_name in self.by_motor_name:
            return self.by_motor_name[motor_name]
        sim_joint = REAL_ARM_KEY_TO_SIM_JOINT.get(real_key)
        if sim_joint is None:
            return None
        motor_id = SIM_JOINT_TO_REAL_MOTOR_ID.get(sim_joint)
        if motor_id is None:
            return None
        return self.by_motor_id.get(int(motor_id))

    def _raw_to_rad(self, raw_pos: float, cfg: dict[str, float]) -> float:
        range_min = float(cfg["range_min"])
        range_max = float(cfg["range_max"])
        mid = 0.5 * (range_min + range_max)
        max_res = max(1.0, float(self.ticks_per_turn - 1))
        deg = (float(raw_pos) - mid) * 360.0 / max_res
        return float(np.deg2rad(deg))

    def normalized_to_rad(self, real_key: str, value: float) -> float | None:
        cfg = self._get_cfg_for_real_key(real_key)
        if cfg is None:
            return None

        range_min = float(cfg["range_min"])
        range_max = float(cfg["range_max"])
        drive_mode = int(cfg["drive_mode"]) != 0
        lo = min(range_min, range_max)
        hi = max(range_min, range_max)

        if _is_gripper_real_key(real_key):
            v = float(np.clip(value, 0.0, 100.0))
            if drive_mode:
                v = 100.0 - v
            raw = (v / 100.0) * (hi - lo) + lo
        else:
            v = float(np.clip(value, -100.0, 100.0))
            if drive_mode:
                v = -v
            raw = ((v + 100.0) / 200.0) * (hi - lo) + lo

        return self._raw_to_rad(raw, cfg)

    def limits_rad_for_real_key(self, real_key: str) -> tuple[float, float] | None:
        if _is_gripper_real_key(real_key):
            low = self.normalized_to_rad(real_key, 0.0)
            high = self.normalized_to_rad(real_key, 100.0)
        else:
            low = self.normalized_to_rad(real_key, -100.0)
            high = self.normalized_to_rad(real_key, 100.0)
        if low is None or high is None:
            return None
        lo = float(min(low, high))
        hi = float(max(low, high))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-8:
            return None
        return (lo, hi)


def _guess_encoder_calibration_path(robot_id: str) -> Path:
    base = os.environ.get("HF_LEROBOT_CALIBRATION", "~/.cache/huggingface/lerobot/calibration")
    return Path(base).expanduser() / "robots" / "lekiwi" / f"{robot_id}.json"


def _load_encoder_mapper(path: str | None, robot_id: str) -> EncoderCalibrationMapper | None:
    p = Path(path).expanduser() if path else _guess_encoder_calibration_path(robot_id)
    if not p.is_file():
        if path:
            print(f"[encoder] calibration file not found: {p}")
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[encoder] failed to load {p}: {exc}")
        return None
    mapper = EncoderCalibrationMapper.from_payload(payload)
    if mapper is None:
        print(f"[encoder] invalid calibration payload: {p}")
        return None
    print(f"[encoder] loaded motor calibration: {p}")
    return mapper


def _extract_tuned_params(payload: dict) -> dict:
    if isinstance(payload.get("best_params"), dict):
        return payload["best_params"]
    if isinstance(payload.get("params"), dict):
        return payload["params"]
    return payload


def _load_tuned_params(path: str | None) -> dict[str, float]:
    defaults = {
        "wheel_stiffness_scale": 1.0,
        "wheel_damping_scale": 1.0,
        "wheel_armature_scale": 1.0,
        "wheel_friction_coeff": 0.0,
        "wheel_dynamic_friction_coeff": 0.0,
        "wheel_viscous_friction_coeff": 0.0,
        "arm_stiffness_scale": 1.0,
        "arm_damping_scale": 1.0,
        "arm_armature_scale": 1.0,
        "lin_cmd_scale": 1.0,
        "ang_cmd_scale": 1.0,
    }
    if not path:
        return defaults

    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        print(f"[dynamics] file not found: {p} -> using defaults")
        return defaults

    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = _extract_tuned_params(payload)
    if not isinstance(raw, dict):
        raise ValueError("Invalid dynamics JSON: expected dict or {'best_params': dict}")

    return {
        "wheel_stiffness_scale": _safe_float(raw, "wheel_stiffness_scale", 1.0),
        "wheel_damping_scale": _safe_float(raw, "wheel_damping_scale", 1.0),
        "wheel_armature_scale": _safe_float(raw, "wheel_armature_scale", 1.0),
        "wheel_friction_coeff": _safe_float(raw, "wheel_friction_coeff", 0.0),
        "wheel_dynamic_friction_coeff": _safe_float(raw, "wheel_dynamic_friction_coeff", 0.0),
        "wheel_viscous_friction_coeff": _safe_float(raw, "wheel_viscous_friction_coeff", 0.0),
        "arm_stiffness_scale": _safe_float(raw, "arm_stiffness_scale", 1.0),
        "arm_damping_scale": _safe_float(raw, "arm_damping_scale", 1.0),
        "arm_armature_scale": _safe_float(raw, "arm_armature_scale", 1.0),
        "lin_cmd_scale": _safe_float(raw, "lin_cmd_scale", 1.0),
        "ang_cmd_scale": _safe_float(raw, "ang_cmd_scale", 1.0),
    }


def _load_json_if_exists(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _resolve_motion_geometry(calibration_json: str | None, dynamics_json: str | None) -> tuple[float, float, str]:
    wr = float(WHEEL_RADIUS)
    br = float(BASE_RADIUS)
    source = "defaults"

    dyn = _load_json_if_exists(dynamics_json)
    if isinstance(dyn, dict):
        # tune_sim_dynamics.py output fields
        wr_dyn = dyn.get("wheel_radius_used")
        br_dyn = dyn.get("base_radius_used")
        if wr_dyn is None:
            wr_dyn = dyn.get("measured_wheel_radius")
        if br_dyn is None:
            br_dyn = dyn.get("measured_base_radius")
        try:
            if wr_dyn is not None:
                wr = float(wr_dyn)
            if br_dyn is not None:
                br = float(br_dyn)
            if (wr_dyn is not None) or (br_dyn is not None):
                source = "dynamics_json"
        except (TypeError, ValueError):
            pass

    cal = _load_json_if_exists(calibration_json)
    if source == "defaults" and isinstance(cal, dict):
        try:
            wr_cal = cal.get("wheel_radius", {}).get("wheel_radius_m")
            br_cal = cal.get("base_radius", {}).get("base_radius_m")
            if wr_cal is not None:
                wr = float(wr_cal)
            if br_cal is not None:
                br = float(br_cal)
            if (wr_cal is not None) or (br_cal is not None):
                source = "calibration_json"
        except (TypeError, ValueError, AttributeError):
            pass

    if (not np.isfinite(wr)) or wr <= 1e-8:
        wr = float(WHEEL_RADIUS)
    if (not np.isfinite(br)) or br <= 1e-8:
        br = float(BASE_RADIUS)
    return wr, br, source


def _apply_tuned_dynamics(robot: Articulation, arm_idx: torch.Tensor, wheel_idx: torch.Tensor, params: dict[str, float]) -> None:
    wheel_ids = wheel_idx.tolist()
    arm_ids = arm_idx.tolist()

    base_wheel_stiff = robot.data.joint_stiffness[:, wheel_idx].clone()
    base_wheel_damping = robot.data.joint_damping[:, wheel_idx].clone()
    base_wheel_armature = robot.data.joint_armature[:, wheel_idx].clone()
    base_arm_stiff = robot.data.joint_stiffness[:, arm_idx].clone()
    base_arm_damping = robot.data.joint_damping[:, arm_idx].clone()
    base_arm_armature = robot.data.joint_armature[:, arm_idx].clone()

    robot.write_joint_stiffness_to_sim(base_wheel_stiff * params["wheel_stiffness_scale"], joint_ids=wheel_ids)
    robot.write_joint_damping_to_sim(base_wheel_damping * params["wheel_damping_scale"], joint_ids=wheel_ids)
    robot.write_joint_armature_to_sim(base_wheel_armature * params["wheel_armature_scale"], joint_ids=wheel_ids)

    if hasattr(robot, "write_joint_friction_coefficient_to_sim"):
        robot.write_joint_friction_coefficient_to_sim(
            torch.full_like(base_wheel_damping, params["wheel_friction_coeff"]), joint_ids=wheel_ids
        )
    if hasattr(robot, "write_joint_dynamic_friction_coefficient_to_sim"):
        robot.write_joint_dynamic_friction_coefficient_to_sim(
            torch.full_like(base_wheel_damping, params["wheel_dynamic_friction_coeff"]), joint_ids=wheel_ids
        )
    if hasattr(robot, "write_joint_viscous_friction_coefficient_to_sim"):
        robot.write_joint_viscous_friction_coefficient_to_sim(
            torch.full_like(base_wheel_damping, params["wheel_viscous_friction_coeff"]), joint_ids=wheel_ids
        )

    robot.write_joint_stiffness_to_sim(base_arm_stiff * params["arm_stiffness_scale"], joint_ids=arm_ids)
    robot.write_joint_damping_to_sim(base_arm_damping * params["arm_damping_scale"], joint_ids=arm_ids)
    robot.write_joint_armature_to_sim(base_arm_armature * params["arm_armature_scale"], joint_ids=arm_ids)


def _kiwi_ik(vx: float, vy: float, wz_rad: float, wheel_radius: float, base_radius: float) -> np.ndarray:
    mat = np.array([[np.cos(a), np.sin(a), base_radius] for a in WHEEL_ANGLES_RAD], dtype=np.float64)
    body = np.array([vx, vy, wz_rad], dtype=np.float64)
    return mat.dot(body) / max(wheel_radius, 1e-6)


def _pick(action: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in action:
            val = action[key]
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
    return float(default)


def _arm_to_rad_scale(value: float, unit: str, arm_action_scale: float) -> float:
    if unit == "rad":
        return float(value)
    if unit == "deg":
        return float(np.deg2rad(value))
    if unit == "auto":
        # conservative fallback if caller forgot to resolve unit
        return float(np.deg2rad(value))
    # m100_100 -> [-1, 1] -> [-arm_action_scale, arm_action_scale]
    return float(np.clip(value / 100.0, -1.0, 1.0) * arm_action_scale)


def _resolve_arm_input_unit(unit: str, encoder_mapper: EncoderCalibrationMapper | None) -> str:
    if unit in ("m100_100", "deg", "rad"):
        return unit
    # auto: with encoder calibration prefer normalized mode, otherwise preserve old heuristic (deg)
    return "m100_100" if encoder_mapper is not None else "deg"


def _normalize_key(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _extract_motor_id_candidates(key: str) -> list[int]:
    out: list[int] = []
    cur = ""
    for ch in key:
        if ch.isdigit():
            cur += ch
        elif cur:
            out.append(int(cur))
            cur = ""
    if cur:
        out.append(int(cur))
    return out


def _infer_sim_arm_from_real_key(key: str) -> str | None:
    nk = _normalize_key(key)
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
    for token, joint_name in token_to_joint.items():
        if token in nk:
            return joint_name
    for motor_id in reversed(_extract_motor_id_candidates(key)):
        if motor_id in REAL_ARM_ID_TO_SIM_JOINT:
            return REAL_ARM_ID_TO_SIM_JOINT[motor_id]
    return None


def _extract_joint_ranges(payload: dict) -> dict[str, tuple[float, float]]:
    block = payload.get("joint_ranges")
    if isinstance(block, dict) and isinstance(block.get("joint_ranges"), dict):
        block = block["joint_ranges"]
    if not isinstance(block, dict):
        return {}

    out: dict[str, tuple[float, float]] = {}
    for key, val in block.items():
        if not isinstance(val, dict):
            continue
        if "min" not in val or "max" not in val:
            continue
        try:
            lo = float(val["min"])
            hi = float(val["max"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi < lo:
            lo, hi = hi, lo
        out[str(key)] = (lo, hi)
    return out


def _load_joint_ranges_from_calibration(path: str | None) -> dict[str, tuple[float, float]]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        print(f"[arm-map] calibration file not found: {p}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _extract_joint_ranges(payload) if isinstance(payload, dict) else {}


def _default_real_range_for_key(real_key: str, unit: str, sim_min: float, sim_max: float) -> tuple[float, float]:
    if unit in ("m100_100", "auto"):
        # LeRobot gripper is commonly [0, 100], other joints are [-100, 100]
        if _is_gripper_real_key(real_key):
            return (0.0, 100.0)
        return (-100.0, 100.0)
    if unit == "deg":
        return (float(np.rad2deg(sim_min)), float(np.rad2deg(sim_max)))
    return (sim_min, sim_max)


def _build_arm_mapper(
    joint_ranges: dict[str, tuple[float, float]],
    arm_name_to_idx: dict[str, int],
    robot: Articulation,
    unit: str,
    encoder_mapper: EncoderCalibrationMapper | None = None,
) -> dict[str, dict[str, float | str]]:
    ranges_by_sim: dict[str, tuple[float, float, str]] = {}
    for real_key, (real_min, real_max) in joint_ranges.items():
        sim_joint = _infer_sim_arm_from_real_key(real_key)
        if sim_joint is None:
            continue
        prev = ranges_by_sim.get(sim_joint)
        span = abs(real_max - real_min)
        prev_span = abs(prev[1] - prev[0]) if prev is not None else -1.0
        if prev is None or span > prev_span:
            ranges_by_sim[sim_joint] = (real_min, real_max, real_key)

    mapper: dict[str, dict[str, float | str]] = {}
    for real_key, sim_joint in REAL_ARM_KEY_TO_SIM_JOINT.items():
        idx = arm_name_to_idx.get(sim_joint)
        if idx is None:
            continue
        sim_min = float(robot.data.soft_joint_pos_limits[0, idx, 0].item())
        sim_max = float(robot.data.soft_joint_pos_limits[0, idx, 1].item())
        sim_source = "usd"
        if encoder_mapper is not None and unit == "m100_100":
            enc_limits = encoder_mapper.limits_rad_for_real_key(real_key)
            if enc_limits is not None:
                sim_min, sim_max = enc_limits
                sim_source = "encoder"
        if (not np.isfinite(sim_min)) or (not np.isfinite(sim_max)) or abs(sim_max - sim_min) < 1e-6:
            sim_min = -1.5
            sim_max = 1.5
            sim_source = "fallback"
        if sim_max < sim_min:
            sim_min, sim_max = sim_max, sim_min

        source = "default"
        if sim_joint in ranges_by_sim:
            real_min, real_max, source_key = ranges_by_sim[sim_joint]
            source = f"calibration:{source_key}"
        else:
            real_min, real_max = _default_real_range_for_key(real_key, unit, sim_min, sim_max)
        if real_max < real_min:
            real_min, real_max = real_max, real_min

        mapper[real_key] = {
            "sim_joint": sim_joint,
            "sim_min": float(sim_min),
            "sim_max": float(sim_max),
            "real_min": float(real_min),
            "real_max": float(real_max),
            "source": source,
            "sim_limit_source": sim_source,
        }
    return mapper


def _arm_to_sim_from_mapper(
    real_key: str,
    value: float,
    mapper: dict[str, dict[str, float | str]] | None,
    unit: str,
    arm_action_scale: float,
    encoder_mapper: EncoderCalibrationMapper | None = None,
) -> float:
    if encoder_mapper is not None and unit == "m100_100":
        v = encoder_mapper.normalized_to_rad(real_key, float(value))
        if v is not None:
            if mapper is not None and real_key in mapper:
                m = mapper[real_key]
                lo = min(float(m["sim_min"]), float(m["sim_max"]))
                hi = max(float(m["sim_min"]), float(m["sim_max"]))
                return float(np.clip(v, lo, hi))
            return float(v)

    if mapper is not None and real_key in mapper:
        m = mapper[real_key]
        real_min = float(m["real_min"])
        real_max = float(m["real_max"])
        sim_min = float(m["sim_min"])
        sim_max = float(m["sim_max"])
        denom = real_max - real_min
        if abs(denom) < 1e-9:
            return float(0.5 * (sim_min + sim_max))
        alpha = (float(value) - real_min) / denom
        sim_val = sim_min + alpha * (sim_max - sim_min)
        lo = min(sim_min, sim_max)
        hi = max(sim_min, sim_max)
        return float(np.clip(sim_val, lo, hi))
    return _arm_to_rad_scale(value, unit, arm_action_scale)


def _apply_arm_limits_from_mapper(
    robot: Articulation,
    arm_idx: torch.Tensor,
    mapper: dict[str, dict[str, float | str]] | None,
) -> int:
    if mapper is None:
        return 0

    sim_to_limits: dict[str, tuple[float, float]] = {}
    for m in mapper.values():
        sim_joint = str(m.get("sim_joint"))
        sim_min = float(m.get("sim_min", np.nan))
        sim_max = float(m.get("sim_max", np.nan))
        if not np.isfinite(sim_min) or not np.isfinite(sim_max):
            continue
        lo = float(min(sim_min, sim_max))
        hi = float(max(sim_min, sim_max))
        if abs(hi - lo) < 1e-8:
            continue
        sim_to_limits[sim_joint] = (lo, hi)

    if not sim_to_limits:
        return 0

    joint_ids = arm_idx.tolist()
    limits = robot.data.joint_pos_limits[:, joint_ids].clone()
    updated = 0
    for local_idx, joint_id in enumerate(joint_ids):
        sim_joint = robot.joint_names[joint_id]
        lim = sim_to_limits.get(sim_joint)
        if lim is None:
            continue
        limits[:, local_idx, 0] = float(lim[0])
        limits[:, local_idx, 1] = float(lim[1])
        updated += 1

    if updated > 0:
        robot.write_joint_position_limit_to_sim(limits, joint_ids=joint_ids, warn_limit_violation=False)
    return updated


def _to_float_dict(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in payload.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            out[str(key)] = float(val)
    return out


def _compute_arm_series(series_acc: dict[tuple[str, str], dict[str, list[float]]], out_path: Path) -> None:
    sequences = []
    for (real_key, sim_joint), data in sorted(series_acc.items()):
        if len(data["t"]) < 3:
            continue
        t = np.asarray(data["t"], dtype=np.float64)
        real = np.asarray(data["real"], dtype=np.float64)
        sim = np.asarray(data["sim"], dtype=np.float64)
        real_delta = real - real[0]
        sim_delta = sim - sim[0]
        err = sim_delta - real_delta
        sequences.append(
            {
                "sequence": "live_arm",
                "real_key": real_key,
                "sim_joint": sim_joint,
                "time_s": t.tolist(),
                "real_delta_rad": real_delta.tolist(),
                "sim_delta_rad": sim_delta.tolist(),
                "mae_rad": float(np.mean(np.abs(err))),
                "rmse_rad": float(np.sqrt(np.mean(err**2))),
                "max_err_rad": float(np.max(np.abs(err))),
            }
        )

    payload = {"mode": "arm_command", "sequences": sequences}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[series] saved: {out_path} ({len(sequences)} sequences)")


def _save_arm_series_if_needed(series_acc: dict[tuple[str, str], dict[str, list[float]]], out_path: Path | None, reason: str) -> None:
    if out_path is None:
        return
    try:
        _compute_arm_series(series_acc, out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[series] save failed ({reason}): {exc}")


def main() -> None:
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = log_dir / f"sim_receiver_{ts}.jsonl"

    print("=" * 72)
    print("LeKiwi live sim receiver/logger")
    print(f"listen: {args.listen_host}:{args.listen_port}")
    print(f"log: {jsonl_path}")
    if args.dynamics_json:
        print(f"dynamics_json: {os.path.expanduser(args.dynamics_json)}")
    if args.calibration_json:
        print(f"calibration_json: {os.path.expanduser(args.calibration_json)}")
    if args.encoder_calibration_json:
        print(f"encoder_calibration_json: {os.path.expanduser(args.encoder_calibration_json)}")
    print(f"arm_mapping: {args.arm_mapping} (unit={args.arm_input_unit})")
    series_out_path = Path(args.series_path).expanduser() if args.series_path else None
    if series_out_path is not None:
        print(f"series_path: {series_out_path} (autosave every {args.series_autosave_packets} packets)")
    print("=" * 72)

    sim_cfg = SimulationCfg(
        dt=float(args.sim_dt),
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        device=args.device,
    )
    sim = SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    )

    robot_cfg = LEKIWI_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.update(sim.get_physics_dt())

    arm_idx = torch.tensor(robot.find_joints(ARM_JOINT_NAMES)[0], dtype=torch.long, device=robot.device)
    wheel_idx = torch.tensor(robot.find_joints(WHEEL_JOINT_NAMES)[0], dtype=torch.long, device=robot.device)
    arm_name_to_idx = {robot.joint_names[i]: i for i in arm_idx.tolist()}
    wheel_name_to_idx = {robot.joint_names[i]: i for i in wheel_idx.tolist()}

    encoder_mapper = _load_encoder_mapper(args.encoder_calibration_json, robot_id=args.encoder_robot_id)
    arm_input_unit = _resolve_arm_input_unit(args.arm_input_unit, encoder_mapper)
    if arm_input_unit != args.arm_input_unit:
        print(f"[arm-map] arm_input_unit auto -> {arm_input_unit}")

    arm_mapper: dict[str, dict[str, float | str]] | None = None
    if args.arm_mapping == "joint_range":
        joint_ranges = _load_joint_ranges_from_calibration(args.calibration_json)
        arm_mapper = _build_arm_mapper(
            joint_ranges=joint_ranges,
            arm_name_to_idx=arm_name_to_idx,
            robot=robot,
            unit=arm_input_unit,
            encoder_mapper=encoder_mapper,
        )
        if args.apply_mapper_limits:
            n_updated = _apply_arm_limits_from_mapper(robot=robot, arm_idx=arm_idx, mapper=arm_mapper)
            print(f"[arm-map] applied arm joint limits from mapper: {n_updated} joints")
        print("[arm-map] active per-joint real->sim range mapping")
        for real_key, m in arm_mapper.items():
            print(
                f"  {real_key} -> {m['sim_joint']} | "
                f"real[{float(m['real_min']):.3f}, {float(m['real_max']):.3f}] "
                f"-> sim[{float(m['sim_min']):.3f}, {float(m['sim_max']):.3f}] "
                f"({m['source']}, limit={m.get('sim_limit_source', 'n/a')})"
            )
    else:
        print(f"[arm-map] legacy scale mode (arm_action_scale={args.arm_action_scale:.3f})")

    params = _load_tuned_params(args.dynamics_json)
    _apply_tuned_dynamics(robot, arm_idx, wheel_idx, params)
    wheel_radius, base_radius, geom_source = _resolve_motion_geometry(
        calibration_json=args.calibration_json,
        dynamics_json=args.dynamics_json,
    )
    lin_scale = params["lin_cmd_scale"] if args.apply_cmd_scale else 1.0
    ang_scale = params["ang_cmd_scale"] if args.apply_cmd_scale else 1.0

    print(
        f"[dynamics] lin_scale={lin_scale:.4f} ang_scale={ang_scale:.4f} "
        f"(apply_cmd_scale={args.apply_cmd_scale})"
    )
    print(f"[motion] wheel_radius={wheel_radius:.6f} base_radius={base_radius:.6f} ({geom_source})")

    arm_target = robot.data.default_joint_pos.clone()
    vel_target = torch.zeros((1, robot.num_joints), dtype=torch.float32, device=robot.device)

    series_acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    t0_mono_ns: int | None = None
    packet_count = 0

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.listen_host, args.listen_port))
    server.listen(1)
    server.settimeout(1.0)

    print("[net] waiting for laptop connection...")

    try:
        with open(jsonl_path, "w", encoding="utf-8") as log_file:
            while sim_app.is_running():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue

                print(f"[net] connected from {addr[0]}:{addr[1]}")
                conn.settimeout(1.0)
                with conn:
                    file = conn.makefile("r", encoding="utf-8")
                    for line in file:
                        if not line:
                            break
                        if args.max_packets > 0 and packet_count >= args.max_packets:
                            break

                        try:
                            packet = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        action = _to_float_dict(packet.get("action", {}))
                        real_obs = _to_float_dict(packet.get("observation", {}))

                        vx = _pick(action, "x.vel", "base.linear_velocity_x.pos", "base.vx", default=0.0)
                        vy = _pick(action, "y.vel", "base.linear_velocity_y.pos", "base.vy", default=0.0)
                        wz_deg = _pick(action, "theta.vel", "base.angular_velocity_z.pos", "base.wz", default=0.0)
                        vx *= lin_scale
                        vy *= lin_scale
                        wz_deg *= ang_scale
                        wz_rad = np.deg2rad(wz_deg)

                        wheel_radps = _kiwi_ik(vx, vy, wz_rad, wheel_radius=wheel_radius, base_radius=base_radius)
                        vel_target.zero_()
                        vel_target[:, wheel_idx] = torch.tensor(wheel_radps, dtype=torch.float32, device=robot.device).unsqueeze(0)

                        for real_key, sim_joint in REAL_ARM_KEY_TO_SIM_JOINT.items():
                            if real_key not in action:
                                continue
                            idx = arm_name_to_idx.get(sim_joint)
                            if idx is None:
                                continue
                            arm_target[:, idx] = _arm_to_sim_from_mapper(
                                real_key=real_key,
                                value=float(action[real_key]),
                                mapper=arm_mapper,
                                unit=arm_input_unit,
                                arm_action_scale=args.arm_action_scale,
                                encoder_mapper=encoder_mapper,
                            )

                        robot.set_joint_velocity_target(vel_target)
                        robot.set_joint_position_target(arm_target)
                        robot.write_data_to_sim()
                        sim.step()
                        robot.update(sim.get_physics_dt())

                        root_quat_w = robot.data.root_quat_w
                        lin_vel_w = robot.data.root_lin_vel_w
                        ang_vel_w = robot.data.root_ang_vel_w
                        lin_vel_b = quat_apply_inverse(root_quat_w, lin_vel_w)[0].detach().cpu().numpy()
                        ang_vel_b = quat_apply_inverse(root_quat_w, ang_vel_w)[0].detach().cpu().numpy()
                        jp = robot.data.joint_pos[0].detach().cpu().numpy()
                        jv = robot.data.joint_vel[0].detach().cpu().numpy()

                        sim_arm_pos = {name: float(jp[idx]) for name, idx in arm_name_to_idx.items()}
                        sim_wheel_vel = {name: float(jv[idx]) for name, idx in wheel_name_to_idx.items()}
                        sim_body = {
                            "x.vel": float(lin_vel_b[0]),
                            "y.vel": float(lin_vel_b[1]),
                            "theta.vel": float(np.rad2deg(ang_vel_b[2])),
                        }

                        t_mono_ns = int(packet.get("t_mono_ns", time.monotonic_ns()))
                        if t0_mono_ns is None:
                            t0_mono_ns = t_mono_ns
                        t_rel = max(0.0, (t_mono_ns - t0_mono_ns) / 1e9)

                        for real_key, sim_joint in REAL_ARM_KEY_TO_SIM_JOINT.items():
                            real_val = real_obs.get(real_key, action.get(real_key))
                            if real_val is None:
                                continue
                            real_rad = _arm_to_sim_from_mapper(
                                real_key=real_key,
                                value=float(real_val),
                                mapper=arm_mapper,
                                unit=arm_input_unit,
                                arm_action_scale=args.arm_action_scale,
                                encoder_mapper=encoder_mapper,
                            )
                            sim_rad = sim_arm_pos.get(sim_joint)
                            if sim_rad is None:
                                continue
                            key = (real_key, sim_joint)
                            if key not in series_acc:
                                series_acc[key] = {"t": [], "real": [], "sim": []}
                            series_acc[key]["t"].append(float(t_rel))
                            series_acc[key]["real"].append(float(real_rad))
                            series_acc[key]["sim"].append(float(sim_rad))

                        record = {
                            "seq": int(packet.get("seq", packet_count)),
                            "t_wall_recv_s": time.time(),
                            "t_mono_ns": t_mono_ns,
                            "action": action,
                            "real_observation": real_obs,
                            "sim": {
                                "arm_pos_rad": sim_arm_pos,
                                "wheel_vel_radps": sim_wheel_vel,
                                "body_vel": sim_body,
                            },
                        }
                        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if packet_count % 20 == 0:
                            log_file.flush()

                        packet_count += 1
                        if packet_count % 30 == 0:
                            print(f"[sim] packets={packet_count}")
                        if args.series_autosave_packets > 0 and packet_count % args.series_autosave_packets == 0:
                            _save_arm_series_if_needed(series_acc, series_out_path, reason="autosave")

                    print("[net] client disconnected")
                    _save_arm_series_if_needed(series_acc, series_out_path, reason="disconnect")
                    if args.max_packets > 0 and packet_count >= args.max_packets:
                        break
    except KeyboardInterrupt:
        print("\n[sim] interrupted")
    finally:
        try:
            server.close()
        except OSError:
            pass
        _save_arm_series_if_needed(series_acc, series_out_path, reason="shutdown")
        sim.clear_all_callbacks()
        sim_app.close()
        print(f"[sim] done. packets={packet_count}, log={jsonl_path}")


if __name__ == "__main__":
    main()
