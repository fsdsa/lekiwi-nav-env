#!/usr/bin/env python3
"""
Build arm joint limits JSON for Isaac Sim from real robot calibration data.

This utility generates `joint_limits_rad` consumable by:
  - LeKiwiNavEnvCfg.arm_limit_json
  - replay_in_sim.py --arm_limit_json

Inputs:
  1) calibration_latest.json (optional): uses `joint_ranges` min/max
  2) LeRobot motor calibration json (optional but recommended for m100_100):
     ~/.cache/huggingface/lerobot/calibration/robots/lekiwi/<robot_id>.json

Usage:
  python build_arm_limits_real2sim.py \
    --calibration_json calibration/calibration_latest.json \
    --encoder_calibration_json ~/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_lekiwi.json \
    --input_unit auto \
    --output calibration/arm_limits_real2sim.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


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
REAL_ARM_KEY_TO_MOTOR_NAME = {
    "arm_shoulder_pan.pos": "arm_shoulder_pan",
    "arm_shoulder_lift.pos": "arm_shoulder_lift",
    "arm_elbow_flex.pos": "arm_elbow_flex",
    "arm_wrist_flex.pos": "arm_wrist_flex",
    "arm_wrist_roll.pos": "arm_wrist_roll",
    "arm_gripper.pos": "arm_gripper",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build arm limit JSON for real2sim")
    parser.add_argument("--calibration_json", type=str, default=None, help="calibration_latest.json path")
    parser.add_argument("--encoder_calibration_json", type=str, default=None, help="LeRobot motor calibration json")
    parser.add_argument("--input_unit", type=str, default="auto", choices=["auto", "m100_100", "deg", "rad"])
    parser.add_argument("--fallback_scale_rad_per_100", type=float, default=float(np.pi))
    parser.add_argument("--limit_margin_rad", type=float, default=0.0, help="expand each side by this margin")
    parser.add_argument("--output", type=str, default="calibration/arm_limits_real2sim.json")
    return parser.parse_args()


def _normalize_key(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _is_gripper_key(real_key: str) -> bool:
    return "gripper" in _normalize_key(real_key)


def _extract_motor_id_candidates(key: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", key)]


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


class EncoderCalibrationMapper:
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
        by_name: dict[str, dict[str, float]] = {}
        for motor_name, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            try:
                mid = int(raw.get("id"))
                drive_mode = int(raw.get("drive_mode", 0))
                range_min = float(raw.get("range_min"))
                range_max = float(raw.get("range_max"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(range_min) or not np.isfinite(range_max):
                continue
            if abs(range_max - range_min) < 1e-9:
                continue
            by_name[str(motor_name)] = {
                "id": float(mid),
                "drive_mode": float(drive_mode),
                "range_min": float(range_min),
                "range_max": float(range_max),
            }
        if not by_name:
            return None
        return cls(by_name)

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
        lo = min(float(cfg["range_min"]), float(cfg["range_max"]))
        hi = max(float(cfg["range_min"]), float(cfg["range_max"]))
        drive_mode = int(cfg["drive_mode"]) != 0

        if _is_gripper_key(real_key):
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


def _default_range_for_unit(real_key: str, unit: str) -> tuple[float, float]:
    if unit == "m100_100":
        if _is_gripper_key(real_key):
            return (0.0, 100.0)
        return (-100.0, 100.0)
    if unit == "deg":
        if _is_gripper_key(real_key):
            return (0.0, 180.0)
        return (-180.0, 180.0)
    if _is_gripper_key(real_key):
        return (0.0, float(np.pi))
    return (float(-np.pi), float(np.pi))


def _infer_input_unit(ranges: dict[str, tuple[float, float]], user_choice: str, encoder_mapper: EncoderCalibrationMapper | None) -> str:
    if user_choice in ("m100_100", "deg", "rad"):
        return user_choice
    vals = []
    for lo, hi in ranges.values():
        vals.append(abs(lo))
        vals.append(abs(hi))
        vals.append(abs(hi - lo))
    if not vals:
        return "m100_100" if encoder_mapper is not None else "deg"
    p95 = float(np.percentile(np.asarray(vals, dtype=np.float64), 95))
    if p95 <= 6.0:
        return "rad"
    if encoder_mapper is not None and p95 <= 120.0:
        return "m100_100"
    return "deg"


def _convert_to_rad(real_key: str, val: float, unit: str, encoder_mapper: EncoderCalibrationMapper | None, fallback_scale: float) -> float:
    if unit == "rad":
        return float(val)
    if unit == "deg":
        return float(np.deg2rad(val))
    if encoder_mapper is not None:
        out = encoder_mapper.normalized_to_rad(real_key, float(val))
        if out is not None:
            return float(out)
    # fallback for normalized values when encoder calibration is unavailable
    return float(val) * float(fallback_scale) / 100.0


def main() -> None:
    args = parse_args()

    joint_ranges: dict[str, tuple[float, float]] = {}
    calibration_path = None
    if args.calibration_json:
        p = Path(args.calibration_json).expanduser()
        calibration_path = str(p)
        if not p.is_file():
            raise FileNotFoundError(f"calibration_json not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        joint_ranges = _extract_joint_ranges(payload if isinstance(payload, dict) else {})

    encoder_mapper = None
    encoder_path = None
    if args.encoder_calibration_json:
        p = Path(args.encoder_calibration_json).expanduser()
        encoder_path = str(p)
        if not p.is_file():
            raise FileNotFoundError(f"encoder_calibration_json not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        encoder_mapper = EncoderCalibrationMapper.from_payload(payload)
        if encoder_mapper is None:
            raise ValueError(f"invalid encoder_calibration_json payload: {p}")

    unit = _infer_input_unit(joint_ranges, args.input_unit, encoder_mapper)
    margin = float(args.limit_margin_rad)

    # Map possibly noisy real keys -> canonical sim joints by widest span.
    per_joint_best: dict[str, dict[str, Any]] = {}
    for real_key, (real_min, real_max) in joint_ranges.items():
        sim_joint = _infer_sim_arm_from_real_key(real_key)
        if sim_joint is None or sim_joint not in SIM_JOINT_TO_REAL_MOTOR_ID:
            continue
        span = abs(real_max - real_min)
        prev = per_joint_best.get(sim_joint)
        prev_span = abs(float(prev["real_max"]) - float(prev["real_min"])) if prev is not None else -1.0
        if prev is None or span > prev_span:
            per_joint_best[sim_joint] = {
                "real_key": real_key,
                "real_min": float(real_min),
                "real_max": float(real_max),
                "source": f"joint_ranges:{real_key}",
            }

    # Fill missing joints with canonical keys + defaults.
    for real_key, sim_joint in REAL_ARM_KEY_TO_SIM_JOINT.items():
        if sim_joint in per_joint_best:
            continue
        dmin, dmax = _default_range_for_unit(real_key, unit)
        per_joint_best[sim_joint] = {
            "real_key": real_key,
            "real_min": float(dmin),
            "real_max": float(dmax),
            "source": "default",
        }

    limits_out: dict[str, dict[str, Any]] = {}
    for sim_joint, row in per_joint_best.items():
        real_key = str(row["real_key"])
        lo = _convert_to_rad(real_key, float(row["real_min"]), unit, encoder_mapper, args.fallback_scale_rad_per_100)
        hi = _convert_to_rad(real_key, float(row["real_max"]), unit, encoder_mapper, args.fallback_scale_rad_per_100)
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        lo -= margin
        hi += margin
        limits_out[sim_joint] = {
            "min": lo,
            "max": hi,
            "real_key": real_key,
            "real_min": float(row["real_min"]),
            "real_max": float(row["real_max"]),
            "source": str(row["source"]),
            "unit": unit,
        }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "calibration_json": calibration_path,
            "encoder_calibration_json": encoder_path,
            "input_unit": args.input_unit,
            "resolved_unit": unit,
            "limit_margin_rad": margin,
            "fallback_scale_rad_per_100": float(args.fallback_scale_rad_per_100),
        },
        "joint_limits_rad": limits_out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[saved] {out_path}")
    print(f"[unit] {unit}")
    for sim_joint in sorted(limits_out.keys()):
        row = limits_out[sim_joint]
        print(
            f"  {sim_joint:<36} "
            f"[{row['min']:+.4f}, {row['max']:+.4f}] rad "
            f"<- {row['real_key']} ({row['source']})"
        )


if __name__ == "__main__":
    main()

