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
import time
from pathlib import Path
from typing import Any

import numpy as np
from calibration_common import (
    EncoderCalibrationMapper,
    REAL_ARM_KEY_TO_SIM_JOINT,
    SIM_JOINT_TO_REAL_MOTOR_ID,
    infer_sim_arm_from_real_key as _infer_sim_arm_from_real_key,
    is_gripper_key as _is_gripper_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build arm limit JSON for real2sim")
    parser.add_argument("--calibration_json", type=str, default=None, help="calibration_latest.json path")
    parser.add_argument("--encoder_calibration_json", type=str, default=None, help="LeRobot motor calibration json")
    parser.add_argument("--input_unit", type=str, default="auto", choices=["auto", "m100_100", "deg", "rad"])
    parser.add_argument("--fallback_scale_rad_per_100", type=float, default=float(np.pi))
    parser.add_argument("--limit_margin_rad", type=float, default=0.0, help="expand each side by this margin")
    parser.add_argument("--output", type=str, default="calibration/arm_limits_real2sim.json")
    return parser.parse_args()


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
