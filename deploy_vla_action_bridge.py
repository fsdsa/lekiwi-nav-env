#!/usr/bin/env python3
"""
VLA action(9D, [-1, 1]) -> real robot command bridge utility.

This script implements the full deployment conversion chain:
  1) action[0:3] denormalization in sim-space
  2) sim-space -> real-space base command transform
  3) action[3:9] -> arm target(rad) mapping (same semantics as LeKiwiNavEnv)

Usage example:
  python deploy_vla_action_bridge.py \
    --action "0.2,0.0,-0.3,0,0,0,0,0,0" \
    --dynamics_json calibration/tuned_dynamics.json \
    --arm_limit_json calibration/arm_limits_real2sim.json \
    --json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from sim_real_command_transform import load_transform_cfg, sim_to_real


ARM_JOINT_NAMES = [
    "STS3215_03a_v1_Revolute_45",
    "STS3215_03a_v1_1_Revolute_49",
    "STS3215_03a_v1_2_Revolute_51",
    "STS3215_03a_v1_3_Revolute_53",
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "STS3215_03a_v1_4_Revolute_57",
]


def _safe_float(v: object, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(f):
        return float(default)
    return float(f)


def _parse_action(raw: str) -> np.ndarray:
    items = [x.strip() for x in str(raw).split(",")]
    if len(items) != 9:
        raise ValueError(f"--action must contain exactly 9 comma-separated values, got {len(items)}")
    vals = np.array([_safe_float(x, 0.0) for x in items], dtype=np.float64)
    return np.clip(vals, -1.0, 1.0)


def _load_best_params(dynamics_json: str | None) -> dict:
    if not dynamics_json:
        return {}
    p = Path(dynamics_json).expanduser()
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if isinstance(payload.get("best_params"), dict):
            return payload["best_params"]
        if isinstance(payload.get("params"), dict):
            return payload["params"]
    return payload if isinstance(payload, dict) else {}


def _load_arm_limits(arm_limit_json: str | None) -> dict[str, tuple[float, float]]:
    if not arm_limit_json:
        return {}
    p = Path(arm_limit_json).expanduser()
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    block = payload.get("joint_limits_rad") if isinstance(payload, dict) else None
    if block is None and isinstance(payload, dict):
        block = payload
    if not isinstance(block, dict):
        return {}

    out: dict[str, tuple[float, float]] = {}
    for joint_name, val in block.items():
        lo, hi = None, None
        if isinstance(val, dict):
            lo = val.get("min")
            hi = val.get("max")
        elif isinstance(val, (list, tuple)) and len(val) >= 2:
            lo, hi = val[0], val[1]
        if lo is None or hi is None:
            continue
        lo_f = _safe_float(lo, float("nan"))
        hi_f = _safe_float(hi, float("nan"))
        if not math.isfinite(lo_f) or not math.isfinite(hi_f):
            continue
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        if abs(hi_f - lo_f) < 1e-8:
            continue
        out[str(joint_name)] = (lo_f, hi_f)
    return out


def _resolve_arm_limits(args) -> tuple[np.ndarray, np.ndarray]:
    limits = _load_arm_limits(args.arm_limit_json)
    lo = np.full((6,), -np.pi, dtype=np.float64)
    hi = np.full((6,), np.pi, dtype=np.float64)

    for i, joint_name in enumerate(ARM_JOINT_NAMES):
        if joint_name in limits:
            lo[i], hi[i] = limits[joint_name]

    margin = _safe_float(args.arm_limit_margin_rad, 0.0)
    lo = np.maximum(lo - margin, -2.0 * np.pi + 1e-6)
    hi = np.minimum(hi + margin, 2.0 * np.pi - 1e-6)
    invalid = (hi - lo) <= 1e-6
    lo[invalid] = -np.pi
    hi[invalid] = np.pi
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description="Convert VLA 9D action into real robot base+arm command")
    parser.add_argument("--action", type=str, required=True, help="9D action in CSV: a0,...,a8")
    parser.add_argument(
        "--dynamics_json",
        type=str,
        default="calibration/tuned_dynamics.json",
        help="tuned dynamics JSON (for lin/ang cmd scale + command_transform)",
    )
    parser.add_argument("--arm_limit_json", type=str, default="", help="arm limit JSON (real2sim converted)")
    parser.add_argument("--arm_limit_margin_rad", type=float, default=0.0)
    parser.add_argument("--base_max_lin_vel", type=float, default=0.5)
    parser.add_argument("--base_max_ang_vel", type=float, default=3.0)
    parser.add_argument("--arm_action_scale", type=float, default=1.5)
    parser.add_argument("--arm_action_to_limits", action="store_true", help="map arm action [-1,1] to limits")
    parser.add_argument("--no_dynamics_cmd_scale", action="store_true", help="disable lin/ang cmd scaling from best_params")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    action = _parse_action(args.action)
    best_params = _load_best_params(args.dynamics_json)
    cmd_tf = load_transform_cfg(args.dynamics_json)

    lin_scale = _safe_float(best_params.get("lin_cmd_scale"), 1.0)
    ang_scale = _safe_float(best_params.get("ang_cmd_scale"), 1.0)
    if args.no_dynamics_cmd_scale:
        lin_scale = 1.0
        ang_scale = 1.0

    max_lin_vel = _safe_float(args.base_max_lin_vel, 0.5) * lin_scale
    max_ang_vel = _safe_float(args.base_max_ang_vel, 3.0) * ang_scale

    vx_sim = float(action[0] * max_lin_vel)
    vy_sim = float(action[1] * max_lin_vel)
    wz_sim = float(action[2] * max_ang_vel)
    vx_real, vy_real, wz_real = sim_to_real(vx_sim, vy_sim, wz_sim, cmd_tf)

    arm_action = action[3:9]
    if args.arm_action_to_limits:
        arm_lo, arm_hi = _resolve_arm_limits(args)
        arm_center = 0.5 * (arm_lo + arm_hi)
        arm_half = 0.5 * (arm_hi - arm_lo)
        arm_targets = arm_center + arm_action * arm_half
        arm_targets = np.clip(arm_targets, arm_lo, arm_hi)
    else:
        arm_targets = arm_action * _safe_float(args.arm_action_scale, 1.5)

    out = {
        "action_9d": action.tolist(),
        "sim_base_cmd": {"vx": vx_sim, "vy": vy_sim, "wz": wz_sim},
        "real_base_cmd": {"vx": float(vx_real), "vy": float(vy_real), "wz": float(wz_real)},
        "arm_target_rad": arm_targets.astype(np.float64).tolist(),
        "max_vel_used": {"max_lin_vel": float(max_lin_vel), "max_ang_vel": float(max_ang_vel)},
        "dynamics_cmd_scale": {"lin_cmd_scale": float(lin_scale), "ang_cmd_scale": float(ang_scale)},
        "command_transform": cmd_tf,
        "arm_action_to_limits": bool(args.arm_action_to_limits),
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"action: {out['action_9d']}")
    print(
        f"sim base : vx={vx_sim:+.6f} vy={vy_sim:+.6f} wz={wz_sim:+.6f} "
        f"(max_lin={max_lin_vel:.4f}, max_ang={max_ang_vel:.4f})"
    )
    print(f"real base: vx={vx_real:+.6f} vy={vy_real:+.6f} wz={wz_real:+.6f}")
    print(f"arm rad  : {out['arm_target_rad']}")


if __name__ == "__main__":
    main()
