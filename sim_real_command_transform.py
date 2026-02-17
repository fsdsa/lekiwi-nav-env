#!/usr/bin/env python3
"""
Apply base command transform between sim-space and real-space.

Default behavior:
  - Load command_transform from calibration/tuned_dynamics.json
  - Convert sim command -> real command (deployment path)

Examples:
  python sim_real_command_transform.py --mode sim_to_real --vx 0.2 --vy 0.0 --wz -1.0
  python sim_real_command_transform.py --mode real_to_sim --vx 0.2 --vy 0.0 --wz 1.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LINEAR_MAPS = {
    "identity": ((1.0, 0.0), (0.0, 1.0)),
    "flip_180": ((-1.0, 0.0), (0.0, -1.0)),
    "rot_cw_90": ((0.0, 1.0), (-1.0, 0.0)),
    "rot_ccw_90": ((0.0, -1.0), (1.0, 0.0)),
}

INVERSE_LINEAR_MAP = {
    "identity": "identity",
    "flip_180": "flip_180",
    "rot_cw_90": "rot_ccw_90",
    "rot_ccw_90": "rot_cw_90",
}


def _to_float_or_default(v: object, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(f):
        return float(default)
    return float(f)


def _normalize_cfg(raw: dict | None) -> dict:
    cfg = {
        "mode": "real_to_sim",
        "linear_map": "identity",
        "lin_scale": 1.0,
        "ang_scale": 1.0,
        "wz_sign": 1.0,
    }
    if not isinstance(raw, dict):
        return cfg

    linear_map = str(raw.get("linear_map", cfg["linear_map"])).strip().lower()
    if linear_map not in LINEAR_MAPS:
        linear_map = cfg["linear_map"]

    cfg["linear_map"] = linear_map
    cfg["lin_scale"] = _to_float_or_default(raw.get("lin_scale"), cfg["lin_scale"])
    cfg["ang_scale"] = _to_float_or_default(raw.get("ang_scale"), cfg["ang_scale"])
    cfg["wz_sign"] = _to_float_or_default(raw.get("wz_sign"), cfg["wz_sign"])
    return cfg


def load_transform_cfg(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        return _normalize_cfg(None)
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("command_transform") if isinstance(payload, dict) else None
    return _normalize_cfg(raw)


def _apply_linear_map(vx: float, vy: float, map_name: str) -> tuple[float, float]:
    m = LINEAR_MAPS[map_name]
    x = m[0][0] * vx + m[0][1] * vy
    y = m[1][0] * vx + m[1][1] * vy
    return float(x), float(y)


def real_to_sim(vx: float, vy: float, wz: float, cfg: dict) -> tuple[float, float, float]:
    x, y = _apply_linear_map(float(vx), float(vy), str(cfg["linear_map"]))
    x *= float(cfg["lin_scale"])
    y *= float(cfg["lin_scale"])
    z = float(wz) * float(cfg["wz_sign"]) * float(cfg["ang_scale"])
    return float(x), float(y), float(z)


def sim_to_real(vx: float, vy: float, wz: float, cfg: dict) -> tuple[float, float, float]:
    lin = float(cfg["lin_scale"])
    ang = float(cfg["ang_scale"])
    wz_sign = float(cfg["wz_sign"])
    if abs(lin) < 1e-8:
        raise ValueError("lin_scale is zero; cannot invert transform.")
    if abs(ang) < 1e-8 or abs(wz_sign) < 1e-8:
        raise ValueError("ang_scale or wz_sign is zero; cannot invert transform.")

    x = float(vx) / lin
    y = float(vy) / lin
    inv_map = INVERSE_LINEAR_MAP[str(cfg["linear_map"])]
    x, y = _apply_linear_map(x, y, inv_map)
    z = float(wz) / (wz_sign * ang)
    return float(x), float(y), float(z)


def main():
    parser = argparse.ArgumentParser(description="sim<->real base command transform utility")
    parser.add_argument(
        "--dynamics_json",
        type=str,
        default="calibration/tuned_dynamics.json",
        help="JSON file with command_transform (default: calibration/tuned_dynamics.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sim_to_real",
        choices=["sim_to_real", "real_to_sim"],
        help="conversion direction",
    )
    parser.add_argument("--vx", type=float, required=True, help="input vx (m/s)")
    parser.add_argument("--vy", type=float, required=True, help="input vy (m/s)")
    parser.add_argument("--wz", type=float, required=True, help="input wz (rad/s)")
    parser.add_argument("--linear_map", type=str, default="auto", choices=["auto"] + list(LINEAR_MAPS.keys()))
    parser.add_argument("--lin_scale", type=float, default=None)
    parser.add_argument("--ang_scale", type=float, default=None)
    parser.add_argument("--wz_sign", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="print output as JSON")
    args = parser.parse_args()

    cfg = load_transform_cfg(args.dynamics_json)

    if args.linear_map != "auto":
        cfg["linear_map"] = args.linear_map
    if args.lin_scale is not None:
        cfg["lin_scale"] = float(args.lin_scale)
    if args.ang_scale is not None:
        cfg["ang_scale"] = float(args.ang_scale)
    if args.wz_sign is not None:
        cfg["wz_sign"] = float(args.wz_sign)

    if args.mode == "real_to_sim":
        ox, oy, oz = real_to_sim(args.vx, args.vy, args.wz, cfg)
    else:
        ox, oy, oz = sim_to_real(args.vx, args.vy, args.wz, cfg)

    out = {
        "mode": args.mode,
        "input": {"vx": float(args.vx), "vy": float(args.vy), "wz": float(args.wz)},
        "output": {"vx": ox, "vy": oy, "wz": oz},
        "command_transform": cfg,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"mode: {args.mode}")
    print(
        "command_transform: "
        f"map={cfg['linear_map']}, lin={cfg['lin_scale']:.6f}, "
        f"ang={cfg['ang_scale']:.6f}, wz_sign={cfg['wz_sign']:+.3f}"
    )
    print(f"in : vx={args.vx:+.6f} vy={args.vy:+.6f} wz={args.wz:+.6f}")
    print(f"out: vx={ox:+.6f} vy={oy:+.6f} wz={oz:+.6f}")


if __name__ == "__main__":
    main()
