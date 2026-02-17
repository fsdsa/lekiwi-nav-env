#!/usr/bin/env python3
"""
Extract Kiwi geometry parameters (wheel/base radius) from USD assets.

What this script does:
1) Reads axle revolute joint anchors from robot USD to compute base radius.
2) Reads Kaya wheel assembly mesh geometry to estimate wheel radius.
3) Optionally updates constants in lekiwi_robot_cfg.py.

Usage:
  # Print extracted values
  python extract_kiwi_geometry_from_usd.py

  # Save report JSON
  python extract_kiwi_geometry_from_usd.py --output calibration/usd_geometry_baseline.json

  # Apply extracted values to lekiwi_robot_cfg.py
  python extract_kiwi_geometry_from_usd.py --apply_to_cfg
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np

try:
    from pxr import Usd, UsdGeom
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "pxr module is required. Run in Isaac Sim / USD-enabled Python environment.\n"
        "Example:\n"
        "  source ~/miniconda3/etc/profile.d/conda.sh\n"
        "  conda activate env_isaaclab\n"
        "  export LD_LIBRARY_PATH=/home/yubin11/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311/bin:$LD_LIBRARY_PATH\n"
        "  export PYTHONPATH=/home/yubin11/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311:$PYTHONPATH\n"
        "  python scripts/lekiwi_nav_env/extract_kiwi_geometry_from_usd.py"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Kiwi geometry from USD")
    parser.add_argument(
        "--robot_usd",
        type=str,
        default="/home/yubin11/Downloads/lekiwi_robot.usd",
        help="LeKiwi robot USD path",
    )
    parser.add_argument(
        "--wheel_asset_url",
        type=str,
        default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Kaya/props/Kaya_WheelAssembly.usd",
        help="Kaya wheel assembly USD URL",
    )
    parser.add_argument(
        "--wheel_asset_local",
        type=str,
        default="calibration/usd_refs/Kaya_WheelAssembly.usd",
        help="Local cache path for wheel assembly USD",
    )
    parser.add_argument(
        "--wheel_subtree",
        type=str,
        default="/Root/MX_12W_Drive_Wheel_ASM__1__1/__25in_Omni_Wheel__1__1",
        help="Wheel subtree path in wheel asset USD",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--apply_to_cfg",
        action="store_true",
        help="Update WHEEL_RADIUS/BASE_RADIUS in scripts/lekiwi_nav_env/lekiwi_robot_cfg.py",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="scripts/lekiwi_nav_env/lekiwi_robot_cfg.py",
        help="Target cfg file for --apply_to_cfg",
    )
    return parser.parse_args()


def _download_if_missing(url: str, local_path: Path):
    if local_path.is_file():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(local_path, "wb") as f:
        f.write(resp.read())


def _extract_base_radius_from_robot_usd(robot_usd: Path) -> dict:
    stage = Usd.Stage.Open(str(robot_usd))
    if not stage:
        raise RuntimeError(f"Failed to open robot USD: {robot_usd}")

    joint_paths = [
        "/World/LeKiwi/joints/axle_2_joint",
        "/World/LeKiwi/joints/axle_1_joint",
        "/World/LeKiwi/joints/axle_0_joint",
    ]
    rows = []
    for jp in joint_paths:
        prim = stage.GetPrimAtPath(jp)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"Missing joint prim in USD: {jp}")
        attr = prim.GetAttribute("physics:localPos0")
        if not attr:
            raise RuntimeError(f"Missing physics:localPos0 at {jp}")
        p = attr.Get()
        x = float(p[0])
        y = float(p[1])
        r = math.sqrt(x * x + y * y)
        rows.append({"joint_path": jp, "local_pos0_xy": [x, y], "radius_m": r})

    mean_r = float(np.mean([row["radius_m"] for row in rows]))
    return {"mean_radius_m": mean_r, "per_joint": rows}


def _resolve_wheel_subtree(stage: Usd.Stage, preferred: str) -> Usd.Prim:
    p = stage.GetPrimAtPath(preferred)
    if p and p.IsValid():
        return p
    for prim in stage.Traverse():
        if "Omni_Wheel" in prim.GetPath().pathString:
            return prim
    raise RuntimeError(f"Wheel subtree not found: {preferred}")


def _extract_wheel_radius_from_asset(asset_usd: Path, wheel_subtree: str) -> dict:
    stage = Usd.Stage.Open(str(asset_usd))
    if not stage:
        raise RuntimeError(f"Failed to open wheel asset USD: {asset_usd}")

    root = _resolve_wheel_subtree(stage, wheel_subtree)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_tf_inv = cache.GetLocalToWorldTransform(root).GetInverse()

    xz_points: list[tuple[float, float]] = []
    mesh_count = 0
    point_count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh_count += 1
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get() or []
        tf = cache.GetLocalToWorldTransform(prim)
        for p in pts:
            q = root_tf_inv.Transform(tf.Transform(p))
            x = float(q[0])
            z = float(q[2])
            xz_points.append((x, z))
        point_count += len(pts)

    if not xz_points:
        raise RuntimeError(f"No mesh points found under wheel subtree: {root.GetPath()}")

    xs = [p[0] for p in xz_points]
    zs = [p[1] for p in xz_points]
    cx = 0.5 * (min(xs) + max(xs))
    cz = 0.5 * (min(zs) + max(zs))
    rs = [math.sqrt((x - cx) ** 2 + (z - cz) ** 2) for x, z in xz_points]

    r_max = float(np.max(rs))
    r_p99 = float(np.percentile(rs, 99.0))
    r_p95 = float(np.percentile(rs, 95.0))
    return {
        "subtree_path": root.GetPath().pathString,
        "mesh_count": mesh_count,
        "point_count": point_count,
        "center_xz": [cx, cz],
        "radius_max_m": r_max,
        "radius_p99_m": r_p99,
        "radius_p95_m": r_p95,
        "recommended_radius_m": r_max,
    }


def _apply_to_cfg(cfg_path: Path, wheel_radius: float, base_radius: float):
    text = cfg_path.read_text(encoding="utf-8")
    text2 = re.sub(
        r"^WHEEL_RADIUS\s*=\s*[-+0-9.eE]+\s*#\s*m\s*$",
        f"WHEEL_RADIUS = {wheel_radius:.12f}      # m",
        text,
        flags=re.MULTILINE,
    )
    text2 = re.sub(
        r"^BASE_RADIUS\s*=\s*[-+0-9.eE]+\s*#\s*m\s*\(center -> wheel\)\s*$",
        f"BASE_RADIUS = {base_radius:.12f}    # m (center -> wheel)",
        text2,
        flags=re.MULTILINE,
    )
    if text2 == text:
        raise RuntimeError(f"Failed to update constants in cfg file: {cfg_path}")
    cfg_path.write_text(text2, encoding="utf-8")


def main():
    args = parse_args()

    robot_usd = Path(args.robot_usd).expanduser().resolve()
    wheel_local = Path(args.wheel_asset_local).expanduser().resolve()
    cfg_path = Path(args.cfg_path).expanduser().resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(f"robot_usd not found: {robot_usd}")

    _download_if_missing(args.wheel_asset_url, wheel_local)

    base = _extract_base_radius_from_robot_usd(robot_usd)
    wheel = _extract_wheel_radius_from_asset(wheel_local, args.wheel_subtree)

    out = {
        "robot_usd": str(robot_usd),
        "wheel_asset_local": str(wheel_local),
        "base_radius_from_axle_joints": base,
        "wheel_radius_from_asset_mesh": wheel,
        "recommended_constants": {
            "WHEEL_RADIUS": wheel["recommended_radius_m"],
            "BASE_RADIUS": base["mean_radius_m"],
        },
    }

    print("=" * 72)
    print("Kiwi USD geometry extraction")
    print(f"robot_usd          : {robot_usd}")
    print(f"wheel_asset_local  : {wheel_local}")
    print(f"recommended WHEEL  : {out['recommended_constants']['WHEEL_RADIUS']:.12f} m")
    print(f"recommended BASE   : {out['recommended_constants']['BASE_RADIUS']:.12f} m")
    print("=" * 72)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {output_path}")

    if args.apply_to_cfg:
        if not cfg_path.is_file():
            raise FileNotFoundError(f"cfg_path not found: {cfg_path}")
        _apply_to_cfg(
            cfg_path=cfg_path,
            wheel_radius=float(out["recommended_constants"]["WHEEL_RADIUS"]),
            base_radius=float(out["recommended_constants"]["BASE_RADIUS"]),
        )
        print(f"[updated] {cfg_path}")


if __name__ == "__main__":
    main()
