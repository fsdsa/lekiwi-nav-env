#!/usr/bin/env python3
"""
Build representative multi-object catalog JSON from USD objects.

Pipeline:
  1) Read USD paths from index JSONL
  2) Auto-extract bbox (AABB) from USD stage
  3) Cluster by bbox/aspect features
  4) Pick representative object per cluster
  5) Write object_catalog.json for multi-object teacher training

Example:
  python build_object_catalog.py \
    --index_jsonl /home/yubin11/isaac-objects/mujoco_obj_usd_index_all.jsonl \
    --output_json scripts/lekiwi_nav_env/object_catalog.json \
    --num_representatives 12
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ObjectEntry:
    name: str
    usd: str
    bbox_xyz: np.ndarray  # (3,), meters
    volume_m3: float
    category: int
    category_name: str
    mass_kg: float
    scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build representative object_catalog.json from USD index")
    parser.add_argument("--index_jsonl", type=str, required=True, help="Path to JSONL index (name/usd/status)")
    parser.add_argument("--output_json", type=str, required=True, help="Output object_catalog.json path")
    parser.add_argument(
        "--all_objects_json",
        type=str,
        default="",
        help="Optional output path for full extracted metadata (all objects)",
    )
    parser.add_argument("--num_representatives", type=int, default=12, help="Number of representative objects")
    parser.add_argument("--max_objects", type=int, default=0, help="Limit number of objects to process (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--default_mass", type=float, default=0.3)
    parser.add_argument(
        "--mass_mode",
        type=str,
        default="constant",
        choices=["constant", "volume_density"],
        help="How to set mass for generated catalog",
    )
    parser.add_argument("--density_kg_m3", type=float, default=350.0, help="Used only in volume_density mode")
    parser.add_argument("--min_mass", type=float, default=0.05, help="Used only in volume_density mode")
    parser.add_argument("--max_mass", type=float, default=1.50, help="Used only in volume_density mode")
    parser.add_argument("--min_bbox_dim", type=float, default=1e-4, help="Minimum valid bbox side length (m)")
    parser.add_argument("--dry_run", action="store_true", help="Print summary only")
    return parser.parse_args()


def _resolve_usd_path(raw_path: str, index_path: Path) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p
    candidate = (index_path.parent / p).resolve()
    return candidate


def load_index(index_jsonl: Path, max_objects: int = 0) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with index_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") not in (None, "ok"):
                continue
            usd = obj.get("usd")
            name = obj.get("name")
            if not usd or not name:
                continue
            items.append(obj)
            if max_objects > 0 and len(items) >= max_objects:
                break
    return items


def compute_bbox_xyz_from_usd(usd_path: Path, min_bbox_dim: float) -> np.ndarray | None:
    try:
        from pxr import Gf, Usd, UsdGeom
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pxr module is required. Run this in Isaac Sim / USD-enabled Python environment."
        ) from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        True,
    )

    def _range_from_prim(prim) -> Gf.Range3d | None:
        if prim is None or not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            return None
        bound = bbox_cache.ComputeWorldBound(prim)
        aligned = bound.ComputeAlignedBox()
        if aligned.IsEmpty():
            return None
        return aligned

    range3d = _range_from_prim(stage.GetDefaultPrim())

    if range3d is None:
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Imageable):
                continue
            aligned = _range_from_prim(prim)
            if aligned is None:
                continue
            pmin = aligned.GetMin()
            pmax = aligned.GetMax()
            mins.append(np.array([float(pmin[0]), float(pmin[1]), float(pmin[2])], dtype=np.float64))
            maxs.append(np.array([float(pmax[0]), float(pmax[1]), float(pmax[2])], dtype=np.float64))
        if not mins:
            return None
        bb_min = np.min(np.stack(mins, axis=0), axis=0)
        bb_max = np.max(np.stack(maxs, axis=0), axis=0)
        dims = np.maximum(bb_max - bb_min, 0.0)
    else:
        pmin = range3d.GetMin()
        pmax = range3d.GetMax()
        dims = np.array(
            [float(pmax[0] - pmin[0]), float(pmax[1] - pmin[1]), float(pmax[2] - pmin[2])],
            dtype=np.float64,
        )

    if not np.all(np.isfinite(dims)):
        return None
    dims = np.maximum(dims, 0.0)
    if float(np.min(dims)) < float(min_bbox_dim):
        return None
    return dims.astype(np.float32)


def categorize_bbox(dims_xyz: np.ndarray) -> tuple[int, str]:
    # Heuristic category from axis-aligned bbox only.
    # 0=cube_like, 1=cylinder_like, 2=round_like, 3=flat_box_like, 4=bottle_like, 5=irregular
    dims = np.sort(np.maximum(dims_xyz.astype(np.float64), 1e-8))
    a, b, c = float(dims[0]), float(dims[1]), float(dims[2])
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return 5, "irregular"

    ratio_ca = c / a
    ratio_ba = b / a
    ratio_cb = c / b

    near_cube = ratio_ca <= 1.25
    flat_like = (a / c <= 0.35) and (abs(b - c) / max(c, 1e-8) <= 0.35)
    long_like = (a / c <= 0.35) and (abs(a - b) / max(b, 1e-8) <= 0.35)

    if near_cube:
        return 0, "cube_like"
    if long_like and c >= 0.09:
        return 4, "bottle_like"
    if long_like:
        return 1, "cylinder_like"
    if flat_like:
        return 3, "flat_box_like"
    if ratio_ca <= 1.6 and ratio_cb <= 1.3 and ratio_ba <= 1.3:
        return 2, "round_like"
    return 5, "irregular"


def build_feature(dims_xyz: np.ndarray) -> np.ndarray:
    # Orientation-invariant feature from sorted bbox dimensions.
    d = np.sort(np.maximum(dims_xyz.astype(np.float64), 1e-8))
    a, b, c = d[0], d[1], d[2]
    return np.array([np.log(a), np.log(b), np.log(c), b / a, c / b, c / a], dtype=np.float64)


def _kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float64)

    first = int(rng.integers(0, n))
    centers[0] = x[first]

    closest_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 1e-12:
            centers[i] = x[int(rng.integers(0, n))]
            continue
        probs = closest_sq / total
        idx = int(rng.choice(n, p=probs))
        centers[i] = x[idx]
        dist_sq = np.sum((x - centers[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)

    return centers


def run_kmeans(x: np.ndarray, k: int, seed: int, max_iter: int = 100, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus_init(x, k, rng)
    labels = np.zeros((x.shape[0],), dtype=np.int64)

    for _ in range(max_iter):
        dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1)

        new_centers = np.zeros_like(centers)
        for ci in range(k):
            mask = new_labels == ci
            if not np.any(mask):
                new_centers[ci] = x[int(rng.integers(0, x.shape[0]))]
            else:
                new_centers[ci] = np.mean(x[mask], axis=0)

        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break

    return labels, centers


def pick_representatives(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> list[int]:
    reps: list[int] = []
    k = centers.shape[0]
    for ci in range(k):
        idxs = np.nonzero(labels == ci)[0]
        if idxs.size == 0:
            continue
        d2 = np.sum((x[idxs] - centers[ci]) ** 2, axis=1)
        best_local = int(np.argmin(d2))
        reps.append(int(idxs[best_local]))
    return reps


def estimate_mass_kg(volume_m3: float, args: argparse.Namespace) -> float:
    if args.mass_mode == "constant":
        return float(args.default_mass)
    est = float(volume_m3) * float(args.density_kg_m3)
    return float(np.clip(est, float(args.min_mass), float(args.max_mass)))


def main() -> None:
    args = parse_args()
    index_path = Path(args.index_jsonl).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    if not index_path.is_file():
        raise FileNotFoundError(f"index_jsonl not found: {index_path}")

    rows = load_index(index_path, max_objects=int(args.max_objects))
    if not rows:
        raise RuntimeError(f"No valid rows loaded from index: {index_path}")

    entries: list[ObjectEntry] = []
    skipped = 0
    for row in rows:
        name = str(row.get("name", "")).strip()
        usd_raw = str(row.get("usd", "")).strip()
        if not name or not usd_raw:
            skipped += 1
            continue
        usd_path = _resolve_usd_path(usd_raw, index_path)
        if not usd_path.is_file():
            skipped += 1
            continue

        bbox = compute_bbox_xyz_from_usd(usd_path, min_bbox_dim=float(args.min_bbox_dim))
        if bbox is None:
            skipped += 1
            continue

        volume_m3 = float(np.prod(bbox))
        category_id, category_name = categorize_bbox(bbox)
        mass_kg = estimate_mass_kg(volume_m3, args)

        entries.append(
            ObjectEntry(
                name=name,
                usd=str(usd_path),
                bbox_xyz=bbox,
                volume_m3=volume_m3,
                category=category_id,
                category_name=category_name,
                mass_kg=mass_kg,
                scale=float(args.scale),
            )
        )

    if not entries:
        raise RuntimeError("No valid objects with bbox extracted. Check USD files / pxr environment.")

    x = np.stack([build_feature(e.bbox_xyz) for e in entries], axis=0)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_norm = (x - mean) / std

    k = int(max(1, min(args.num_representatives, len(entries))))
    labels, centers = run_kmeans(x_norm, k=k, seed=int(args.seed))
    rep_indices = pick_representatives(x_norm, labels, centers)

    # Keep deterministic output order by cluster id then name.
    rep_indices = sorted(rep_indices, key=lambda i: (int(labels[i]), entries[i].name))

    cluster_sizes = {int(ci): int(np.sum(labels == ci)) for ci in range(k)}

    catalog: list[dict[str, Any]] = []
    for idx in rep_indices:
        e = entries[idx]
        ci = int(labels[idx])
        catalog.append(
            {
                "name": e.name,
                "usd": e.usd,
                "bbox": [float(e.bbox_xyz[0]), float(e.bbox_xyz[1]), float(e.bbox_xyz[2])],
                "mass": float(e.mass_kg),
                "scale": float(e.scale),
                "category": int(e.category),
                "category_name": e.category_name,
                "cluster_id": ci,
                "cluster_size": int(cluster_sizes.get(ci, 0)),
            }
        )

    cat_count: dict[int, int] = {}
    for e in entries:
        cat_count[e.category] = cat_count.get(e.category, 0) + 1

    print("=" * 72)
    print("Object catalog build summary")
    print(f"index: {index_path}")
    print(f"total rows loaded: {len(rows)}")
    print(f"valid bbox objects: {len(entries)}")
    print(f"skipped rows: {skipped}")
    print(f"k (representatives): {k}")
    print(f"category distribution (all valid): {cat_count}")
    print("=" * 72)

    if args.all_objects_json:
        all_out = Path(args.all_objects_json).expanduser().resolve()
        all_out.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "name": e.name,
                "usd": e.usd,
                "bbox": [float(e.bbox_xyz[0]), float(e.bbox_xyz[1]), float(e.bbox_xyz[2])],
                "volume_m3": float(e.volume_m3),
                "mass": float(e.mass_kg),
                "scale": float(e.scale),
                "category": int(e.category),
                "category_name": e.category_name,
                "cluster_id": int(labels[i]),
            }
            for i, e in enumerate(entries)
        ]
        if not args.dry_run:
            all_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote all-object metadata: {all_out}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        output_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote representative catalog: {output_path}")


if __name__ == "__main__":
    main()
