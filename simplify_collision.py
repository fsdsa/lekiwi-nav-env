"""
약병 USD의 32개 convex hull collision을 단순 cylinder로 교체.

Usage:
    python simplify_collision.py \
        --src ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
        --dst ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_cylinder_col.usd
"""
import argparse
import shutil
from pxr import Usd, UsdGeom, UsdPhysics, Gf


def simplify(src, dst):
    shutil.copy2(src, dst)

    stage = Usd.Stage.Open(dst)

    # 1. 기존 collider 비활성화
    removed = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "collider" in path.lower() or "collision" in path.lower():
            prim.SetActive(False)
            removed += 1
        elif prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
            prim.SetActive(False)
            removed += 1

    print(f"  Deactivated {removed} collision prims")

    # 2. bbox 계산 (visual mesh 기준)
    cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    root = stage.GetPseudoRoot()
    bbox = cache.ComputeWorldBound(root)
    r = bbox.ComputeAlignedRange()
    mn = r.GetMin()
    mx = r.GetMax()

    sx = float(mx[0] - mn[0])
    sy = float(mx[1] - mn[1])
    sz = float(mx[2] - mn[2])
    cx = float(mn[0] + mx[0]) * 0.5
    cy = float(mn[1] + mx[1]) * 0.5
    cz = float(mn[2] + mx[2]) * 0.5

    print(f"  BBox: {sx:.4f} x {sy:.4f} x {sz:.4f}")
    print(f"  Center: ({cx:.4f}, {cy:.4f}, {cz:.4f})")

    # 3. Cylinder collision 추가
    radius = max(sx, sy) * 0.5 * 0.95  # 5% 축소 (gripper가 잡을 수 있게)
    height = sz

    col_path = "/CollisionCylinder"
    col_prim = stage.DefinePrim(col_path, "Cylinder")
    cylinder = UsdGeom.Cylinder(col_prim)
    cylinder.GetRadiusAttr().Set(float(radius))
    cylinder.GetHeightAttr().Set(float(height))
    cylinder.GetAxisAttr().Set("Z")

    # 위치: bbox center
    xf = UsdGeom.Xformable(col_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))

    # visual 숨기기 (collision only)
    cylinder.GetPurposeAttr().Set("guide")

    # collision API 추가
    UsdPhysics.CollisionAPI.Apply(col_prim)

    stage.GetRootLayer().Save()

    print(f"  Cylinder: radius={radius:.4f}, height={height:.4f}")
    print(f"  Saved: {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    simplify(args.src, args.dst)
