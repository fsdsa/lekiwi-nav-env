# Script Editor: EE spawn position marker

Wrist_Roll_08c_v1 아래에 0.001 scale cube를 생성해서 물체 스폰 위치를 표시.

```python
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

parent = "/World/envs/env_0/Robot/LeKiwi/Wrist_Roll_08c_v1"
marker_path = f"{parent}/SpawnMarker"

cube = UsdGeom.Cube.Define(stage, marker_path)
cube.GetSizeAttr().Set(1.0)

xform = UsdGeom.Xformable(cube.GetPrim())
xform.ClearXformOpOrder()

# EE_LOCAL_OFFSET = (0.04029, 0.0309, -0.06419)
translate = xform.AddTranslateOp()
translate.Set(Gf.Vec3d(0.04029, 0.0309, -0.06419))

scale = xform.AddScaleOp()
scale.Set(Gf.Vec3d(0.001, 0.001, 0.001))

# 빨간색
cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

print(f"Marker created at {marker_path}")
```
