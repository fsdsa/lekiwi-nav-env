"""
Isaac Sim Script Editor — REST_POSE 측정

이미 원격 조종 중인 상태에서, 현재 sim 관절값을 읽고 저장한다.

사용법:
  1. 기존 텔레옵으로 팔을 tucked pose로 만든다
  2. Script Editor에 이 코드 붙여넣고 Run → 현재 관절값 출력
  3. 콘솔에서:
       show_pose()              # 다시 출력
       save_pose()              # ~/rest_pose.json 저장
       save_pose("/tmp/p.json") # 경로 지정

  로봇 prim 경로가 다르면:
       init(prim="/World/lekiwi")
"""

import json
import math
import os

from omni.isaac.dynamic_control import _dynamic_control

# ══════════ 설정 ══════════
ROBOT_PRIM = "/World/LeKiwi/base_plate_layer1_v5"

ARM_JOINT_NAMES = [
    "STS3215_03a_v1_Revolute_45",
    "STS3215_03a_v1_1_Revolute_49",
    "STS3215_03a_v1_2_Revolute_51",
    "STS3215_03a_v1_3_Revolute_53",
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",
    "STS3215_03a_v1_4_Revolute_57",
]
FRIENDLY = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

_dc = None
_art = None
_dofs = []


def init(prim=None):
    global _dc, _art, _dofs, ROBOT_PRIM
    if prim:
        ROBOT_PRIM = prim

    _dc = _dynamic_control.acquire_dynamic_control_interface()
    _art = _dc.get_articulation(ROBOT_PRIM)

    if _art == _dynamic_control.INVALID_HANDLE:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        found = []
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                found.append(str(p.GetPath()))
        if found:
            print(f"[REST] '{ROBOT_PRIM}' 못 찾음. 발견된 articulation:")
            for a in found:
                print(f"  {a}")
            print(f"  → init(prim='{found[0]}') 로 재시도")
        else:
            print("[REST] articulation 없음. Play 상태인지 확인")
        return False

    _dofs = []
    for jn in ARM_JOINT_NAMES:
        dof = _dc.find_articulation_dof(_art, jn)
        _dofs.append(dof if dof != _dynamic_control.INVALID_HANDLE else None)

    n = sum(1 for d in _dofs if d is not None)
    print(f"[REST] DOF {n}/{len(ARM_JOINT_NAMES)} 발견")
    return True


def show_pose():
    if not _dofs:
        if not init():
            return
    print(f"\n{'='*55}")
    print("  ARM JOINT POSITIONS (sim)")
    print(f"  {'─'*45}")
    for i, (jn, fn) in enumerate(zip(ARM_JOINT_NAMES, FRIENDLY)):
        d = _dofs[i]
        if d is not None:
            s = _dc.get_dof_state(d, _dynamic_control.STATE_POS)
            v = s.pos
            print(f"    [{i+1}] {fn:16s} {v:+8.4f} rad ({math.degrees(v):+7.2f} deg)")
    print(f"{'='*55}")


def save_pose(path=None):
    if not _dofs:
        if not init():
            return
    if path is None:
        path = os.path.expanduser("~/rest_pose.json")

    vals = {}
    vals_sim = {}
    for i, (jn, fn) in enumerate(zip(ARM_JOINT_NAMES, FRIENDLY)):
        d = _dofs[i]
        if d is not None:
            s = _dc.get_dof_state(d, _dynamic_control.STATE_POS)
            vals[fn] = round(s.pos, 6)
            vals_sim[jn] = round(s.pos, 6)

    data = {
        "description": "LeKiwi tucked rest pose (teleop in Isaac Sim)",
        "unit": "radians",
        "joints": vals,
        "joints_sim_names": vals_sim,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    show_pose()
    print(f"\n  저장: {path}")
    print(f"\n  SIM_REST_RAD6 = [")
    for fn in FRIENDLY:
        print(f"      {vals.get(fn, 0):+.6f},   # {fn}")
    print("  ]")


# ══════════ 자동 실행 ══════════
if init():
    show_pose()
