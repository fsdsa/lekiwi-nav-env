"""
LeKiwi Robot — Isaac Lab ArticulationCfg.

6-DOF SO-100 arm + 3-wheel Kiwi omni-drive.
모든 drive 값은 Isaac Sim USD에서 검증된 값을 그대로 사용 (stiffness=None, damping=None).

Drive 속성 (04_read_drive_props.py로 확인):
  Arm joints:   type=force,        stiffness~[0.05‥1.8], damping~[0.001‥0.1], maxForce=10
  Wheel joints: type=acceleration, stiffness=0,           damping=174.53,       maxForce=inf
  Rollers:      DriveAPI 없음 (패시브)
"""
from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ── 조인트 이름 (01_inspect_scene.py에서 확인) ──────────────────────────
ARM_JOINT_NAMES = [
    "STS3215_03a_v1_Revolute_45",        # shoulder_pan
    "STS3215_03a_v1_1_Revolute_49",      # shoulder_lift
    "STS3215_03a_v1_2_Revolute_51",      # elbow_flex
    "STS3215_03a_v1_3_Revolute_53",      # wrist_flex
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",  # wrist_roll
    "STS3215_03a_v1_4_Revolute_57",      # gripper
]
GRIPPER_JOINT_NAME = "STS3215_03a_v1_4_Revolute_57"
GRIPPER_JOINT_IDX_IN_ARM = 5

WHEEL_JOINT_NAMES = [
    "axle_2_joint",  # Front-Left
    "axle_1_joint",  # Front-Right
    "axle_0_joint",  # Back
]

NUM_ARM_JOINTS = len(ARM_JOINT_NAMES)    # 6
NUM_WHEEL_JOINTS = len(WHEEL_JOINT_NAMES)  # 3
NUM_DOF = NUM_ARM_JOINTS + NUM_WHEEL_JOINTS  # 9 (제어 대상)

# ── Kiwi IK 파라미터 (시뮬레이터 실측 기하 상수) ────────────────────────────
# measured in simulator:
#   center -> wheel distance = 10.85 cm
#   wheel radius             = 4.90 cm
WHEEL_RADIUS = 0.049000000000      # m
BASE_RADIUS = 0.108500000000    # m (center -> wheel)
WHEEL_ANGLES_DEG = [-30.0, -150.0, 90.0]  # FL, FR, Back
WHEEL_ANGLES_RAD = [a * math.pi / 180.0 for a in WHEEL_ANGLES_DEG]

# Kiwi IK 행렬 M:  wheel_radps = M @ [vx, vy, wz]^T / r
#   M[i] = [cos(θ_i), sin(θ_i), L]
KIWI_M = [
    [math.cos(WHEEL_ANGLES_RAD[0]), math.sin(WHEEL_ANGLES_RAD[0]), BASE_RADIUS],
    [math.cos(WHEEL_ANGLES_RAD[1]), math.sin(WHEEL_ANGLES_RAD[1]), BASE_RADIUS],
    [math.cos(WHEEL_ANGLES_RAD[2]), math.sin(WHEEL_ANGLES_RAD[2]), BASE_RADIUS],
]

# ── Arm soft limits (REAL 측정값 베이크, rad) ────────────────────────────
# source: calibration/arm_limits_real2sim.json
ARM_LIMITS_BAKED_RAD = {
    "STS3215_03a_v1_Revolute_45": (-1.7453292519943295, 1.7208288783054544),
    "STS3215_03a_v1_1_Revolute_49": (-1.7453292519943295, 1.7453292519943295),
    "STS3215_03a_v1_2_Revolute_51": (-1.7173293174703563, 1.73444038856834),
    "STS3215_03a_v1_3_Revolute_53": (-1.7423760722109212, 1.7069379148100212),
    "STS3215_03a_Wrist_Roll_v1_Revolute_55": (-1.680545357903209, 1.5680259629028417),
    "STS3215_03a_v1_4_Revolute_57": (0.006566325252047892, 1.7453292519943295),
}

# ── 로봇 설정 ──────────────────────────────────────────────────────────
LEKIWI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.environ.get("LEKIWI_USD_PATH", "/home/yubin11/Downloads/lekiwi_robot.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=5.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # 5cm 위에서 드롭 (안정화 후 ~3cm)
        joint_pos={
            # arm rest
            "STS3215_03a_v1_Revolute_45": 0.0,
            "STS3215_03a_v1_1_Revolute_49": 0.0,
            "STS3215_03a_v1_2_Revolute_51": 0.0,
            "STS3215_03a_v1_3_Revolute_53": 0.0,
            "STS3215_03a_Wrist_Roll_v1_Revolute_55": 0.0,
            "STS3215_03a_v1_4_Revolute_57": 0.0,
            # wheels
            "axle_2_joint": 0.0,
            "axle_1_joint": 0.0,
            "axle_0_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            stiffness=None,   # USD force-drive 값 사용 (0.05~1.8)
            damping=None,     # USD force-drive 값 사용 (0.001~0.1)
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            stiffness=None,   # USD acceleration-drive 값 사용 (0.0)
            damping=None,     # USD acceleration-drive 값 사용 (174.53)
        ),
        "rollers": ImplicitActuatorCfg(
            joint_names_expr=["roller_.*"],
            stiffness=None,   # 패시브 (DriveAPI 없음)
            damping=None,
        ),
    },
)
