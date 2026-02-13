"""
LeKiwi Fetch-Navigation — Isaac Lab DirectRLEnv.

Task phase:
  SEARCH -> APPROACH -> GRASP -> RETURN

Design goal:
  - RL 학습은 state-based(카메라 원본 미사용)로 유지
  - 대신 detector/depth에서 얻는 추상 상태를 observation에 포함
  - BC warm-start + sparse-dominant reward로 "공 찾아서 가져오기" 파이프라인에 맞춤

Observation:
  Default (33D):
    [0:2]   target_xy_body       phase별 타깃 방향 (search/object/home)
    [2:4]   object_xy_body       물체 상대 위치 (비가시 시 0)
    [4:6]   home_xy_body         홈 상대 위치
    [6:10]  phase_onehot         [search, approach, grasp, return]
    [10]    object_visible       0/1
    [11]    object_grasped       0/1
    [12:15] base_lin_vel_body
    [15:18] base_ang_vel_body
    [18:24] arm_joint_pos
    [24:30] arm_joint_vel
    [30:33] wheel_vel
  Multi-object privileged (37D):
    + [33:36] object_bbox_normalized (x, y, z)
    + [36]    object_category_normalized

Action (9D):
  [0:3]   body_vel_cmd         (vx, vy, wz) -> Kiwi IK -> 바퀴 속도
  [3:9]   arm_pos_target       팔 관절 목표 위치 (absolute, rad)
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from lekiwi_robot_cfg import (
    ARM_JOINT_NAMES,
    BASE_RADIUS,
    GRIPPER_JOINT_NAME,
    GRIPPER_JOINT_IDX_IN_ARM,
    LEKIWI_CFG,
    WHEEL_ANGLES_RAD,
    WHEEL_JOINT_NAMES,
    WHEEL_RADIUS,
)


# Phase IDs
PHASE_SEARCH = 0
PHASE_APPROACH = 1
PHASE_GRASP = 2
PHASE_RETURN = 3
NUM_PHASES = 4


@configclass
class LeKiwiNavEnvCfg(DirectRLEnvCfg):
    """Fetch + Navigation 환경 설정."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.02,               # 50 Hz physics
        render_interval=2,     # 25 Hz render
        gravity=(0.0, 0.0, -9.81),
        device="cpu",
    )
    decimation: int = 2        # 25 Hz control
    episode_length_s: float = 20.0

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=10.0,
        replicate_physics=True,
    )

    # Robot
    robot_cfg = LEKIWI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Spaces
    observation_space: int = 33
    action_space: int = 9
    state_space: int = 0

    # Action scaling
    max_lin_vel: float = 0.5
    max_ang_vel: float = 3.0
    arm_action_scale: float = 1.5
    # Optional: tune_sim_dynamics.py output JSON path
    dynamics_json: str | None = None
    # If true, apply lin_cmd_scale / ang_cmd_scale to max_lin_vel / max_ang_vel
    dynamics_apply_cmd_scale: bool = True
    # Optional: per-joint arm limit JSON. Expected format:
    #   {"joint_limits_rad": {"<sim_joint_name>": {"min": <rad>, "max": <rad>}, ...}}
    # or
    #   {"<sim_joint_name>": [min_rad, max_rad], ...}
    arm_limit_json: str | None = None
    arm_limit_margin_rad: float = 0.0
    # Map arm action [-1,1] to joint soft limits when finite.
    arm_action_to_limits: bool = True

    # Fetch task geometry
    object_dist_min: float = 1.0
    object_dist_max: float = 2.5
    search_dist_min: float = 0.6
    search_dist_max: float = 2.0
    vision_fov_deg: float = 120.0
    vision_max_dist: float = 3.0
    search_reveal_dist: float = 0.6
    search_reveal_scan: float = 1.2
    approach_thresh: float = 0.35
    grasp_thresh: float = 0.20
    return_thresh: float = 0.30
    object_height: float = 0.03
    # Optional: physics-based grasp mode. If empty, legacy proximity grasp is used.
    object_usd: str = ""
    object_mass: float = 0.3
    object_scale: float = 1.0
    # Contact sensor prim path for gripper body (single body prim per env recommended).
    gripper_contact_prim_path: str = ""
    object_prim_path: str = "/World/envs/env_.*/Object"
    object_filter_prim_expr: str = "/World/envs/env_.*/Object"
    grasp_gripper_threshold: float = -0.3
    grasp_contact_threshold: float = 0.5
    grasp_max_object_dist: float = 0.25
    grasp_attach_height: float = 0.15
    grasp_timeout_steps: int = 75  # GRASP phase 최대 지속 step (0 = 무제한)
    # Optional: representative multi-object catalog JSON for privileged teacher observations.
    # Format: [{"usd": "...", "bbox": [x, y, z], "mass": 0.2, "scale": 1.0, "category": 0}, ...]
    multi_object_json: str = ""
    num_object_categories: int = 6

    # Backward compatibility (기존 스크립트에서 참조)
    goal_reached_thresh: float = 0.30

    # Rewards
    rew_time_penalty: float = -0.01
    rew_effort_weight: float = -0.01
    rew_arm_move_weight: float = -0.02

    rew_search_progress_weight: float = 2.0
    rew_search_move_weight: float = 0.05
    rew_search_turn_weight: float = 0.02
    rew_detect_bonus: float = 4.0

    rew_approach_progress_weight: float = 6.0
    rew_approach_heading_weight: float = 0.2
    rew_approach_vel_weight: float = 0.5
    rew_approach_ready_bonus: float = 1.5

    rew_grasp_success_bonus: float = 8.0

    rew_return_progress_weight: float = 7.0
    rew_return_heading_weight: float = 0.2
    rew_return_vel_weight: float = 0.6
    rew_carry_bonus: float = 0.05
    rew_return_success_bonus: float = 15.0

    # Termination
    max_dist_from_origin: float = 6.0


class LeKiwiNavEnv(DirectRLEnv):
    """LeKiwi Fetch Navigation RL 환경."""

    cfg: LeKiwiNavEnvCfg

    def __init__(self, cfg: LeKiwiNavEnvCfg, render_mode: str | None = None, **kwargs):
        self._multi_object = bool(str(getattr(cfg, "multi_object_json", "")).strip())
        if self._multi_object:
            cfg.observation_space = 37
        super().__init__(cfg, render_mode, **kwargs)

        self._multi_object = bool(str(self.cfg.multi_object_json).strip())
        self._physics_grasp = bool(str(self.cfg.object_usd).strip()) or self._multi_object
        self.object_rigids: list[RigidObject] = list(getattr(self, "object_rigids", []))
        self.object_rigid: RigidObject | None = getattr(self, "object_rigid", None)
        self.contact_sensor: ContactSensor | None = getattr(self, "contact_sensor", None)
        self._object_catalog: list[dict] = list(getattr(self, "_object_catalog", []))
        self._num_object_types: int = int(getattr(self, "_num_object_types", 0))

        # Joint indices
        self.arm_idx, _ = self.robot.find_joints(ARM_JOINT_NAMES)
        self.wheel_idx, _ = self.robot.find_joints(WHEEL_JOINT_NAMES)
        self.arm_idx = torch.tensor(self.arm_idx, device=self.device)
        self.wheel_idx = torch.tensor(self.wheel_idx, device=self.device)
        gripper_ids, _ = self.robot.find_joints([GRIPPER_JOINT_NAME])
        if len(gripper_ids) != 1:
            raise RuntimeError(f"Expected single gripper joint: {GRIPPER_JOINT_NAME}, got {len(gripper_ids)}")
        self.gripper_idx = int(gripper_ids[0])
        self.gripper_arm_offset = int(GRIPPER_JOINT_IDX_IN_ARM)
        if self.gripper_arm_offset < 0 or self.gripper_arm_offset >= len(self.arm_idx):
            self.gripper_arm_offset = len(self.arm_idx) - 1
        self._base_max_lin_vel = float(self.cfg.max_lin_vel)
        self._base_max_ang_vel = float(self.cfg.max_ang_vel)
        self._applied_dynamics_params: dict[str, float] | None = None

        # Kiwi IK matrix
        self.base_radius = float(BASE_RADIUS)
        self.wheel_radius = float(WHEEL_RADIUS)
        angles = torch.tensor(WHEEL_ANGLES_RAD, dtype=torch.float32, device=self.device)
        self.kiwi_M = torch.stack(
            [
                torch.cos(angles),
                torch.sin(angles),
                torch.full_like(angles, self.base_radius),
            ],
            dim=-1,
        )
        self._vision_half_fov_rad = math.radians(self.cfg.vision_fov_deg * 0.5)

        # Task buffers
        self.home_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.search_target_w = torch.zeros(self.num_envs, 3, device=self.device)
        # test_env.py 및 기존 코드 호환: goal_pos를 object_pos alias로 유지
        self.goal_pos_w = self.object_pos_w
        self.active_object_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_bbox = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_category_id = torch.zeros(self.num_envs, device=self.device)
        self._bbox_norm_scale = 0.2

        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_revealed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.search_scan_score = torch.zeros(self.num_envs, device=self.device)
        self.prev_search_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_object_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_home_dist = torch.zeros(self.num_envs, device=self.device)

        self.just_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_reached_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_entry_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Action / metrics buffers
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.episode_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self.reached_count = 0
        self._cached_metrics: Dict[str, torch.Tensor] | None = None

        if self._multi_object:
            if self._num_object_types <= 0 or len(self._object_catalog) == 0:
                raise RuntimeError("multi_object_json was provided but catalog is empty.")
            catalog_bbox_rows: list[list[float]] = []
            for obj in self._object_catalog:
                raw_bbox = obj.get("bbox", [0.05, 0.05, 0.05])
                if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 3:
                    raw_bbox = [0.05, 0.05, 0.05]
                try:
                    bx = float(raw_bbox[0])
                    by = float(raw_bbox[1])
                    bz = float(raw_bbox[2])
                except (TypeError, ValueError):
                    bx, by, bz = 0.05, 0.05, 0.05
                try:
                    scale = float(obj.get("scale", 1.0))
                except (TypeError, ValueError):
                    scale = 1.0
                if not math.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
                catalog_bbox_rows.append(
                    [
                        max(bx, 1e-6) * scale,
                        max(by, 1e-6) * scale,
                        max(bz, 1e-6) * scale,
                    ]
                )
            self._catalog_bbox = torch.tensor(
                catalog_bbox_rows,
                dtype=torch.float32,
                device=self.device,
            )
            self._catalog_category = torch.tensor(
                [float(obj.get("category", 0)) for obj in self._object_catalog],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.object_bbox[:] = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=self.device)
            self.object_category_id[:] = 0.0

        self._maybe_apply_tuned_dynamics_from_cfg()
        self._maybe_apply_arm_limits_from_cfg()

        print(f"  [LeKiwiFetchEnv] obs={self.cfg.observation_space} act={self.cfg.action_space}")
        print(f"  [LeKiwiFetchEnv] arm_idx={self.arm_idx.tolist()} wheel_idx={self.wheel_idx.tolist()}")

    # Scene setup
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object_rigids = []
        self.object_rigid = None
        self.contact_sensor = None
        self._object_catalog = []
        self._num_object_types = 0
        self._multi_object = bool(str(self.cfg.multi_object_json).strip())
        self._physics_grasp = bool(str(self.cfg.object_usd).strip()) or self._multi_object

        if self._multi_object:
            if not str(self.cfg.gripper_contact_prim_path).strip():
                raise ValueError(
                    "Multi-object physics grasp requires cfg.gripper_contact_prim_path. "
                    "Set --gripper_contact_prim_path to a single gripper body prim path."
                )
            mo_path = os.path.expanduser(self.cfg.multi_object_json)
            if not os.path.isfile(mo_path):
                raise FileNotFoundError(f"multi_object_json not found: {mo_path}")
            with open(mo_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list) or len(payload) == 0:
                raise ValueError("multi_object_json must be a non-empty JSON list.")
            self._object_catalog = payload
            self._num_object_types = len(self._object_catalog)

            filter_exprs: list[str] = []
            for oi, obj_info in enumerate(self._object_catalog):
                if not isinstance(obj_info, dict):
                    raise ValueError(f"multi_object_json[{oi}] must be an object.")
                obj_usd = os.path.expanduser(str(obj_info.get("usd", "")).strip())
                if not obj_usd:
                    raise ValueError(f"multi_object_json[{oi}].usd is missing.")
                if not os.path.isfile(obj_usd):
                    raise FileNotFoundError(f"multi_object_json[{oi}].usd not found: {obj_usd}")
                obj_mass = float(obj_info.get("mass", self.cfg.object_mass))
                obj_scale = float(obj_info.get("scale", self.cfg.object_scale))
                prim_path = f"/World/envs/env_.*/Object_{oi}"
                filter_exprs.append(prim_path)

                obj_cfg = RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=obj_usd,
                        scale=(obj_scale, obj_scale, obj_scale),
                        activate_contact_sensors=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=True,
                            kinematic_enabled=False,
                            disable_gravity=False,
                            max_linear_velocity=2.0,
                            max_angular_velocity=5.0,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj_mass),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    ),
                    # Spawn hidden by default; reset selects one object type per env.
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
                )
                self.object_rigids.append(RigidObject(obj_cfg))

            contact_cfg = ContactSensorCfg(
                prim_path=str(self.cfg.gripper_contact_prim_path),
                update_period=0.0,
                history_length=2,
                filter_prim_paths_expr=filter_exprs,
            )
            self.contact_sensor = ContactSensor(contact_cfg)

        elif self._physics_grasp:
            object_usd = os.path.expanduser(self.cfg.object_usd)
            if not os.path.isfile(object_usd):
                raise FileNotFoundError(f"object_usd not found: {object_usd}")
            if not str(self.cfg.gripper_contact_prim_path).strip():
                raise ValueError(
                    "Physics grasp requires cfg.gripper_contact_prim_path. "
                    "Set train/collect argument --gripper_contact_prim_path to a single gripper body prim path."
                )

            object_cfg = RigidObjectCfg(
                prim_path=self.cfg.object_prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=object_usd,
                    scale=(float(self.cfg.object_scale), float(self.cfg.object_scale), float(self.cfg.object_scale)),
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=False,
                        disable_gravity=False,
                        max_linear_velocity=2.0,
                        max_angular_velocity=5.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=float(self.cfg.object_mass)),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, float(self.cfg.object_height))),
            )
            self.object_rigid = RigidObject(object_cfg)
            contact_cfg = ContactSensorCfg(
                prim_path=str(self.cfg.gripper_contact_prim_path),
                update_period=0.0,
                history_length=2,
                filter_prim_paths_expr=[str(self.cfg.object_filter_prim_expr)],
            )
            self.contact_sensor = ContactSensor(contact_cfg)

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        for oi, rigid in enumerate(self.object_rigids):
            self.scene.rigid_objects[f"object_{oi}"] = rigid
        if self.object_rigid is not None:
            self.scene.rigid_objects["object"] = self.object_rigid
        if self.contact_sensor is not None:
            self.scene.sensors["gripper_contact"] = self.contact_sensor

    # Tuned dynamics
    def _maybe_apply_tuned_dynamics_from_cfg(self):
        if not self.cfg.dynamics_json:
            return

        self.apply_tuned_dynamics(self.cfg.dynamics_json)

    def _extract_tuned_params(self, payload: dict) -> dict:
        if isinstance(payload.get("best_params"), dict):
            return payload["best_params"]
        if isinstance(payload.get("params"), dict):
            return payload["params"]
        return payload

    @staticmethod
    def _extract_arm_limits_payload(payload: dict) -> dict[str, tuple[float, float]]:
        if not isinstance(payload, dict):
            return {}

        block = payload.get("joint_limits_rad")
        if block is None:
            block = payload.get("arm_joint_limits_rad", payload)
        if not isinstance(block, dict):
            return {}

        out: dict[str, tuple[float, float]] = {}
        for sim_joint, val in block.items():
            lo, hi = None, None
            if isinstance(val, dict) and "min" in val and "max" in val:
                lo = val.get("min")
                hi = val.get("max")
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                lo, hi = val[0], val[1]
            if lo is None or hi is None:
                continue
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(lo_f) or not math.isfinite(hi_f):
                continue
            if hi_f < lo_f:
                lo_f, hi_f = hi_f, lo_f
            if abs(hi_f - lo_f) < 1e-8:
                continue
            out[str(sim_joint)] = (lo_f, hi_f)
        return out

    def _maybe_apply_arm_limits_from_cfg(self):
        if not self.cfg.arm_limit_json:
            return

        path = os.path.expanduser(self.cfg.arm_limit_json)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"arm_limit_json not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        limits_by_joint = self._extract_arm_limits_payload(payload)
        if not limits_by_joint:
            raise ValueError(
                f"arm_limit_json has no valid limits: {path}. "
                "Expected {'joint_limits_rad': {'<joint>': {'min': ..., 'max': ...}}}."
            )

        margin = float(self.cfg.arm_limit_margin_rad)
        joint_ids = []
        limits_rows = []
        for sim_joint, (lo, hi) in limits_by_joint.items():
            idxs, _ = self.robot.find_joints([sim_joint])
            if len(idxs) != 1:
                continue
            joint_ids.append(int(idxs[0]))
            limits_rows.append([lo - margin, hi + margin])

        if not joint_ids:
            raise ValueError(f"No valid sim joints found from arm_limit_json: {path}")

        limits = torch.tensor(limits_rows, dtype=torch.float32, device=self.device).unsqueeze(0)
        limits = limits.repeat(self.num_envs, 1, 1)
        self.robot.write_joint_position_limit_to_sim(limits, joint_ids=joint_ids, warn_limit_violation=False)
        print(f"  [ArmLimits] applied from {path}: {len(joint_ids)} joints (margin={margin:.4f} rad)")

    @staticmethod
    def _safe_float(params: dict, key: str, default: float) -> float:
        try:
            return float(params.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def apply_tuned_dynamics(self, dynamics_json: str):
        path = os.path.expanduser(dynamics_json)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"dynamics_json not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        raw_params = self._extract_tuned_params(payload)
        if not isinstance(raw_params, dict):
            raise ValueError("Invalid dynamics JSON: expected dict or {'best_params': dict}")

        params = {
            "wheel_stiffness_scale": self._safe_float(raw_params, "wheel_stiffness_scale", 1.0),
            "wheel_damping_scale": self._safe_float(raw_params, "wheel_damping_scale", 1.0),
            "wheel_armature_scale": self._safe_float(raw_params, "wheel_armature_scale", 1.0),
            "wheel_friction_coeff": self._safe_float(raw_params, "wheel_friction_coeff", 0.0),
            "wheel_dynamic_friction_coeff": self._safe_float(raw_params, "wheel_dynamic_friction_coeff", 0.0),
            "wheel_viscous_friction_coeff": self._safe_float(raw_params, "wheel_viscous_friction_coeff", 0.0),
            "arm_stiffness_scale": self._safe_float(raw_params, "arm_stiffness_scale", 1.0),
            "arm_damping_scale": self._safe_float(raw_params, "arm_damping_scale", 1.0),
            "arm_armature_scale": self._safe_float(raw_params, "arm_armature_scale", 1.0),
            "lin_cmd_scale": self._safe_float(raw_params, "lin_cmd_scale", 1.0),
            "ang_cmd_scale": self._safe_float(raw_params, "ang_cmd_scale", 1.0),
        }

        def _pick_positive_finite(*vals: object) -> float | None:
            for v in vals:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f) and f > 1e-8:
                    return f
            return None

        tuned_wheel_radius = _pick_positive_finite(
            payload.get("wheel_radius_used"),
            payload.get("measured_wheel_radius"),
            raw_params.get("wheel_radius_used"),
            raw_params.get("measured_wheel_radius"),
        )
        tuned_base_radius = _pick_positive_finite(
            payload.get("base_radius_used"),
            payload.get("measured_base_radius"),
            raw_params.get("base_radius_used"),
            raw_params.get("measured_base_radius"),
        )
        if tuned_wheel_radius is not None:
            self.wheel_radius = tuned_wheel_radius
        if tuned_base_radius is not None:
            self.base_radius = tuned_base_radius
            angles = torch.tensor(WHEEL_ANGLES_RAD, dtype=torch.float32, device=self.device)
            self.kiwi_M = torch.stack(
                [
                    torch.cos(angles),
                    torch.sin(angles),
                    torch.full_like(angles, self.base_radius),
                ],
                dim=-1,
            )

        wheel_ids = self.wheel_idx.tolist()
        arm_ids = self.arm_idx.tolist()

        base_wheel_stiff = self.robot.data.joint_stiffness[:, self.wheel_idx].clone()
        base_wheel_damping = self.robot.data.joint_damping[:, self.wheel_idx].clone()
        base_wheel_armature = self.robot.data.joint_armature[:, self.wheel_idx].clone()
        base_arm_stiff = self.robot.data.joint_stiffness[:, self.arm_idx].clone()
        base_arm_damping = self.robot.data.joint_damping[:, self.arm_idx].clone()
        base_arm_armature = self.robot.data.joint_armature[:, self.arm_idx].clone()

        self.robot.write_joint_stiffness_to_sim(base_wheel_stiff * params["wheel_stiffness_scale"], joint_ids=wheel_ids)
        self.robot.write_joint_damping_to_sim(base_wheel_damping * params["wheel_damping_scale"], joint_ids=wheel_ids)
        self.robot.write_joint_armature_to_sim(base_wheel_armature * params["wheel_armature_scale"], joint_ids=wheel_ids)

        # friction 계수 API는 Isaac Lab 0.44.x에서 제공됨
        if hasattr(self.robot, "write_joint_friction_coefficient_to_sim"):
            self.robot.write_joint_friction_coefficient_to_sim(
                torch.full_like(base_wheel_damping, params["wheel_friction_coeff"]), joint_ids=wheel_ids
            )
        if hasattr(self.robot, "write_joint_dynamic_friction_coefficient_to_sim"):
            self.robot.write_joint_dynamic_friction_coefficient_to_sim(
                torch.full_like(base_wheel_damping, params["wheel_dynamic_friction_coeff"]), joint_ids=wheel_ids
            )
        if hasattr(self.robot, "write_joint_viscous_friction_coefficient_to_sim"):
            self.robot.write_joint_viscous_friction_coefficient_to_sim(
                torch.full_like(base_wheel_damping, params["wheel_viscous_friction_coeff"]), joint_ids=wheel_ids
            )

        self.robot.write_joint_stiffness_to_sim(base_arm_stiff * params["arm_stiffness_scale"], joint_ids=arm_ids)
        self.robot.write_joint_damping_to_sim(base_arm_damping * params["arm_damping_scale"], joint_ids=arm_ids)
        self.robot.write_joint_armature_to_sim(base_arm_armature * params["arm_armature_scale"], joint_ids=arm_ids)

        if self.cfg.dynamics_apply_cmd_scale:
            self.cfg.max_lin_vel = self._base_max_lin_vel * params["lin_cmd_scale"]
            self.cfg.max_ang_vel = self._base_max_ang_vel * params["ang_cmd_scale"]

        self._applied_dynamics_params = params
        print("\n  [Dynamics] tuned parameters applied")
        print(f"  [Dynamics] source: {path}")
        print(
            f"  [Dynamics] max_lin_vel={self.cfg.max_lin_vel:.4f} "
            f"max_ang_vel={self.cfg.max_ang_vel:.4f}"
        )
        print(
            f"  [Dynamics] wheel_radius={self.wheel_radius:.6f} "
            f"base_radius={self.base_radius:.6f}"
        )

    # Action pipeline
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        vx = self.actions[:, 0] * self.cfg.max_lin_vel
        vy = self.actions[:, 1] * self.cfg.max_lin_vel
        wz = self.actions[:, 2] * self.cfg.max_ang_vel

        body_cmd = torch.stack([vx, vy, wz], dim=-1)
        wheel_radps = body_cmd @ self.kiwi_M.T / self.wheel_radius

        vel_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        vel_target[:, self.wheel_idx] = wheel_radps
        self.robot.set_joint_velocity_target(vel_target)

        arm_action = self.actions[:, 3:9]
        if self.cfg.arm_action_to_limits:
            arm_limits = self.robot.data.soft_joint_pos_limits[:, self.arm_idx]
            arm_lo = arm_limits[..., 0]
            arm_hi = arm_limits[..., 1]
            finite = torch.isfinite(arm_lo) & torch.isfinite(arm_hi) & ((arm_hi - arm_lo) > 1e-6)

            # finite limit가 있으면 [-1,1]을 limit 범위로 직접 매핑하고, 없으면 기존 scale을 유지.
            center = 0.5 * (arm_lo + arm_hi)
            half = 0.5 * (arm_hi - arm_lo)
            mapped = center + arm_action * half
            fallback = arm_action * self.cfg.arm_action_scale
            arm_targets = torch.where(finite, mapped, fallback)
            arm_targets = torch.where(finite, torch.clamp(arm_targets, arm_lo, arm_hi), arm_targets)
        else:
            arm_targets = arm_action * self.cfg.arm_action_scale
        pos_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        pos_target[:, self.arm_idx] = arm_targets
        self.robot.set_joint_position_target(pos_target)

        # In physics grasp mode, keep grasped object attached to robot body for stable carry behavior.
        if self._multi_object and bool(self.object_grasped.any()) and len(self.object_rigids) > 0:
            grasped_ids = self.object_grasped.nonzero(as_tuple=False).squeeze(-1)
            if len(grasped_ids) > 0:
                for oi, rigid in enumerate(self.object_rigids):
                    oi_mask = self.active_object_idx[grasped_ids] == oi
                    if not oi_mask.any():
                        continue
                    env_ids = grasped_ids[oi_mask]
                    obj_pose = rigid.data.root_pose_w[env_ids].clone()
                    robot_pose = self.robot.data.root_state_w[env_ids, :7]
                    obj_pose[:, :2] = robot_pose[:, :2]
                    obj_pose[:, 2] = float(self.cfg.grasp_attach_height)
                    obj_pose[:, 3:7] = robot_pose[:, 3:7]
                    rigid.write_root_pose_to_sim(obj_pose, env_ids=env_ids)
                    zero_vel = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device)
                    rigid.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
                    self.object_pos_w[env_ids] = obj_pose[:, :3]
        elif self._physics_grasp and self.object_rigid is not None and bool(self.object_grasped.any()):
            grasped_ids = self.object_grasped.nonzero(as_tuple=False).squeeze(-1)
            if len(grasped_ids) > 0:
                obj_pose = self.object_rigid.data.root_pose_w[grasped_ids].clone()
                robot_pose = self.robot.data.root_state_w[grasped_ids, :7]
                obj_pose[:, :2] = robot_pose[:, :2]
                obj_pose[:, 2] = float(self.cfg.grasp_attach_height)
                obj_pose[:, 3:7] = robot_pose[:, 3:7]
                self.object_rigid.write_root_pose_to_sim(obj_pose, env_ids=grasped_ids)
                zero_vel = torch.zeros((len(grasped_ids), 6), dtype=torch.float32, device=self.device)
                self.object_rigid.write_root_velocity_to_sim(zero_vel, env_ids=grasped_ids)
                self.object_pos_w[grasped_ids] = obj_pose[:, :3]

    def _contact_force_per_env(self) -> torch.Tensor:
        force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not self._physics_grasp or self.contact_sensor is None:
            return force

        # Preferred: filtered force matrix (sensor body vs filtered object bodies).
        force_matrix = self.contact_sensor.data.force_matrix_w
        if force_matrix is not None:
            mag = torch.norm(force_matrix, dim=-1)
            mag = mag.reshape(mag.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag
            if mag.numel() % self.num_envs == 0:
                return mag.reshape(self.num_envs, -1).sum(dim=-1)

        # Fallback: net contact force on sensor body/bodies.
        net = self.contact_sensor.data.net_forces_w
        if net is not None:
            mag = torch.norm(net, dim=-1)
            mag = mag.reshape(mag.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag
            if mag.numel() % self.num_envs == 0:
                return mag.reshape(self.num_envs, -1).sum(dim=-1)
        return force

    # Task metric helpers
    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        if self._multi_object and len(self.object_rigids) > 0:
            not_grasped = ~self.object_grasped
            if not_grasped.any():
                for oi, rigid in enumerate(self.object_rigids):
                    mask = (self.active_object_idx == oi) & not_grasped
                    if not mask.any():
                        continue
                    ids = mask.nonzero(as_tuple=False).squeeze(-1)
                    self.object_pos_w[ids] = rigid.data.root_pos_w[ids]
        elif self._physics_grasp and self.object_rigid is not None:
            not_grasped = ~self.object_grasped
            if not_grasped.any():
                self.object_pos_w[not_grasped] = self.object_rigid.data.root_pos_w[not_grasped]

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        object_delta_w = self.object_pos_w - root_pos_w
        object_pos_b = quat_apply_inverse(root_quat_w, object_delta_w)
        object_xy_b = object_pos_b[:, :2]
        object_dist = torch.norm(object_xy_b, dim=-1)
        heading_object = torch.atan2(object_xy_b[:, 1], object_xy_b[:, 0])

        home_delta_w = self.home_pos_w - root_pos_w
        home_pos_b = quat_apply_inverse(root_quat_w, home_delta_w)
        home_xy_b = home_pos_b[:, :2]
        home_dist = torch.norm(home_xy_b, dim=-1)
        heading_home = torch.atan2(home_xy_b[:, 1], home_xy_b[:, 0])

        search_delta_w = self.search_target_w - root_pos_w
        search_pos_b = quat_apply_inverse(root_quat_w, search_delta_w)
        search_xy_b = search_pos_b[:, :2]
        search_dist = torch.norm(search_xy_b, dim=-1)

        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        lin_speed = torch.norm(lin_vel_b[:, :2], dim=-1)

        object_dir_b = object_xy_b / (object_dist.unsqueeze(-1) + 1e-6)
        home_dir_b = home_xy_b / (home_dist.unsqueeze(-1) + 1e-6)
        vel_toward_object = (lin_vel_b[:, :2] * object_dir_b).sum(dim=-1)
        vel_toward_home = (lin_vel_b[:, :2] * home_dir_b).sum(dim=-1)

        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        arm_vel = self.robot.data.joint_vel[:, self.arm_idx]
        wheel_vel = self.robot.data.joint_vel[:, self.wheel_idx]

        return {
            "root_pos_w": root_pos_w,
            "object_xy_b": object_xy_b,
            "object_dist": object_dist,
            "heading_object": heading_object,
            "home_xy_b": home_xy_b,
            "home_dist": home_dist,
            "heading_home": heading_home,
            "search_xy_b": search_xy_b,
            "search_dist": search_dist,
            "lin_vel_b": lin_vel_b,
            "ang_vel_b": ang_vel_b,
            "lin_speed": lin_speed,
            "vel_toward_object": vel_toward_object,
            "vel_toward_home": vel_toward_home,
            "arm_pos": arm_pos,
            "arm_vel": arm_vel,
            "wheel_vel": wheel_vel,
        }

    def _sample_targets_around(self, env_ids: torch.Tensor, base_xy: torch.Tensor, dist_min: float, dist_max: float):
        n = len(env_ids)
        angle = torch.rand(n, device=self.device) * 2.0 * math.pi
        dist = torch.rand(n, device=self.device) * (dist_max - dist_min) + dist_min
        x = base_xy[:, 0] + dist * torch.cos(angle)
        y = base_xy[:, 1] + dist * torch.sin(angle)
        z = torch.full((n,), self.cfg.object_height, device=self.device)
        return torch.stack([x, y, z], dim=-1)

    def _resample_search_targets(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        base_xy = self.home_pos_w[env_ids, :2]
        self.search_target_w[env_ids] = self._sample_targets_around(
            env_ids=env_ids,
            base_xy=base_xy,
            dist_min=self.cfg.search_dist_min,
            dist_max=self.cfg.search_dist_max,
        )

    def _update_phase_state(self, metrics: Dict[str, torch.Tensor]):
        phase = self.phase.clone()

        # 이벤트 플래그 초기화
        self.just_detected[:] = False
        self.just_reached_grasp[:] = False
        self.just_grasped[:] = False
        self.task_success[:] = False

        # SEARCH 동안 scan score 누적
        search_mask = phase == PHASE_SEARCH
        self.search_scan_score[search_mask] += torch.abs(self.actions[search_mask, 2]) * self.step_dt

        # "벽 뒤 가림"을 단순화한 reveal 조건: 일정 이동/스캔 이후에만 검출 가능
        reveal_cond = search_mask & (~self.object_revealed) & (
            (metrics["home_dist"] > self.cfg.search_reveal_dist)
            | (self.search_scan_score > self.cfg.search_reveal_scan)
        )
        self.object_revealed[reveal_cond] = True

        # Search 힌트 지점 도달 시 새 힌트 샘플링
        reached_hint = search_mask & (~self.object_revealed) & (metrics["search_dist"] < 0.35)
        reached_hint_ids = reached_hint.nonzero(as_tuple=False).squeeze(-1)
        if len(reached_hint_ids) > 0:
            self._resample_search_targets(reached_hint_ids)

        # Visibility (detector + depth 추상화)
        bearing_object = metrics["heading_object"]
        in_fov = (metrics["object_xy_b"][:, 0] > 0.0) & (torch.abs(bearing_object) < self._vision_half_fov_rad)
        in_range = metrics["object_dist"] < self.cfg.vision_max_dist
        visible = (self.object_revealed & in_fov & in_range & (~self.object_grasped)) | self.object_grasped
        self.object_visible[:] = visible

        # SEARCH -> APPROACH
        detect = search_mask & visible
        if detect.any():
            phase[detect] = PHASE_APPROACH
            self.just_detected[detect] = True

        # APPROACH -> GRASP
        approach_mask = phase == PHASE_APPROACH
        ready_grasp = approach_mask & (metrics["object_dist"] < self.cfg.approach_thresh)
        if ready_grasp.any():
            phase[ready_grasp] = PHASE_GRASP
            self.just_reached_grasp[ready_grasp] = True
            self.grasp_entry_step[ready_grasp] = self.episode_length_buf[ready_grasp]

        # GRASP timeout → APPROACH (재시도)
        grasp_mask = phase == PHASE_GRASP
        if self.cfg.grasp_timeout_steps > 0:
            grasp_elapsed = self.episode_length_buf - self.grasp_entry_step
            timed_out = grasp_mask & (grasp_elapsed > self.cfg.grasp_timeout_steps) & (~self.object_grasped)
            if timed_out.any():
                phase[timed_out] = PHASE_APPROACH

        # GRASP -> RETURN
        grasp_mask = phase == PHASE_GRASP
        if self._physics_grasp and self.contact_sensor is not None:
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
            gripper_closed = gripper_pos < float(self.cfg.grasp_gripper_threshold)
            contact_force = self._contact_force_per_env()
            has_contact = contact_force > float(self.cfg.grasp_contact_threshold)
            bbox_max_dim = self.object_bbox.max(dim=-1).values
            adaptive_dist = torch.clamp(
                float(self.cfg.grasp_max_object_dist) + bbox_max_dim * 0.5,
                min=0.10,
                max=0.40,
            )
            close_enough = metrics["object_dist"] < adaptive_dist
            grasp_ok = grasp_mask & gripper_closed & has_contact & close_enough
        else:
            grasp_ok = (
                grasp_mask
                & (metrics["object_dist"] < self.cfg.grasp_thresh)
                & (metrics["lin_speed"] < 0.35)
            )
        newly_grasped = grasp_ok & (~self.object_grasped)
        if grasp_ok.any():
            self.object_grasped[grasp_ok] = True
            phase[grasp_ok] = PHASE_RETURN
        if newly_grasped.any():
            newly_grasped_ids = newly_grasped.nonzero(as_tuple=False).squeeze(-1)
            self.just_grasped[newly_grasped] = True
            # Attach to robot for stable carry once grasp is successful.
            self.object_pos_w[newly_grasped, :2] = metrics["root_pos_w"][newly_grasped, :2]
            self.object_pos_w[newly_grasped, 2] = float(self.cfg.grasp_attach_height)
            if self._multi_object and len(self.object_rigids) > 0:
                for oi, rigid in enumerate(self.object_rigids):
                    oi_mask = self.active_object_idx[newly_grasped_ids] == oi
                    if not oi_mask.any():
                        continue
                    env_ids = newly_grasped_ids[oi_mask]
                    pose = rigid.data.root_pose_w[env_ids].clone()
                    pose[:, :3] = self.object_pos_w[env_ids]
                    pose[:, 3:7] = self.robot.data.root_state_w[env_ids, 3:7]
                    rigid.write_root_pose_to_sim(pose, env_ids=env_ids)
                    zero_vel = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device)
                    rigid.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
            elif self._physics_grasp and self.object_rigid is not None:
                pose = self.object_rigid.data.root_pose_w[newly_grasped_ids].clone()
                pose[:, :3] = self.object_pos_w[newly_grasped_ids]
                pose[:, 3:7] = self.robot.data.root_state_w[newly_grasped_ids, 3:7]
                self.object_rigid.write_root_pose_to_sim(pose, env_ids=newly_grasped_ids)
                zero_vel = torch.zeros((len(newly_grasped_ids), 6), dtype=torch.float32, device=self.device)
                self.object_rigid.write_root_velocity_to_sim(zero_vel, env_ids=newly_grasped_ids)

        # RETURN 성공
        return_success = (phase == PHASE_RETURN) & self.object_grasped & (metrics["home_dist"] < self.cfg.return_thresh)
        self.task_success[return_success] = True

        self.phase[:] = phase

    def _get_target_xy_body(self, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        # SEARCH: search hint, APPROACH/GRASP: object, RETURN: home
        target_xy = metrics["search_xy_b"].clone()
        approach_like = (self.phase == PHASE_APPROACH) | (self.phase == PHASE_GRASP)
        target_xy[approach_like] = metrics["object_xy_b"][approach_like]
        return_mask = self.phase == PHASE_RETURN
        target_xy[return_mask] = metrics["home_xy_b"][return_mask]
        return target_xy

    # Observations
    def _get_observations(self) -> dict:
        # _get_dones → _get_rewards → _get_observations 순서에서 캐시 재사용
        metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()

        target_xy = self._get_target_xy_body(metrics)
        object_xy = metrics["object_xy_b"].clone()
        object_xy[~self.object_visible] = 0.0

        phase_onehot = torch.nn.functional.one_hot(self.phase, NUM_PHASES).float()

        parts = [
            target_xy,                           # 2
            object_xy,                           # 2
            metrics["home_xy_b"],               # 2
            phase_onehot,                        # 4
            self.object_visible.float().unsqueeze(-1),  # 1
            self.object_grasped.float().unsqueeze(-1),  # 1
            metrics["lin_vel_b"],               # 3
            metrics["ang_vel_b"],               # 3
            metrics["arm_pos"],                 # 6
            metrics["arm_vel"],                 # 6
            metrics["wheel_vel"],               # 3
        ]
        if self._multi_object:
            bbox_normalized = self.object_bbox / float(self._bbox_norm_scale)
            cat_denom = max(int(self.cfg.num_object_categories) - 1, 1)
            cat_normalized = (self.object_category_id / float(cat_denom)).unsqueeze(-1)
            parts.extend([bbox_normalized, cat_normalized])  # +4

        obs = torch.cat(parts, dim=-1)
        self._cached_metrics = None
        return {"policy": obs}

    # Rewards
    def _get_rewards(self) -> torch.Tensor:
        metrics = self._cached_metrics
        if metrics is None:
            metrics = self._compute_metrics()
            self._update_phase_state(metrics)

        search_progress = torch.clamp(self.prev_search_dist - metrics["search_dist"], -0.2, 0.2)
        object_progress = torch.clamp(self.prev_object_dist - metrics["object_dist"], -0.2, 0.2)
        home_progress = torch.clamp(self.prev_home_dist - metrics["home_dist"], -0.2, 0.2)

        search_mask = self.phase == PHASE_SEARCH
        approach_mask = self.phase == PHASE_APPROACH
        grasp_mask = self.phase == PHASE_GRASP
        return_mask = self.phase == PHASE_RETURN

        effort_pen = self.cfg.rew_effort_weight * (self.actions[:, :3] ** 2).sum(dim=-1)
        arm_pen = self.cfg.rew_arm_move_weight * (metrics["arm_vel"] ** 2).sum(dim=-1)

        reward = torch.full((self.num_envs,), self.cfg.rew_time_penalty, device=self.device)
        reward += effort_pen + arm_pen

        reward += search_mask.float() * (
            self.cfg.rew_search_progress_weight * search_progress
            + self.cfg.rew_search_move_weight * metrics["lin_speed"]
            + self.cfg.rew_search_turn_weight * torch.abs(self.actions[:, 2])
        )
        reward += self.just_detected.float() * self.cfg.rew_detect_bonus

        reward += approach_mask.float() * (
            self.cfg.rew_approach_progress_weight * object_progress
            + self.cfg.rew_approach_heading_weight * torch.cos(metrics["heading_object"])
            + self.cfg.rew_approach_vel_weight * metrics["vel_toward_object"]
        )
        reward += self.just_reached_grasp.float() * self.cfg.rew_approach_ready_bonus

        if self._physics_grasp:
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
            if self._multi_object:
                bbox_max = self.object_bbox.max(dim=-1).values
                close_target = torch.clamp(
                    float(self.cfg.grasp_gripper_threshold) * (1.0 - bbox_max * 2.0),
                    min=-1.0,
                    max=0.0,
                )
                close_progress = torch.clamp(close_target - gripper_pos, min=0.0, max=1.0)
            else:
                close_progress = torch.clamp(float(self.cfg.grasp_gripper_threshold) - gripper_pos, min=0.0, max=1.0)
            open_progress = torch.clamp(gripper_pos - float(self.cfg.grasp_gripper_threshold), min=0.0, max=1.0)
            reward += grasp_mask.float() * (0.5 * close_progress)
            reward += approach_mask.float() * (0.1 * open_progress)

        reward += self.just_grasped.float() * self.cfg.rew_grasp_success_bonus
        reward += return_mask.float() * (
            self.cfg.rew_return_progress_weight * home_progress
            + self.cfg.rew_return_heading_weight * torch.cos(metrics["heading_home"])
            + self.cfg.rew_return_vel_weight * metrics["vel_toward_home"]
            + self.cfg.rew_carry_bonus * self.object_grasped.float()
        )
        reward += self.task_success.float() * self.cfg.rew_return_success_bonus

        self.prev_search_dist[:] = metrics["search_dist"]
        self.prev_object_dist[:] = metrics["object_dist"]
        self.prev_home_dist[:] = metrics["home_dist"]

        self.episode_reward_sum += reward
        self.reached_count += self.task_success.sum().item()
        # 캐시는 _get_observations에서 재사용 후 해제됨
        return reward

    # Dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self._compute_metrics()
        self._update_phase_state(metrics)
        self._cached_metrics = metrics

        root_pos = metrics["root_pos_w"]
        # home_pos_w 기준 상대 거리 — 절대 좌표 사용 시 env 0 이외 환경이 즉시 terminated됨
        out_of_bounds = torch.norm(root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1) > self.cfg.max_dist_from_origin
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)
        terminated = out_of_bounds | fell

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        truncated = self.task_success | time_out

        # 로그용 지표
        self.extras["task_success_rate"] = self.task_success.float().mean()
        self.extras["object_visible_rate"] = self.object_visible.float().mean()
        self.extras["phase_mean"] = self.phase.float().mean()
        return terminated, truncated

    # Reset
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        num = len(env_ids)

        # Root reset
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:2] += torch.randn(num, 2, device=self.device) * 0.1

        random_yaw = torch.rand(num, device=self.device) * 2.0 * math.pi - math.pi
        half_yaw = random_yaw * 0.5
        default_root_state[:, 3] = torch.cos(half_yaw)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(half_yaw)
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # Joint reset
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Home/object/search target
        self.home_pos_w[env_ids] = default_root_state[:, :3]
        base_xy = self.home_pos_w[env_ids, :2]

        self.object_pos_w[env_ids] = self._sample_targets_around(
            env_ids=env_ids,
            base_xy=base_xy,
            dist_min=self.cfg.object_dist_min,
            dist_max=self.cfg.object_dist_max,
        )
        if self._multi_object and len(self.object_rigids) > 0:
            chosen = torch.randint(0, self._num_object_types, (num,), device=self.device)
            self.active_object_idx[env_ids] = chosen
            self.object_bbox[env_ids] = self._catalog_bbox[chosen]
            self.object_category_id[env_ids] = self._catalog_category[chosen]
            # Keep object resting on ground based on per-object bbox z (scaled).
            self.object_pos_w[env_ids, 2] = torch.clamp(
                self.object_bbox[env_ids, 2] * 0.5,
                min=float(self.cfg.object_height),
            )

            # Hide all object types for the resetting envs.
            for rigid in self.object_rigids:
                hide_pose = rigid.data.default_root_state[env_ids, :7].clone()
                hide_pose[:, 0] = 0.0
                hide_pose[:, 1] = 0.0
                hide_pose[:, 2] = -10.0
                rigid.write_root_pose_to_sim(hide_pose, env_ids=env_ids)
                obj_vel = torch.zeros((num, 6), dtype=torch.float32, device=self.device)
                rigid.write_root_velocity_to_sim(obj_vel, env_ids=env_ids)

            # Activate only the sampled object type for each env.
            for oi, rigid in enumerate(self.object_rigids):
                mask = chosen == oi
                if not mask.any():
                    continue
                selected_env_ids = env_ids[mask]
                selected_num = int(mask.sum().item())
                pose = rigid.data.default_root_state[selected_env_ids, :7].clone()
                pose[:, :3] = self.object_pos_w[selected_env_ids]
                obj_yaw = torch.rand(selected_num, device=self.device) * 2.0 * math.pi - math.pi
                half = obj_yaw * 0.5
                pose[:, 3] = torch.cos(half)
                pose[:, 4] = 0.0
                pose[:, 5] = 0.0
                pose[:, 6] = torch.sin(half)
                rigid.write_root_pose_to_sim(pose, env_ids=selected_env_ids)
                obj_vel = torch.zeros((selected_num, 6), dtype=torch.float32, device=self.device)
                rigid.write_root_velocity_to_sim(obj_vel, env_ids=selected_env_ids)
        elif self._physics_grasp and self.object_rigid is not None:
            self.active_object_idx[env_ids] = 0
            self.object_bbox[env_ids] = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=self.device)
            self.object_category_id[env_ids] = 0.0
            obj_pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
            obj_pose[:, :3] = self.object_pos_w[env_ids]
            obj_yaw = torch.rand(num, device=self.device) * 2.0 * math.pi - math.pi
            half = obj_yaw * 0.5
            obj_pose[:, 3] = torch.cos(half)
            obj_pose[:, 4] = 0.0
            obj_pose[:, 5] = 0.0
            obj_pose[:, 6] = torch.sin(half)
            self.object_rigid.write_root_pose_to_sim(obj_pose, env_ids=env_ids)
            obj_vel = torch.zeros((num, 6), dtype=torch.float32, device=self.device)
            self.object_rigid.write_root_velocity_to_sim(obj_vel, env_ids=env_ids)
        else:
            self.active_object_idx[env_ids] = 0
            self.object_bbox[env_ids] = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=self.device)
            self.object_category_id[env_ids] = 0.0
        self._resample_search_targets(env_ids)

        # Task state reset
        self.phase[env_ids] = PHASE_SEARCH
        self.object_revealed[env_ids] = False
        self.object_visible[env_ids] = False
        self.object_grasped[env_ids] = False
        self.task_success[env_ids] = False

        self.search_scan_score[env_ids] = 0.0
        self.just_detected[env_ids] = False
        self.just_reached_grasp[env_ids] = False
        self.just_grasped[env_ids] = False
        self.grasp_entry_step[env_ids] = 0

        # Progress buffers init
        # 리셋 직후 "이전 거리"는 실제 시작 root 위치 기준으로 맞춘다.
        start_xy = default_root_state[:, :2]
        obj_dist = torch.norm(self.object_pos_w[env_ids, :2] - start_xy, dim=-1)
        search_dist = torch.norm(self.search_target_w[env_ids, :2] - start_xy, dim=-1)
        self.prev_object_dist[env_ids] = obj_dist
        self.prev_search_dist[env_ids] = search_dist
        self.prev_home_dist[env_ids] = 0.0

        # Common buffers reset
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.episode_reward_sum[env_ids] = 0.0
