"""
LeKiwi Skill-2 — ApproachAndGrasp Isaac Lab DirectRLEnv.

3-Skill 파이프라인의 두 번째 스킬: 물체를 향해 접근하고 잡기(Grasp).
FSM 없이 단일 목표(approach→grasp→lift) 수행.

Observation (Actor 30D):
  [0:5]   arm joint pos (5)
  [5:6]   gripper pos (1)
  [6:9]   base body velocity (vx, vy, wz) (3)
  [9:12]  base linear vel body (3)
  [12:15] base angular vel body (3)
  [15:21] arm+grip joint vel (6)
  [21:24] object relative pos body (3)
  [24:26] contact L/R (2)
  [26:29] object bbox normalized (3)
  [29:30] object category normalized (1)

Observation (Critic 37D, AAC):
  Actor 30D + object_bbox(3) + mass(1) + object_dist(1) + heading(1) + vel_toward(1)

Action (9D — lekiwi_v6 순서):
  [0:5]   arm joint position target
  [5]     gripper position target
  [6:8]   base linear velocity (vx, vy)
  [8]     base angular velocity (wz)
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict

import omni.usd
import torch
from pxr import Gf, Sdf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from lekiwi_robot_cfg import (
    ARM_LIMITS_BAKED_RAD,
    ARM_JOINT_NAMES,
    BASE_RADIUS,
    GRIPPER_JOINT_NAME,
    GRIPPER_JOINT_IDX_IN_ARM,
    LEKIWI_CFG,
    WHEEL_ANGLES_RAD,
    WHEEL_JOINT_NAMES,
    WHEEL_RADIUS,
)


@configclass
class Skill2EnvCfg(DirectRLEnvCfg):
    """ApproachAndGrasp 환경 설정."""

    # === Simulation (v8 동일) ===
    sim: SimulationCfg = SimulationCfg(
        dt=0.02, render_interval=2,
        gravity=(0.0, 0.0, -9.81), device="cpu",
    )
    decimation: int = 2
    episode_length_s: float = 20.0

    # === Scene (v8 동일) ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=10.0, replicate_physics=True,
    )

    # === Robot (v8 동일) ===
    robot_cfg = LEKIWI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # === Spaces (변경) ===
    observation_space: int = 30     # v8은 33/37
    action_space: int = 9           # 동일하지만 순서 다름
    state_space: int = 37           # Critic용 (AAC)

    # === Action (v8 동일 값, 순서만 다름) ===
    max_lin_vel: float = 0.5
    max_ang_vel: float = 3.0
    arm_action_scale: float = 1.5
    arm_action_to_limits: bool = True

    # === Calibration (v8 동일) ===
    calibration_json: str | None = None
    dynamics_json: str | None = None
    dynamics_apply_cmd_scale: bool = True
    arm_limit_json: str | None = None
    arm_limit_margin_rad: float = 0.0
    arm_limit_write_to_sim: bool = True    # USD의 inf limit 시 팔이 몸체 관통 방지

    # === Task Geometry (변경) ===
    object_dist_min: float = 0.5    # Curriculum 시작값 (v8은 1.0)
    object_dist_max: float = 2.5    # 최대 (v8 동일)
    approach_thresh: float = 0.35   # v8 동일
    grasp_thresh: float = 0.20      # v8 동일
    object_height: float = 0.03     # v8 동일

    # === Curriculum (신규) ===
    curriculum_success_threshold: float = 0.7
    curriculum_dist_increment: float = 0.25
    curriculum_current_max_dist: float = 0.5  # 런타임에 변경됨

    # === Physics Grasp (v8 동일, break_force만 변경) ===
    object_usd: str = ""
    object_mass: float = 0.3
    object_scale: float = 1.0
    gripper_contact_prim_path: str = ""
    object_prim_path: str = "/World/envs/env_.*/Object"
    object_filter_prim_expr: str = "/World/envs/env_.*/Object"
    grasp_gripper_threshold: float = 0.7
    grasp_contact_threshold: float = 0.5
    grasp_max_object_dist: float = 0.25
    grasp_attach_height: float = 0.15
    grasp_attach_mode: str = "fixed_joint"
    grasp_joint_break_force: float = 30.0    # v8의 1e8에서 변경 (mass*g*10)
    grasp_joint_break_torque: float = 30.0
    grasp_drop_detect_dist: float = 0.15     # gripper-object 거리 > 이 값이면 drop 판정
    grasp_timeout_steps: int = 75

    # === Multi-object (v8 동일) ===
    multi_object_json: str = ""
    num_object_categories: int = 6

    # === Reward (approach/grasp/lift 전용) ===
    rew_time_penalty: float = -0.01
    rew_effort_weight: float = -0.01
    rew_arm_move_weight: float = -0.02
    rew_action_smoothness_weight: float = -0.005  # action delta penalty (sim2real)
    rew_approach_progress_weight: float = 6.0
    rew_approach_heading_weight: float = 0.2
    rew_approach_vel_weight: float = 0.5
    rew_proximity_tanh_weight: float = 2.0  # tanh proximity kernel (가까울수록 강한 gradient)
    rew_proximity_tanh_sigma: float = 0.5   # tanh kernel sigma
    rew_grasp_success_bonus: float = 10.0
    rew_lift_bonus: float = 5.0
    rew_collision: float = -1.0

    # === Termination (v8 동일) ===
    max_dist_from_origin: float = 6.0

    # === DR (sim2real gap 핵심 — 범위 확대) ===
    enable_domain_randomization: bool = True
    dr_root_xy_noise_std: float = 0.12
    dr_root_yaw_jitter_rad: float = 0.2
    dr_wheel_stiffness_scale_range: tuple[float, float] = (0.75, 1.5)
    dr_wheel_damping_scale_range: tuple[float, float] = (0.3, 3.0)   # 가장 중요: 실기 damping 편차 큼
    dr_wheel_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_wheel_dynamic_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_wheel_viscous_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_arm_stiffness_scale_range: tuple[float, float] = (0.8, 1.25)
    dr_arm_damping_scale_range: tuple[float, float] = (0.5, 2.0)
    dr_object_mass_scale_range: tuple[float, float] = (0.5, 2.0)     # 실제 물체 질량 편차 큼
    dr_object_static_friction_scale_range: tuple[float, float] = (0.6, 1.5)
    dr_object_dynamic_friction_scale_range: tuple[float, float] = (0.6, 1.5)

    # Grasp DR (sim2real gap 핵심)
    dr_grasp_break_force_range: tuple[float, float] = (15.0, 45.0)
    dr_grasp_break_torque_range: tuple[float, float] = (15.0, 45.0)

    # Observation noise (sim2real: 센서 노이즈 시뮬레이션)
    dr_obs_noise_joint_pos: float = 0.01     # rad — arm joint position noise
    dr_obs_noise_base_vel: float = 0.02      # m/s — base velocity noise
    dr_obs_noise_object_rel: float = 0.02    # m — object relative position noise

    # Action delay (sim2real: 통신 지연 시뮬레이션)
    dr_action_delay_steps: int = 1           # 0=없음, 1-2=권장 (SSH/ZMQ 10-50ms)


class Skill2Env(DirectRLEnv):
    """LeKiwi Skill-2: ApproachAndGrasp RL 환경."""

    cfg: Skill2EnvCfg

    def __init__(self, cfg: Skill2EnvCfg, render_mode: str | None = None, **kwargs):
        self._multi_object = bool(str(getattr(cfg, "multi_object_json", "")).strip())
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
        self._dynamics_command_transform: dict[str, float | str] | None = None

        # Kiwi IK matrix
        self.base_radius = float(BASE_RADIUS)
        self.wheel_radius = float(WHEEL_RADIUS)
        self._maybe_apply_calibration_geometry_from_cfg()
        angles = torch.tensor(WHEEL_ANGLES_RAD, dtype=torch.float32, device=self.device)
        self.kiwi_M = torch.stack(
            [
                torch.cos(angles),
                torch.sin(angles),
                torch.full_like(angles, self.base_radius),
            ],
            dim=-1,
        )

        # Task buffers
        self.home_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.active_object_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_bbox = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_category_id = torch.zeros(self.num_envs, device=self.device)
        self._bbox_norm_scale = 0.2

        self.object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_entry_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.prev_object_dist = torch.zeros(self.num_envs, device=self.device)

        # Curriculum 상태 추적 (신규)
        # curriculum_current_max_dist가 설정되었으면 해당 값에서 시작 (handoff/navigate 수집 시 사용)
        if (hasattr(self.cfg, 'curriculum_current_max_dist')
                and float(self.cfg.curriculum_current_max_dist) > float(self.cfg.object_dist_min)):
            self._curriculum_dist = min(
                float(self.cfg.curriculum_current_max_dist),
                float(self.cfg.object_dist_max),
            )
        else:
            self._curriculum_dist = float(self.cfg.object_dist_min)
        self._curriculum_success_window = torch.zeros(100, device=self.device)
        self._curriculum_idx = 0

        # Action / metrics buffers
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.episode_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self.reached_count = 0

        # Action delay buffer (sim2real: 통신 지연 시뮬레이션)
        delay = max(int(self.cfg.dr_action_delay_steps), 0)
        if delay > 0:
            self._action_delay_buf = torch.zeros(
                delay, self.num_envs, self.cfg.action_space, device=self.device
            )
        else:
            self._action_delay_buf = None
        self._cached_metrics: Dict[str, torch.Tensor] | None = None
        self._contact_shape_warned = False
        self._grasp_attach_mode = str(getattr(self.cfg, "grasp_attach_mode", "fixed_joint")).strip().lower()
        self._grasp_fixed_joints: dict[tuple[int, int], UsdPhysics.FixedJoint] = {}
        self._grasp_fixed_joint_warned = False

        self._dr_base_wheel_stiffness: torch.Tensor | None = None
        self._dr_base_wheel_damping: torch.Tensor | None = None
        self._dr_base_arm_stiffness: torch.Tensor | None = None
        self._dr_base_arm_damping: torch.Tensor | None = None
        self._dr_curr_wheel_stiffness: torch.Tensor | None = None
        self._dr_curr_wheel_damping: torch.Tensor | None = None
        self._dr_curr_arm_stiffness: torch.Tensor | None = None
        self._dr_curr_arm_damping: torch.Tensor | None = None
        self._dr_base_wheel_friction: torch.Tensor | None = None
        self._dr_base_wheel_dynamic_friction: torch.Tensor | None = None
        self._dr_base_wheel_viscous_friction: torch.Tensor | None = None
        self._dr_curr_wheel_friction: torch.Tensor | None = None
        self._dr_curr_wheel_dynamic_friction: torch.Tensor | None = None
        self._dr_curr_wheel_viscous_friction: torch.Tensor | None = None
        self._dr_object_material_base: dict[str, tuple[float, float]] = {}
        self._arm_action_limits_override: torch.Tensor | None = None

        # Per-env grasp break force/torque (DR용)
        self._per_env_break_force = torch.full((self.num_envs,), float(self.cfg.grasp_joint_break_force), device=self.device)
        self._per_env_break_torque = torch.full((self.num_envs,), float(self.cfg.grasp_joint_break_torque), device=self.device)

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
                catalog_bbox_rows, dtype=torch.float32, device=self.device,
            )
            self._catalog_category = torch.tensor(
                [float(obj.get("category", 0)) for obj in self._object_catalog],
                dtype=torch.float32, device=self.device,
            )
            self._catalog_mass = torch.tensor(
                [max(float(obj.get("mass", self.cfg.object_mass)), 1e-5) for obj in self._object_catalog],
                dtype=torch.float32, device=self.device,
            )
        else:
            self.object_bbox[:] = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=self.device)
            self.object_category_id[:] = 0.0
            self._catalog_bbox = torch.tensor([[0.05, 0.05, 0.05]], dtype=torch.float32, device=self.device)
            self._catalog_category = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            self._catalog_mass = torch.tensor([max(float(self.cfg.object_mass), 1e-5)], dtype=torch.float32, device=self.device)

        self._maybe_apply_tuned_dynamics_from_cfg()
        self._apply_baked_arm_limits()
        self._maybe_apply_arm_limits_from_cfg()
        self._init_domain_randomization_buffers()

        print(f"  [Skill2Env] obs={self.cfg.observation_space} act={self.cfg.action_space} critic={self.cfg.state_space}")
        print(f"  [Skill2Env] arm_idx={self.arm_idx.tolist()} wheel_idx={self.wheel_idx.tolist()}")

    # ═══════════════════════════════════════════════════════════════════
    #  Scene setup — v8 그대로 복사
    # ═══════════════════════════════════════════════════════════════════

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
                    raise FileNotFoundError(
                        f"multi_object_json[{oi}].usd not found: {obj_usd}\n"
                        "Hint: regenerate catalog with build_object_catalog.py using real USD paths."
                    )
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

        # Grasp break 감지를 위한 gripper body index
        _gripper_body_names = ["Moving_Jaw_08d_v1"]
        try:
            body_ids, _ = self.robot.find_bodies(_gripper_body_names)
            self._gripper_body_idx = body_ids[0]
        except (IndexError, RuntimeError):
            self._gripper_body_idx = 0

    # ═══════════════════════════════════════════════════════════════════
    #  Calibration / Dynamics — v8 그대로 복사
    # ═══════════════════════════════════════════════════════════════════

    def _maybe_apply_calibration_geometry_from_cfg(self):
        raw_path = str(getattr(self.cfg, "calibration_json", "") or "").strip()
        if not raw_path:
            return

        path = os.path.expanduser(raw_path)
        if not os.path.isabs(path) and not os.path.isfile(path):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            alt = os.path.join(repo_root, raw_path)
            if os.path.isfile(alt):
                path = alt

        if not os.path.isfile(path):
            print(f"  [Calibration] geometry JSON not found: {raw_path} (keep defaults)")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  [Calibration] failed to load {path}: {exc} (keep defaults)")
            return

        wr = payload.get("wheel_radius", {}).get("wheel_radius_m")
        br = payload.get("base_radius", {}).get("base_radius_m")

        applied = False
        try:
            wr_f = float(wr)
            if math.isfinite(wr_f) and wr_f > 1e-8:
                self.wheel_radius = wr_f
                applied = True
        except (TypeError, ValueError):
            pass

        try:
            br_f = float(br)
            if math.isfinite(br_f) and br_f > 1e-8:
                self.base_radius = br_f
                applied = True
        except (TypeError, ValueError):
            pass

        if applied:
            print(
                f"  [Calibration] geometry applied from {path}: "
                f"wheel_radius={self.wheel_radius:.6f}, base_radius={self.base_radius:.6f}"
            )
        else:
            print(f"  [Calibration] no valid geometry fields in {path} (keep defaults)")

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

    def _apply_arm_limits(self, limits_by_joint: dict[str, tuple[float, float]], source: str, margin: float = 0.0):
        margin = float(margin)
        joint_ids = []
        limits_rows = []
        min_limit = -2.0 * math.pi + 1e-6
        max_limit = 2.0 * math.pi - 1e-6
        arm_limits = self.robot.data.soft_joint_pos_limits[:, self.arm_idx].clone()
        arm_name_to_offset = {str(name): i for i, name in enumerate(ARM_JOINT_NAMES)}
        applied_count = 0
        for sim_joint, (lo, hi) in limits_by_joint.items():
            idxs, _ = self.robot.find_joints([sim_joint])
            if len(idxs) != 1:
                continue
            lo_c = max(float(lo - margin), min_limit)
            hi_c = min(float(hi + margin), max_limit)
            if not math.isfinite(lo_c) or not math.isfinite(hi_c) or abs(hi_c - lo_c) < 1e-8:
                continue
            joint_ids.append(int(idxs[0]))
            limits_rows.append([lo_c, hi_c])
            off = arm_name_to_offset.get(str(sim_joint))
            if off is not None:
                arm_limits[:, off, 0] = lo_c
                arm_limits[:, off, 1] = hi_c
                applied_count += 1

        if not joint_ids:
            raise ValueError(f"No valid sim joints found from arm limits source: {source}")

        self._arm_action_limits_override = arm_limits

        if bool(getattr(self.cfg, "arm_limit_write_to_sim", False)):
            limits = torch.tensor(limits_rows, dtype=torch.float32, device=self.device).unsqueeze(0)
            limits = limits.repeat(self.num_envs, 1, 1)
            self.robot.write_joint_position_limit_to_sim(limits, joint_ids=joint_ids, warn_limit_violation=False)
            print(
                f"  [ArmLimits] applied from {source}: {len(joint_ids)} joints "
                f"(margin={margin:.4f} rad, write_to_sim=True)"
            )
        else:
            print(
                f"  [ArmLimits] loaded from {source}: {applied_count} arm joints "
                f"(margin={margin:.4f} rad, write_to_sim=False)"
            )

    def _apply_baked_arm_limits(self):
        if not ARM_LIMITS_BAKED_RAD:
            return
        self._apply_arm_limits(
            limits_by_joint={str(k): (float(v[0]), float(v[1])) for k, v in ARM_LIMITS_BAKED_RAD.items()},
            source="lekiwi_robot_cfg.ARM_LIMITS_BAKED_RAD",
            margin=0.0,
        )

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
        self._apply_arm_limits(
            limits_by_joint=limits_by_joint,
            source=path,
            margin=float(self.cfg.arm_limit_margin_rad),
        )

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

        cmd_tf_raw = payload.get("command_transform") if isinstance(payload, dict) else None
        if isinstance(cmd_tf_raw, dict):
            self._dynamics_command_transform = {
                "mode": str(cmd_tf_raw.get("mode", "none")),
                "linear_map": str(cmd_tf_raw.get("linear_map", "identity")),
                "lin_scale": self._safe_float(cmd_tf_raw, "lin_scale", 1.0),
                "ang_scale": self._safe_float(cmd_tf_raw, "ang_scale", 1.0),
                "wz_sign": self._safe_float(cmd_tf_raw, "wz_sign", 1.0),
            }
        else:
            self._dynamics_command_transform = None

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
        for ji in range(len(ARM_JOINT_NAMES)):
            params[f"arm_stiffness_scale_j{ji}"] = self._safe_float(raw_params, f"arm_stiffness_scale_j{ji}", 1.0)
            params[f"arm_damping_scale_j{ji}"] = self._safe_float(raw_params, f"arm_damping_scale_j{ji}", 1.0)

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

        arm_stiff_joint_scale = torch.ones_like(base_arm_stiff)
        arm_damp_joint_scale = torch.ones_like(base_arm_damping)
        for ji in range(len(ARM_JOINT_NAMES)):
            arm_stiff_joint_scale[:, ji] = float(params.get(f"arm_stiffness_scale_j{ji}", 1.0))
            arm_damp_joint_scale[:, ji] = float(params.get(f"arm_damping_scale_j{ji}", 1.0))
        self.robot.write_joint_stiffness_to_sim(
            base_arm_stiff * params["arm_stiffness_scale"] * arm_stiff_joint_scale,
            joint_ids=arm_ids,
        )
        self.robot.write_joint_damping_to_sim(
            base_arm_damping * params["arm_damping_scale"] * arm_damp_joint_scale,
            joint_ids=arm_ids,
        )
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
        if self._dynamics_command_transform is not None:
            print(f"  [Dynamics] command_transform(meta)={self._dynamics_command_transform}")
            print(
                "  [Dynamics] note: command_transform is for real<->sim log conversion; "
                "RL env action path stays in sim space."
            )

    # ═══════════════════════════════════════════════════════════════════
    #  Grasp attach helpers (fixed joint) — v8 그대로 복사
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_env_pattern_path(path_pattern: str, env_id: int) -> str:
        return str(path_pattern).replace("env_.*", f"env_{int(env_id)}")

    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _gripper_body_prim_path(self, env_id: int) -> str:
        return self._resolve_env_pattern_path(self.cfg.gripper_contact_prim_path, env_id)

    def _object_body_prim_path(self, env_id: int, object_type: int) -> str:
        if self._multi_object:
            return self._resolve_env_pattern_path(f"/World/envs/env_.*/Object_{int(object_type)}", env_id)
        return self._resolve_env_pattern_path(self.cfg.object_prim_path, env_id)

    @staticmethod
    def _quatd_to_quatf(quat_d: Gf.Quatd) -> Gf.Quatf:
        imag = quat_d.GetImaginary()
        return Gf.Quatf(float(quat_d.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))

    def _get_or_create_grasp_fixed_joint(
        self,
        env_id: int,
        object_type: int,
        gripper_path: str,
        object_path: str,
    ) -> UsdPhysics.FixedJoint | None:
        key = (int(env_id), int(object_type))
        existing = self._grasp_fixed_joints.get(key)
        if existing is not None and existing.GetPrim().IsValid():
            return existing

        stage = self._get_stage()
        if stage is None:
            return None

        try:
            joint_path = Sdf.Path(f"/World/envs/env_{int(env_id)}/GraspFixedJoint_{int(object_type)}")
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            if not joint:
                return None
        except Exception:
            return None

        joint.GetBody0Rel().SetTargets([Sdf.Path(gripper_path)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(object_path)])
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateBreakForceAttr(float(self.cfg.grasp_joint_break_force))
        joint.CreateBreakTorqueAttr(float(self.cfg.grasp_joint_break_torque))
        joint.CreateJointEnabledAttr(False)

        self._grasp_fixed_joints[key] = joint
        return joint

    def _attach_grasp_fixed_joint_for_envs(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        stage = self._get_stage()
        if stage is None:
            return

        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            object_type = int(self.active_object_idx[env_id].item()) if self._multi_object else 0
            gripper_path = self._gripper_body_prim_path(env_id)
            object_path = self._object_body_prim_path(env_id, object_type)

            gripper_prim = stage.GetPrimAtPath(gripper_path)
            object_prim = stage.GetPrimAtPath(object_path)
            if not gripper_prim.IsValid() or not object_prim.IsValid():
                if not self._grasp_fixed_joint_warned:
                    self._grasp_fixed_joint_warned = True
                    print(
                        "  [WARN] grasp fixed-joint prim not found. "
                        f"gripper={gripper_path}, object={object_path}. Falling back to teleport attach."
                    )
                self._teleport_attach_for_envs(torch.tensor([env_id], device=self.device, dtype=torch.long))
                continue

            joint = self._get_or_create_grasp_fixed_joint(
                env_id=env_id,
                object_type=object_type,
                gripper_path=gripper_path,
                object_path=object_path,
            )
            if joint is None:
                if not self._grasp_fixed_joint_warned:
                    self._grasp_fixed_joint_warned = True
                    print("  [WARN] failed to create grasp fixed-joint. Falling back to teleport attach.")
                self._teleport_attach_for_envs(torch.tensor([env_id], device=self.device, dtype=torch.long))
                continue

            try:
                tf_gripper = omni.usd.get_world_transform_matrix(gripper_prim)
                tf_object = omni.usd.get_world_transform_matrix(object_prim)
                local_tf0 = tf_gripper.GetInverse() * tf_object
                local_pos0 = local_tf0.ExtractTranslation()
                local_rot0 = self._quatd_to_quatf(local_tf0.ExtractRotationQuat())

                joint.GetLocalPos0Attr().Set(local_pos0)
                joint.GetLocalRot0Attr().Set(local_rot0)
                joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                # Per-env break force/torque (DR 적용)
                joint.CreateBreakForceAttr(self._per_env_break_force[env_id].item())
                joint.CreateBreakTorqueAttr(self._per_env_break_torque[env_id].item())
                joint.GetJointEnabledAttr().Set(True)
            except Exception:
                if not self._grasp_fixed_joint_warned:
                    self._grasp_fixed_joint_warned = True
                    print("  [WARN] failed to configure/enable grasp fixed-joint. Falling back to teleport attach.")
                self._teleport_attach_for_envs(torch.tensor([env_id], device=self.device, dtype=torch.long))

    def _disable_grasp_fixed_joint_for_envs(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        max_object_types = max(int(self._num_object_types), 1)
        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            for object_type in range(max_object_types):
                joint = self._grasp_fixed_joints.get((env_id, object_type))
                if joint is None or not joint.GetPrim().IsValid():
                    continue
                joint.GetJointEnabledAttr().Set(False)

    def _teleport_attach_for_envs(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        metrics_root = self.robot.data.root_state_w[:, :3]
        self.object_pos_w[env_ids, :2] = metrics_root[env_ids, :2]
        self.object_pos_w[env_ids, 2] = metrics_root[env_ids, 2] + float(self.cfg.grasp_attach_height)
        if self._multi_object and len(self.object_rigids) > 0:
            for oi, rigid in enumerate(self.object_rigids):
                oi_mask = self.active_object_idx[env_ids] == oi
                if not oi_mask.any():
                    continue
                ids = env_ids[oi_mask]
                pose = rigid.data.root_pose_w[ids].clone()
                pose[:, :3] = self.object_pos_w[ids]
                pose[:, 3:7] = self.robot.data.root_state_w[ids, 3:7]
                rigid.write_root_pose_to_sim(pose, env_ids=ids)
                zero_vel = torch.zeros((len(ids), 6), dtype=torch.float32, device=self.device)
                rigid.write_root_velocity_to_sim(zero_vel, env_ids=ids)
        elif self._physics_grasp and self.object_rigid is not None:
            pose = self.object_rigid.data.root_pose_w[env_ids].clone()
            pose[:, :3] = self.object_pos_w[env_ids]
            pose[:, 3:7] = self.robot.data.root_state_w[env_ids, 3:7]
            self.object_rigid.write_root_pose_to_sim(pose, env_ids=env_ids)
            zero_vel = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device)
            self.object_rigid.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    # ═══════════════════════════════════════════════════════════════════
    #  Domain randomization — v8 그대로 복사
    # ═══════════════════════════════════════════════════════════════════

    def _init_domain_randomization_buffers(self):
        self._dr_base_wheel_stiffness = self.robot.data.joint_stiffness[:, self.wheel_idx].clone()
        self._dr_base_wheel_damping = self.robot.data.joint_damping[:, self.wheel_idx].clone()
        self._dr_base_arm_stiffness = self.robot.data.joint_stiffness[:, self.arm_idx].clone()
        self._dr_base_arm_damping = self.robot.data.joint_damping[:, self.arm_idx].clone()

        self._dr_curr_wheel_stiffness = self._dr_base_wheel_stiffness.clone()
        self._dr_curr_wheel_damping = self._dr_base_wheel_damping.clone()
        self._dr_curr_arm_stiffness = self._dr_base_arm_stiffness.clone()
        self._dr_curr_arm_damping = self._dr_base_arm_damping.clone()

        friction_base = 0.0
        dynamic_friction_base = 0.0
        viscous_friction_base = 0.0
        if self._applied_dynamics_params is not None:
            friction_base = float(self._applied_dynamics_params.get("wheel_friction_coeff", 0.0))
            dynamic_friction_base = float(self._applied_dynamics_params.get("wheel_dynamic_friction_coeff", 0.0))
            viscous_friction_base = float(self._applied_dynamics_params.get("wheel_viscous_friction_coeff", 0.0))

        self._dr_base_wheel_friction = torch.full_like(self._dr_base_wheel_damping, friction_base)
        self._dr_base_wheel_dynamic_friction = torch.full_like(self._dr_base_wheel_damping, dynamic_friction_base)
        self._dr_base_wheel_viscous_friction = torch.full_like(self._dr_base_wheel_damping, viscous_friction_base)
        self._dr_curr_wheel_friction = self._dr_base_wheel_friction.clone()
        self._dr_curr_wheel_dynamic_friction = self._dr_base_wheel_dynamic_friction.clone()
        self._dr_curr_wheel_viscous_friction = self._dr_base_wheel_viscous_friction.clone()

    @staticmethod
    def _parse_scale_range(v: tuple[float, float]) -> tuple[float, float]:
        lo = float(v[0])
        hi = float(v[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            return 1.0, 1.0
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _sample_scale(self, v: tuple[float, float], n: int) -> torch.Tensor:
        lo, hi = self._parse_scale_range(v)
        if n <= 0:
            return torch.empty((0, 1), device=self.device)
        if abs(hi - lo) < 1e-8:
            return torch.full((n, 1), lo, dtype=torch.float32, device=self.device)
        return torch.empty((n, 1), dtype=torch.float32, device=self.device).uniform_(lo, hi)

    def _apply_object_mass_randomization(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        lo, hi = self._parse_scale_range(self.cfg.dr_object_mass_scale_range)
        sf_lo, sf_hi = self._parse_scale_range(self.cfg.dr_object_static_friction_scale_range)
        df_lo, df_hi = self._parse_scale_range(self.cfg.dr_object_dynamic_friction_scale_range)
        if (
            abs(hi - lo) < 1e-8
            and abs(lo - 1.0) < 1e-8
            and abs(sf_hi - sf_lo) < 1e-8
            and abs(sf_lo - 1.0) < 1e-8
            and abs(df_hi - df_lo) < 1e-8
            and abs(df_lo - 1.0) < 1e-8
        ):
            return
        stage = self._get_stage()
        if stage is None:
            return
        scales = self._sample_scale(self.cfg.dr_object_mass_scale_range, len(env_ids)).squeeze(-1)
        sf_scales = self._sample_scale(self.cfg.dr_object_static_friction_scale_range, len(env_ids)).squeeze(-1)
        df_scales = self._sample_scale(self.cfg.dr_object_dynamic_friction_scale_range, len(env_ids)).squeeze(-1)
        for i, env_id_t in enumerate(env_ids):
            env_id = int(env_id_t.item())
            object_type = int(self.active_object_idx[env_id].item()) if self._multi_object else 0
            base_mass = float(self._catalog_mass[min(object_type, len(self._catalog_mass) - 1)].item())
            mass_val = max(base_mass * float(scales[i].item()), 1e-5)
            prim_path = self._object_body_prim_path(env_id, object_type)
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            try:
                if not prim.HasAPI(UsdPhysics.MassAPI):
                    UsdPhysics.MassAPI.Apply(prim)
                mass_api = UsdPhysics.MassAPI(prim)
                attr = mass_api.GetMassAttr()
                if not attr:
                    attr = mass_api.CreateMassAttr()
                attr.Set(mass_val)

                if not prim.HasAPI(UsdPhysics.MaterialAPI):
                    UsdPhysics.MaterialAPI.Apply(prim)
                mat_api = UsdPhysics.MaterialAPI(prim)
                sf_attr = mat_api.GetStaticFrictionAttr()
                df_attr = mat_api.GetDynamicFrictionAttr()
                if not sf_attr:
                    sf_attr = mat_api.CreateStaticFrictionAttr()
                if not df_attr:
                    df_attr = mat_api.CreateDynamicFrictionAttr()

                base_sf, base_df = self._dr_object_material_base.get(prim_path, (0.8, 0.6))
                cur_sf = sf_attr.Get()
                cur_df = df_attr.Get()
                if prim_path not in self._dr_object_material_base:
                    if cur_sf is not None:
                        try:
                            base_sf = float(cur_sf)
                        except (TypeError, ValueError):
                            base_sf = 0.8
                    if cur_df is not None:
                        try:
                            base_df = float(cur_df)
                        except (TypeError, ValueError):
                            base_df = 0.6
                    self._dr_object_material_base[prim_path] = (base_sf, base_df)

                sf_val = max(base_sf * float(sf_scales[i].item()), 0.0)
                df_val = max(base_df * float(df_scales[i].item()), 0.0)
                sf_attr.Set(sf_val)
                df_attr.Set(min(df_val, sf_val))
            except Exception:
                continue

    def _apply_domain_randomization(self, env_ids: torch.Tensor):
        if not bool(self.cfg.enable_domain_randomization) or len(env_ids) == 0:
            return
        if self._dr_base_wheel_stiffness is None or self._dr_curr_wheel_stiffness is None:
            return

        n = len(env_ids)
        ids = env_ids
        ws = self._sample_scale(self.cfg.dr_wheel_stiffness_scale_range, n)
        wd = self._sample_scale(self.cfg.dr_wheel_damping_scale_range, n)
        as_ = self._sample_scale(self.cfg.dr_arm_stiffness_scale_range, n)
        ad = self._sample_scale(self.cfg.dr_arm_damping_scale_range, n)

        self._dr_curr_wheel_stiffness[ids] = torch.clamp(self._dr_base_wheel_stiffness[ids] * ws, min=0.0)
        self._dr_curr_wheel_damping[ids] = torch.clamp(self._dr_base_wheel_damping[ids] * wd, min=0.0)
        self._dr_curr_arm_stiffness[ids] = torch.clamp(self._dr_base_arm_stiffness[ids] * as_, min=0.0)
        self._dr_curr_arm_damping[ids] = torch.clamp(self._dr_base_arm_damping[ids] * ad, min=0.0)

        wheel_ids = self.wheel_idx.tolist()
        arm_ids = self.arm_idx.tolist()
        self.robot.write_joint_stiffness_to_sim(self._dr_curr_wheel_stiffness, joint_ids=wheel_ids)
        self.robot.write_joint_damping_to_sim(self._dr_curr_wheel_damping, joint_ids=wheel_ids)
        self.robot.write_joint_stiffness_to_sim(self._dr_curr_arm_stiffness, joint_ids=arm_ids)
        self.robot.write_joint_damping_to_sim(self._dr_curr_arm_damping, joint_ids=arm_ids)

        wf = self._sample_scale(self.cfg.dr_wheel_friction_scale_range, n)
        wdf = self._sample_scale(self.cfg.dr_wheel_dynamic_friction_scale_range, n)
        wvf = self._sample_scale(self.cfg.dr_wheel_viscous_friction_scale_range, n)
        self._dr_curr_wheel_friction[ids] = torch.clamp(self._dr_base_wheel_friction[ids] * wf, min=0.0)
        self._dr_curr_wheel_dynamic_friction[ids] = torch.clamp(
            self._dr_base_wheel_dynamic_friction[ids] * wdf, min=0.0
        )
        self._dr_curr_wheel_viscous_friction[ids] = torch.clamp(
            self._dr_base_wheel_viscous_friction[ids] * wvf, min=0.0
        )

        if hasattr(self.robot, "write_joint_friction_coefficient_to_sim"):
            self.robot.write_joint_friction_coefficient_to_sim(self._dr_curr_wheel_friction, joint_ids=wheel_ids)
        if hasattr(self.robot, "write_joint_dynamic_friction_coefficient_to_sim"):
            self.robot.write_joint_dynamic_friction_coefficient_to_sim(
                self._dr_curr_wheel_dynamic_friction, joint_ids=wheel_ids
            )
        if hasattr(self.robot, "write_joint_viscous_friction_coefficient_to_sim"):
            self.robot.write_joint_viscous_friction_coefficient_to_sim(
                self._dr_curr_wheel_viscous_friction, joint_ids=wheel_ids
            )

        self._apply_object_mass_randomization(env_ids=env_ids)

        # Grasp break force/torque 랜덤화 (신규)
        if hasattr(self.cfg, 'dr_grasp_break_force_range'):
            bf_lo, bf_hi = self.cfg.dr_grasp_break_force_range
            bt_lo, bt_hi = self.cfg.dr_grasp_break_torque_range
            self._per_env_break_force[env_ids] = (
                torch.rand(n, device=self.device) * (bf_hi - bf_lo) + bf_lo
            )
            self._per_env_break_torque[env_ids] = (
                torch.rand(n, device=self.device) * (bt_hi - bt_lo) + bt_lo
            )

    # ═══════════════════════════════════════════════════════════════════
    #  Contact force — v8 그대로 복사
    # ═══════════════════════════════════════════════════════════════════

    def _contact_force_per_env(self) -> torch.Tensor:
        force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not self._physics_grasp or self.contact_sensor is None:
            return force

        def _warn_shape_once(name: str, tensor: torch.Tensor):
            if self._contact_shape_warned:
                return
            self._contact_shape_warned = True
            print(
                "  [WARN] contact tensor shape mismatch: "
                f"{name}.shape={tuple(tensor.shape)}, num_envs={self.num_envs}. "
                "Using fallback contact source (or zero if unavailable)."
            )

        force_matrix = self.contact_sensor.data.force_matrix_w
        if force_matrix is not None:
            mag = torch.norm(force_matrix, dim=-1)
            mag = mag.reshape(mag.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag
            if mag.numel() % self.num_envs == 0:
                return mag.reshape(self.num_envs, -1).sum(dim=-1)
            _warn_shape_once("force_matrix_w", force_matrix)

        net = self.contact_sensor.data.net_forces_w
        if net is not None:
            mag = torch.norm(net, dim=-1)
            mag = mag.reshape(mag.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag
            if mag.numel() % self.num_envs == 0:
                return mag.reshape(self.num_envs, -1).sum(dim=-1)
            _warn_shape_once("net_forces_w", net)
        return force

    # ═══════════════════════════════════════════════════════════════════
    #  Utility — v8 그대로 복사
    # ═══════════════════════════════════════════════════════════════════

    def _sample_targets_around(
        self,
        env_ids: torch.Tensor,
        base_xy: torch.Tensor,
        dist_min: float,
        dist_max: float,
        base_z: torch.Tensor | None = None,
    ):
        n = len(env_ids)
        angle = torch.rand(n, device=self.device) * 2.0 * math.pi
        dist = torch.rand(n, device=self.device) * (dist_max - dist_min) + dist_min
        x = base_xy[:, 0] + dist * torch.cos(angle)
        y = base_xy[:, 1] + dist * torch.sin(angle)
        if base_z is None:
            z = torch.full((n,), self.cfg.object_height, device=self.device)
        else:
            z = base_z + float(self.cfg.object_height)
        return torch.stack([x, y, z], dim=-1)

    # ═══════════════════════════════════════════════════════════════════
    #  Base body velocity — v3.0 (displacement 계산 대체)
    # ═══════════════════════════════════════════════════════════════════

    def _read_base_body_vel(self):
        """Body-frame velocity 직접 읽기 (3D: vx, vy, wz).

        Isaac Sim의 root_lin_vel_b, root_ang_vel_b에서 직접 추출.
        이전 설계의 pose delta → body-frame 변환 계산이 불필요하다.
        단위: m/s (vx, vy), rad/s (wz) — 실제 로봇과 동일.
        """
        vx_body = self.robot.data.root_lin_vel_b[:, 0:1]   # x.vel (m/s)
        vy_body = self.robot.data.root_lin_vel_b[:, 1:2]   # y.vel (m/s)
        wz_body = self.robot.data.root_ang_vel_b[:, 2:3]   # theta.vel (rad/s)
        return torch.cat([vx_body, vy_body, wz_body], dim=-1)  # (N, 3)

    # ═══════════════════════════════════════════════════════════════════
    #  Task metrics — v8에서 search/home 관련 제거
    # ═══════════════════════════════════════════════════════════════════

    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        if self._multi_object and len(self.object_rigids) > 0:
            for oi, rigid in enumerate(self.object_rigids):
                mask = self.active_object_idx == oi
                if not mask.any():
                    continue
                ids = mask.nonzero(as_tuple=False).squeeze(-1)
                self.object_pos_w[ids] = rigid.data.root_pos_w[ids]
        elif self._physics_grasp and self.object_rigid is not None:
            self.object_pos_w[:] = self.object_rigid.data.root_pos_w

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        object_delta_w = self.object_pos_w - root_pos_w
        object_pos_b = quat_apply_inverse(root_quat_w, object_delta_w)
        object_dist = torch.norm(object_pos_b[:, :2], dim=-1)
        heading_object = torch.atan2(object_pos_b[:, 1], object_pos_b[:, 0])

        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        lin_speed = torch.norm(lin_vel_b[:, :2], dim=-1)

        object_dir_b = object_pos_b[:, :2] / (object_dist.unsqueeze(-1) + 1e-6)
        vel_toward_object = (lin_vel_b[:, :2] * object_dir_b).sum(dim=-1)

        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        arm_vel = self.robot.data.joint_vel[:, self.arm_idx]

        return {
            "root_pos_w": root_pos_w,
            "object_pos_b": object_pos_b,
            "object_dist": object_dist,
            "heading_object": heading_object,
            "lin_vel_b": lin_vel_b,
            "ang_vel_b": ang_vel_b,
            "lin_speed": lin_speed,
            "vel_toward_object": vel_toward_object,
            "arm_pos": arm_pos,
            "arm_vel": arm_vel,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  Grasp state — FSM 없이 직접 판정
    # ═══════════════════════════════════════════════════════════════════

    def _update_grasp_state(self, metrics: Dict[str, torch.Tensor]):
        """Grasp 판정 — FSM 없이 직접 처리."""
        self.just_grasped[:] = False
        self.just_dropped[:] = False
        self.task_success[:] = False

        if not self._physics_grasp or self.contact_sensor is None:
            can_grasp = (
                (metrics["object_dist"] < self.cfg.grasp_thresh)
                & (metrics["lin_speed"] < 0.35)
                & (~self.object_grasped)
            )
        else:
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
            gripper_closed = gripper_pos < float(self.cfg.grasp_gripper_threshold)
            contact_force = self._contact_force_per_env()
            has_contact = contact_force > float(self.cfg.grasp_contact_threshold)

            bbox_max_dim = self.object_bbox.max(dim=-1).values
            adaptive_dist = torch.clamp(
                float(self.cfg.grasp_max_object_dist) + bbox_max_dim * 0.5,
                min=0.10, max=0.40,
            )
            close_enough = metrics["object_dist"] < adaptive_dist

            can_grasp = gripper_closed & has_contact & close_enough & (~self.object_grasped)

        newly_grasped = can_grasp
        if newly_grasped.any():
            self.object_grasped[newly_grasped] = True
            self.just_grasped[newly_grasped] = True
            grasped_ids = newly_grasped.nonzero(as_tuple=False).squeeze(-1)
            if self._physics_grasp and self._grasp_attach_mode == "fixed_joint":
                self._attach_grasp_fixed_joint_for_envs(grasped_ids)
            else:
                self._teleport_attach_for_envs(grasped_ids)

        # Lift 성공 판정
        if self.object_grasped.any():
            obj_z = self.object_pos_w[:, 2]
            env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
            lifted = self.object_grasped & ((obj_z - env_z) > self.cfg.grasp_attach_height)
            self.task_success[lifted] = True

        # Grasp break 감지 (fixed joint 파손 시 drop 판정)
        if self.object_grasped.any() and self._physics_grasp:
            grip_pos_w = self.robot.data.body_pos_w[:, self._gripper_body_idx]
            obj_delta = self.object_pos_w - grip_pos_w
            grip_obj_dist = torch.norm(obj_delta, dim=-1)
            drop_detected = self.object_grasped & (grip_obj_dist > float(self.cfg.grasp_drop_detect_dist))
            if drop_detected.any():
                self.object_grasped[drop_detected] = False
                self.just_dropped[drop_detected] = True
                drop_ids = drop_detected.nonzero(as_tuple=False).squeeze(-1)
                if self._grasp_attach_mode == "fixed_joint":
                    self._disable_grasp_fixed_joint_for_envs(drop_ids)

        # GRASP timeout
        if self.cfg.grasp_timeout_steps > 0:
            near_object = metrics["object_dist"] < self.cfg.approach_thresh
            in_grasp_zone = near_object & (~self.object_grasped)
            first_entry = in_grasp_zone & (self.grasp_entry_step == 0)
            self.grasp_entry_step[first_entry] = self.episode_length_buf[first_entry]
            grasp_elapsed = self.episode_length_buf - self.grasp_entry_step
            timed_out = in_grasp_zone & (self.grasp_entry_step > 0) & (grasp_elapsed > self.cfg.grasp_timeout_steps)
            self.grasp_entry_step[timed_out] = 0

    # ═══════════════════════════════════════════════════════════════════
    #  Action pipeline — 순서 반전
    # ═══════════════════════════════════════════════════════════════════

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        raw = actions.clone().clamp(-1.0, 1.0)
        # Action delay: FIFO buffer로 1-N step 지연된 action 적용
        if self._action_delay_buf is not None:
            self.actions = self._action_delay_buf[0].clone()
            self._action_delay_buf = torch.cat(
                [self._action_delay_buf[1:], raw.unsqueeze(0)], dim=0
            )
        else:
            self.actions = raw

    def _apply_action(self):
        # 새 순서: [arm5, grip1, base3]
        arm_grip_action = self.actions[:, 0:6]

        base_vx = self.actions[:, 6] * self.cfg.max_lin_vel
        base_vy = self.actions[:, 7] * self.cfg.max_lin_vel
        base_wz = self.actions[:, 8] * self.cfg.max_ang_vel

        # Base -> Kiwi IK -> Wheel (v8 로직 그대로)
        body_cmd = torch.stack([base_vx, base_vy, base_wz], dim=-1)
        wheel_radps = body_cmd @ self.kiwi_M.T / self.wheel_radius

        vel_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        vel_target[:, self.wheel_idx] = wheel_radps
        self.robot.set_joint_velocity_target(vel_target)

        # Arm -> Position Target (v8 arm_action_to_limits 로직 그대로)
        if self.cfg.arm_action_to_limits:
            if self._arm_action_limits_override is not None:
                arm_limits = self._arm_action_limits_override
            else:
                arm_limits = self.robot.data.soft_joint_pos_limits[:, self.arm_idx]
            arm_lo = arm_limits[..., 0]
            arm_hi = arm_limits[..., 1]
            finite = torch.isfinite(arm_lo) & torch.isfinite(arm_hi) & ((arm_hi - arm_lo) > 1e-6)

            center = 0.5 * (arm_lo + arm_hi)
            half = 0.5 * (arm_hi - arm_lo)
            mapped = center + arm_grip_action * half
            fallback = arm_grip_action * self.cfg.arm_action_scale
            arm_targets = torch.where(finite, mapped, fallback)
            arm_targets = torch.where(finite, torch.clamp(arm_targets, arm_lo, arm_hi), arm_targets)
        else:
            arm_targets = arm_grip_action * self.cfg.arm_action_scale

        pos_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        pos_target[:, self.arm_idx] = arm_targets
        self.robot.set_joint_position_target(pos_target)

    # ═══════════════════════════════════════════════════════════════════
    #  Observations — 30D Actor + 37D Critic
    # ═══════════════════════════════════════════════════════════════════

    def _get_observations(self) -> dict:
        metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()
        base_body_vel = self._read_base_body_vel()  # (N, 3) — vx, vy, wz

        arm_pos = metrics["arm_pos"]
        arm_vel = metrics["arm_vel"]
        lin_vel = metrics["lin_vel_b"]
        ang_vel = metrics["ang_vel_b"]

        rel_object = metrics["object_pos_b"]

        contact_force = self._contact_force_per_env()
        contact_binary = (contact_force > float(self.cfg.grasp_contact_threshold)).float()
        contact_lr = torch.stack([contact_binary, contact_binary], dim=-1)

        bbox_norm = self.object_bbox / float(self._bbox_norm_scale)
        cat_denom = max(int(self.cfg.num_object_categories) - 1, 1)
        cat_norm = (self.object_category_id / float(cat_denom)).unsqueeze(-1)

        # Observation noise (sim2real: 센서 노이즈 시뮬레이션, 학습 시에만)
        if bool(self.cfg.enable_domain_randomization):
            jp_noise = float(self.cfg.dr_obs_noise_joint_pos)
            bv_noise = float(self.cfg.dr_obs_noise_base_vel)
            or_noise = float(self.cfg.dr_obs_noise_object_rel)
            if jp_noise > 0:
                arm_pos = arm_pos + torch.randn_like(arm_pos) * jp_noise
            if bv_noise > 0:
                base_body_vel = base_body_vel + torch.randn_like(base_body_vel) * bv_noise
                lin_vel = lin_vel + torch.randn_like(lin_vel) * bv_noise
                ang_vel = ang_vel + torch.randn_like(ang_vel) * bv_noise
            if or_noise > 0:
                rel_object = rel_object + torch.randn_like(rel_object) * or_noise

        # Actor Observation (30D)
        actor_obs = torch.cat([
            arm_pos[:, :5],             # [0:5]   arm joint pos 5D
            arm_pos[:, 5:6],            # [5:6]   gripper pos 1D
            base_body_vel,              # [6:9]   base body velocity 3D (m/s, rad/s)
            lin_vel,                    # [9:12]  base linear vel 3D
            ang_vel,                    # [12:15] base angular vel 3D
            arm_vel,                    # [15:21] arm+grip joint vel 6D
            rel_object,                 # [21:24] object relative pos 3D
            contact_lr,                 # [24:26] contact L/R 2D
            bbox_norm,                  # [26:29] object bbox 3D
            cat_norm,                   # [29:30] object category 1D
        ], dim=-1)  # 30D

        self._cached_metrics = None

        # Critic Observation (37D, AAC)
        obj_mass_per_env = self._catalog_mass[
            self.active_object_idx.clamp(max=len(self._catalog_mass) - 1)
        ].unsqueeze(-1)
        critic_extra = torch.cat([
            self.object_bbox,                            # 3D (원본, 비정규화)
            obj_mass_per_env,                            # 1D
            metrics["object_dist"].unsqueeze(-1),        # 1D
            metrics["heading_object"].unsqueeze(-1),     # 1D
            metrics["vel_toward_object"].unsqueeze(-1),  # 1D
        ], dim=-1)  # 7D
        critic_obs = torch.cat([actor_obs, critic_extra], dim=-1)  # 37D
        self._critic_obs = critic_obs  # AAC wrapper에서 state()로 접근

        return {"policy": actor_obs, "critic": critic_obs}

    # ═══════════════════════════════════════════════════════════════════
    #  Rewards — Approach + Grasp + Lift
    # ═══════════════════════════════════════════════════════════════════

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._cached_metrics
        if metrics is None:
            metrics = self._compute_metrics()
            self._update_grasp_state(metrics)

        reward = torch.full((self.num_envs,), self.cfg.rew_time_penalty, device=self.device)

        # Effort penalty (새 action 순서: base는 [6:9])
        effort_pen = self.cfg.rew_effort_weight * (self.actions[:, 6:9] ** 2).sum(dim=-1)
        arm_pen = self.cfg.rew_arm_move_weight * (metrics["arm_vel"] ** 2).sum(dim=-1)
        reward += effort_pen + arm_pen

        # Action smoothness penalty (sim2real: 실기에서 부드러운 동작 유도)
        action_delta = self.actions - self.prev_actions
        reward += self.cfg.rew_action_smoothness_weight * (action_delta ** 2).sum(dim=-1)

        # Approach: 물체 방향 진전
        approach_progress = torch.clamp(
            self.prev_object_dist - metrics["object_dist"], -0.2, 0.2
        )
        reward += self.cfg.rew_approach_progress_weight * approach_progress

        # Tanh proximity kernel (목표 가까울수록 강한 gradient로 수렴 가속)
        proximity_bonus = 1.0 - torch.tanh(
            metrics["object_dist"] / self.cfg.rew_proximity_tanh_sigma
        )
        reward += self.cfg.rew_proximity_tanh_weight * proximity_bonus

        # Heading alignment
        reward += self.cfg.rew_approach_heading_weight * torch.cos(metrics["heading_object"])

        # Velocity toward object
        reward += self.cfg.rew_approach_vel_weight * metrics["vel_toward_object"]

        # Grasp 성공 보너스
        reward += self.just_grasped.float() * self.cfg.rew_grasp_success_bonus

        # Lift 성공 보너스
        reward += self.task_success.float() * self.cfg.rew_lift_bonus

        # Gripper shaping (v8 그대로)
        if self._physics_grasp:
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
            if self._multi_object:
                bbox_max = self.object_bbox.max(dim=-1).values
                close_target = torch.clamp(
                    float(self.cfg.grasp_gripper_threshold) * (1.0 - bbox_max * 2.0),
                    min=0.0, max=1.0,
                )
                close_progress = torch.clamp(close_target - gripper_pos, min=0.0, max=1.0)
            else:
                close_progress = torch.clamp(
                    float(self.cfg.grasp_gripper_threshold) - gripper_pos, min=0.0, max=1.0
                )
            near_object = metrics["object_dist"] < self.cfg.approach_thresh
            reward += near_object.float() * (0.5 * close_progress)
            far_from_object = ~near_object
            open_progress = torch.clamp(
                gripper_pos - float(self.cfg.grasp_gripper_threshold), min=0.0, max=1.0
            )
            reward += far_from_object.float() * (0.1 * open_progress)

        self.prev_object_dist[:] = metrics["object_dist"]
        self.episode_reward_sum += reward
        return reward

    # ═══════════════════════════════════════════════════════════════════
    #  Dones — Curriculum 업데이트 포함
    # ═══════════════════════════════════════════════════════════════════

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self._compute_metrics()
        self._update_grasp_state(metrics)
        self._cached_metrics = metrics

        root_pos = metrics["root_pos_w"]
        out_of_bounds = torch.norm(
            root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1
        ) > self.cfg.max_dist_from_origin
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)
        terminated = out_of_bounds | fell

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        truncated = self.task_success | time_out

        # Logging
        self.extras["task_success_rate"] = self.task_success.float().mean()

        # Curriculum 업데이트
        if self.task_success.any() or time_out.any():
            done_mask = self.task_success | time_out | terminated
            if done_mask.any():
                batch_success = self.task_success[done_mask].float().mean().item()
                self._curriculum_success_window[self._curriculum_idx % 100] = batch_success
                self._curriculum_idx += 1
                if self._curriculum_idx >= 100:
                    avg = self._curriculum_success_window.mean().item()
                    if avg > self.cfg.curriculum_success_threshold:
                        old = self._curriculum_dist
                        self._curriculum_dist = min(
                            self._curriculum_dist + self.cfg.curriculum_dist_increment,
                            self.cfg.object_dist_max,
                        )
                        if self._curriculum_dist != old:
                            print(f"  [Curriculum] dist: {old:.2f} -> {self._curriculum_dist:.2f} (avg success: {avg:.2f})")

        return terminated, truncated

    # ═══════════════════════════════════════════════════════════════════
    #  Reset — Curriculum 적용
    # ═══════════════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        num = len(env_ids)
        if num == 0:
            return

        # Grasp joint 해제 (v8 그대로)
        if self._physics_grasp and self._grasp_attach_mode == "fixed_joint":
            self._disable_grasp_fixed_joint_for_envs(env_ids)

        # Root reset (v8 그대로)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        root_xy_std = float(self.cfg.dr_root_xy_noise_std) if bool(self.cfg.enable_domain_randomization) else 0.1
        default_root_state[:, 0:2] += torch.randn(num, 2, device=self.device) * root_xy_std

        random_yaw = torch.rand(num, device=self.device) * 2.0 * math.pi - math.pi
        if bool(self.cfg.enable_domain_randomization) and float(self.cfg.dr_root_yaw_jitter_rad) > 0.0:
            random_yaw += torch.randn(num, device=self.device) * float(self.cfg.dr_root_yaw_jitter_rad)
            random_yaw = torch.atan2(torch.sin(random_yaw), torch.cos(random_yaw))
        half_yaw = random_yaw * 0.5
        default_root_state[:, 3] = torch.cos(half_yaw)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(half_yaw)
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # Joint reset (v8 그대로)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Home / Object 설정
        self.home_pos_w[env_ids] = default_root_state[:, :3]
        base_xy = self.home_pos_w[env_ids, :2]

        # Curriculum: object_dist_max 대신 _curriculum_dist 사용
        self.object_pos_w[env_ids] = self._sample_targets_around(
            env_ids=env_ids,
            base_xy=base_xy,
            dist_min=self.cfg.object_dist_min,
            dist_max=self._curriculum_dist,
            base_z=self.home_pos_w[env_ids, 2],
        )

        # Multi-object hide/show (v8 그대로)
        if self._multi_object and len(self.object_rigids) > 0:
            chosen = torch.randint(0, self._num_object_types, (num,), device=self.device)
            self.active_object_idx[env_ids] = chosen
            self.object_bbox[env_ids] = self._catalog_bbox[chosen]
            self.object_category_id[env_ids] = self._catalog_category[chosen]
            self.object_pos_w[env_ids, 2] = self.home_pos_w[env_ids, 2] + torch.clamp(
                self.object_bbox[env_ids, 2] * 0.5, min=float(self.cfg.object_height),
            )
            for rigid in self.object_rigids:
                hide_pose = rigid.data.default_root_state[env_ids, :7].clone()
                hide_pose[:, 0] = 0.0
                hide_pose[:, 1] = 0.0
                hide_pose[:, 2] = -10.0
                rigid.write_root_pose_to_sim(hide_pose, env_ids=env_ids)
                obj_vel = torch.zeros((num, 6), dtype=torch.float32, device=self.device)
                rigid.write_root_velocity_to_sim(obj_vel, env_ids=env_ids)

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

        # Task 버퍼 리셋
        self.object_grasped[env_ids] = False
        self.task_success[env_ids] = False
        self.just_grasped[env_ids] = False
        self.just_dropped[env_ids] = False
        self.prev_object_dist[env_ids] = 10.0
        self.grasp_entry_step[env_ids] = 0
        self.episode_reward_sum[env_ids] = 0.0

        # Common buffers reset
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        if self._action_delay_buf is not None:
            self._action_delay_buf[:, env_ids] = 0.0

        # DR 적용 (v8 그대로)
        self._apply_domain_randomization(env_ids)
