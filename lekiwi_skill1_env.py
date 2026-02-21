"""
LeKiwi Skill-1 — Navigate Isaac Lab DirectRLEnv.

3-Skill pipeline first skill: navigate toward target object while avoiding obstacles.
Arm held at TUCKED_POSE, gripper open. RL controls base only (3D effective action).

Observation (Actor 20D):
  [0:5]   arm joint pos (5) — fixed TUCKED_POSE, included for VLA format
  [5:6]   gripper pos (1) — fixed open
  [6:9]   base body velocity (vx, vy, wz) (3)
  [9:12]  object relative pos body (3)
  [12:20] pseudo-lidar scan (8 rays, normalized)

Observation (Critic 25D, AAC):
  Actor 20D + abs_object_dist(1) + heading_to_object(1) +
  vel_toward_object(1) + closest_obstacle_dist(1) + closest_obstacle_angle(1)

Action (9D — lekiwi_v6 order):
  [0:5]   arm joint target (IGNORED — forced to TUCKED_POSE)
  [5]     gripper target (IGNORED — forced to open)
  [6:8]   base linear velocity (vx, vy)
  [8]     base angular velocity (wz)
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

# TUCKED_POSE (from calibration/tucked_pose.json, sim joint names order)
# arm 5D only (no gripper)
_TUCKED_POSE_RAD = [
    -0.02966,    # shoulder_pan
    -0.213839,   # shoulder_lift
    0.09066,     # elbow_flex
    0.120177,    # wrist_flex
    0.058418,    # wrist_roll
]
_TUCKED_GRIPPER_RAD = -0.201554  # gripper joint value at tucked (closed-ish)
_GRIPPER_OPEN_RAD = 1.0  # Navigate: gripper fully open


@configclass
class Skill1EnvCfg(DirectRLEnvCfg):
    """Navigate RL environment config."""

    # === Simulation (same as Skill-2) ===
    sim: SimulationCfg = SimulationCfg(
        dt=0.02, render_interval=2,
        gravity=(0.0, 0.0, -9.81), device="cuda:0",
    )
    decimation: int = 2
    episode_length_s: float = 15.0

    # === Scene (same as Skill-2) ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=10.0, replicate_physics=True,
    )

    # === Robot (same as Skill-2) ===
    robot_cfg = LEKIWI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # === Spaces ===
    observation_space: int = 20
    action_space: int = 9
    state_space: int = 25  # Critic (AAC)

    # === Action (same as Skill-2) ===
    max_lin_vel: float = 0.5
    max_ang_vel: float = 3.0
    arm_action_scale: float = 1.5
    arm_action_to_limits: bool = True

    # === Calibration (same as Skill-2) ===
    calibration_json: str | None = None
    dynamics_json: str | None = None
    dynamics_apply_cmd_scale: bool = True
    arm_limit_json: str | None = None
    arm_limit_margin_rad: float = 0.0
    arm_limit_write_to_sim: bool = True

    # === Task Geometry ===
    object_dist_min: float = 1.0
    object_dist_max: float = 4.0
    arrival_thresh: float = 0.5  # = Skill-2 curriculum start
    object_height: float = 0.03

    # === Object (for target spawning, no grasp) ===
    object_usd: str = ""
    object_mass: float = 0.3
    object_scale: float = 1.0
    object_prim_path: str = "/World/envs/env_.*/Object"
    multi_object_json: str = ""
    num_object_categories: int = 6

    # === Obstacle (tensor-based, no USD prims) ===
    num_obstacles_min: int = 3
    num_obstacles_max: int = 8
    obstacle_size_min: float = 0.15
    obstacle_size_max: float = 0.5
    obstacle_dist_min: float = 0.6
    obstacle_dist_max: float = 3.8
    collision_dist: float = 0.20
    obstacle_clearance_from_object: float = 0.4

    # === Pseudo-Lidar ===
    lidar_num_rays: int = 8
    lidar_max_range: float = 2.0

    # === Reward ===
    rew_time_penalty: float = -0.01
    rew_approach_weight: float = 3.0
    rew_arrival_bonus: float = 15.0
    rew_collision_penalty: float = -2.0
    rew_speed_bonus: float = 0.5
    rew_action_smoothness: float = -0.005
    rew_decel_weight: float = 0.15
    rew_decel_dist: float = 0.6  # Just above arrival_thresh (0.5m)
    rew_heading_weight: float = 0.3   # Face the target before moving
    rew_diagonal_penalty: float = -1.0  # Prevent simultaneous vx+vy

    # === Termination ===
    max_dist_from_origin: float = 6.0

    # === DR (wheel only, no arm/object/grasp DR) ===
    enable_domain_randomization: bool = True
    dr_root_xy_noise_std: float = 0.12
    dr_root_yaw_jitter_rad: float = 0.2
    dr_wheel_stiffness_scale_range: tuple[float, float] = (0.75, 1.5)
    dr_wheel_damping_scale_range: tuple[float, float] = (0.3, 3.0)
    dr_wheel_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_wheel_dynamic_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_wheel_viscous_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_arm_stiffness_scale_range: tuple[float, float] = (0.8, 1.25)
    dr_arm_damping_scale_range: tuple[float, float] = (0.5, 2.0)

    # Obs noise
    dr_obs_noise_base_vel: float = 0.02
    dr_obs_noise_object_rel: float = 0.02
    dr_obs_noise_lidar: float = 0.05

    # Action delay
    dr_action_delay_steps: int = 1

    # === TUCKED_POSE JSON (optional override) ===
    tucked_pose_json: str | None = None


class Skill1Env(DirectRLEnv):
    """LeKiwi Skill-1: Navigate RL environment."""

    cfg: Skill1EnvCfg

    def __init__(self, cfg: Skill1EnvCfg, render_mode: str | None = None, **kwargs):
        self._multi_object = bool(str(getattr(cfg, "multi_object_json", "")).strip())
        super().__init__(cfg, render_mode, **kwargs)

        self._multi_object = bool(str(self.cfg.multi_object_json).strip())
        self.object_rigids: list[RigidObject] = list(getattr(self, "object_rigids", []))
        self.object_rigid: RigidObject | None = getattr(self, "object_rigid", None)
        self._object_catalog: list[dict] = list(getattr(self, "_object_catalog", []))
        self._num_object_types: int = int(getattr(self, "_num_object_types", 0))

        # Joint indices
        self.arm_idx, _ = self.robot.find_joints(ARM_JOINT_NAMES)
        self.wheel_idx, _ = self.robot.find_joints(WHEEL_JOINT_NAMES)
        self.arm_idx = torch.tensor(self.arm_idx, device=self.device)
        self.wheel_idx = torch.tensor(self.wheel_idx, device=self.device)
        gripper_ids, _ = self.robot.find_joints([GRIPPER_JOINT_NAME])
        if len(gripper_ids) != 1:
            raise RuntimeError(f"Expected single gripper joint: {GRIPPER_JOINT_NAME}")
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

        # TUCKED_POSE
        self._tucked_pose = self._load_tucked_pose()

        # Task buffers
        self.home_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.active_object_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_bbox = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_category_id = torch.zeros(self.num_envs, device=self.device)
        self._bbox_norm_scale = 0.2

        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_object_dist = torch.zeros(self.num_envs, device=self.device)

        # Action / metrics buffers
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.episode_reward_sum = torch.zeros(self.num_envs, device=self.device)

        # Action delay
        delay = max(int(self.cfg.dr_action_delay_steps), 0)
        if delay > 0:
            self._action_delay_buf = torch.zeros(
                delay, self.num_envs, self.cfg.action_space, device=self.device
            )
        else:
            self._action_delay_buf = None
        self._cached_metrics: Dict[str, torch.Tensor] | None = None

        # Obstacle tensors
        self._max_obstacles = int(self.cfg.num_obstacles_max)
        self._obstacle_pos = torch.zeros(
            self.num_envs, self._max_obstacles, 2, device=self.device
        )
        self._obstacle_radius = torch.zeros(
            self.num_envs, self._max_obstacles, device=self.device
        )
        self._obstacle_valid = torch.zeros(
            self.num_envs, self._max_obstacles, dtype=torch.bool, device=self.device
        )

        # DR buffers
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
        self._arm_action_limits_override: torch.Tensor | None = None

        # Multi-object catalog
        if self._multi_object:
            if self._num_object_types <= 0 or len(self._object_catalog) == 0:
                raise RuntimeError("multi_object_json was provided but catalog is empty.")
            catalog_bbox_rows: list[list[float]] = []
            for obj in self._object_catalog:
                raw_bbox = obj.get("bbox", [0.05, 0.05, 0.05])
                if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 3:
                    raw_bbox = [0.05, 0.05, 0.05]
                try:
                    bx, by, bz = float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2])
                except (TypeError, ValueError):
                    bx, by, bz = 0.05, 0.05, 0.05
                try:
                    scale = float(obj.get("scale", 1.0))
                except (TypeError, ValueError):
                    scale = 1.0
                if not math.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
                catalog_bbox_rows.append([max(bx, 1e-6) * scale, max(by, 1e-6) * scale, max(bz, 1e-6) * scale])
            self._catalog_bbox = torch.tensor(catalog_bbox_rows, dtype=torch.float32, device=self.device)
            self._catalog_category = torch.tensor(
                [float(obj.get("category", 0)) for obj in self._object_catalog],
                dtype=torch.float32, device=self.device,
            )
        else:
            self.object_bbox[:] = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=self.device)
            self.object_category_id[:] = 0.0
            self._catalog_bbox = torch.tensor([[0.05, 0.05, 0.05]], dtype=torch.float32, device=self.device)
            self._catalog_category = torch.tensor([0.0], dtype=torch.float32, device=self.device)

        # Precompute lidar ray angles (body frame, uniform)
        self._lidar_ray_angles = torch.linspace(
            0, 2 * math.pi, self.cfg.lidar_num_rays + 1, device=self.device
        )[:-1]  # (num_rays,)

        self._maybe_apply_tuned_dynamics_from_cfg()
        self._apply_baked_arm_limits()
        self._maybe_apply_arm_limits_from_cfg()
        self._init_domain_randomization_buffers()

        print(f"  [Skill1Env] obs={self.cfg.observation_space} act={self.cfg.action_space} critic={self.cfg.state_space}")
        print(f"  [Skill1Env] arm_idx={self.arm_idx.tolist()} wheel_idx={self.wheel_idx.tolist()}")
        print(f"  [Skill1Env] obstacles={self.cfg.num_obstacles_min}-{self.cfg.num_obstacles_max}, lidar={self.cfg.lidar_num_rays} rays")

    # ═══════════════════════════════════════════════════════════════════
    #  TUCKED_POSE loading
    # ═══════════════════════════════════════════════════════════════════

    def _load_tucked_pose(self) -> torch.Tensor:
        """Load TUCKED_POSE from JSON or use hardcoded defaults. Returns (5,) arm targets."""
        pose_vals = list(_TUCKED_POSE_RAD)

        json_path = getattr(self.cfg, "tucked_pose_json", None)
        if not json_path:
            # Try default path
            default_path = os.path.join(os.path.dirname(__file__), "calibration", "tucked_pose.json")
            if os.path.isfile(default_path):
                json_path = default_path

        if json_path and os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                sim_names = payload.get("joints_sim_names", {})
                if sim_names:
                    for i, jname in enumerate(ARM_JOINT_NAMES[:5]):
                        if jname in sim_names:
                            pose_vals[i] = float(sim_names[jname])
                    print(f"  [Skill1Env] TUCKED_POSE loaded from {json_path}")
            except Exception as exc:
                print(f"  [Skill1Env] WARN: failed to load tucked_pose: {exc}, using defaults")

        return torch.tensor(pose_vals, dtype=torch.float32, device=self.device)

    # ═══════════════════════════════════════════════════════════════════
    #  Scene setup
    # ═══════════════════════════════════════════════════════════════════

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object_rigids = []
        self.object_rigid = None
        self._object_catalog = []
        self._num_object_types = 0
        self._multi_object = bool(str(self.cfg.multi_object_json).strip())

        # Object spawning (target object, no grasp)
        if self._multi_object:
            mo_path = os.path.expanduser(self.cfg.multi_object_json)
            if not os.path.isfile(mo_path):
                raise FileNotFoundError(f"multi_object_json not found: {mo_path}")
            with open(mo_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list) or len(payload) == 0:
                raise ValueError("multi_object_json must be a non-empty JSON list.")
            self._object_catalog = payload
            self._num_object_types = len(self._object_catalog)

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
                obj_cfg = RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=obj_usd,
                        scale=(obj_scale, obj_scale, obj_scale),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=True, kinematic_enabled=False,
                            disable_gravity=False, max_linear_velocity=2.0,
                            max_angular_velocity=5.0, max_depenetration_velocity=1.0,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj_mass),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
                )
                self.object_rigids.append(RigidObject(obj_cfg))

        elif str(self.cfg.object_usd).strip():
            object_usd = os.path.expanduser(self.cfg.object_usd)
            if not os.path.isfile(object_usd):
                raise FileNotFoundError(f"object_usd not found: {object_usd}")
            object_cfg = RigidObjectCfg(
                prim_path=self.cfg.object_prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=object_usd,
                    scale=(float(self.cfg.object_scale),) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True, kinematic_enabled=False,
                        disable_gravity=False, max_linear_velocity=2.0,
                        max_angular_velocity=5.0, max_depenetration_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=float(self.cfg.object_mass)),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, float(self.cfg.object_height))),
            )
            self.object_rigid = RigidObject(object_cfg)

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

    # ═══════════════════════════════════════════════════════════════════
    #  Calibration / Dynamics — copied from Skill2Env
    # ═══════════════════════════════════════════════════════════════════

    def _maybe_apply_calibration_geometry_from_cfg(self):
        raw_path = str(getattr(self.cfg, "calibration_json", "") or "").strip()
        if not raw_path:
            return
        path = os.path.expanduser(raw_path)
        if not os.path.isfile(path):
            print(f"  [Calibration] geometry JSON not found: {raw_path} (keep defaults)")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
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
            print(f"  [Calibration] geometry applied: wheel={self.wheel_radius:.6f}, base={self.base_radius:.6f}")

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
    def _safe_float(params: dict, key: str, default: float) -> float:
        try:
            return float(params.get(key, default))
        except (TypeError, ValueError):
            return float(default)

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
                lo, hi = val.get("min"), val.get("max")
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                lo, hi = val[0], val[1]
            if lo is None or hi is None:
                continue
            try:
                lo_f, hi_f = float(lo), float(hi)
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
            print(f"  [ArmLimits] applied from {source}: {len(joint_ids)} joints (write_to_sim=True)")
        else:
            print(f"  [ArmLimits] loaded from {source}: {applied_count} arm joints (write_to_sim=False)")

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
            raise ValueError(f"arm_limit_json has no valid limits: {path}")
        self._apply_arm_limits(limits_by_joint=limits_by_joint, source=path, margin=float(self.cfg.arm_limit_margin_rad))

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
            raise ValueError("Invalid dynamics JSON")
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

        tuned_wr = _pick_positive_finite(
            payload.get("wheel_radius_used"), raw_params.get("wheel_radius_used"),
        )
        tuned_br = _pick_positive_finite(
            payload.get("base_radius_used"), raw_params.get("base_radius_used"),
        )
        if tuned_wr is not None:
            self.wheel_radius = tuned_wr
        if tuned_br is not None:
            self.base_radius = tuned_br
            angles = torch.tensor(WHEEL_ANGLES_RAD, dtype=torch.float32, device=self.device)
            self.kiwi_M = torch.stack(
                [torch.cos(angles), torch.sin(angles), torch.full_like(angles, self.base_radius)],
                dim=-1,
            )

        wheel_ids = self.wheel_idx.tolist()
        arm_ids = self.arm_idx.tolist()
        base_ws = self.robot.data.joint_stiffness[:, self.wheel_idx].clone()
        base_wd = self.robot.data.joint_damping[:, self.wheel_idx].clone()
        base_wa = self.robot.data.joint_armature[:, self.wheel_idx].clone()
        base_as = self.robot.data.joint_stiffness[:, self.arm_idx].clone()
        base_ad = self.robot.data.joint_damping[:, self.arm_idx].clone()
        base_aa = self.robot.data.joint_armature[:, self.arm_idx].clone()

        self.robot.write_joint_stiffness_to_sim(base_ws * params["wheel_stiffness_scale"], joint_ids=wheel_ids)
        self.robot.write_joint_damping_to_sim(base_wd * params["wheel_damping_scale"], joint_ids=wheel_ids)
        self.robot.write_joint_armature_to_sim(base_wa * params["wheel_armature_scale"], joint_ids=wheel_ids)
        if hasattr(self.robot, "write_joint_friction_coefficient_to_sim"):
            self.robot.write_joint_friction_coefficient_to_sim(
                torch.full_like(base_wd, params["wheel_friction_coeff"]), joint_ids=wheel_ids)
        if hasattr(self.robot, "write_joint_dynamic_friction_coefficient_to_sim"):
            self.robot.write_joint_dynamic_friction_coefficient_to_sim(
                torch.full_like(base_wd, params["wheel_dynamic_friction_coeff"]), joint_ids=wheel_ids)
        if hasattr(self.robot, "write_joint_viscous_friction_coefficient_to_sim"):
            self.robot.write_joint_viscous_friction_coefficient_to_sim(
                torch.full_like(base_wd, params["wheel_viscous_friction_coeff"]), joint_ids=wheel_ids)

        arm_stiff_scale = torch.ones_like(base_as)
        arm_damp_scale = torch.ones_like(base_ad)
        for ji in range(len(ARM_JOINT_NAMES)):
            arm_stiff_scale[:, ji] = float(params.get(f"arm_stiffness_scale_j{ji}", 1.0))
            arm_damp_scale[:, ji] = float(params.get(f"arm_damping_scale_j{ji}", 1.0))
        self.robot.write_joint_stiffness_to_sim(
            base_as * params["arm_stiffness_scale"] * arm_stiff_scale, joint_ids=arm_ids)
        self.robot.write_joint_damping_to_sim(
            base_ad * params["arm_damping_scale"] * arm_damp_scale, joint_ids=arm_ids)
        self.robot.write_joint_armature_to_sim(base_aa * params["arm_armature_scale"], joint_ids=arm_ids)

        if self.cfg.dynamics_apply_cmd_scale:
            self.cfg.max_lin_vel = self._base_max_lin_vel * params["lin_cmd_scale"]
            self.cfg.max_ang_vel = self._base_max_ang_vel * params["ang_cmd_scale"]
        self._applied_dynamics_params = params
        print(f"\n  [Dynamics] tuned parameters applied from {path}")
        print(f"  [Dynamics] max_lin_vel={self.cfg.max_lin_vel:.4f} max_ang_vel={self.cfg.max_ang_vel:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    #  Domain Randomization — wheel only
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
        lo, hi = float(v[0]), float(v[1])
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

    def _apply_domain_randomization(self, env_ids: torch.Tensor):
        if not bool(self.cfg.enable_domain_randomization) or len(env_ids) == 0:
            return
        if self._dr_base_wheel_stiffness is None:
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
        self._dr_curr_wheel_dynamic_friction[ids] = torch.clamp(self._dr_base_wheel_dynamic_friction[ids] * wdf, min=0.0)
        self._dr_curr_wheel_viscous_friction[ids] = torch.clamp(self._dr_base_wheel_viscous_friction[ids] * wvf, min=0.0)
        if hasattr(self.robot, "write_joint_friction_coefficient_to_sim"):
            self.robot.write_joint_friction_coefficient_to_sim(self._dr_curr_wheel_friction, joint_ids=wheel_ids)
        if hasattr(self.robot, "write_joint_dynamic_friction_coefficient_to_sim"):
            self.robot.write_joint_dynamic_friction_coefficient_to_sim(self._dr_curr_wheel_dynamic_friction, joint_ids=wheel_ids)
        if hasattr(self.robot, "write_joint_viscous_friction_coefficient_to_sim"):
            self.robot.write_joint_viscous_friction_coefficient_to_sim(self._dr_curr_wheel_viscous_friction, joint_ids=wheel_ids)

    # ═══════════════════════════════════════════════════════════════════
    #  Utility
    # ═══════════════════════════════════════════════════════════════════

    def _sample_targets_around(self, env_ids, base_xy, dist_min, dist_max, base_z=None):
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

    def _read_base_body_vel(self):
        """Body-frame velocity (3D: vx, vy, wz) in m/s, rad/s."""
        vx = self.robot.data.root_lin_vel_b[:, 0:1]
        vy = self.robot.data.root_lin_vel_b[:, 1:2]
        wz = self.robot.data.root_ang_vel_b[:, 2:3]
        return torch.cat([vx, vy, wz], dim=-1)

    def _get_robot_heading(self) -> torch.Tensor:
        """Robot heading angle from quaternion (N,)."""
        quat = self.robot.data.root_quat_w  # (N, 4) [w,x,y,z]
        return 2.0 * torch.atan2(quat[:, 3], quat[:, 0])

    # ═══════════════════════════════════════════════════════════════════
    #  Obstacles (tensor-based, no USD prims)
    # ═══════════════════════════════════════════════════════════════════

    def _reset_obstacles(self, env_ids: torch.Tensor):
        """Randomize obstacle positions and sizes for given envs."""
        n = len(env_ids)
        if n == 0:
            return
        cfg = self.cfg
        M = self._max_obstacles

        # Random number of obstacles per env
        num_obs = torch.randint(
            cfg.num_obstacles_min, cfg.num_obstacles_max + 1, (n,), device=self.device
        )

        # Generate all obstacle positions (polar sampling around robot)
        r = torch.rand(n, M, device=self.device) * (cfg.obstacle_dist_max - cfg.obstacle_dist_min) + cfg.obstacle_dist_min
        theta = torch.rand(n, M, device=self.device) * 2.0 * math.pi

        robot_xy = self.home_pos_w[env_ids, :2]  # (n, 2)
        ox = robot_xy[:, 0:1] + r * torch.cos(theta)  # (n, M)
        oy = robot_xy[:, 1:2] + r * torch.sin(theta)

        # Random radius (half of obstacle size)
        radius = torch.rand(n, M, device=self.device) * (cfg.obstacle_size_max - cfg.obstacle_size_min) / 2 + cfg.obstacle_size_min / 2

        # Validity mask: only first num_obs obstacles per env are active
        idx_range = torch.arange(M, device=self.device).unsqueeze(0).expand(n, M)
        valid = idx_range < num_obs.unsqueeze(1)

        # Invalidate obstacles too close to target object
        obj_xy = self.object_pos_w[env_ids, :2]  # (n, 2)
        obs_pos = torch.stack([ox, oy], dim=-1)  # (n, M, 2)
        obj_dist = torch.norm(obs_pos - obj_xy.unsqueeze(1), dim=-1)  # (n, M)
        too_close_to_obj = obj_dist < cfg.obstacle_clearance_from_object
        valid = valid & (~too_close_to_obj)

        # Invalidate obstacles too close to robot start
        robot_dist = torch.norm(obs_pos - robot_xy.unsqueeze(1), dim=-1)
        too_close_to_robot = robot_dist < 0.3  # 30cm clearance
        valid = valid & (~too_close_to_robot)

        self._obstacle_pos[env_ids] = obs_pos
        self._obstacle_radius[env_ids] = radius
        self._obstacle_valid[env_ids] = valid

    # ═══════════════════════════════════════════════════════════════════
    #  Pseudo-Lidar (fully vectorized)
    # ═══════════════════════════════════════════════════════════════════

    def _compute_lidar_scan(self) -> torch.Tensor:
        """
        8-directional pseudo-lidar from GT obstacle positions.
        Returns: (N, num_rays) normalized [0=contact, 1=max_range].
        """
        N = self.num_envs
        num_rays = self.cfg.lidar_num_rays
        max_range = self.cfg.lidar_max_range
        M = self._max_obstacles

        robot_xy = self.robot.data.root_pos_w[:, :2]  # (N, 2)
        heading = self._get_robot_heading()  # (N,)

        # World-frame ray angles: (N, num_rays)
        world_angles = self._lidar_ray_angles.unsqueeze(0) + heading.unsqueeze(1)

        # Delta from robot to each obstacle: (N, M, 2)
        delta = self._obstacle_pos - robot_xy.unsqueeze(1)

        # Distance and angle to each obstacle: (N, M)
        obs_dist = torch.norm(delta, dim=-1)
        obs_angle = torch.atan2(delta[:, :, 1], delta[:, :, 0])

        # Distance to obstacle surface
        dist_to_surface = obs_dist - self._obstacle_radius  # (N, M)
        dist_to_surface = dist_to_surface.clamp(min=0.0)

        # Expand for broadcasting: angles (N, num_rays, 1) vs obstacles (N, 1, M)
        world_angles_exp = world_angles.unsqueeze(2)  # (N, num_rays, 1)
        obs_angle_exp = obs_angle.unsqueeze(1)  # (N, 1, M)

        # Angular difference (wrapped to [-pi, pi])
        angle_diff = obs_angle_exp - world_angles_exp  # (N, num_rays, M)
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        # Beam half-width
        beam_half_width = math.pi / num_rays

        # Obstacles within beam: (N, num_rays, M)
        in_beam = torch.abs(angle_diff) < beam_half_width

        # Valid obstacles within beam
        valid_exp = self._obstacle_valid.unsqueeze(1).expand(N, num_rays, M)
        mask = in_beam & valid_exp

        # Distance for valid obstacles (replace invalid with max_range)
        dist_exp = dist_to_surface.unsqueeze(1).expand(N, num_rays, M)
        dist_masked = torch.where(mask, dist_exp, torch.tensor(max_range, device=self.device))

        # Minimum distance per ray
        scan = dist_masked.min(dim=-1).values  # (N, num_rays)

        # Normalize to [0, 1]
        scan = (scan / max_range).clamp(0.0, 1.0)

        return scan

    # ═══════════════════════════════════════════════════════════════════
    #  Task metrics
    # ═══════════════════════════════════════════════════════════════════

    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        # Update object positions from sim
        if self._multi_object and len(self.object_rigids) > 0:
            for oi, rigid in enumerate(self.object_rigids):
                mask = self.active_object_idx == oi
                if not mask.any():
                    continue
                ids = mask.nonzero(as_tuple=False).squeeze(-1)
                self.object_pos_w[ids] = rigid.data.root_pos_w[ids]
        elif self.object_rigid is not None:
            self.object_pos_w[:] = self.object_rigid.data.root_pos_w

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w

        # Object relative (body frame)
        object_delta_w = self.object_pos_w - root_pos_w
        object_pos_b = quat_apply_inverse(root_quat_w, object_delta_w)
        object_dist = torch.norm(object_pos_b[:, :2], dim=-1)
        heading_object = torch.atan2(object_pos_b[:, 1], object_pos_b[:, 0])

        # Body-frame velocity
        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        lin_speed = torch.norm(lin_vel_b[:, :2], dim=-1)

        # Velocity toward object
        object_dir_b = object_pos_b[:, :2] / (object_dist.unsqueeze(-1) + 1e-6)
        vel_toward_object = (lin_vel_b[:, :2] * object_dir_b).sum(dim=-1)

        # Closest obstacle
        robot_xy = root_pos_w[:, :2].unsqueeze(1)  # (N, 1, 2)
        delta_obs = self._obstacle_pos - robot_xy  # (N, M, 2)
        obs_dist_all = torch.norm(delta_obs, dim=-1) - self._obstacle_radius  # (N, M)
        obs_dist_all = torch.where(
            self._obstacle_valid, obs_dist_all,
            torch.tensor(float("inf"), device=self.device)
        )
        min_obs_dist, min_obs_idx = obs_dist_all.min(dim=-1)  # (N,)

        # Angle to closest obstacle (body frame)
        closest_delta = torch.gather(delta_obs, 1, min_obs_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)).squeeze(1)
        closest_delta_b = quat_apply_inverse(root_quat_w, torch.cat([closest_delta, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1))
        closest_obs_angle = torch.atan2(closest_delta_b[:, 1], closest_delta_b[:, 0])

        return {
            "root_pos_w": root_pos_w,
            "object_pos_b": object_pos_b,
            "object_dist": object_dist,
            "heading_object": heading_object,
            "lin_vel_b": lin_vel_b,
            "lin_speed": lin_speed,
            "vel_toward_object": vel_toward_object,
            "min_obs_dist": min_obs_dist,
            "closest_obs_angle": closest_obs_angle,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  Action pipeline — arm forced to TUCKED_POSE
    # ═══════════════════════════════════════════════════════════════════

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        raw = actions.clone().clamp(-1.0, 1.0)
        if self._action_delay_buf is not None:
            self.actions = self._action_delay_buf[0].clone()
            self._action_delay_buf = torch.cat(
                [self._action_delay_buf[1:], raw.unsqueeze(0)], dim=0
            )
        else:
            self.actions = raw

    def _apply_action(self):
        # Base control from RL action [6:9]
        base_vx = self.actions[:, 6] * self.cfg.max_lin_vel
        base_vy = self.actions[:, 7] * self.cfg.max_lin_vel
        base_wz = self.actions[:, 8] * self.cfg.max_ang_vel

        # Kiwi Drive IK
        body_cmd = torch.stack([base_vx, base_vy, base_wz], dim=-1)
        wheel_radps = body_cmd @ self.kiwi_M.T / self.wheel_radius
        vel_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        vel_target[:, self.wheel_idx] = wheel_radps
        self.robot.set_joint_velocity_target(vel_target)

        # Arm forced to TUCKED_POSE (position target)
        pos_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        tucked = self._tucked_pose.unsqueeze(0).expand(self.num_envs, -1)  # (N, 5)
        pos_target[:, self.arm_idx[:5]] = tucked
        pos_target[:, self.arm_idx[5]] = _GRIPPER_OPEN_RAD  # gripper open
        self.robot.set_joint_position_target(pos_target)

    # ═══════════════════════════════════════════════════════════════════
    #  Observations — 20D Actor + 25D Critic
    # ═══════════════════════════════════════════════════════════════════

    def _get_observations(self) -> dict:
        metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()
        base_body_vel = self._read_base_body_vel()

        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        rel_object = metrics["object_pos_b"]

        # Pseudo-lidar
        lidar_scan = self._compute_lidar_scan()  # (N, 8)

        # Observation noise
        if bool(self.cfg.enable_domain_randomization):
            bv_noise = float(self.cfg.dr_obs_noise_base_vel)
            or_noise = float(self.cfg.dr_obs_noise_object_rel)
            li_noise = float(self.cfg.dr_obs_noise_lidar)
            if bv_noise > 0:
                base_body_vel = base_body_vel + torch.randn_like(base_body_vel) * bv_noise
            if or_noise > 0:
                rel_object = rel_object + torch.randn_like(rel_object) * or_noise
            if li_noise > 0:
                lidar_scan = (lidar_scan + torch.randn_like(lidar_scan) * li_noise).clamp(0.0, 1.0)

        # Actor Observation (20D)
        actor_obs = torch.cat([
            arm_pos[:, :5],       # [0:5]   arm joint pos (fixed TUCKED_POSE)
            arm_pos[:, 5:6],      # [5:6]   gripper pos (fixed open)
            base_body_vel,        # [6:9]   base body velocity (vx, vy, wz)
            rel_object,           # [9:12]  object relative pos (body frame)
            lidar_scan,           # [12:20] pseudo-lidar (8 rays, normalized)
        ], dim=-1)  # 20D

        self._cached_metrics = None

        # Critic Observation (25D, AAC)
        critic_extra = torch.cat([
            metrics["object_dist"].unsqueeze(-1),         # 1D
            metrics["heading_object"].unsqueeze(-1),      # 1D
            metrics["vel_toward_object"].unsqueeze(-1),   # 1D
            metrics["min_obs_dist"].unsqueeze(-1),        # 1D
            metrics["closest_obs_angle"].unsqueeze(-1),   # 1D
        ], dim=-1)  # 5D
        critic_obs = torch.cat([actor_obs, critic_extra], dim=-1)  # 25D
        self._critic_obs = critic_obs

        return {"policy": actor_obs, "critic": critic_obs}

    # ═══════════════════════════════════════════════════════════════════
    #  Rewards — Navigate
    # ═══════════════════════════════════════════════════════════════════

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._cached_metrics
        if metrics is None:
            metrics = self._compute_metrics()

        reward = torch.full((self.num_envs,), self.cfg.rew_time_penalty, device=self.device)

        curr_dist = metrics["object_dist"]

        # 1. Approach progress
        progress = torch.clamp(self.prev_object_dist - curr_dist, -0.2, 0.2)
        rew_approach = self.cfg.rew_approach_weight * progress

        # 2. Arrival bonus
        arrived = curr_dist < self.cfg.arrival_thresh
        rew_arrival = torch.where(arrived, self.cfg.rew_arrival_bonus, 0.0)

        # 3. Collision penalty
        min_obs_dist = metrics["min_obs_dist"]
        collision = min_obs_dist < self.cfg.collision_dist
        rew_collision = torch.where(collision, self.cfg.rew_collision_penalty, 0.0)

        # 4. Speed bonus (velocity toward object)
        rew_speed = self.cfg.rew_speed_bonus * metrics["vel_toward_object"].clamp(0.0, 0.5)

        # 5. Action smoothness
        delta_action = self.actions[:, 6:9] - self.prev_actions[:, 6:9]
        rew_smooth = self.cfg.rew_action_smoothness * (delta_action ** 2).sum(dim=-1)

        # 6. Deceleration reward (slow down near target)
        near_target = curr_dist < self.cfg.rew_decel_dist
        speed = metrics["lin_speed"]
        max_speed = self.cfg.max_lin_vel
        decel_bonus = self.cfg.rew_decel_weight * (1.0 - speed / max_speed).clamp(0.0, 1.0)
        rew_decel = torch.where(near_target, decel_bonus, 0.0)

        # 7. Heading alignment (reward facing the target)
        heading_to_obj = metrics["heading_object"]  # body-frame angle to object
        rew_heading = self.cfg.rew_heading_weight * torch.cos(heading_to_obj)

        # 8. Diagonal penalty (prevent simultaneous vx + vy)
        act_vx_abs = self.actions[:, 6].abs()  # actions are [-1, 1] normalized
        act_vy_abs = self.actions[:, 7].abs()
        rew_diagonal = self.cfg.rew_diagonal_penalty * act_vx_abs * act_vy_abs

        total = (reward + rew_approach + rew_arrival + rew_collision + rew_speed
                 + rew_smooth + rew_decel + rew_heading + rew_diagonal)
        self.prev_object_dist[:] = curr_dist

        # Logging
        self.extras["log"] = {
            "rew_approach": rew_approach.mean(),
            "rew_arrival": rew_arrival.mean(),
            "rew_collision": rew_collision.mean(),
            "rew_speed": rew_speed.mean(),
            "rew_smooth": rew_smooth.mean(),
            "rew_decel": rew_decel.mean(),
            "rew_heading": rew_heading.mean(),
            "rew_diagonal": rew_diagonal.mean(),
            "dist_to_target": curr_dist.mean(),
            "min_obstacle_dist": min_obs_dist.mean(),
            "arrival_rate": arrived.float().mean(),
            "collision_rate": collision.float().mean(),
            "avg_speed": speed.mean(),
            "diagonal_ratio": (act_vx_abs * act_vy_abs).mean(),
        }

        self.episode_reward_sum += total
        return total

    # ═══════════════════════════════════════════════════════════════════
    #  Dones
    # ═══════════════════════════════════════════════════════════════════

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self._compute_metrics()
        self._cached_metrics = metrics

        root_pos = metrics["root_pos_w"]
        out_of_bounds = torch.norm(
            root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1
        ) > self.cfg.max_dist_from_origin
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)
        terminated = out_of_bounds | fell

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        dist_to_target = metrics["object_dist"]
        arrived = dist_to_target < self.cfg.arrival_thresh
        self.task_success = arrived

        # Arrived = truncated (success), not terminated
        truncated = arrived | time_out

        self.extras["task_success_rate"] = arrived.float().mean()

        return terminated, truncated

    # ═══════════════════════════════════════════════════════════════════
    #  Reset
    # ═══════════════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        num = len(env_ids)
        if num == 0:
            return

        # Root reset (random position + yaw)
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

        # Joint reset (arm to TUCKED_POSE)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        for i in range(5):
            joint_pos[:, self.arm_idx[i]] = self._tucked_pose[i]
        joint_pos[:, self.arm_idx[5]] = _GRIPPER_OPEN_RAD
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Home / Object
        self.home_pos_w[env_ids] = default_root_state[:, :3]
        base_xy = self.home_pos_w[env_ids, :2]
        self.object_pos_w[env_ids] = self._sample_targets_around(
            env_ids=env_ids, base_xy=base_xy,
            dist_min=self.cfg.object_dist_min, dist_max=self.cfg.object_dist_max,
            base_z=self.home_pos_w[env_ids, 2],
        )

        # Multi-object hide/show
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
                hide_pose[:, 2] = -100.0
                rigid.write_root_pose_to_sim(hide_pose, env_ids=env_ids)
                rigid.write_root_velocity_to_sim(torch.zeros((num, 6), device=self.device), env_ids=env_ids)
            for oi, rigid in enumerate(self.object_rigids):
                mask = chosen == oi
                if not mask.any():
                    continue
                sel_ids = env_ids[mask]
                sel_num = int(mask.sum().item())
                pose = rigid.data.default_root_state[sel_ids, :7].clone()
                pose[:, :3] = self.object_pos_w[sel_ids]
                obj_yaw = torch.rand(sel_num, device=self.device) * 2.0 * math.pi - math.pi
                half = obj_yaw * 0.5
                pose[:, 3] = torch.cos(half)
                pose[:, 4:6] = 0.0
                pose[:, 6] = torch.sin(half)
                rigid.write_root_pose_to_sim(pose, env_ids=sel_ids)
                rigid.write_root_velocity_to_sim(torch.zeros((sel_num, 6), device=self.device), env_ids=sel_ids)
        elif self.object_rigid is not None:
            self.active_object_idx[env_ids] = 0
            obj_pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
            obj_pose[:, :3] = self.object_pos_w[env_ids]
            obj_yaw = torch.rand(num, device=self.device) * 2.0 * math.pi - math.pi
            half = obj_yaw * 0.5
            obj_pose[:, 3] = torch.cos(half)
            obj_pose[:, 4:6] = 0.0
            obj_pose[:, 6] = torch.sin(half)
            self.object_rigid.write_root_pose_to_sim(obj_pose, env_ids=env_ids)
            self.object_rigid.write_root_velocity_to_sim(torch.zeros((num, 6), device=self.device), env_ids=env_ids)
        else:
            self.active_object_idx[env_ids] = 0

        # Obstacles (tensor-based)
        self._reset_obstacles(env_ids)

        # Task state reset
        self.task_success[env_ids] = False
        self.prev_object_dist[env_ids] = self.object_pos_w[env_ids, :2].sub(
            self.robot.data.root_pos_w[env_ids, :2]
        ).norm(dim=-1)
        self.episode_reward_sum[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        if self._action_delay_buf is not None:
            self._action_delay_buf[:, env_ids] = 0.0

        # DR
        self._apply_domain_randomization(env_ids)
