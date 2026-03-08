"""
LeKiwi Skill-3: CarryAndPlace — Isaac Lab DirectRLEnv.

Asymmetric Actor-Critic (AAC):
  Actor obs  13D: arm_pos(5) + grip(1) + base_disp(3) + home_rel(3) + grip_force(1)
  Critic obs 20D: Actor 13D + obj_dims(3D) + obj_mass(1D) + gripper_rel_pos(3D)
  Action      9D: arm_target(5) + gripper_cmd(1) + base_cmd(3, vx/vy/wz)

Reset from Handoff Buffer (Skill-2 success states).
break_force = mass × g × 10.
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
class CarryPlaceEnvCfg(DirectRLEnvCfg):
    """Skill-3 CarryAndPlace config."""

    sim: SimulationCfg = SimulationCfg(
        dt=0.02,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        device="cpu",
    )
    decimation: int = 2
    episode_length_s: float = 15.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=10.0,
        replicate_physics=True,
    )

    robot_cfg = LEKIWI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Spaces (AAC)
    observation_space: int = 13  # actor obs
    action_space: int = 9
    state_space: int = 20  # critic obs (privileged)

    # Action scaling
    max_lin_vel: float = 0.5
    max_ang_vel: float = 3.0
    arm_action_scale: float = 1.5
    arm_action_to_limits: bool = True

    # Calibration / dynamics
    calibration_json: str | None = None
    dynamics_json: str | None = None
    dynamics_apply_cmd_scale: bool = True
    arm_limit_json: str | None = None
    arm_limit_margin_rad: float = 0.0
    arm_limit_write_to_sim: bool = False

    # Object / physics grasp
    object_usd: str = ""
    object_mass: float = 0.3
    object_scale: float = 1.0
    gripper_contact_prim_path: str = ""
    object_prim_path: str = "/World/envs/env_.*/Object"
    object_filter_prim_expr: str = "/World/envs/env_.*/Object"
    grasp_gripper_threshold: float = -0.3
    grasp_contact_threshold: float = 0.5
    grasp_max_object_dist: float = 0.25
    grasp_attach_height: float = 0.15
    grasp_attach_mode: str = "fixed_joint"
    object_height: float = 0.03
    multi_object_json: str = ""
    num_object_categories: int = 6

    # Skill-3 specific
    handoff_buffer_path: str = ""
    home_dist_min: float = 1.0
    home_dist_max: float = 3.0
    place_thresh: float = 0.30

    # Rewards
    rew_carry: float = -1.0
    rew_hold: float = 0.1
    rew_place_success: float = 20.0
    rew_drop: float = -10.0
    rew_collision: float = -1.0
    rew_time_penalty: float = -0.01

    # Termination
    max_dist_from_origin: float = 6.0

    # Domain randomization
    enable_domain_randomization: bool = True
    dr_root_xy_noise_std: float = 0.12
    dr_root_yaw_jitter_rad: float = 0.2
    dr_wheel_stiffness_scale_range: tuple[float, float] = (0.9, 1.1)
    dr_wheel_damping_scale_range: tuple[float, float] = (0.85, 1.15)
    dr_wheel_friction_scale_range: tuple[float, float] = (0.8, 1.2)
    dr_wheel_dynamic_friction_scale_range: tuple[float, float] = (0.8, 1.2)
    dr_wheel_viscous_friction_scale_range: tuple[float, float] = (0.8, 1.2)
    dr_arm_stiffness_scale_range: tuple[float, float] = (0.9, 1.1)
    dr_arm_damping_scale_range: tuple[float, float] = (0.85, 1.15)
    dr_object_mass_scale_range: tuple[float, float] = (0.8, 1.2)
    dr_object_static_friction_scale_range: tuple[float, float] = (0.8, 1.2)
    dr_object_dynamic_friction_scale_range: tuple[float, float] = (0.8, 1.2)


class CarryPlaceEnv(DirectRLEnv):
    """Skill-3: CarryAndPlace RL environment."""

    cfg: CarryPlaceEnvCfg

    def __init__(self, cfg: CarryPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        self._multi_object = bool(str(getattr(cfg, "multi_object_json", "")).strip())
        self._physics_grasp = bool(str(cfg.object_usd).strip()) or self._multi_object
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
        self._maybe_apply_calibration_geometry()
        angles = torch.tensor(WHEEL_ANGLES_RAD, dtype=torch.float32, device=self.device)
        self.kiwi_M = torch.stack(
            [torch.cos(angles), torch.sin(angles), torch.full_like(angles, self.base_radius)],
            dim=-1,
        )

        # Task buffers
        self.home_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.active_object_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_bbox = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_mass_per_env = torch.full((self.num_envs,), self.cfg.object_mass, device=self.device)
        self.object_held = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_home_dist = torch.zeros(self.num_envs, device=self.device)

        # Action buffers
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self._cached_metrics: Dict[str, torch.Tensor] | None = None
        self._grasp_attach_mode = str(getattr(self.cfg, "grasp_attach_mode", "fixed_joint")).strip().lower()
        self._grasp_fixed_joints: dict[tuple[int, int], UsdPhysics.FixedJoint] = {}
        self._grasp_fixed_joint_warned = False
        self._arm_action_limits_override: torch.Tensor | None = None
        self._contact_shape_warned = False

        # DR buffers
        self._dr_base_wheel_stiffness = None
        self._dr_base_wheel_damping = None
        self._dr_base_arm_stiffness = None
        self._dr_base_arm_damping = None
        self._dr_curr_wheel_stiffness = None
        self._dr_curr_wheel_damping = None
        self._dr_curr_arm_stiffness = None
        self._dr_curr_arm_damping = None
        self._dr_base_wheel_friction = None
        self._dr_base_wheel_dynamic_friction = None
        self._dr_base_wheel_viscous_friction = None
        self._dr_curr_wheel_friction = None
        self._dr_curr_wheel_dynamic_friction = None
        self._dr_curr_wheel_viscous_friction = None
        self._dr_object_material_base: dict[str, tuple[float, float]] = {}

        # Multi-object catalog
        if self._multi_object:
            catalog_bbox_rows = []
            catalog_mass_list = []
            for obj in self._object_catalog:
                raw_bbox = obj.get("bbox", [0.05, 0.05, 0.05])
                if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 3:
                    raw_bbox = [0.05, 0.05, 0.05]
                scale = max(float(obj.get("scale", 1.0)), 1e-6)
                catalog_bbox_rows.append([max(float(raw_bbox[i]), 1e-6) * scale for i in range(3)])
                catalog_mass_list.append(max(float(obj.get("mass", self.cfg.object_mass)), 1e-5))
            self._catalog_bbox = torch.tensor(catalog_bbox_rows, dtype=torch.float32, device=self.device)
            self._catalog_mass = torch.tensor(catalog_mass_list, dtype=torch.float32, device=self.device)
        else:
            self.object_bbox[:] = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=self.device)
            self._catalog_mass = torch.tensor([max(float(self.cfg.object_mass), 1e-5)], dtype=torch.float32, device=self.device)

        self._maybe_apply_tuned_dynamics()
        self._apply_baked_arm_limits()
        self._maybe_apply_arm_limits_from_cfg()
        self._init_domain_randomization_buffers()

        # Load handoff buffer
        self._handoff_buffer = None
        if str(self.cfg.handoff_buffer_path).strip():
            try:
                from handoff_buffer import HandoffBuffer
                buf_path = os.path.expanduser(self.cfg.handoff_buffer_path)
                if os.path.isfile(buf_path):
                    self._handoff_buffer = HandoffBuffer.load(buf_path, device=str(self.device))
                    print(f"  [Skill3] Handoff buffer loaded: {buf_path} ({len(self._handoff_buffer)} entries)")
                else:
                    print(f"  [Skill3] Handoff buffer not found: {buf_path} (using random init)")
            except Exception as e:
                print(f"  [Skill3] Failed to load handoff buffer: {e}")

        print(f"  [Skill3-CarryPlace] obs={self.cfg.observation_space} "
              f"critic={self.cfg.state_space} act={self.cfg.action_space}")

    # ── Scene setup (same as Skill-2) ────────────────────────────────
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
                raise ValueError("Multi-object mode requires cfg.gripper_contact_prim_path.")
            mo_path = os.path.expanduser(self.cfg.multi_object_json)
            if not os.path.isfile(mo_path):
                raise FileNotFoundError(f"multi_object_json not found: {mo_path}")
            with open(mo_path, "r", encoding="utf-8") as f:
                self._object_catalog = json.load(f)
            self._num_object_types = len(self._object_catalog)
            filter_exprs = []
            for oi, obj_info in enumerate(self._object_catalog):
                obj_usd = os.path.expanduser(str(obj_info.get("usd", "")).strip())
                if not obj_usd or not os.path.isfile(obj_usd):
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
                            rigid_body_enabled=True, kinematic_enabled=False,
                            disable_gravity=False, max_linear_velocity=2.0, max_angular_velocity=5.0,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=obj_mass),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
                )
                self.object_rigids.append(RigidObject(obj_cfg))
            contact_cfg = ContactSensorCfg(
                prim_path=str(self.cfg.gripper_contact_prim_path),
                update_period=0.0, history_length=2, filter_prim_paths_expr=filter_exprs,
            )
            self.contact_sensor = ContactSensor(contact_cfg)
        elif self._physics_grasp:
            object_usd = os.path.expanduser(self.cfg.object_usd)
            if not os.path.isfile(object_usd):
                raise FileNotFoundError(f"object_usd not found: {object_usd}")
            if not str(self.cfg.gripper_contact_prim_path).strip():
                raise ValueError("Physics grasp requires cfg.gripper_contact_prim_path.")
            object_cfg = RigidObjectCfg(
                prim_path=self.cfg.object_prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=object_usd,
                    scale=(float(self.cfg.object_scale),) * 3,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True, kinematic_enabled=False,
                        disable_gravity=False, max_linear_velocity=2.0, max_angular_velocity=5.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=float(self.cfg.object_mass)),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, float(self.cfg.object_height))),
            )
            self.object_rigid = RigidObject(object_cfg)
            contact_cfg = ContactSensorCfg(
                prim_path=str(self.cfg.gripper_contact_prim_path),
                update_period=0.0, history_length=2,
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

    # ── Calibration / dynamics / arm limits (same pattern as Skill-2) ─
    def _maybe_apply_calibration_geometry(self):
        raw_path = str(getattr(self.cfg, "calibration_json", "") or "").strip()
        if not raw_path:
            return
        path = os.path.expanduser(raw_path)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        wr = payload.get("wheel_radius", {}).get("wheel_radius_m")
        br = payload.get("base_radius", {}).get("base_radius_m")
        try:
            if wr and float(wr) > 1e-8:
                self.wheel_radius = float(wr)
        except (TypeError, ValueError):
            pass
        try:
            if br and float(br) > 1e-8:
                self.base_radius = float(br)
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _safe_float(params, key, default):
        try:
            return float(params.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _maybe_apply_tuned_dynamics(self):
        if not self.cfg.dynamics_json:
            return
        path = os.path.expanduser(self.cfg.dynamics_json)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"dynamics_json not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_params = payload.get("best_params", payload.get("params", payload))
        if not isinstance(raw_params, dict):
            return
        params = {k: self._safe_float(raw_params, k, d) for k, d in [
            ("wheel_stiffness_scale", 1.0), ("wheel_damping_scale", 1.0),
            ("wheel_armature_scale", 1.0), ("wheel_friction_coeff", 0.0),
            ("arm_stiffness_scale", 1.0), ("arm_damping_scale", 1.0),
            ("arm_armature_scale", 1.0), ("lin_cmd_scale", 1.0), ("ang_cmd_scale", 1.0),
        ]}
        for ji in range(len(ARM_JOINT_NAMES)):
            params[f"arm_stiffness_scale_j{ji}"] = self._safe_float(raw_params, f"arm_stiffness_scale_j{ji}", 1.0)
            params[f"arm_damping_scale_j{ji}"] = self._safe_float(raw_params, f"arm_damping_scale_j{ji}", 1.0)
        wheel_ids = self.wheel_idx.tolist()
        arm_ids = self.arm_idx.tolist()
        bws = self.robot.data.joint_stiffness[:, self.wheel_idx].clone()
        bwd = self.robot.data.joint_damping[:, self.wheel_idx].clone()
        bwa = self.robot.data.joint_armature[:, self.wheel_idx].clone()
        bas = self.robot.data.joint_stiffness[:, self.arm_idx].clone()
        bad = self.robot.data.joint_damping[:, self.arm_idx].clone()
        baa = self.robot.data.joint_armature[:, self.arm_idx].clone()
        self.robot.write_joint_stiffness_to_sim(bws * params["wheel_stiffness_scale"], joint_ids=wheel_ids)
        self.robot.write_joint_damping_to_sim(bwd * params["wheel_damping_scale"], joint_ids=wheel_ids)
        self.robot.write_joint_armature_to_sim(bwa * params["wheel_armature_scale"], joint_ids=wheel_ids)
        ass = torch.ones_like(bas)
        ads = torch.ones_like(bad)
        for ji in range(len(ARM_JOINT_NAMES)):
            ass[:, ji] = params.get(f"arm_stiffness_scale_j{ji}", 1.0)
            ads[:, ji] = params.get(f"arm_damping_scale_j{ji}", 1.0)
        self.robot.write_joint_stiffness_to_sim(bas * params["arm_stiffness_scale"] * ass, joint_ids=arm_ids)
        self.robot.write_joint_damping_to_sim(bad * params["arm_damping_scale"] * ads, joint_ids=arm_ids)
        self.robot.write_joint_armature_to_sim(baa * params["arm_armature_scale"], joint_ids=arm_ids)
        if self.cfg.dynamics_apply_cmd_scale:
            self.cfg.max_lin_vel = self._base_max_lin_vel * params["lin_cmd_scale"]
            self.cfg.max_ang_vel = self._base_max_ang_vel * params["ang_cmd_scale"]
        self._applied_dynamics_params = params

    def _apply_baked_arm_limits(self):
        if ARM_LIMITS_BAKED_RAD:
            arm_limits = self.robot.data.soft_joint_pos_limits[:, self.arm_idx].clone()
            arm_name_to_offset = {str(name): i for i, name in enumerate(ARM_JOINT_NAMES)}
            for sim_joint, (lo, hi) in ARM_LIMITS_BAKED_RAD.items():
                off = arm_name_to_offset.get(str(sim_joint))
                if off is not None:
                    arm_limits[:, off, 0] = float(lo)
                    arm_limits[:, off, 1] = float(hi)
            self._arm_action_limits_override = arm_limits

    def _maybe_apply_arm_limits_from_cfg(self):
        if not self.cfg.arm_limit_json:
            return
        path = os.path.expanduser(self.cfg.arm_limit_json)
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        block = payload.get("joint_limits_rad", payload.get("arm_joint_limits_rad", payload))
        if not isinstance(block, dict):
            return
        arm_limits = self.robot.data.soft_joint_pos_limits[:, self.arm_idx].clone()
        arm_name_to_offset = {str(name): i for i, name in enumerate(ARM_JOINT_NAMES)}
        margin = float(self.cfg.arm_limit_margin_rad)
        for sim_joint, val in block.items():
            lo, hi = None, None
            if isinstance(val, dict):
                lo, hi = val.get("min"), val.get("max")
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                lo, hi = val[0], val[1]
            if lo is None or hi is None:
                continue
            try:
                lo_f, hi_f = float(lo) - margin, float(hi) + margin
            except (TypeError, ValueError):
                continue
            off = arm_name_to_offset.get(str(sim_joint))
            if off is not None:
                arm_limits[:, off, 0] = lo_f
                arm_limits[:, off, 1] = hi_f
        self._arm_action_limits_override = arm_limits

    # ── Domain Randomization ──────────────────────────────────────────
    def _init_domain_randomization_buffers(self):
        self._dr_base_wheel_stiffness = self.robot.data.joint_stiffness[:, self.wheel_idx].clone()
        self._dr_base_wheel_damping = self.robot.data.joint_damping[:, self.wheel_idx].clone()
        self._dr_base_arm_stiffness = self.robot.data.joint_stiffness[:, self.arm_idx].clone()
        self._dr_base_arm_damping = self.robot.data.joint_damping[:, self.arm_idx].clone()
        self._dr_curr_wheel_stiffness = self._dr_base_wheel_stiffness.clone()
        self._dr_curr_wheel_damping = self._dr_base_wheel_damping.clone()
        self._dr_curr_arm_stiffness = self._dr_base_arm_stiffness.clone()
        self._dr_curr_arm_damping = self._dr_base_arm_damping.clone()
        friction_base = float((self._applied_dynamics_params or {}).get("wheel_friction_coeff", 0.0))
        self._dr_base_wheel_friction = torch.full_like(self._dr_base_wheel_damping, friction_base)
        self._dr_curr_wheel_friction = self._dr_base_wheel_friction.clone()

    @staticmethod
    def _parse_scale_range(v):
        lo, hi = float(v[0]), float(v[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            return 1.0, 1.0
        return (lo, hi) if lo <= hi else (hi, lo)

    def _sample_scale(self, v, n):
        lo, hi = self._parse_scale_range(v)
        if n <= 0:
            return torch.empty((0, 1), device=self.device)
        if abs(hi - lo) < 1e-8:
            return torch.full((n, 1), lo, dtype=torch.float32, device=self.device)
        return torch.empty((n, 1), dtype=torch.float32, device=self.device).uniform_(lo, hi)

    def _apply_domain_randomization(self, env_ids):
        if not self.cfg.enable_domain_randomization or len(env_ids) == 0 or self._dr_base_wheel_stiffness is None:
            return
        n = len(env_ids)
        ws = self._sample_scale(self.cfg.dr_wheel_stiffness_scale_range, n)
        wd = self._sample_scale(self.cfg.dr_wheel_damping_scale_range, n)
        as_ = self._sample_scale(self.cfg.dr_arm_stiffness_scale_range, n)
        ad = self._sample_scale(self.cfg.dr_arm_damping_scale_range, n)
        self._dr_curr_wheel_stiffness[env_ids] = torch.clamp(self._dr_base_wheel_stiffness[env_ids] * ws, min=0.0)
        self._dr_curr_wheel_damping[env_ids] = torch.clamp(self._dr_base_wheel_damping[env_ids] * wd, min=0.0)
        self._dr_curr_arm_stiffness[env_ids] = torch.clamp(self._dr_base_arm_stiffness[env_ids] * as_, min=0.0)
        self._dr_curr_arm_damping[env_ids] = torch.clamp(self._dr_base_arm_damping[env_ids] * ad, min=0.0)
        wheel_ids = self.wheel_idx.tolist()
        arm_ids = self.arm_idx.tolist()
        self.robot.write_joint_stiffness_to_sim(self._dr_curr_wheel_stiffness, joint_ids=wheel_ids)
        self.robot.write_joint_damping_to_sim(self._dr_curr_wheel_damping, joint_ids=wheel_ids)
        self.robot.write_joint_stiffness_to_sim(self._dr_curr_arm_stiffness, joint_ids=arm_ids)
        self.robot.write_joint_damping_to_sim(self._dr_curr_arm_damping, joint_ids=arm_ids)

    # ── Physics grasp helpers ─────────────────────────────────────────
    @staticmethod
    def _resolve_env_pattern_path(path_pattern, env_id):
        return str(path_pattern).replace("env_.*", f"env_{int(env_id)}")

    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _gripper_body_prim_path(self, env_id):
        return self._resolve_env_pattern_path(self.cfg.gripper_contact_prim_path, env_id)

    def _object_body_prim_path(self, env_id, object_type):
        if self._multi_object:
            return self._resolve_env_pattern_path(f"/World/envs/env_.*/Object_{int(object_type)}", env_id)
        return self._resolve_env_pattern_path(self.cfg.object_prim_path, env_id)

    @staticmethod
    def _quatd_to_quatf(quat_d):
        imag = quat_d.GetImaginary()
        return Gf.Quatf(float(quat_d.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))

    def _get_or_create_grasp_fixed_joint(self, env_id, object_type, gripper_path, object_path):
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
        mass = float(self.object_mass_per_env[env_id].item()) if env_id < self.num_envs else self.cfg.object_mass
        break_force = mass * 9.81 * 10.0
        joint.CreateBreakForceAttr(break_force)
        joint.CreateBreakTorqueAttr(break_force)
        joint.CreateJointEnabledAttr(False)
        self._grasp_fixed_joints[key] = joint
        return joint

    def _attach_grasp_for_envs(self, env_ids):
        """Attach object to gripper via FixedJoint for all given env_ids."""
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
                continue
            joint = self._get_or_create_grasp_fixed_joint(env_id, object_type, gripper_path, object_path)
            if joint is None:
                continue
            try:
                tf_gripper = omni.usd.get_world_transform_matrix(gripper_prim)
                tf_object = omni.usd.get_world_transform_matrix(object_prim)
                local_tf0 = tf_gripper.GetInverse() * tf_object
                local_pos0 = local_tf0.ExtractTranslation()
                local_rot0 = self._quatd_to_quatf(local_tf0.ExtractRotationQuat())
                joint.GetLocalPos0Attr().Set(local_pos0)
                joint.GetLocalRot0Attr().Set(local_rot0)
                joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
                joint.GetJointEnabledAttr().Set(True)
            except Exception:
                pass

    def _disable_grasp_fixed_joint_for_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        max_ot = max(int(self._num_object_types), 1)
        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            for ot in range(max_ot):
                joint = self._grasp_fixed_joints.get((env_id, ot))
                if joint and joint.GetPrim().IsValid():
                    joint.GetJointEnabledAttr().Set(False)

    def _contact_force_per_env(self):
        force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not self._physics_grasp or self.contact_sensor is None:
            return force
        force_matrix = self.contact_sensor.data.force_matrix_w
        if force_matrix is not None:
            mag = torch.norm(force_matrix, dim=-1).reshape(force_matrix.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag
        net = self.contact_sensor.data.net_forces_w
        if net is not None:
            mag = torch.norm(net, dim=-1).reshape(net.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag
        return force

    # ── Action pipeline ──────────────────────────────────────────────
    def _pre_physics_step(self, actions):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        vx = self.actions[:, 6] * self.cfg.max_lin_vel
        vy = self.actions[:, 7] * self.cfg.max_lin_vel
        wz = self.actions[:, 8] * self.cfg.max_ang_vel
        body_cmd = torch.stack([vx, vy, wz], dim=-1)
        wheel_radps = body_cmd @ self.kiwi_M.T / self.wheel_radius
        vel_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        vel_target[:, self.wheel_idx] = wheel_radps
        self.robot.set_joint_velocity_target(vel_target)

        arm_action = self.actions[:, :6]
        if self.cfg.arm_action_to_limits and self._arm_action_limits_override is not None:
            arm_limits = self._arm_action_limits_override
            arm_lo = arm_limits[..., 0]
            arm_hi = arm_limits[..., 1]
            finite = torch.isfinite(arm_lo) & torch.isfinite(arm_hi) & ((arm_hi - arm_lo) > 1e-6)
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

    # ── Observations ─────────────────────────────────────────────────
    def _compute_metrics(self):
        if self._multi_object and len(self.object_rigids) > 0:
            for oi, rigid in enumerate(self.object_rigids):
                mask = self.active_object_idx == oi
                if mask.any():
                    ids = mask.nonzero(as_tuple=False).squeeze(-1)
                    self.object_pos_w[ids] = rigid.data.root_pos_w[ids]
        elif self._physics_grasp and self.object_rigid is not None:
            self.object_pos_w[:] = self.object_rigid.data.root_pos_w

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        home_delta_w = self.home_pos_w - root_pos_w
        home_pos_b = quat_apply_inverse(root_quat_w, home_delta_w)
        home_dist = torch.norm(home_pos_b[:, :2], dim=-1)
        base_disp_w = root_pos_w - self.home_pos_w
        base_disp_b = quat_apply_inverse(root_quat_w, base_disp_w)
        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        contact_force = self._contact_force_per_env()

        # Gripper-to-object relative position
        obj_delta_w = self.object_pos_w - root_pos_w
        obj_pos_b = quat_apply_inverse(root_quat_w, obj_delta_w)

        return {
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "home_pos_b": home_pos_b,
            "home_dist": home_dist,
            "base_disp_b": base_disp_b,
            "arm_pos": arm_pos,
            "contact_force": contact_force,
            "obj_pos_b": obj_pos_b,
        }

    def _get_observations(self):
        metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()
        arm_pos = metrics["arm_pos"]
        # Actor obs 13D: arm(5) + grip(1) + base_disp(3) + home_rel(3) + grip_force(1)
        arm_no_gripper = torch.cat([arm_pos[:, :self.gripper_arm_offset],
                                     arm_pos[:, self.gripper_arm_offset+1:]], dim=-1)[:, :5]
        grip_pos = self.robot.data.joint_pos[:, self.gripper_idx].unsqueeze(-1)
        grip_force = metrics["contact_force"].unsqueeze(-1)

        actor_obs = torch.cat([
            arm_no_gripper,                        # 5
            grip_pos,                               # 1
            metrics["base_disp_b"],                 # 3
            metrics["home_pos_b"],                  # 3
            grip_force,                             # 1
        ], dim=-1)  # 13D

        # Critic obs 20D: actor 13D + obj_dims(3D) + obj_mass(1D) + grip_rel_pos(3D)
        critic_obs = torch.cat([
            actor_obs,                              # 13
            self.object_bbox,                       # 3
            self.object_mass_per_env.unsqueeze(-1), # 1
            metrics["obj_pos_b"],                   # 3
        ], dim=-1)  # 20D

        self._cached_metrics = None
        return {"policy": actor_obs, "critic": critic_obs}

    # ── Rewards ──────────────────────────────────────────────────────
    def _get_rewards(self):
        metrics = self._cached_metrics
        if metrics is None:
            metrics = self._compute_metrics()
        home_dist = metrics["home_dist"]
        contact_force = metrics["contact_force"]

        reward = torch.full((self.num_envs,), self.cfg.rew_time_penalty, device=self.device)

        # Carry reward (distance to home)
        reward += self.cfg.rew_carry * home_dist

        # Hold bonus (object still held)
        reward += self.object_held.float() * self.cfg.rew_hold

        # Drop detection: object fell below home z or no contact force when should be held
        obj_z = self.object_pos_w[:, 2]
        home_z = self.home_pos_w[:, 2]
        obj_fell = obj_z < (home_z - 0.05)
        newly_dropped = self.object_held & obj_fell
        if newly_dropped.any():
            self.object_dropped[newly_dropped] = True
            self.object_held[newly_dropped] = False
            reward[newly_dropped] += self.cfg.rew_drop

        # Place success: near home and still holding
        place_ok = self.object_held & (home_dist < self.cfg.place_thresh)
        self.task_success = place_ok
        reward[place_ok] += self.cfg.rew_place_success

        self.prev_home_dist[:] = home_dist
        return reward

    # ── Dones ────────────────────────────────────────────────────────
    def _get_dones(self):
        metrics = self._compute_metrics()
        self._cached_metrics = metrics
        root_pos = metrics["root_pos_w"]
        out_of_bounds = torch.norm(root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1) > self.cfg.max_dist_from_origin
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)
        terminated = out_of_bounds | fell | self.object_dropped
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        truncated = self.task_success | time_out
        self.extras["task_success_rate"] = self.task_success.float().mean()
        return terminated, truncated

    # ── Reset ────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        num = len(env_ids)
        if num == 0:
            return

        self._disable_grasp_fixed_joint_for_envs(env_ids)

        use_handoff = (self._handoff_buffer is not None and self._handoff_buffer.is_ready)

        if use_handoff:
            # Sample from handoff buffer
            entries = self._handoff_buffer.sample(num)
            # Restore robot state
            root_state = entries["root_state"].to(self.device)
            self.robot.write_root_state_to_sim(root_state, env_ids)
            joint_pos = entries["joint_pos"].to(self.device)
            joint_vel = entries["joint_vel"].to(self.device)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            # Restore object
            self.object_pos_w[env_ids] = entries["object_pos_w"].to(self.device)
            self.home_pos_w[env_ids] = entries["home_pos_w"].to(self.device)
            if "active_object_idx" in entries:
                self.active_object_idx[env_ids] = entries["active_object_idx"].to(self.device).squeeze(-1)
            if "object_bbox" in entries:
                self.object_bbox[env_ids] = entries["object_bbox"].to(self.device)
            if "object_mass" in entries:
                self.object_mass_per_env[env_ids] = entries["object_mass"].to(self.device).squeeze(-1)
            # Write object pose to sim
            if self._multi_object and len(self.object_rigids) > 0:
                for rigid in self.object_rigids:
                    hide_pose = rigid.data.default_root_state[env_ids, :7].clone()
                    hide_pose[:, 2] = -10.0
                    rigid.write_root_pose_to_sim(hide_pose, env_ids=env_ids)
                    rigid.write_root_velocity_to_sim(torch.zeros((num, 6), device=self.device), env_ids=env_ids)
                for oi, rigid in enumerate(self.object_rigids):
                    mask = self.active_object_idx[env_ids] == oi
                    if not mask.any():
                        continue
                    sel_ids = env_ids[mask]
                    pose = rigid.data.default_root_state[sel_ids, :7].clone()
                    pose[:, :3] = self.object_pos_w[sel_ids]
                    if "object_quat_w" in entries:
                        obj_quats = entries["object_quat_w"].to(self.device)
                        pose[:, 3:7] = obj_quats[mask]
                    rigid.write_root_pose_to_sim(pose, env_ids=sel_ids)
            elif self._physics_grasp and self.object_rigid is not None:
                obj_pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
                obj_pose[:, :3] = self.object_pos_w[env_ids]
                if "object_quat_w" in entries:
                    obj_pose[:, 3:7] = entries["object_quat_w"].to(self.device)
                self.object_rigid.write_root_pose_to_sim(obj_pose, env_ids=env_ids)
        else:
            # Random init with object pre-grasped
            default_root = self.robot.data.default_root_state[env_ids].clone()
            root_xy_std = float(self.cfg.dr_root_xy_noise_std) if self.cfg.enable_domain_randomization else 0.1
            default_root[:, 0:2] += torch.randn(num, 2, device=self.device) * root_xy_std
            random_yaw = torch.rand(num, device=self.device) * 2 * math.pi - math.pi
            half_yaw = random_yaw * 0.5
            default_root[:, 3] = torch.cos(half_yaw)
            default_root[:, 6] = torch.sin(half_yaw)
            self.robot.write_root_state_to_sim(default_root, env_ids)
            joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            self.home_pos_w[env_ids] = default_root[:, :3]
            # Place object near gripper (already grasped)
            self.object_pos_w[env_ids, :2] = default_root[:, :2]
            self.object_pos_w[env_ids, 2] = default_root[:, 2] + float(self.cfg.grasp_attach_height)

            if self._multi_object and len(self.object_rigids) > 0:
                chosen = torch.randint(0, self._num_object_types, (num,), device=self.device)
                self.active_object_idx[env_ids] = chosen
                self.object_bbox[env_ids] = self._catalog_bbox[chosen]
                self.object_mass_per_env[env_ids] = self._catalog_mass[chosen]
                for rigid in self.object_rigids:
                    hide_pose = rigid.data.default_root_state[env_ids, :7].clone()
                    hide_pose[:, 2] = -10.0
                    rigid.write_root_pose_to_sim(hide_pose, env_ids=env_ids)
                for oi, rigid in enumerate(self.object_rigids):
                    mask = chosen == oi
                    if not mask.any():
                        continue
                    sel_ids = env_ids[mask]
                    pose = rigid.data.default_root_state[sel_ids, :7].clone()
                    pose[:, :3] = self.object_pos_w[sel_ids]
                    rigid.write_root_pose_to_sim(pose, env_ids=sel_ids)
            elif self._physics_grasp and self.object_rigid is not None:
                self.object_mass_per_env[env_ids] = float(self.cfg.object_mass)
                obj_pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
                obj_pose[:, :3] = self.object_pos_w[env_ids]
                self.object_rigid.write_root_pose_to_sim(obj_pose, env_ids=env_ids)

            # Sample home position away from current position
            angle = torch.rand(num, device=self.device) * 2 * math.pi
            dist = torch.rand(num, device=self.device) * (self.cfg.home_dist_max - self.cfg.home_dist_min) + self.cfg.home_dist_min
            self.home_pos_w[env_ids, 0] = default_root[:, 0] + dist * torch.cos(angle)
            self.home_pos_w[env_ids, 1] = default_root[:, 1] + dist * torch.sin(angle)

        # Attach object via FixedJoint (it's already grasped at start of Skill-3)
        self._attach_grasp_for_envs(env_ids)

        self._apply_domain_randomization(env_ids)

        # Reset task state
        self.object_held[env_ids] = True
        self.object_dropped[env_ids] = False
        self.task_success[env_ids] = False
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_home_dist[env_ids] = 0.0
