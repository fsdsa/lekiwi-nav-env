"""
LeKiwi Skill-2 Environment v7 — Exploit-aware wrapper for Residual PPO.

Extends Skill2Env with:
  1. Object quaternion tracking → toppling detection
  2. EE world-position tracking → floor-press detection
  3. Clean grasp state machine (establish N steps, confirmed drop)
  4. Terminations: sustained ground-pressing, persistent object topple
  5. Rich extras dict for external reward computation
  6. Clean success = lifted + near lifted pose + base still
"""
from __future__ import annotations

import math
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from lekiwi_skill2_env import Skill2EnvCfg, Skill2Env, EE_LOCAL_OFFSET


@configclass
class Skill2V7EnvCfg(Skill2EnvCfg):
    """Configuration additions for v7 exploit-aware environment."""

    # ── Grasp state machine ──
    grasp_establish_steps: int = 4     # consecutive live-hold frames to confirm grasp
    drop_confirm_steps: int = 3        # consecutive non-hold frames after grasp → drop

    # ── Ground-press termination ──
    ground_press_force_thresh: float = 5.0   # N — gripper+wrist ground contact force
    ground_press_steps: int = 5              # consecutive frames → terminate
    ee_floor_z_thresh: float = 0.020         # m — EE below this = pressing floor

    # ── Object topple termination ──
    # object upright → object_up_z ≈ 1.0; fallen → ≈ 0.0
    # Also use height: upright center ~0.033, fallen ~0.020
    object_fallen_height: float = 0.025      # m — object center below this (not grasped) = fallen
    fallen_terminate_steps: int = 15         # consecutive fallen frames → terminate

    # ── Clean success ──
    clean_lift_height: float = 0.10          # m — object must be this high
    clean_pose_err_thresh: float = 0.25      # rad — joint err from lifted pose
    clean_base_speed_thresh: float = 0.10    # m/s — base must be ~still
    clean_hold_steps: int = 6               # consecutive clean frames → success

    # Lifted pose target (from teleop demos)
    lift_pose_targets: tuple = (-0.0457, -0.1930, 0.2771, -0.9146, 0.0201)


class Skill2V7Env(Skill2Env):
    """Exploit-aware Skill-2 environment for residual PPO."""

    cfg: Skill2V7EnvCfg

    def __init__(self, cfg: Skill2V7EnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        dev = self.device

        # Object orientation
        self.object_quat_w = torch.zeros(N, 4, device=dev)
        self.object_quat_w[:, 0] = 1.0

        # State machine buffers
        self._grasp_hold_counter = torch.zeros(N, dtype=torch.long, device=dev)
        self._drop_counter = torch.zeros(N, dtype=torch.long, device=dev)
        self._ground_press_counter = torch.zeros(N, dtype=torch.long, device=dev)
        self._fallen_counter = torch.zeros(N, dtype=torch.long, device=dev)
        self._success_counter = torch.zeros(N, dtype=torch.long, device=dev)

        # Flags exposed via extras
        self.live_hold = torch.zeros(N, dtype=torch.bool, device=dev)
        self.confirmed_drop = torch.zeros(N, dtype=torch.bool, device=dev)
        self.object_fallen = torch.zeros(N, dtype=torch.bool, device=dev)
        self.bad_ground_press = torch.zeros(N, dtype=torch.bool, device=dev)
        self.clean_success = torch.zeros(N, dtype=torch.bool, device=dev)
        self.gripper_was_open = torch.zeros(N, dtype=torch.bool, device=dev)

        # Lifted pose target tensor
        self._lift_pose = torch.tensor(
            list(self.cfg.lift_pose_targets), dtype=torch.float32, device=dev,
        ).unsqueeze(0)  # (1, 5)

        self._world_z = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=dev)

    # ─────────────────────────────────────────────────────────────────
    # Override _compute_metrics to also track object quat and EE pos
    # ─────────────────────────────────────────────────────────────────
    def _compute_metrics(self):
        # Update object_pos_w and object_quat_w
        if self._multi_object and len(self.object_rigids) > 0:
            for oi, rigid in enumerate(self.object_rigids):
                mask = self.active_object_idx == oi
                if not mask.any():
                    continue
                ids = mask.nonzero(as_tuple=False).squeeze(-1)
                self.object_pos_w[ids] = rigid.data.root_pos_w[ids]
                self.object_quat_w[ids] = rigid.data.root_quat_w[ids]
        elif self._physics_grasp and self.object_rigid is not None:
            self.object_pos_w[:] = self.object_rigid.data.root_pos_w
            self.object_quat_w[:] = self.object_rigid.data.root_quat_w

        # bbox center correction
        bbox_offset = getattr(self, '_object_bbox_center_local', None)
        if bbox_offset is not None and bbox_offset.any():
            if self._multi_object and len(self.object_rigids) > 0:
                for oi, rigid in enumerate(self.object_rigids):
                    mask = self.active_object_idx == oi
                    if not mask.any():
                        continue
                    ids = mask.nonzero(as_tuple=False).squeeze(-1)
                    self.object_pos_w[ids] += quat_apply(
                        rigid.data.root_quat_w[ids], bbox_offset.expand(ids.shape[0], -1))
            elif self._physics_grasp and self.object_rigid is not None:
                self.object_pos_w += quat_apply(
                    self.object_quat_w, bbox_offset.unsqueeze(0).expand_as(self.object_pos_w))

        if self._dest_object_rigid is not None:
            self.dest_object_pos_w[:] = self._dest_object_rigid.data.root_pos_w

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w

        object_delta_w = self.object_pos_w - root_pos_w
        object_pos_b = quat_apply_inverse(root_quat_w, object_delta_w)
        object_dist = torch.norm(object_pos_b[:, :2], dim=-1)
        heading_object = torch.atan2(object_pos_b[:, 0], object_pos_b[:, 1])

        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        lin_speed = torch.norm(lin_vel_b[:, :2], dim=-1)

        object_dir_b = object_pos_b[:, :2] / (object_dist.unsqueeze(-1) + 1e-6)
        vel_toward_object = (lin_vel_b[:, :2] * object_dir_b).sum(dim=-1)

        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        arm_vel = self.robot.data.joint_vel[:, self.arm_idx]

        # EE world position
        if self._fixed_jaw_body_idx >= 0:
            wrist_pos = self.robot.data.body_pos_w[:, self._fixed_jaw_body_idx, :]
            wrist_quat = self.robot.data.body_quat_w[:, self._fixed_jaw_body_idx, :]
            ee_pos_w = wrist_pos + quat_apply(wrist_quat, self._ee_local_offset.expand_as(wrist_pos))
        else:
            ee_pos_w = root_pos_w.clone()

        ee_to_obj = torch.norm(ee_pos_w - self.object_pos_w, dim=-1)

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
            "ee_pos_w": ee_pos_w,
            "ee_to_obj": ee_to_obj,
        }

    # ─────────────────────────────────────────────────────────────────
    # Override _update_grasp_state with establish/drop counters
    # ─────────────────────────────────────────────────────────────────
    def _update_grasp_state(self, metrics):
        self.just_grasped[:] = False
        self.just_dropped[:] = False
        self.confirmed_drop[:] = False
        self.task_success[:] = False
        self.clean_success[:] = False

        gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]

        # Track if gripper was open at any point (prevents wedge exploit)
        self.gripper_was_open |= (gripper_pos > 0.85)

        # Live hold = gripper closed + EE near object + (optionally) contact
        gripper_closed = gripper_pos < float(self.cfg.grasp_gripper_threshold)
        ee_near = metrics["ee_to_obj"] < float(self.cfg.grasp_ee_max_dist)

        if self._physics_grasp and self.contact_sensor is not None:
            cf = self._contact_force_per_env()
            has_contact = cf > float(self.cfg.grasp_contact_threshold)
            if bool(self.cfg.grasp_require_contact):
                live = gripper_closed & has_contact & ee_near
            else:
                live = gripper_closed & ee_near
        else:
            live = gripper_closed & ee_near

        self.live_hold[:] = live

        # Establish grasp: N consecutive live-hold steps + gripper was open
        establishing = (~self.object_grasped) & live & self.gripper_was_open
        self._grasp_hold_counter[establishing] += 1
        self._grasp_hold_counter[~establishing & (~self.object_grasped)] = 0

        newly_grasped = (
            (~self.object_grasped)
            & (self._grasp_hold_counter >= int(self.cfg.grasp_establish_steps))
        )
        if newly_grasped.any():
            self.object_grasped[newly_grasped] = True
            self.just_grasped[newly_grasped] = True
            self._drop_counter[newly_grasped] = 0

        # Confirmed drop: object was grasped but live_hold lost for N steps
        dropping = self.object_grasped & (~live)
        self._drop_counter[dropping] += 1
        self._drop_counter[~dropping & self.object_grasped] = 0

        dropped = self.object_grasped & (self._drop_counter >= int(self.cfg.drop_confirm_steps))
        if dropped.any():
            self.object_grasped[dropped] = False
            self.just_dropped[dropped] = True
            self.confirmed_drop[dropped] = True
            self._grasp_hold_counter[dropped] = 0
            self._success_counter[dropped] = 0

        # Clean success: grasped + lifted + near pose + base still
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        obj_height = self.object_pos_w[:, 2] - env_z
        arm5 = metrics["arm_pos"][:, :5]
        pose_err = torch.norm(arm5 - self._lift_pose.expand(self.num_envs, -1), dim=-1)

        clean = (
            self.object_grasped & live
            & (obj_height > float(self.cfg.clean_lift_height))
            & (pose_err < float(self.cfg.clean_pose_err_thresh))
            & (metrics["lin_speed"] < float(self.cfg.clean_base_speed_thresh))
        )
        self._success_counter[clean] += 1
        self._success_counter[~clean] = 0
        self.clean_success[:] = self._success_counter >= int(self.cfg.clean_hold_steps)
        self.task_success[:] = self.clean_success

        # Also set task_success for basic lift (env reward uses this)
        basic_lift = self.object_grasped & (obj_height > float(self.cfg.grasp_success_height))
        self.task_success |= basic_lift

    # ─────────────────────────────────────────────────────────────────
    # Override _get_dones: add exploit terminations + rich extras
    # ─────────────────────────────────────────────────────────────────
    def _get_dones(self):
        metrics = self._compute_metrics()
        self._update_grasp_state(metrics)
        self._cached_metrics = metrics

        root_pos = metrics["root_pos_w"]
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0

        # Standard terminations
        oob = torch.norm(root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1) > self.cfg.max_dist_from_origin
        fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)

        # Ground press termination
        gcf = self._ground_contact_force_per_env()
        ee_z = metrics["ee_pos_w"][:, 2] - env_z
        pressing = (gcf > float(self.cfg.ground_press_force_thresh)) | (ee_z < float(self.cfg.ee_floor_z_thresh))
        self._ground_press_counter[pressing] += 1
        self._ground_press_counter[~pressing] = 0
        self.bad_ground_press[:] = self._ground_press_counter >= int(self.cfg.ground_press_steps)

        # Object fallen termination (only when not grasped)
        obj_h = self.object_pos_w[:, 2] - env_z
        fallen_now = (obj_h < float(self.cfg.object_fallen_height)) & (~self.object_grasped)
        self.object_fallen[:] = fallen_now
        self._fallen_counter[fallen_now] += 1
        self._fallen_counter[~fallen_now] = 0
        persistent_fallen = self._fallen_counter >= int(self.cfg.fallen_terminate_steps)

        terminated = oob | fell | self.bad_ground_press | persistent_fallen
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        truncated = self.clean_success | time_out

        # ── Rich extras (all pre-auto-reset) ──
        gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
        arm5 = metrics["arm_pos"][:, :5]
        pose_err = torch.norm(arm5 - self._lift_pose.expand(self.num_envs, -1), dim=-1)

        self.extras.update({
            "task_success_rate": self.task_success.float().mean(),
            "task_success_mask": self.task_success.clone(),
            "clean_success_mask": self.clean_success.clone(),
            "just_grasped_mask": self.just_grasped.clone(),
            "just_dropped_mask": self.just_dropped.clone(),
            "confirmed_drop_mask": self.confirmed_drop.clone(),
            "object_grasped_mask": self.object_grasped.clone(),
            "live_hold_mask": self.live_hold.clone(),
            "gripper_was_open_mask": self.gripper_was_open.clone(),
            "bad_ground_press_mask": self.bad_ground_press.clone(),
            "object_fallen_mask": self.object_fallen.clone(),
            "persistent_fallen_mask": persistent_fallen.clone(),
            "object_height_mask": obj_h.clone(),
            "object_height": obj_h.clone(),
            "gripper_pos_raw": gripper_pos.clone(),
            "ee_z": ee_z.clone(),
            "ee_to_obj_3d": metrics["ee_to_obj"].clone(),
            "object_dist": metrics["object_dist"].clone(),
            "heading_object": metrics["heading_object"].clone(),
            "base_speed": metrics["lin_speed"].clone(),
            "lift_pose_err": pose_err.clone(),
            "contact_force_raw": self._contact_force_per_env().clone(),
            "ground_contact_force_raw": gcf.clone(),
            "ground_press_counter": self._ground_press_counter.clone().float(),
            "fallen_counter": self._fallen_counter.clone().float(),
        })

        return terminated, truncated

    # ─────────────────────────────────────────────────────────────────
    # Override _reset_idx to clear new buffers
    # ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        self.object_quat_w[env_ids] = 0.0
        self.object_quat_w[env_ids, 0] = 1.0

        self._grasp_hold_counter[env_ids] = 0
        self._drop_counter[env_ids] = 0
        self._ground_press_counter[env_ids] = 0
        self._fallen_counter[env_ids] = 0
        self._success_counter[env_ids] = 0

        self.live_hold[env_ids] = False
        self.confirmed_drop[env_ids] = False
        self.object_fallen[env_ids] = False
        self.bad_ground_press[env_ids] = False
        self.clean_success[env_ids] = False
        self.gripper_was_open[env_ids] = False