
from __future__ import annotations

import math
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from lekiwi_skill2_env import Skill2EnvCfg, Skill2Env


@configclass
class Skill2ResiPEnvCfg(Skill2EnvCfg):
    """Skill-2 env specialized for residual PPO on top of BC.

    Goal:
      - keep BC's coarse sequence
      - remove sticky-grasp bug
      - terminate bad exploits early (ground press, toppled-object pressing, drop)
      - define success as clean lift + return to lifted pose
    """

    # clean grasp / drop state machine
    grasp_establish_steps: int = 3
    drop_steps: int = 2
    allow_regrasp: bool = False
    drop_terminate: bool = True

    # exploit blockers
    ground_press_force_thresh: float = 6.0
    ground_press_steps: int = 4
    fallen_cos_z_thresh: float = 0.45
    fallen_steps: int = 10

    # clean-success definition
    lift_success_height: float = 0.12
    lift_pose_joint_targets: tuple[float, float, float, float, float] = (
        -0.0457, -0.1930, 0.2771, -0.9146, 0.0201
    )
    lift_pose_joint_err_thresh: float = 0.22
    lift_success_hold_steps: int = 8
    lift_success_base_speed_thresh: float = 0.08


class Skill2ResiPEnv(Skill2Env):
    """Drop-aware / exploit-aware wrapper over the user's Skill-2 env."""

    cfg: Skill2ResiPEnvCfg

    def __init__(self, cfg: Skill2ResiPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._physics_grasp = bool(str(self.cfg.object_usd).strip()) or bool(str(self.cfg.multi_object_json).strip())

        self.object_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_quat_w[:, 0] = 1.0

        self.live_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.clean_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bad_ground_press = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._grasp_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._drop_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ground_press_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fallen_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.lift_pose_joint_err = torch.full((self.num_envs,), 9.0, dtype=torch.float32, device=self.device)

        if len(self.cfg.lift_pose_joint_targets) != 5:
            raise ValueError("cfg.lift_pose_joint_targets must have length 5.")
        self._lift_pose_joint_targets = torch.tensor(
            list(self.cfg.lift_pose_joint_targets),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device)

    # ---------------------------------------------------------------------
    # contact utilities
    # ---------------------------------------------------------------------
    def _contact_force_per_env(self) -> torch.Tensor:
        force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not self._physics_grasp or self.contact_sensor is None:
            return force

        force_matrix = self.contact_sensor.data.force_matrix_w
        if force_matrix is not None:
            mag = torch.norm(force_matrix, dim=-1)
            mag = mag.reshape(mag.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag
            if mag.numel() % self.num_envs == 0:
                return mag.reshape(self.num_envs, -1).sum(dim=-1)

        net = self.contact_sensor.data.net_forces_w
        if net is not None:
            mag = torch.norm(net, dim=-1)
            if mag.ndim > 1:
                mag = mag.reshape(mag.shape[0], -1).sum(dim=-1)
            if mag.shape[0] == self.num_envs:
                return mag

        return force

    # ---------------------------------------------------------------------
    # metrics
    # ---------------------------------------------------------------------
    def _compute_metrics(self):
        # active object's world pose
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
        else:
            self.object_quat_w[:] = 0.0
            self.object_quat_w[:, 0] = 1.0

        # origin -> bbox center correction
        bbox_offset = getattr(self, "_object_bbox_center_local", None)
        if bbox_offset is not None and bbox_offset.numel() == 3 and bool(torch.any(bbox_offset != 0.0)):
            if self._multi_object and len(self.object_rigids) > 0:
                for oi, rigid in enumerate(self.object_rigids):
                    mask = self.active_object_idx == oi
                    if not mask.any():
                        continue
                    ids = mask.nonzero(as_tuple=False).squeeze(-1)
                    obj_quat = rigid.data.root_quat_w[ids]
                    self.object_pos_w[ids] += quat_apply(obj_quat, bbox_offset.expand(ids.shape[0], -1))
            elif self._physics_grasp and self.object_rigid is not None:
                self.object_pos_w += quat_apply(
                    self.object_quat_w,
                    bbox_offset.unsqueeze(0).expand_as(self.object_pos_w),
                )

        if self._dest_object_rigid is not None:
            self.dest_object_pos_w[:] = self._dest_object_rigid.data.root_pos_w

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        object_delta_w = self.object_pos_w - root_pos_w
        object_pos_b = quat_apply_inverse(root_quat_w, object_delta_w)
        object_dist = torch.norm(object_pos_b[:, :2], dim=-1)
        heading_object = torch.atan2(object_pos_b[:, 0], object_pos_b[:, 1])

        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        lin_speed = torch.norm(lin_vel_b[:, :2], dim=-1)

        object_dir_b = object_pos_b[:, :2] / (object_dist.unsqueeze(-1) + 1e-6)
        vel_toward_object = (lin_vel_b[:, :2] * object_dir_b).sum(dim=-1)

        arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
        arm_vel = self.robot.data.joint_vel[:, self.arm_idx]

        if self._fixed_jaw_body_idx >= 0:
            wrist_pos = self.robot.data.body_pos_w[:, self._fixed_jaw_body_idx, :]
            wrist_quat = self.robot.data.body_quat_w[:, self._fixed_jaw_body_idx, :]
            ee_pos_w = wrist_pos + quat_apply(wrist_quat, self._ee_local_offset.expand_as(wrist_pos))
        else:
            ee_pos_w = root_pos_w

        ee_to_obj_dist = torch.norm(ee_pos_w - self.object_pos_w, dim=-1)

        object_up = quat_apply(self.object_quat_w, self._world_up.expand(self.num_envs, -1))
        object_up_z = object_up[:, 2]

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
            "ee_to_obj_dist": ee_to_obj_dist,
            "object_up_z": object_up_z,
        }

    # ---------------------------------------------------------------------
    # grasp / drop / success state machine
    # ---------------------------------------------------------------------
    def _update_grasp_state(self, metrics):
        self.just_grasped[:] = False
        self.just_dropped[:] = False
        self.clean_success[:] = False
        self.task_success[:] = False
        self.live_hold[:] = False

        if not self._physics_grasp or self.contact_sensor is None:
            live_hold = (
                (metrics["object_dist"] < float(self.cfg.grasp_thresh))
                & (metrics["lin_speed"] < 0.35)
            )
        else:
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
            gripper_closed = gripper_pos < float(self.cfg.grasp_gripper_threshold)
            contact_force = self._contact_force_per_env()
            has_contact = contact_force > float(self.cfg.grasp_contact_threshold)
            between_jaws = metrics["ee_to_obj_dist"] < float(self.cfg.grasp_ee_max_dist)

            if bool(self.cfg.grasp_require_contact):
                live_hold = gripper_closed & has_contact & between_jaws
            else:
                live_hold = gripper_closed & between_jaws

        self.live_hold[:] = live_hold

        # establish grasp only after N consecutive live-hold steps
        can_establish = (~self.object_grasped) & live_hold
        self._grasp_counter[can_establish] += 1
        self._grasp_counter[~can_establish] = 0

        newly_grasped = (~self.object_grasped) & (self._grasp_counter >= int(self.cfg.grasp_establish_steps))
        if newly_grasped.any():
            self.object_grasped[newly_grasped] = True
            self.just_grasped[newly_grasped] = True
            self._drop_counter[newly_grasped] = 0

        # confirmed drop after grasp
        dropping = self.object_grasped & (~live_hold)
        self._drop_counter[dropping] += 1
        self._drop_counter[~dropping] = 0

        confirmed_drop = self.object_grasped & (self._drop_counter >= int(self.cfg.drop_steps))
        if confirmed_drop.any():
            self.object_grasped[confirmed_drop] = False
            self.just_dropped[confirmed_drop] = True
            self._grasp_counter[confirmed_drop] = 0
            self._success_hold_counter[confirmed_drop] = 0

        # clean success = live hold + lifted + back to lifted pose + base almost still
        arm_pos5 = metrics["arm_pos"][:, :5]
        self.lift_pose_joint_err[:] = torch.norm(
            arm_pos5 - self._lift_pose_joint_targets.expand(self.num_envs, -1),
            dim=-1,
        )
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        object_height = self.object_pos_w[:, 2] - env_z
        clean_hold = (
            self.object_grasped
            & live_hold
            & (object_height > float(self.cfg.lift_success_height))
            & (self.lift_pose_joint_err < float(self.cfg.lift_pose_joint_err_thresh))
            & (metrics["lin_speed"] < float(self.cfg.lift_success_base_speed_thresh))
        )
        self._success_hold_counter[clean_hold] += 1
        self._success_hold_counter[~clean_hold] = 0
        self.clean_success[:] = self._success_hold_counter >= int(self.cfg.lift_success_hold_steps)
        self.task_success[:] = self.clean_success

        # preserve original timeout bookkeeping
        if int(self.cfg.grasp_timeout_steps) > 0:
            near_object = metrics["object_dist"] < float(self.cfg.approach_thresh)
            in_grasp_zone = near_object & (~self.object_grasped)
            first_entry = in_grasp_zone & (self.grasp_entry_step == 0)
            self.grasp_entry_step[first_entry] = self.episode_length_buf[first_entry]
            grasp_elapsed = self.episode_length_buf - self.grasp_entry_step
            timed_out = (
                in_grasp_zone
                & (self.grasp_entry_step > 0)
                & (grasp_elapsed > int(self.cfg.grasp_timeout_steps))
            )
            self.grasp_entry_step[timed_out] = 0

    # ---------------------------------------------------------------------
    # dones + extras
    # ---------------------------------------------------------------------
    def _get_dones(self):
        metrics = self._compute_metrics()
        self._update_grasp_state(metrics)
        self._cached_metrics = metrics

        root_pos = metrics["root_pos_w"]
        out_of_bounds = torch.norm(root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1) > float(self.cfg.max_dist_from_origin)
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        fell_robot = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)

        ground_force = self._ground_contact_force_per_env()
        pressing_ground = ground_force > float(self.cfg.ground_press_force_thresh)
        self._ground_press_counter[pressing_ground] += 1
        self._ground_press_counter[~pressing_ground] = 0
        self.bad_ground_press[:] = self._ground_press_counter >= int(self.cfg.ground_press_steps)

        fallen_now = (metrics["object_up_z"] < float(self.cfg.fallen_cos_z_thresh)) & (~self.object_grasped)
        self.object_fallen[:] = fallen_now
        self._fallen_counter[fallen_now] += 1
        self._fallen_counter[~fallen_now] = 0
        persistent_fallen = self._fallen_counter >= int(self.cfg.fallen_steps)

        terminated = out_of_bounds | fell_robot | self.bad_ground_press | persistent_fallen
        if bool(self.cfg.drop_terminate):
            terminated = terminated | self.just_dropped

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        truncated = self.task_success | time_out

        # extras consumed by residual RL
        self.extras["task_success_rate"] = self.task_success.float().mean()
        self.extras["task_success_mask"] = self.task_success.clone()
        self.extras["clean_success_mask"] = self.clean_success.clone()
        self.extras["just_grasped_mask"] = self.just_grasped.clone()
        self.extras["just_dropped_mask"] = self.just_dropped.clone()
        self.extras["object_grasped_mask"] = self.object_grasped.clone()
        self.extras["live_hold_mask"] = self.live_hold.clone()
        self.extras["bad_ground_press_mask"] = self.bad_ground_press.clone()
        self.extras["persistent_fallen_mask"] = persistent_fallen.clone()

        self.extras["object_height_mask"] = (self.object_pos_w[:, 2] - env_z).clone()
        self.extras["object_height"] = (self.object_pos_w[:, 2] - env_z).clone()
        self.extras["object_up_z"] = metrics["object_up_z"].clone()
        self.extras["object_dist"] = metrics["object_dist"].clone()
        self.extras["heading_object"] = metrics["heading_object"].clone()
        self.extras["base_speed"] = metrics["lin_speed"].clone()
        self.extras["ee_to_object_3d"] = metrics["ee_to_obj_dist"].clone()
        self.extras["lift_pose_joint_err"] = self.lift_pose_joint_err.clone()
        self.extras["gripper_pos_raw"] = self.robot.data.joint_pos[:, self.gripper_idx].clone()

        self.extras["contact_force_raw"] = self._contact_force_per_env().clone()
        self.extras["ground_contact_force_raw"] = ground_force.clone()

        return terminated, truncated

    # ---------------------------------------------------------------------
    # reset
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        self.object_quat_w[env_ids] = 0.0
        self.object_quat_w[env_ids, 0] = 1.0

        self.live_hold[env_ids] = False
        self.clean_success[env_ids] = False
        self.object_fallen[env_ids] = False
        self.bad_ground_press[env_ids] = False
        self.lift_pose_joint_err[env_ids] = 9.0

        self._grasp_counter[env_ids] = 0
        self._drop_counter[env_ids] = 0
        self._ground_press_counter[env_ids] = 0
        self._fallen_counter[env_ids] = 0
        self._success_hold_counter[env_ids] = 0