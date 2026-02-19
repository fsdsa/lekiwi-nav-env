"""
LeKiwi Skill-3 — CarryAndPlace Isaac Lab DirectRLEnv.

3-Skill 파이프라인의 세 번째 스킬: 잡은 물체를 들고 home까지 운반 후 놓기(Place).
Handoff Buffer에서 초기 상태(물체가 이미 잡힌 상태)를 로드하여 시작.

Observation (Actor 29D):
  [0:5]   arm joint pos (5)
  [5:6]   gripper pos (1)
  [6:9]   base body velocity (vx, vy, wz) (3)
  [9:12]  base linear vel body (3)
  [12:15] base angular vel body (3)
  [15:21] arm+grip joint vel (6)
  [21:24] home relative pos body (3)
  [24:25] grip force (1)
  [25:28] object bbox normalized (3)
  [28:29] object category normalized (1)

Observation (Critic 36D, AAC):
  Actor 29D + object_bbox(3) + mass(1) + gripper_rel_pos(3)

Action (9D — lekiwi_v6 순서):
  [0:5]   arm joint position target
  [5]     gripper position target
  [6:8]   base linear velocity (vx, vy)
  [8]     base angular velocity (wz)
"""
from __future__ import annotations

import math
import pickle
from typing import Dict

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg


@configclass
class Skill3EnvCfg(Skill2EnvCfg):
    """CarryAndPlace 환경 설정."""

    observation_space: int = 29
    state_space: int = 36

    # Task
    return_thresh: float = 0.30
    place_dist_thresh: float = 0.05

    # Handoff
    handoff_buffer_path: str = ""

    # Reward (CarryAndPlace 전용)
    rew_carry_progress_weight: float = 3.0
    rew_carry_heading_weight: float = 0.2
    rew_hold_bonus: float = 0.1
    rew_place_success_bonus: float = 20.0
    rew_drop_penalty: float = -10.0

    # Curriculum 제거 (Skill-3는 불필요)
    curriculum_success_threshold: float = 1.0


class Skill3Env(Skill2Env):
    """LeKiwi Skill-3: CarryAndPlace RL 환경."""

    cfg: Skill3EnvCfg

    def __init__(self, cfg: Skill3EnvCfg, render_mode: str | None = None, **kwargs):
        # Handoff buffer 로드
        self.handoff_buffer = None
        if cfg.handoff_buffer_path:
            import os
            buf_path = os.path.expanduser(cfg.handoff_buffer_path)
            if os.path.isfile(buf_path):
                with open(buf_path, "rb") as f:
                    self.handoff_buffer = pickle.load(f)
                print(f"  [Skill3Env] Loaded handoff buffer: {len(self.handoff_buffer)} entries from {buf_path}")
            else:
                print(f"  [WARN] Handoff buffer not found: {buf_path}")

        super().__init__(cfg, render_mode, **kwargs)

        # Skill-3 추가 버퍼
        self.prev_home_dist = torch.zeros(self.num_envs, device=self.device)
        # Handoff buffer에서 읽은 object orientation (identity default)
        self._handoff_object_ori = torch.zeros(self.num_envs, 4, device=self.device)
        self._handoff_object_ori[:, 0] = 1.0  # qw=1 identity

        print(f"  [Skill3Env] obs={self.cfg.observation_space} act={self.cfg.action_space} critic={self.cfg.state_space}")

    # ═══════════════════════════════════════════════════════════════════
    #  Metrics — home 관련 추가
    # ═══════════════════════════════════════════════════════════════════

    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        metrics = super()._compute_metrics()

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        home_delta_w = self.home_pos_w - root_pos_w
        home_pos_b = quat_apply_inverse(root_quat_w, home_delta_w)
        home_dist = torch.norm(home_pos_b[:, :2], dim=-1)
        heading_home = torch.atan2(home_pos_b[:, 1], home_pos_b[:, 0])

        metrics["home_pos_b"] = home_pos_b
        metrics["home_dist"] = home_dist
        metrics["heading_home"] = heading_home
        return metrics

    # ═══════════════════════════════════════════════════════════════════
    #  Observations — 29D Actor
    # ═══════════════════════════════════════════════════════════════════

    def _get_observations(self) -> dict:
        metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()
        base_body_vel = self._read_base_body_vel()  # (N, 3) — vx, vy, wz

        arm_pos = metrics["arm_pos"]
        arm_vel = metrics["arm_vel"]
        lin_vel = metrics["lin_vel_b"]
        ang_vel = metrics["ang_vel_b"]

        # home_rel: body-frame 상대 벡터 3D
        home_delta_w = self.home_pos_w - self.robot.data.root_pos_w
        home_rel = quat_apply_inverse(self.robot.data.root_quat_w, home_delta_w)

        # grip_force: 스칼라 1D
        contact_force = self._contact_force_per_env()
        grip_force = contact_force.unsqueeze(-1)

        # BBox / Category
        bbox_norm = self.object_bbox / float(self._bbox_norm_scale)
        cat_denom = max(int(self.cfg.num_object_categories) - 1, 1)
        cat_norm = (self.object_category_id / float(cat_denom)).unsqueeze(-1)

        actor_obs = torch.cat([
            arm_pos[:, :5],             # [0:5]   arm 5D
            arm_pos[:, 5:6],            # [5:6]   gripper 1D
            base_body_vel,              # [6:9]   base body velocity 3D (m/s, rad/s)
            lin_vel,                    # [9:12]  base_lin_vel 3D
            ang_vel,                    # [12:15] base_ang_vel 3D
            arm_vel,                    # [15:21] arm+grip vel 6D
            home_rel,                   # [21:24] home relative 3D
            grip_force,                 # [24:25] grip force 1D
            bbox_norm,                  # [25:28] bbox 3D
            cat_norm,                   # [28:29] category 1D
        ], dim=-1)  # 29D

        self._cached_metrics = None

        # Critic Observation (36D, AAC)
        # Actor 29D + obj_dimensions(3D) + obj_mass(1D) + gripper_rel_pos(3D) = 36D
        obj_mass_per_env = self._catalog_mass[
            self.active_object_idx.clamp(max=len(self._catalog_mass) - 1)
        ].unsqueeze(-1)
        # gripper_rel_pos: object position relative to gripper body (world-frame)
        grip_pos_w = self.robot.data.body_pos_w[:, self._gripper_body_idx]
        gripper_rel_pos = self.object_pos_w - grip_pos_w  # (N, 3)
        critic_extra = torch.cat([
            self.object_bbox,                            # 3D (obj_dimensions)
            obj_mass_per_env,                            # 1D
            gripper_rel_pos,                             # 3D (gripper-to-object)
        ], dim=-1)  # 7D
        critic_obs = torch.cat([actor_obs, critic_extra], dim=-1)  # 36D

        return {"policy": actor_obs, "critic": critic_obs}

    # ═══════════════════════════════════════════════════════════════════
    #  Rewards — CarryAndPlace
    # ═══════════════════════════════════════════════════════════════════

    def _get_rewards(self) -> torch.Tensor:
        # _cached_metrics 사용 — _get_dones()에서 이미 _update_grasp_state() 실행됨
        metrics = self._cached_metrics
        if metrics is None:
            metrics = self._compute_metrics()
            self._update_grasp_state(metrics)

        reward = torch.full((self.num_envs,), self.cfg.rew_time_penalty, device=self.device)

        # Carry: home까지 거리 줄이기
        home_progress = torch.clamp(self.prev_home_dist - metrics["home_dist"], -0.2, 0.2)
        reward += self.cfg.rew_carry_progress_weight * home_progress
        reward += self.cfg.rew_carry_heading_weight * torch.cos(metrics["heading_home"])

        # Hold bonus — 물체를 아직 들고 있을 때만
        reward += self.cfg.rew_hold_bonus * self.object_grasped.float()

        # Drop penalty — grasp break 감지 시 즉시 발동
        reward += self.just_dropped.float() * self.cfg.rew_drop_penalty

        # Place 성공: home 근처에서 의도적으로 놓음 (drop이 아닌 경우)
        place_dist = torch.norm(self.object_pos_w[:, :2] - self.home_pos_w[:, :2], dim=-1)
        near_home = place_dist < self.cfg.return_thresh
        place_success = (~self.object_grasped) & near_home & (~self.just_dropped)
        reward += place_success.float() * self.cfg.rew_place_success_bonus
        self.task_success = place_success

        # Effort
        reward += self.cfg.rew_effort_weight * (self.actions[:, 6:9] ** 2).sum(dim=-1)
        reward += self.cfg.rew_arm_move_weight * (metrics["arm_vel"] ** 2).sum(dim=-1)

        self.prev_home_dist[:] = metrics["home_dist"]
        self.episode_reward_sum += reward
        return reward

    # ═══════════════════════════════════════════════════════════════════
    #  Dones
    # ═══════════════════════════════════════════════════════════════════

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self._compute_metrics()
        self._update_grasp_state(metrics)  # drop 감지 포함
        self._cached_metrics = metrics

        root_pos = metrics["root_pos_w"]
        out_of_bounds = torch.norm(
            root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1
        ) > self.cfg.max_dist_from_origin
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)

        # Skill-3 핵심: drop → terminated (에피소드 즉시 종료)
        dropped = self.just_dropped
        terminated = out_of_bounds | fell | dropped

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)
        truncated = self.task_success | time_out

        # Logging
        self.extras["task_success_rate"] = self.task_success.float().mean()
        self.extras["drop_rate"] = dropped.float().mean()

        return terminated, truncated

    # ═══════════════════════════════════════════════════════════════════
    #  Reset — Handoff Buffer 기반
    # ═══════════════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: torch.Tensor):
        # DirectRLEnv._reset_idx (Skill2Env._reset_idx를 건너뜀)
        DirectRLEnv_reset = super(Skill2Env, self)._reset_idx
        DirectRLEnv_reset(env_ids)
        num = len(env_ids)
        if num == 0:
            return

        if self._physics_grasp and self._grasp_attach_mode == "fixed_joint":
            self._disable_grasp_fixed_joint_for_envs(env_ids)

        if self.handoff_buffer is not None and len(self.handoff_buffer) > 0:
            self._reset_from_handoff(env_ids, num)
        else:
            # Fallback: Skill-2 방식으로 리셋 (물체 잡힌 상태로 시작)
            self._reset_fallback(env_ids, num)

    def _reset_from_handoff(self, env_ids: torch.Tensor, num: int):
        """Handoff Buffer에서 랜덤 샘플하여 리셋."""
        buf_size = len(self.handoff_buffer)
        indices = torch.randint(0, buf_size, (num,))

        root_states = self.robot.data.default_root_state[env_ids].clone()
        joint_positions = self.robot.data.default_joint_pos[env_ids].clone()

        for i in range(num):
            entry = self.handoff_buffer[indices[i].item()]
            eid = env_ids[i]

            # Robot root state
            root_states[i, 0:3] = torch.tensor(entry["base_pos"], device=self.device, dtype=torch.float32)
            root_states[i, 3:7] = torch.tensor(entry["base_ori"], device=self.device, dtype=torch.float32)

            # Arm+gripper joints
            arm_vals = entry["arm_joints"]
            grip_val = entry["gripper_state"]
            for j in range(5):
                joint_positions[i, self.arm_idx[j]] = float(arm_vals[j])
            joint_positions[i, self.arm_idx[5]] = float(grip_val)

            # Object position + orientation
            self.object_pos_w[eid] = torch.tensor(entry["object_pos"], device=self.device, dtype=torch.float32)
            if "object_ori" in entry:
                self._handoff_object_ori[eid] = torch.tensor(entry["object_ori"], device=self.device, dtype=torch.float32)
            obj_idx = int(entry["object_type_idx"])
            self.active_object_idx[eid] = obj_idx
            if self._multi_object:
                self.object_bbox[eid] = self._catalog_bbox[min(obj_idx, len(self._catalog_bbox) - 1)]
                self.object_category_id[eid] = self._catalog_category[min(obj_idx, len(self._catalog_category) - 1)]

        self.robot.write_root_state_to_sim(root_states, env_ids)
        joint_vel = torch.zeros_like(joint_positions)
        self.robot.write_joint_state_to_sim(joint_positions, joint_vel, env_ids=env_ids)

        self.home_pos_w[env_ids, 0:2] = 0.0
        self.home_pos_w[env_ids, 2] = root_states[:, 2]

        # Multi-object hide/show
        if self._multi_object and len(self.object_rigids) > 0:
            for rigid in self.object_rigids:
                hide_pose = rigid.data.default_root_state[env_ids, :7].clone()
                hide_pose[:, 2] = -10.0
                rigid.write_root_pose_to_sim(hide_pose, env_ids=env_ids)

            for i in range(num):
                eid = env_ids[i]
                oi = int(self.active_object_idx[eid].item())
                if oi < len(self.object_rigids):
                    rigid = self.object_rigids[oi]
                    pose = rigid.data.default_root_state[eid:eid+1, :7].clone()
                    pose[0, :3] = self.object_pos_w[eid]
                    pose[0, 3:7] = self._handoff_object_ori[eid]
                    rigid.write_root_pose_to_sim(pose, torch.tensor([eid.item()], device=self.device))

        # 물체가 이미 잡힌 상태로 시작
        self.object_grasped[env_ids] = True
        # DR을 joint attach 전에 실행 → break_force DR이 적용된 joint 생성
        self._apply_domain_randomization(env_ids)
        self._attach_grasp_fixed_joint_for_envs(env_ids)

        self._finish_reset(env_ids, num)

    def _reset_fallback(self, env_ids: torch.Tensor, num: int):
        """Handoff buffer 없이 fallback: 물체 가까이 + 잡힌 상태."""
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        root_xy_std = float(self.cfg.dr_root_xy_noise_std) if bool(self.cfg.enable_domain_randomization) else 0.1
        default_root_state[:, 0:2] += torch.randn(num, 2, device=self.device) * root_xy_std

        random_yaw = torch.rand(num, device=self.device) * 2.0 * math.pi - math.pi
        half_yaw = random_yaw * 0.5
        default_root_state[:, 3] = torch.cos(half_yaw)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(half_yaw)
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.home_pos_w[env_ids] = default_root_state[:, :3]

        # 물체를 잡힌 위치에 배치
        self.object_pos_w[env_ids, :2] = default_root_state[:, :2]
        self.object_pos_w[env_ids, 2] = default_root_state[:, 2] + float(self.cfg.grasp_attach_height)

        self.object_grasped[env_ids] = True
        self._apply_domain_randomization(env_ids)
        self._attach_grasp_fixed_joint_for_envs(env_ids)

        self._finish_reset(env_ids, num)

    def _finish_reset(self, env_ids: torch.Tensor, num: int):
        """공통 리셋 후처리. DR은 caller가 attach 전에 실행해야 함."""
        self.task_success[env_ids] = False
        self.just_grasped[env_ids] = False
        self.just_dropped[env_ids] = False
        self.prev_home_dist[env_ids] = 10.0
        self.prev_object_dist[env_ids] = 10.0
        self.grasp_entry_step[env_ids] = 0
        self.episode_reward_sum[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
