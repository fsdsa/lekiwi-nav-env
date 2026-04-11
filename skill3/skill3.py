"""
LeKiwi Skill-3 — CarryAndPlace Isaac Lab DirectRLEnv.

3-Skill 파이프라인의 세 번째 스킬: 잡은 물체를 들고 목적지 물체(destination object)
옆까지 운반 후 내려놓기(Place).
Handoff Buffer에서 초기 상태(물체가 이미 잡힌 상태)를 로드하여 시작.

Observation (Actor 29D):
  [0:5]   arm joint pos (5)
  [5:6]   gripper pos (1)
  [6:9]   base body velocity (vx, vy, wz) (3)
  [9:12]  base linear vel body (3)
  [12:15] base angular vel body (3)
  [15:21] arm+grip joint vel (6)
  [21:24] dest object relative pos body (3)
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

Changes vs previous version:
  - Drop → terminated (RL 학습 효율 + 명확한 실패 신호)
  - 2-stage place: preliminary(obj upright+near) → final(grace_done 시점 재확인)
  - preliminary_success extras 출력 (train script에서 rest reward 조건으로 사용)
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Dict

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_mul

from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg


@configclass
class Skill3EnvCfg(Skill2EnvCfg):
    """CarryAndPlace 환경 설정."""

    observation_space: int = 29
    state_space: int = 36

    # Task
    place_gripper_threshold: float = 0.85  # gripper pos > 이 값이면 open 판정

    # Destination object — Skill2EnvCfg의 기본값 override
    dest_object_usd: str = "/home/yubin11/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd"
    dest_object_fixed: bool = False       # dynamic rigid body (scale 적용 위해 kinematic 비활성)
    dest_object_mass: float = 50.0        # 무거운 mass로 밀림 방지

    # Place 판정 (데모 기반: obj-dest 0.119±0.018m, obj_z 0.033)
    place_radius: float = 0.172          # object-dest XY 거리 (데모 max)
    place_obj_z_min: float = 0.032       # 약병 서있는 상태 obj_z 하한
    place_obj_z_max: float = 0.034       # 약병 서있는 상태 obj_z 상한
    place_grace_steps: int = 500         # place 성공 후 rest pose 복귀 여유 step

    # Dest spawn geometry
    dest_spawn_dist_min: float = 0.5
    dest_spawn_dist_max: float = 0.8
    dest_spawn_min_separation: float = 0.3

    # Handoff
    handoff_buffer_path: str = ""

    # Reward (CarryAndPlace 전용 — env 내부 reward, RL은 train에서 별도 계산)
    rew_carry_progress_weight: float = 3.0
    rew_carry_heading_weight: float = 0.2
    rew_hold_bonus: float = 0.1
    rew_place_success_bonus: float = 20.0
    rew_drop_penalty: float = -10.0

    # Handoff noise (per-load: 같은 entry를 여러 번 로드해도 다른 state)
    handoff_arm_noise_std: float = 0.02       # ~1deg joint noise
    handoff_base_pos_noise_std: float = 0.01  # 1cm position noise
    handoff_base_yaw_noise_std: float = 0.02  # ~1deg heading noise

    # Curriculum 제거 (Skill-3는 불필요)
    curriculum_success_threshold: float = 1.0

    # Drop → terminate (RL 학습 효율)
    terminate_on_drop: bool = True


class Skill3Env(Skill2Env):
    """LeKiwi Skill-3: CarryAndPlace RL 환경."""

    cfg: Skill3EnvCfg

    def __init__(self, cfg: Skill3EnvCfg, render_mode: str | None = None, **kwargs):
        # Handoff buffer 로드
        self.handoff_buffer = None
        if cfg.handoff_buffer_path:
            buf_path = os.path.expanduser(cfg.handoff_buffer_path)
            if os.path.isfile(buf_path):
                with open(buf_path, "rb") as f:
                    self.handoff_buffer = pickle.load(f)
                print(f"  [Skill3Env] Loaded handoff buffer: {len(self.handoff_buffer)} entries from {buf_path}")
            else:
                print(f"  [WARN] Handoff buffer not found: {buf_path}")

        super().__init__(cfg, render_mode, **kwargs)

        # Skill-3 추가 버퍼
        self.prev_dest_dist = torch.zeros(self.num_envs, device=self.device)
        self.intentional_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.place_success_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.preliminary_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Fallback: handoff buffer 없을 때 매 step teleport carry 사용
        self._fallback_teleport_carry = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Combined mode: record_teleop에서 Skill-2→3 연속 레코딩 시 사용
        self._combined_mode: bool = False
        # Handoff buffer에서 읽은 object orientation
        self._handoff_object_ori = torch.zeros(self.num_envs, 4, device=self.device)
        self._handoff_object_ori[:, 0] = 1.0  # qw=1 identity

        print(f"  [Skill3Env] obs={self.cfg.observation_space} act={self.cfg.action_space} critic={self.cfg.state_space}")
        print(f"  [Skill3Env] terminate_on_drop={self.cfg.terminate_on_drop}")
        print(f"  [Skill3Env] place: radius={self.cfg.place_radius}m "
              f"obj_z=[{self.cfg.place_obj_z_min},{self.cfg.place_obj_z_max}]")

    # ═══════════════════════════════════════════════════════════════════
    #  Action — teleport carry (fallback 모드)
    # ═══════════════════════════════════════════════════════════════════

    def _apply_action(self):
        super()._apply_action()
        tc_mask = self._fallback_teleport_carry & self.object_grasped
        if tc_mask.any():
            tc_ids = tc_mask.nonzero(as_tuple=False).squeeze(-1)
            if self._gripper_body_idx is None:
                try:
                    body_ids, _ = self.robot.find_bodies(["Moving_Jaw_08d_v1"])
                    self._gripper_body_idx = body_ids[0]
                except (IndexError, RuntimeError, AttributeError):
                    self._gripper_body_idx = 0
            grip_pos = self.robot.data.body_pos_w[tc_ids, self._gripper_body_idx]
            self.object_pos_w[tc_ids] = grip_pos
            if not self._multi_object and self.object_rigid is not None:
                pose = self.object_rigid.data.root_pose_w[tc_ids].clone()
                pose[:, :3] = grip_pos
                self.object_rigid.write_root_pose_to_sim(pose, env_ids=tc_ids)
                zero_vel = torch.zeros(len(tc_ids), 6, dtype=torch.float32, device=self.device)
                self.object_rigid.write_root_velocity_to_sim(zero_vel, env_ids=tc_ids)

    # ═══════════════════════════════════════════════════════════════════
    #  Metrics — destination object 관련
    # ═══════════════════════════════════════════════════════════════════

    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        metrics = super()._compute_metrics()

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        dest_delta_w = self.dest_object_pos_w - root_pos_w
        dest_pos_b = quat_apply_inverse(root_quat_w, dest_delta_w)
        dest_dist = torch.norm(dest_pos_b[:, :2], dim=-1)
        heading_dest = torch.atan2(dest_pos_b[:, 1], dest_pos_b[:, 0])

        metrics["dest_pos_b"] = dest_pos_b
        metrics["dest_dist"] = dest_dist
        metrics["heading_dest"] = heading_dest
        return metrics

    # ═══════════════════════════════════════════════════════════════════
    #  Grasp state — intentional place 추가
    # ═══════════════════════════════════════════════════════════════════

    def _update_grasp_state(self, metrics: Dict[str, torch.Tensor]):
        """Skill-3 grasp state: intentional place 먼저 체크 후 부모 로직."""
        self.intentional_placed[:] = False

        # Intentional place: dest 근처에서 gripper open → place, not drop
        if self.object_grasped.any():
            gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
            gripper_open = gripper_pos > float(self.cfg.place_gripper_threshold)
            dest_dist = metrics.get("dest_dist", None)
            if dest_dist is None:
                dest_delta_w = self.dest_object_pos_w - self.robot.data.root_pos_w
                dest_pos_b = quat_apply_inverse(self.robot.data.root_quat_w, dest_delta_w)
                dest_dist = torch.norm(dest_pos_b[:, :2], dim=-1)
            near_dest = dest_dist < (self.cfg.place_radius * 3.0)

            intentional_place = self.object_grasped & gripper_open & near_dest
            if intentional_place.any():
                self.object_grasped[intentional_place] = False
                self.intentional_placed[intentional_place] = True

        # 부모(Skill2) grasp 로직 (can_grasp, lift, drop detection)
        super()._update_grasp_state(metrics)

    # ═══════════════════════════════════════════════════════════════════
    #  Observations — 29D Actor
    # ═══════════════════════════════════════════════════════════════════

    def _get_observations(self) -> dict:
        metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()
        base_body_vel = self._read_base_body_vel()

        arm_pos = metrics["arm_pos"]
        arm_vel = metrics["arm_vel"]
        lin_vel = metrics["lin_vel_b"]
        ang_vel = metrics["ang_vel_b"]

        dest_delta_w = self.dest_object_pos_w - self.robot.data.root_pos_w
        dest_object_rel = quat_apply_inverse(self.robot.data.root_quat_w, dest_delta_w)

        contact_force = self._contact_force_per_env()
        grip_force = contact_force.unsqueeze(-1)

        # Observation noise
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
                dest_object_rel = dest_object_rel + torch.randn_like(dest_object_rel) * or_noise

        bbox_norm = self.object_bbox / float(self._bbox_norm_scale)
        cat_denom = max(int(self.cfg.num_object_categories) - 1, 1)
        cat_norm = (self.object_category_id / float(cat_denom)).unsqueeze(-1)

        actor_obs = torch.cat([
            arm_pos[:, :5],             # [0:5]   arm 5D
            arm_pos[:, 5:6],            # [5:6]   gripper 1D
            base_body_vel,              # [6:9]   base body velocity 3D
            lin_vel,                    # [9:12]  base_lin_vel 3D
            ang_vel,                    # [12:15] base_ang_vel 3D
            arm_vel,                    # [15:21] arm+grip vel 6D
            dest_object_rel,            # [21:24] dest object relative 3D
            grip_force,                 # [24:25] grip force 1D
            bbox_norm,                  # [25:28] bbox 3D
            cat_norm,                   # [28:29] category 1D
        ], dim=-1)  # 29D

        self._cached_metrics = None

        # Critic Observation (36D)
        obj_mass_per_env = self._catalog_mass[
            self.active_object_idx.clamp(max=len(self._catalog_mass) - 1)
        ].unsqueeze(-1)
        if self._gripper_body_idx is None:
            try:
                body_ids, _ = self.robot.find_bodies(["Moving_Jaw_08d_v1"])
                self._gripper_body_idx = body_ids[0]
            except (IndexError, RuntimeError, AttributeError):
                self._gripper_body_idx = 0
        grip_pos_w = self.robot.data.body_pos_w[:, self._gripper_body_idx]
        gripper_rel_pos = self.object_pos_w - grip_pos_w
        critic_extra = torch.cat([
            self.object_bbox,
            obj_mass_per_env,
            gripper_rel_pos,
        ], dim=-1)  # 7D
        critic_obs = torch.cat([actor_obs, critic_extra], dim=-1)  # 36D
        self._critic_obs = critic_obs

        return {"policy": actor_obs, "critic": critic_obs}

    # ═══════════════════════════════════════════════════════════════════
    #  Combined mode — Skill-2 (30D) obs 계산
    # ═══════════════════════════════════════════════════════════════════

    def _compute_skill2_actor_obs(self) -> torch.Tensor:
        """Combined teleop Phase 1용: Skill-2 (30D) actor obs."""
        metrics = self._compute_metrics()
        base_body_vel = self._read_base_body_vel()

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

        return torch.cat([
            arm_pos[:, :5],
            arm_pos[:, 5:6],
            base_body_vel,
            lin_vel,
            ang_vel,
            arm_vel,
            rel_object,
            contact_lr,
            bbox_norm,
            cat_norm,
        ], dim=-1)  # 30D

    # ═══════════════════════════════════════════════════════════════════
    #  Rewards — CarryAndPlace (env 내부 reward)
    # ═══════════════════════════════════════════════════════════════════

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._cached_metrics
        if metrics is None:
            metrics = self._compute_metrics()
            self._update_grasp_state(metrics)

        reward = torch.full((self.num_envs,), self.cfg.rew_time_penalty, device=self.device)

        action_delta = self.actions - self.prev_actions
        reward += self.cfg.rew_action_smoothness_weight * (action_delta ** 2).sum(dim=-1)

        dest_progress = torch.clamp(self.prev_dest_dist - metrics["dest_dist"], -0.2, 0.2)
        reward += self.cfg.rew_carry_progress_weight * dest_progress
        reward += self.cfg.rew_carry_heading_weight * torch.cos(metrics["heading_dest"])

        reward += self.cfg.rew_hold_bonus * self.object_grasped.float()
        reward += self.just_dropped.float() * self.cfg.rew_drop_penalty

        place_success = self._check_place_condition() & (~self.object_grasped) & (~self.just_dropped)
        reward += place_success.float() * self.cfg.rew_place_success_bonus
        self.task_success = place_success

        reward += self.cfg.rew_effort_weight * (self.actions[:, 6:9] ** 2).sum(dim=-1)
        reward += self.cfg.rew_arm_move_weight * (metrics["arm_vel"] ** 2).sum(dim=-1)

        self.prev_dest_dist[:] = metrics["dest_dist"]
        self.episode_reward_sum += reward
        return reward

    # ═══════════════════════════════════════════════════════════════════
    #  Place condition
    # ═══════════════════════════════════════════════════════════════════

    def _check_place_condition(self) -> torch.Tensor:
        """Place 조건: 약병 서있음(obj_z) + dest 근처(XY)."""
        obj_pos = self.object_pos_w
        dest_pos = self.dest_object_pos_w
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0

        xy_dist = torch.norm(obj_pos[:, :2] - dest_pos[:, :2], dim=-1)
        obj_z = obj_pos[:, 2] - env_z

        upright = (obj_z >= self.cfg.place_obj_z_min) & (obj_z <= self.cfg.place_obj_z_max)
        near_dest = xy_dist < self.cfg.place_radius

        return near_dest & upright

    # ═══════════════════════════════════════════════════════════════════
    #  Dones — drop terminate + 2-stage place
    # ═══════════════════════════════════════════════════════════════════

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self._compute_metrics()
        self._update_grasp_state(metrics)
        self._cached_metrics = metrics

        # ── Place 2단계 판정 ──
        place_cond = self._check_place_condition()

        # 1차 성공: 최초로 obj upright + near dest 진입
        newly_prelim = place_cond & (~self.object_grasped) & (~self.just_dropped) & (~self.preliminary_success)
        if newly_prelim.any():
            self.preliminary_success[newly_prelim] = True
            self.place_success_step[newly_prelim] = self.episode_length_buf[newly_prelim]

        # ── Terminated 조건 ──
        root_pos = metrics["root_pos_w"]
        out_of_bounds = torch.norm(
            root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1
        ) > self.cfg.max_dist_from_origin
        env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
        fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)

        # Drop → terminated (RL 학습: 낙하는 즉시 에피소드 종료)
        dropped = self.just_dropped if bool(self.cfg.terminate_on_drop) else \
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        terminated = out_of_bounds | fell | dropped

        # ── Truncated 조건 ──
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        # 1차 성공 후 grace period (rest pose 복귀 시간)
        place_grace_done = (self.place_success_step > 0) & (
            (self.episode_length_buf - self.place_success_step) >= self.cfg.place_grace_steps
        )

        # 최종 성공: grace 종료 시점에서도 place 조건 유지
        if place_grace_done.any():
            final_still_ok = self.preliminary_success & place_cond
            self.task_success = self.task_success | (final_still_ok & place_grace_done)

        truncated = place_grace_done | time_out

        # ── 진단 출력 ──
        if (terminated.any() or truncated.any()):
            n_oob = out_of_bounds.sum().item()
            n_fell = fell.sum().item()
            n_drop = dropped.sum().item() if bool(self.cfg.terminate_on_drop) else 0
            n_prelim = self.preliminary_success.sum().item()
            n_final = self.task_success.sum().item()
            n_pg = place_grace_done.sum().item()
            n_to = time_out.sum().item()
            obj_z_val = (self.object_pos_w[:, 2] - env_z)
            od_xy = torch.norm(self.object_pos_w[:, :2] - self.dest_object_pos_w[:, :2], dim=-1)
            print(f"  [S3_DONE] oob={n_oob} fell={n_fell} drop={n_drop} "
                  f"prelim={n_prelim} final={n_final} grace={n_pg} to={n_to} "
                  f"obj_z={obj_z_val[0]:.4f} od_xy={od_xy[0]:.3f}",
                  flush=True)

        # ── Extras (train script에서 읽는 key들) ──
        self.extras["place_success_mask"] = self.task_success.clone()
        self.extras["task_success_rate"] = self.task_success.float().mean()
        self.extras["preliminary_success"] = self.preliminary_success.clone()
        self.extras["object_grasped_mask"] = self.object_grasped.clone()
        self.extras["just_dropped_mask"] = self.just_dropped.clone()
        self.extras["object_height_mask"] = (self.object_pos_w[:, 2] - env_z).clone()
        self.extras["dest_contact_force"] = self._dest_contact_force_per_env().clone()

        return terminated, truncated

    # ═══════════════════════════════════════════════════════════════════
    #  Reset — Handoff Buffer 기반
    # ═══════════════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: torch.Tensor):
        if self._combined_mode:
            Skill2Env._reset_idx(self, env_ids)
            self.prev_dest_dist[env_ids] = 10.0
            self.intentional_placed[env_ids] = False
            self.place_success_step[env_ids] = 0
            self.preliminary_success[env_ids] = False
            self._fallback_teleport_carry[env_ids] = False
            env_origins = self.scene.env_origins[env_ids]
            self.home_pos_w[env_ids, 0:2] = env_origins[:, 0:2]
            self.home_pos_w[env_ids, 2] = self.robot.data.root_pos_w[env_ids, 2]
            return

        # DirectRLEnv._reset_idx (Skill2Env._reset_idx를 건너뜀)
        DirectRLEnv_reset = super(Skill2Env, self)._reset_idx
        DirectRLEnv_reset(env_ids)
        num = len(env_ids)
        if num == 0:
            return

        if self.handoff_buffer is not None and len(self.handoff_buffer) > 0:
            self._reset_from_handoff(env_ids, num)
        else:
            self._reset_fallback(env_ids, num)

    def _reset_from_handoff(self, env_ids: torch.Tensor, num: int):
        """Handoff Buffer에서 랜덤 샘플하여 리셋."""
        buf_size = len(self.handoff_buffer)
        indices = torch.randint(0, buf_size, (num,))
        entries = [self.handoff_buffer[indices[i].item()] for i in range(num)]

        root_states = self.robot.data.default_root_state[env_ids].clone()
        joint_positions = self.robot.data.default_joint_pos[env_ids].clone()

        base_pos = torch.tensor([e["base_pos"] for e in entries], device=self.device, dtype=torch.float32)
        base_ori = torch.tensor([e["base_ori"] for e in entries], device=self.device, dtype=torch.float32)
        obj_pos = torch.tensor([e["object_pos"] for e in entries], device=self.device, dtype=torch.float32)
        obj_ori = torch.tensor(
            [e.get("object_ori", [1.0, 0.0, 0.0, 0.0]) for e in entries],
            device=self.device, dtype=torch.float32)
        arm_joints = torch.tensor([e["arm_joints"] for e in entries], device=self.device, dtype=torch.float32)
        grip_states = torch.tensor([e["gripper_state"] for e in entries], device=self.device, dtype=torch.float32)
        obj_type_indices = torch.tensor([int(e["object_type_idx"]) for e in entries], device=self.device, dtype=torch.long)

        env_origins = self.scene.env_origins[env_ids]
        base_pos = base_pos + env_origins
        obj_pos = obj_pos + env_origins

        # Per-load noise
        if self.cfg.handoff_arm_noise_std > 0:
            arm_joints = arm_joints + torch.randn_like(arm_joints) * self.cfg.handoff_arm_noise_std
            if self.cfg.arm_action_to_limits:
                arm_lo = self.robot.data.soft_joint_pos_limits[0, self.arm_idx[:5], 0]
                arm_hi = self.robot.data.soft_joint_pos_limits[0, self.arm_idx[:5], 1]
                arm_joints = torch.clamp(arm_joints, arm_lo, arm_hi)

        if self.cfg.handoff_base_pos_noise_std > 0:
            base_pos[:, :2] += torch.randn(num, 2, device=self.device) * self.cfg.handoff_base_pos_noise_std

        if self.cfg.handoff_base_yaw_noise_std > 0:
            dyaw = torch.randn(num, device=self.device) * self.cfg.handoff_base_yaw_noise_std
            half = dyaw * 0.5
            dq = torch.zeros(num, 4, device=self.device)
            dq[:, 0] = torch.cos(half)
            dq[:, 3] = torch.sin(half)
            base_ori = quat_mul(dq, base_ori)

        root_states[:, 0:3] = base_pos
        root_states[:, 3:7] = base_ori

        for j in range(5):
            joint_positions[:, self.arm_idx[j]] = arm_joints[:, j]
        joint_positions[:, self.arm_idx[5]] = grip_states

        self.object_pos_w[env_ids] = obj_pos
        self._handoff_object_ori[env_ids] = obj_ori
        self.active_object_idx[env_ids] = obj_type_indices

        if not self._multi_object and self.object_rigid is not None:
            pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
            pose[:, :3] = obj_pos
            pose[:, 3:7] = obj_ori
            self.object_rigid.write_root_pose_to_sim(pose, env_ids)
            zero_vel = torch.zeros(num, 6, dtype=torch.float32, device=self.device)
            self.object_rigid.write_root_velocity_to_sim(zero_vel, env_ids)

        if self._multi_object:
            clamped_idx = obj_type_indices.clamp(max=len(self._catalog_bbox) - 1)
            self.object_bbox[env_ids] = self._catalog_bbox[clamped_idx]
            self.object_category_id[env_ids] = self._catalog_category[
                clamped_idx.clamp(max=len(self._catalog_category) - 1)]

        self.robot.write_root_state_to_sim(root_states, env_ids)
        joint_vel = torch.zeros_like(joint_positions)
        self.robot.write_joint_state_to_sim(joint_positions, joint_vel, env_ids=env_ids)

        self.home_pos_w[env_ids, 0:2] = env_origins[:, 0:2]
        self.home_pos_w[env_ids, 2] = root_states[:, 2]
        self._spawn_dest_object(env_ids)

        # Multi-object hide/show
        if self._multi_object and len(self.object_rigids) > 0:
            zero_vel = torch.zeros(num, 6, dtype=torch.float32, device=self.device)
            for oi, rigid in enumerate(self.object_rigids):
                hide_pose = rigid.data.default_root_state[env_ids, :7].clone()
                hide_pose[:, 2] = -100.0 - oi * 2.0
                rigid.write_root_pose_to_sim(hide_pose, env_ids=env_ids)
                rigid.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

            for i in range(num):
                eid = env_ids[i]
                oi = int(self.active_object_idx[eid].item())
                if oi < len(self.object_rigids):
                    rigid = self.object_rigids[oi]
                    pose = rigid.data.default_root_state[eid:eid+1, :7].clone()
                    pose[0, :3] = self.object_pos_w[eid]
                    pose[0, 3:7] = self._handoff_object_ori[eid]
                    rigid.write_root_pose_to_sim(pose, torch.tensor([eid.item()], device=self.device))

        self.object_grasped[env_ids] = True
        self._apply_domain_randomization(env_ids)
        self._fallback_teleport_carry[env_ids] = False
        self._finish_reset(env_ids, num)

    def _reset_fallback(self, env_ids: torch.Tensor, num: int):
        """Handoff buffer 없이 fallback: tucked pose + gripper closed."""
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        root_xy_std = float(self.cfg.dr_root_xy_noise_std) if bool(self.cfg.enable_domain_randomization) else 0.1
        default_root_state[:, 0:2] += torch.randn(num, 2, device=self.device) * root_xy_std
        default_root_state[:, 7:13] = 0.0

        random_yaw = torch.rand(num, device=self.device) * 2.0 * math.pi - math.pi
        half_yaw = random_yaw * 0.5
        default_root_state[:, 3] = torch.cos(half_yaw)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(half_yaw)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        carry_joints = [-0.03, -0.21, 0.09, 0.12, 0.06, 0.50]
        for i, val in enumerate(carry_joints):
            joint_pos[:, self.arm_idx[i]] = val
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        env_origins = self.scene.env_origins[env_ids] if hasattr(self.scene, "env_origins") else torch.zeros(num, 3, device=self.device)
        self.home_pos_w[env_ids, 0:2] = env_origins[:, 0:2]
        self.home_pos_w[env_ids, 2] = default_root_state[:, 2]
        self._spawn_dest_object(env_ids)

        self.object_pos_w[env_ids, :2] = default_root_state[:, :2]
        self.object_pos_w[env_ids, 2] = default_root_state[:, 2] + float(self.cfg.grasp_success_height)

        if not self._multi_object and self.object_rigid is not None:
            pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
            pose[:, :3] = self.object_pos_w[env_ids]
            self.object_rigid.write_root_pose_to_sim(pose, env_ids)
            zero_vel = torch.zeros(num, 6, dtype=torch.float32, device=self.device)
            self.object_rigid.write_root_velocity_to_sim(zero_vel, env_ids)

        self.object_grasped[env_ids] = True
        self._apply_domain_randomization(env_ids)
        self._fallback_teleport_carry[env_ids] = False
        self._finish_reset(env_ids, num)

    def _finish_reset(self, env_ids: torch.Tensor, num: int):
        """공통 리셋 후처리."""
        self.task_success[env_ids] = False
        self.just_grasped[env_ids] = False
        self.just_dropped[env_ids] = False
        self.intentional_placed[env_ids] = False
        self.place_success_step[env_ids] = 0
        self.preliminary_success[env_ids] = False
        self.prev_dest_dist[env_ids] = 10.0
        self.prev_object_dist[env_ids] = 10.0
        self.grasp_entry_step[env_ids] = 0
        self.episode_reward_sum[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        if self._action_delay_buf is not None:
            self._action_delay_buf[:, env_ids] = 0.0
