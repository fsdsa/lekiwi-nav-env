# LeKiwi 3-Skill 파이프라인 리팩토링 구현 가이드

> **이 문서의 목적**: 기존 v8 코드베이스(`lekiwi_nav_env.py` 중심)를 3-Skill 파이프라인으로 리팩토링하기 위한 완전한 구현 명세.
>
> **핵심 원칙**: v8의 검증된 물리·캘리브레이션·DR·grasp 코드를 **함수 단위로 그대로 복사**하되, monolithic 4-phase FSM을 독립 Skill 환경 3개(Navigate, ApproachAndGrasp, CarryAndPlace)로 분리한다.
>
> **서버 환경 설치/전송/학습 가이드**: `feedback/server_guide.md` 참조

---

## 0. 프로젝트 전체 구조

### 변경하지 않는 파일 (한 줄도 수정 금지)

| 파일 | 이유 |
|------|------|
| `lekiwi_robot_cfg.py` | 모든 물리 상수(wheel_radius, base_radius, Kiwi IK, arm limits 등) 검증 완료 |
| `spawn_manager.py` | USD 물체 스폰 + 조명 DR |
| `calibration_common.py` | 캘리브레이션 유틸 |
| `calibrate_real_robot.py` | 실로봇 측정 |
| `check_calibration_gate.py` | 품질 게이트 |
| `tune_sim_dynamics.py` | Sim dynamics CMA-ES 튜닝 |
| `replay_in_sim.py` | Sim-Real replay |
| `compare_real_sim.py` | 비교 플롯 |
| `sim_real_command_transform.py` | sim↔real 변환 |
| `sim_real_calibration_test.py` | Script Editor 검증 |
| `extract_kiwi_geometry_from_usd.py` | USD geometry 추출 |
| `build_object_catalog.py` | 물체 카탈로그 빌드 |
| `leader_to_home_tcp_rest_matched_with_keyboard_base.py` | 리더암+키보드 |
| `object_catalog.json`, `object_catalog_all.json` | 물체 데이터 |

### 새로 만들 파일

| 파일 | 설명 |
|------|------|
| `lekiwi_skill1_env.py` | Navigate RL 환경 (장애물 회피 + pseudo-lidar + 감속, 20D actor / 25D critic) |
| `lekiwi_skill2_env.py` | ApproachAndGrasp 환경 (30D obs) |
| `lekiwi_skill3_env.py` | CarryAndPlace 환경 (29D obs) |
| `generate_handoff_buffer.py` | Skill-2 종료 상태 → Skill-3 초기 상태 |
| `collect_navigate_data.py` | Navigate 스크립트 정책 데이터 수집 (fallback, RL Expert rollout 우선) |
| `calibrate_tucked_pose.py` | Tucked Pose 측정 (리더암 TCP, self-collision 방지 한계) |
| `calibrate_arm_limits.py` | Arm Joint Limits 측정 (리더암 TCP, 관절별 min/max) |
| `aac_wrapper.py` | IsaacLabWrapper monkey-patch — critic obs 노출 (`state()` 메서드) |
| `aac_ppo.py` | PPO 상속 — `critic_states` memory tensor 관리 |
| `aac_trainer.py` | SequentialTrainer 상속 — critic states 매 step 추적 |

### 수정할 파일

| 파일 | 변경 범위 |
|------|-----------|
| `models.py` | `CriticNet` 클래스 추가 (기존 코드 유지) |
| `train_lekiwi.py` | `--skill` 분기 추가 (navigate / approach_and_grasp / carry_and_place / legacy) |
| `train_bc.py` | `--expected_obs_dim` required화 |
| `collect_demos.py` | robot_state 추출 + gripper binary + skill 분기 + `Skill1EnvWithCam`/`Skill2EnvWithCam`/`Skill3EnvWithCam` 카메라 서브클래스 |
| `convert_hdf5_to_lerobot_v3.py` | 채널명 v3.0 업데이트 + 단위 변환 제거 + `infer_robot_state_from_obs()` dim==20 Navigate 지원 |
| `record_teleop.py` | privileged obs 동시 기록 |
| `deploy_vla_action_bridge.py` | `--action_format v6/legacy` 플래그 추가 |

---

## 1. 핵심 설계 변경 3가지

### 1-A. Action 순서 반전

v8에서 가장 큰 구조적 변경. **모든 새 파일에서 아래 순서를 사용한다.**

```
v8 (현재):     [base_vx, base_vy, base_wz, arm0, arm1, arm2, arm3, arm4, gripper]
                 ↓ 0      ↓ 1      ↓ 2     ↓3    ↓4    ↓5    ↓6    ↓7    ↓8

새 파이프라인:  [arm0, arm1, arm2, arm3, arm4, gripper, base_vx, base_vy, base_wz]
                 ↓ 0   ↓ 1   ↓ 2   ↓ 3   ↓ 4   ↓ 5      ↓ 6      ↓ 7      ↓ 8
```

이유: HuggingFace `yubinnn11/lekiwi3` (LeRobot v3.0) 실제 로봇 데이터의 포맷이 `[arm_shoulder_pan.pos, arm_shoulder_lift.pos, arm_elbow_flex.pos, arm_wrist_flex.pos, arm_wrist_roll.pos, arm_gripper.pos, x.vel, y.vel, theta.vel]`이고, VLA(π0-FAST/GR00T)가 이 포맷으로 학습하므로 RL Expert도 동일 순서로 출력해야 변환 없이 사용 가능.

### 1-B. Observation 재구성

v8의 37D(FSM 전용 채널 포함)에서 FSM 관련 채널을 제거하고, **velocity 정보**를 추가. base state의 마지막 3채널은 body-frame velocity(m/s, rad/s)로 sim에서 `root_lin_vel_b`와 `root_ang_vel_b`를 직접 읽는다.

### 1-C. Gripper 처리

RL 학습: continuous position target (v8과 동일).
VLA 데이터 저장: `action[5] = 1.0 if raw > 0.5 else 0.0` (binary 변환).

---

## 2. `lekiwi_skill2_env.py` — ApproachAndGrasp (신규)

### 2-1. 만드는 방법

1. `lekiwi_nav_env.py`를 통째로 복사
2. 아래 "제거" 목록의 코드를 삭제
3. 아래 "변경" 목록의 코드를 교체
4. 아래 "추가" 목록의 코드를 삽입

### 2-2. 제거할 것

**상수 및 import에서:**
```python
# 삭제:
PHASE_SEARCH = 0
PHASE_APPROACH = 1
PHASE_GRASP = 2
PHASE_RETURN = 3
NUM_PHASES = 4
```

**Config 클래스에서 삭제할 필드:**
```python
# 삭제 (search 관련):
search_dist_min, search_dist_max, vision_fov_deg, vision_max_dist,
search_reveal_dist, search_reveal_scan

# 삭제 (return 관련):
return_thresh, goal_reached_thresh

# 삭제 (search/return reward 관련):
rew_search_progress_weight, rew_search_move_weight, rew_search_turn_weight,
rew_detect_bonus, rew_return_progress_weight, rew_return_heading_weight,
rew_return_vel_weight, rew_carry_bonus, rew_return_success_bonus
```

**`__init__`에서 삭제할 버퍼:**
```python
# 삭제:
self.search_target_w = ...
self.phase = ...
self.object_revealed = ...
self.object_visible = ...   # Skill-2는 항상 물체 있음
self.search_scan_score = ...
self.prev_search_dist = ...
self.just_detected = ...
self.just_reached_grasp = ...
self._phase_before_update = ...
self.prev_home_dist = ...  # Skill-2는 home 불필요
```

**메서드 전체 삭제:**
```python
_update_phase_state()     # FSM 전체
_get_target_xy_body()     # phase별 타깃 전환
_resample_search_targets()  # search 힌트 포인트
```

### 2-3. Config 클래스 변경

```python
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
    arm_limit_write_to_sim: bool = True    # RL: True (PhysX 제약), 텔레옵: False (USD 기본 리밋)

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
    grasp_joint_break_force: float = 30.0    # ← v8의 1e8에서 변경 (mass*g*10)
    grasp_joint_break_torque: float = 30.0   # ← 동일
    grasp_drop_detect_dist: float = 0.15     # ★ 신규: gripper-object 거리 > 이 값이면 drop 판정
    grasp_timeout_steps: int = 75            # v8 동일

    # === Multi-object (v8 동일) ===
    multi_object_json: str = ""
    num_object_categories: int = 6

    # === Reward (변경 — approach/grasp/lift 전용) ===
    rew_time_penalty: float = -0.01
    rew_effort_weight: float = -0.01
    rew_arm_move_weight: float = -0.02
    rew_action_smoothness_weight: float = -0.005  # action delta penalty (sim2real)
    rew_approach_progress_weight: float = 6.0
    rew_approach_heading_weight: float = 0.2
    rew_approach_vel_weight: float = 0.5
    rew_proximity_tanh_weight: float = 2.0  # tanh proximity kernel
    rew_proximity_tanh_sigma: float = 0.5   # tanh kernel sigma
    rew_grasp_success_bonus: float = 10.0
    rew_lift_bonus: float = 5.0
    rew_collision: float = -1.0

    # === Termination (v8 동일) ===
    max_dist_from_origin: float = 6.0

    # === DR (v8 동일) ===
    enable_domain_randomization: bool = True
    dr_root_xy_noise_std: float = 0.12
    dr_root_yaw_jitter_rad: float = 0.2
    dr_wheel_stiffness_scale_range: tuple[float, float] = (0.75, 1.5)
    dr_wheel_damping_scale_range: tuple[float, float] = (0.3, 3.0)   # sim2real 가장 중요
    dr_wheel_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_wheel_dynamic_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_wheel_viscous_friction_scale_range: tuple[float, float] = (0.7, 1.3)
    dr_arm_stiffness_scale_range: tuple[float, float] = (0.8, 1.25)
    dr_arm_damping_scale_range: tuple[float, float] = (0.5, 2.0)
    dr_object_mass_scale_range: tuple[float, float] = (0.5, 2.0)     # 실제 물체 편차 큼
    dr_object_static_friction_scale_range: tuple[float, float] = (0.6, 1.5)
    dr_object_dynamic_friction_scale_range: tuple[float, float] = (0.6, 1.5)

    # Observation noise (sim2real: 센서 노이즈)
    dr_obs_noise_joint_pos: float = 0.01     # rad
    dr_obs_noise_base_vel: float = 0.02      # m/s
    dr_obs_noise_object_rel: float = 0.02    # m

    # Action delay (sim2real: 통신 지연)
    dr_action_delay_steps: int = 1           # 0=없음, 1-2=권장

    # ★ Grasp DR (신규 — sim2real gap 핵심)
    dr_grasp_break_force_range: tuple[float, float] = (15.0, 45.0)   # break_force 랜덤화
    dr_grasp_break_torque_range: tuple[float, float] = (15.0, 45.0)  # break_torque 랜덤화
```

### 2-4. `__init__` 변경

v8의 `__init__`을 복사한 후:

**삭제할 버퍼** (위 2-2에 나열된 것들)

**추가할 버퍼:**
```python
# Curriculum 상태 추적 (신규) — config에서 초기값 읽기
if (hasattr(self.cfg, 'curriculum_current_max_dist')
        and float(self.cfg.curriculum_current_max_dist) > float(self.cfg.object_dist_min)):
    self._curriculum_dist = min(
        float(self.cfg.curriculum_current_max_dist), float(self.cfg.object_dist_max))
else:
    self._curriculum_dist = float(self.cfg.object_dist_min)
self._curriculum_success_window = torch.zeros(100, device=self.device)  # 최근 100 에피소드
self._curriculum_idx = 0

# ★ Grasp break 감지용 (신규)
self.just_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
```

**`_setup_scene()` 끝에 추가** (gripper body index 초기화):
```python
# ★ Grasp break 감지를 위한 gripper body index
# USD 확인 결과: 그리퍼 움직이는 부분의 body = "Moving_Jaw_08d_v1"
# (joint: STS3215_03a_v1_4_Revolute_57, 고정부 제외)
# ★ 주의: find_bodies() 반환 index(=39)와 USD Inspector 순서(=6)가 다름
#   Isaac Lab이 articulation tree를 재정렬하기 때문. 반드시 find_bodies()로 동적 취득.
_gripper_body_names = ["Moving_Jaw_08d_v1"]
body_ids, _ = self.robot.find_bodies(_gripper_body_names)
self._gripper_body_idx = body_ids[0]
# 검증: print(f"Gripper body idx: {self._gripper_body_idx}, name: Moving_Jaw_08d_v1")
```

**주의: displacement 관련 버퍼는 생성하지 않는다.** 이전 설계의 `prev_root_pos_w`, `prev_root_quat_w`, `body_displacement` 버퍼는 velocity 직접 읽기(`root_lin_vel_b`, `root_ang_vel_b`)로 대체되어 불필요하다.

**나머지는 v8과 동일** — 특히 다음은 그대로 복사:
- Kiwi IK matrix 생성
- joint index 초기화 (`arm_idx`, `wheel_idx`, `gripper_idx`)
- multi_object catalog 파싱 (`_catalog_bbox`, `_catalog_category`, `_catalog_mass`)
- `_maybe_apply_tuned_dynamics_from_cfg()`, `_apply_baked_arm_limits()`, `_maybe_apply_arm_limits_from_cfg()`
- `_init_domain_randomization_buffers()`
- 모든 DR 버퍼

### 2-5. `_setup_scene()` — v8 전체 복사, 변경 없음

v8의 `_setup_scene()`을 **그대로** 복사한다. 물체 pre-spawn, contact sensor, ground plane, lighting 전부 동일.

### 2-6. v8에서 그대로 복사할 메서드 목록

다음 메서드들은 **한 줄도 수정하지 않고** 그대로 복사:

```python
# Calibration / Dynamics
_maybe_apply_calibration_geometry_from_cfg()
_maybe_apply_tuned_dynamics_from_cfg()
_extract_tuned_params()
_extract_arm_limits_payload()
_apply_arm_limits()
_apply_baked_arm_limits()
_maybe_apply_arm_limits_from_cfg()
_safe_float()
apply_tuned_dynamics()

# Physics Grasp (전체)
_resolve_env_pattern_path()
_get_stage()
_gripper_body_prim_path()
_object_body_prim_path()
_quatd_to_quatf()
_get_or_create_grasp_fixed_joint()
_attach_grasp_fixed_joint_for_envs()
_disable_grasp_fixed_joint_for_envs()
_teleport_attach_for_envs()

# Domain Randomization (전체)
_init_domain_randomization_buffers()
_parse_scale_range()
_sample_scale()
_apply_object_mass_randomization()
_apply_domain_randomization()

# Contact
_contact_force_per_env()

# Utility
_sample_targets_around()
```

### 2-6b. `_apply_domain_randomization()` — 확장 (Grasp DR 추가)

v8의 `_apply_domain_randomization()`을 복사한 후, **끝에 다음을 추가**:

```python
# ★ Grasp break force/torque 랜덤화 (신규)
# 에피소드마다 break_force를 다르게 → "약한 grasp~강한 grasp" 모두 경험
if hasattr(self.cfg, 'dr_grasp_break_force_range'):
    bf_lo, bf_hi = self.cfg.dr_grasp_break_force_range
    bt_lo, bt_hi = self.cfg.dr_grasp_break_torque_range
    num = len(env_ids)
    self._per_env_break_force[env_ids] = (
        torch.rand(num, device=self.device) * (bf_hi - bf_lo) + bf_lo
    )
    self._per_env_break_torque[env_ids] = (
        torch.rand(num, device=self.device) * (bt_hi - bt_lo) + bt_lo
    )
```

**`__init__`에 추가할 버퍼:**
```python
self._per_env_break_force = torch.full((self.num_envs,), float(self.cfg.grasp_joint_break_force), device=self.device)
self._per_env_break_torque = torch.full((self.num_envs,), float(self.cfg.grasp_joint_break_torque), device=self.device)
```

**`_attach_grasp_fixed_joint_for_envs()`에서 참조 변경:**
```python
# 기존: break_force = float(self.cfg.grasp_joint_break_force)
# 변경: break_force = self._per_env_break_force[env_id].item()
```

### 2-7. Base Body Velocity 읽기 — `_compute_body_displacement()` 대신

이전 설계에서는 매 프레임 world pose delta를 body-frame으로 변환하는 `_compute_body_displacement()` 메서드가 필요했다. **실제 로봇(yubinnn11/lekiwi3, v3.0) 확인 결과 base state가 velocity(m/s, rad/s)이므로, sim에서도 velocity를 직접 읽는다.**

```python
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
```

**삭제된 것들:**
- `_compute_body_displacement()` 메서드 (pose delta 계산) — 불필요
- `self.prev_root_pos_w` 버퍼 — 불필요
- `self.prev_root_quat_w` 버퍼 — 불필요  
- `self.body_displacement` 버퍼 — 불필요
- `_reset_idx()`에서 prev_pos/prev_quat 초기화 — 불필요

### 2-8. `_compute_metrics()` — 변경

v8의 `_compute_metrics()`를 복사한 후 **search 관련만 제거**:

```python
def _compute_metrics(self) -> Dict[str, torch.Tensor]:
    # === Multi-object position 업데이트 (v8 그대로) ===
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

    # Object relative (v8 그대로)
    object_delta_w = self.object_pos_w - root_pos_w
    object_pos_b = quat_apply_inverse(root_quat_w, object_delta_w)
    object_dist = torch.norm(object_pos_b[:, :2], dim=-1)
    heading_object = torch.atan2(object_pos_b[:, 1], object_pos_b[:, 0])

    # Velocity (v8 그대로)
    lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
    ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
    lin_speed = torch.norm(lin_vel_b[:, :2], dim=-1)

    object_dir_b = object_pos_b[:, :2] / (object_dist.unsqueeze(-1) + 1e-6)
    vel_toward_object = (lin_vel_b[:, :2] * object_dir_b).sum(dim=-1)

    arm_pos = self.robot.data.joint_pos[:, self.arm_idx]
    arm_vel = self.robot.data.joint_vel[:, self.arm_idx]

    return {
        "root_pos_w": root_pos_w,
        "object_pos_b": object_pos_b,   # 3D (v8은 2D xy만 반환했음)
        "object_dist": object_dist,
        "heading_object": heading_object,
        "lin_vel_b": lin_vel_b,          # 3D
        "ang_vel_b": ang_vel_b,          # 3D
        "lin_speed": lin_speed,
        "vel_toward_object": vel_toward_object,
        "arm_pos": arm_pos,              # 6D (5 arm + 1 grip)
        "arm_vel": arm_vel,              # 6D
    }
    # 삭제된 것: home_xy_b, home_dist, heading_home, vel_toward_home,
    #           search_xy_b, search_dist, wheel_vel
```

### 2-9. `_apply_action()` — Action 순서 반전

```python
def _pre_physics_step(self, actions: torch.Tensor):
    self.prev_actions = self.actions.clone()
    self.actions = actions.clone().clamp(-1.0, 1.0)

def _apply_action(self):
    # ★ 새 순서: [arm5, grip1, base3]
    # indices 0:5 = arm joints, 5 = gripper, 6:9 = base
    arm_grip_action = self.actions[:, 0:6]   # 6D (arm5 + grip1)

    base_vx = self.actions[:, 6] * self.cfg.max_lin_vel
    base_vy = self.actions[:, 7] * self.cfg.max_lin_vel
    base_wz = self.actions[:, 8] * self.cfg.max_ang_vel

    # === Base → Kiwi IK → Wheel (v8 로직 그대로) ===
    body_cmd = torch.stack([base_vx, base_vy, base_wz], dim=-1)
    wheel_radps = body_cmd @ self.kiwi_M.T / self.wheel_radius

    vel_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
    vel_target[:, self.wheel_idx] = wheel_radps
    self.robot.set_joint_velocity_target(vel_target)

    # === Arm → Position Target (v8 arm_action_to_limits 로직 그대로) ===
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
```

### 2-10. `_get_observations()` — 30D Actor + 37D Critic

```python
def _get_observations(self) -> dict:
    metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()
    
    # ★ Body-frame velocity 직접 읽기 (displacement 계산 대신)
    base_body_vel = self._read_base_body_vel()  # (N, 3) — vx, vy, wz

    arm_pos = metrics["arm_pos"]       # 6D
    arm_vel = metrics["arm_vel"]       # 6D
    lin_vel = metrics["lin_vel_b"]     # 3D
    ang_vel = metrics["ang_vel_b"]     # 3D

    # rel_object: 3D body-frame (v8은 2D였음)
    rel_object = metrics["object_pos_b"]  # 3D

    # Contact: v8의 _contact_force_per_env() 반환값을 2채널로 확장
    # 현재 v8은 단일 스칼라 → 좌/우 동일값으로 채움
    # TODO: 물리적 좌/우 분리는 contact sensor 2개 필요, 지금은 동일값
    contact_force = self._contact_force_per_env()  # (N,) scalar
    contact_binary = (contact_force > float(self.cfg.grasp_contact_threshold)).float()
    contact_lr = torch.stack([contact_binary, contact_binary], dim=-1)  # (N, 2)

    # BBox / Category (v8 그대로)
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

    # === Actor Observation (30D) ===
    actor_obs = torch.cat([
        arm_pos[:, :5],             # [0:5]   arm joint pos 5D
        arm_pos[:, 5:6],            # [5:6]   gripper pos 1D
        base_body_vel,              # [6:9]   ★ base body velocity 3D (m/s, rad/s)
        lin_vel,                    # [9:12]  base linear vel 3D
        ang_vel,                    # [12:15] base angular vel 3D
        arm_vel,                    # [15:21] arm+grip joint vel 6D
        rel_object,                 # [21:24] object relative pos 3D
        contact_lr,                 # [24:26] contact L/R 2D
        bbox_norm,                  # [26:29] object bbox 3D
        cat_norm,                   # [29:30] object category 1D
    ], dim=-1)  # 총 30D

    self._cached_metrics = None

    # === Critic Observation (37D, AAC) ===
    # Actor 30D + bbox_full 3D (비정규화) + mass 1D + object_dist 1D
    #          + heading_object 1D + vel_toward_object 1D = 37D
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

    self._critic_obs = critic_obs   # ★ AAC wrapper가 state()로 참조
    return {"policy": actor_obs, "critic": critic_obs}
```

### 2-11. Grasp 판정 (FSM 없이 직접)

v8의 `_update_phase_state()`에서 grasp 관련 부분만 추출:

```python
def _update_grasp_state(self, metrics: Dict[str, torch.Tensor]):
    """Grasp 판정 — FSM 없이 직접 처리."""
    self.just_grasped[:] = False
    self.just_dropped[:] = False    # ★ 신규
    self.task_success[:] = False

    if not self._physics_grasp or self.contact_sensor is None:
        # Legacy proximity grasp
        can_grasp = (
            (metrics["object_dist"] < self.cfg.grasp_thresh)
            & (metrics["lin_speed"] < 0.35)
            & (~self.object_grasped)
        )
    else:
        # Physics-based grasp (v8 로직 그대로)
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

    # ★ Grasp break 감지 (신규 — fixed joint 파손 시 drop 판정)
    # break_force 초과로 fixed joint가 끊어지면 물체가 gripper에서 이탈하지만,
    # object_grasped는 자동으로 False가 되지 않는다. 수동으로 감지해야 한다.
    if self.object_grasped.any() and self._physics_grasp:
        # gripper TCP와 물체 사이 거리 체크
        grip_pos_w = self.robot.data.body_pos_w[:, self._gripper_body_idx]  # gripper body 위치
        obj_delta = self.object_pos_w - grip_pos_w
        grip_obj_dist = torch.norm(obj_delta, dim=-1)
        
        # 잡고 있다고 기록되어 있지만 실제로 물체가 멀어진 경우 → drop
        drop_detected = self.object_grasped & (grip_obj_dist > float(self.cfg.grasp_drop_detect_dist))
        if drop_detected.any():
            self.object_grasped[drop_detected] = False
            self.just_dropped[drop_detected] = True
            # 끊어진 joint 정리
            drop_ids = drop_detected.nonzero(as_tuple=False).squeeze(-1)
            if self._grasp_attach_mode == "fixed_joint":
                self._disable_grasp_fixed_joint_for_envs(drop_ids)

    # GRASP timeout (v8 로직)
    if self.cfg.grasp_timeout_steps > 0:
        # grasp 시도 중(물체 가까이 접근했지만 아직 못 잡음)인 환경 추적
        near_object = metrics["object_dist"] < self.cfg.approach_thresh
        in_grasp_zone = near_object & (~self.object_grasped)
        # grasp_entry_step 업데이트: 처음 grasp zone 진입 시점 기록
        first_entry = in_grasp_zone & (self.grasp_entry_step == 0)
        self.grasp_entry_step[first_entry] = self.episode_length_buf[first_entry]
        # timeout check
        grasp_elapsed = self.episode_length_buf - self.grasp_entry_step
        timed_out = in_grasp_zone & (self.grasp_entry_step > 0) & (grasp_elapsed > self.cfg.grasp_timeout_steps)
        # timeout된 환경은 grasp_entry_step 리셋 (재시도 가능)
        self.grasp_entry_step[timed_out] = 0
```

### 2-12. `_get_rewards()` — Approach + Grasp + Lift

```python
def _get_rewards(self) -> torch.Tensor:
    metrics = self._cached_metrics
    if metrics is None:
        metrics = self._compute_metrics()
        self._update_grasp_state(metrics)

    reward = torch.full((self.num_envs,), self.cfg.rew_time_penalty, device=self.device)

    # Effort penalty (v8 그대로)
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

    # Tanh proximity kernel (목표 가까울수록 강한 gradient로 수렴 가속)
    proximity_bonus = 1.0 - torch.tanh(
        metrics["object_dist"] / self.cfg.rew_proximity_tanh_sigma
    )
    reward += self.cfg.rew_proximity_tanh_weight * proximity_bonus

    self.prev_object_dist[:] = metrics["object_dist"]
    self.episode_reward_sum += reward
    return reward
```

### 2-13. `_get_dones()`

```python
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
                        print(f"  [Curriculum] dist: {old:.2f} → {self._curriculum_dist:.2f} (avg success: {avg:.2f})")
                        # 이전 난이도의 성공률이 남아 조기 재상승하는 것을 방지
                        self._curriculum_success_window.zero_()
                        self._curriculum_idx = 0

    return terminated, truncated
```

### 2-14. `_reset_idx()` — 변경

v8의 `_reset_idx()`를 복사한 후 수정:

```python
def _reset_idx(self, env_ids: torch.Tensor):
    super()._reset_idx(env_ids)
    num = len(env_ids)
    if num == 0:
        return

    # Grasp joint 해제 (v8 그대로)
    if self._physics_grasp and self._grasp_attach_mode == "fixed_joint":
        self._disable_grasp_fixed_joint_for_envs(env_ids)

    # === Root reset (v8 그대로) ===
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

    # === Joint reset (v8 그대로) ===
    joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # === Home / Object 설정 ===
    self.home_pos_w[env_ids] = default_root_state[:, :3]
    base_xy = self.home_pos_w[env_ids, :2]

    # ★ Curriculum: object_dist_max 대신 _curriculum_dist 사용
    self.object_pos_w[env_ids] = self._sample_targets_around(
        env_ids=env_ids,
        base_xy=base_xy,
        dist_min=self.cfg.object_dist_min,
        dist_max=self._curriculum_dist,   # ← 여기가 v8과 다름
        base_z=self.home_pos_w[env_ids, 2],
    )

    # === Multi-object hide/show (v8 그대로 전체 복사) ===
    if self._multi_object and len(self.object_rigids) > 0:
        chosen = torch.randint(0, self._num_object_types, (num,), device=self.device)
        self.active_object_idx[env_ids] = chosen
        self.object_bbox[env_ids] = self._catalog_bbox[chosen]
        self.object_category_id[env_ids] = self._catalog_category[chosen]
        self.object_pos_w[env_ids, 2] = self.home_pos_w[env_ids, 2] + torch.clamp(
            self.object_bbox[env_ids, 2] * 0.5, min=float(self.cfg.object_height),
        )
        # Hide all, show chosen (v8 코드 그대로)
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
        # ... v8 single-object 로직 그대로 ...
        pass

    # === Task 버퍼 리셋 ===
    self.object_grasped[env_ids] = False
    self.task_success[env_ids] = False
    self.just_grasped[env_ids] = False
    self.just_dropped[env_ids] = False    # ★ 신규
    self.prev_object_dist[env_ids] = 10.0
    self.grasp_entry_step[env_ids] = 0
    self.episode_reward_sum[env_ids] = 0.0
    self.actions[env_ids] = 0.0
    self.prev_actions[env_ids] = 0.0
    if self._action_delay_buf is not None:
        self._action_delay_buf[:, env_ids] = 0.0

    # === DR 적용 (v8 그대로) ===
    self._apply_domain_randomization(env_ids)
```

---

## 3. `lekiwi_skill3_env.py` — CarryAndPlace (신규)

### 3-1. 만드는 방법

`lekiwi_skill2_env.py`를 복사한 후 다음을 변경:

### 3-2. Config 변경

```python
@configclass
class Skill3EnvCfg(Skill2EnvCfg):  # Skill2 상속하여 공통 설정 재사용
    """CarryAndPlace 환경 설정."""

    observation_space: int = 29     # Skill-2의 30이 아님
    state_space: int = 36           # Critic용

    # Task
    return_thresh: float = 0.30
    place_dist_thresh: float = 0.05
    place_gripper_threshold: float = 0.3  # ★ 이 값 이상 열리면 의도적 place로 판정

    # Handoff
    handoff_buffer_path: str = ""   # pickle 파일 경로

    # Reward (CarryAndPlace 전용)
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
    curriculum_success_threshold: float = 1.0  # 비활성
```

### 3-3. Observation 변경 (29D Actor)

`_get_observations()`에서 `rel_object` + `contact_lr` 대신 `home_rel` + `grip_force`:

```python
def _get_observations(self) -> dict:
    metrics = self._cached_metrics if self._cached_metrics is not None else self._compute_metrics()
    
    # ★ Body-frame velocity 직접 읽기
    base_body_vel = self._read_base_body_vel()  # (N, 3) — vx, vy, wz

    arm_pos = metrics["arm_pos"]
    arm_vel = metrics["arm_vel"]
    lin_vel = metrics["lin_vel_b"]
    ang_vel = metrics["ang_vel_b"]

    # home_rel: home의 body-frame 상대 벡터 3D
    home_delta_w = self.home_pos_w - self.robot.data.root_pos_w
    home_rel = quat_apply_inverse(self.robot.data.root_quat_w, home_delta_w)

    # grip_force: 스칼라 1D
    contact_force = self._contact_force_per_env()
    grip_force = contact_force.unsqueeze(-1)  # (N, 1)

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
            home_rel = home_rel + torch.randn_like(home_rel) * or_noise

    # BBox / Category (v8 그대로)
    bbox_norm = self.object_bbox / float(self._bbox_norm_scale)
    cat_denom = max(int(self.cfg.num_object_categories) - 1, 1)
    cat_norm = (self.object_category_id / float(cat_denom)).unsqueeze(-1)

    actor_obs = torch.cat([
        arm_pos[:, :5],             # [0:5]   arm 5D
        arm_pos[:, 5:6],            # [5:6]   gripper 1D
        base_body_vel,              # [6:9]   ★ base body velocity 3D (m/s, rad/s)
        lin_vel,                    # [9:12]  base_lin_vel 3D
        ang_vel,                    # [12:15] base_ang_vel 3D
        arm_vel,                    # [15:21] arm+grip vel 6D
        home_rel,                   # [21:24] home relative 3D
        grip_force,                 # [24:25] grip force 1D
        bbox_norm,                  # [25:28] bbox 3D
        cat_norm,                   # [28:29] category 1D
    ], dim=-1)  # 총 29D

    self._cached_metrics = None

    # === Critic Observation (36D, AAC) ===
    # Actor 29D + obj_dimensions(3D) + obj_mass(1D) + gripper_rel_pos(3D) = 36D
    # gripper_rel_pos: object position relative to gripper body (world-frame)
    grip_pos_w = self.robot.data.body_pos_w[:, self._gripper_body_idx]
    gripper_rel_pos = self.object_pos_w - grip_pos_w  # (N, 3) — world frame
    obj_mass_per_env = self._catalog_mass[
        self.active_object_idx.clamp(max=len(self._catalog_mass) - 1)
    ].unsqueeze(-1)
    critic_extra = torch.cat([
        self.object_bbox,          # 3D (원본, 비정규화)
        obj_mass_per_env,          # 1D
        gripper_rel_pos,           # 3D
    ], dim=-1)  # 7D
    critic_obs = torch.cat([actor_obs, critic_extra], dim=-1)  # 36D
    self._critic_obs = critic_obs   # ★ AAC wrapper가 state()로 참조

    return {"policy": actor_obs, "critic": critic_obs}
```

### 3-3b. `_update_grasp_state()` 오버라이드 — 의도적 place 구분

Skill-3는 Skill-2의 `_update_grasp_state()`를 오버라이드하여 의도적 place를 구분한다:

```python
def _update_grasp_state(self, metrics):
    """Skill-3 전용: 의도적 place vs 비의도적 drop 구분."""
    self.intentional_placed[:] = False

    # 부모(Skill2) grasp 로직 먼저 실행 (can_grasp, lift, drop detection)
    super()._update_grasp_state(metrics)

    # 그 후 intentional place 판정: gripper open + home 근처 → 의도적 place
    if self.object_grasped.any():
        gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
        gripper_open = gripper_pos > float(self.cfg.place_gripper_threshold)
        home_dist = metrics.get("home_dist", None)
        if home_dist is None:
            home_delta_w = self.home_pos_w - self.robot.data.root_pos_w
            home_pos_b = quat_apply_inverse(self.robot.data.root_quat_w, home_delta_w)
            home_dist = torch.norm(home_pos_b[:, :2], dim=-1)
        near_home = home_dist < self.cfg.return_thresh
        intentional = self.object_grasped & gripper_open & near_home
        if intentional.any():
            place_ids = intentional.nonzero(as_tuple=False).squeeze(-1)
            if self._physics_grasp and self._grasp_attach_mode == "fixed_joint":
                self._disable_grasp_fixed_joint_for_envs(place_ids)
            self.object_grasped[intentional] = False
            self.intentional_placed[intentional] = True
            # just_dropped은 False 유지 → place_success 조건 충족 가능
```

`__init__`에 추가할 버퍼:
```python
self.intentional_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
```

### 3-4. `_compute_metrics()` — home 관련 추가

Skill-2의 metrics에 home 관련 추가:
```python
# Skill-2의 metrics에 추가:
home_delta_w = self.home_pos_w - root_pos_w
home_pos_b = quat_apply_inverse(root_quat_w, home_delta_w)
home_dist = torch.norm(home_pos_b[:, :2], dim=-1)
heading_home = torch.atan2(home_pos_b[:, 1], home_pos_b[:, 0])

# return dict에 추가:
"home_dist": home_dist,
"heading_home": heading_home,
```

### 3-5. `_reset_idx()` — Handoff Buffer 기반

```python
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
    # Handoff Buffer에서 랜덤 샘플
    buf_size = len(self.handoff_buffer)
    indices = torch.randint(0, buf_size, (num,))

    # ★ 배치 텐서 생성 (per-loop torch.tensor 제거 — 2048 env 성능 개선)
    entries = [self.handoff_buffer[indices[i].item()] for i in range(num)]

    root_states = self.robot.data.default_root_state[env_ids].clone()
    joint_positions = self.robot.data.default_joint_pos[env_ids].clone()

    base_pos = torch.tensor([e["base_pos"] for e in entries], device=self.device, dtype=torch.float32)
    base_ori = torch.tensor([e["base_ori"] for e in entries], device=self.device, dtype=torch.float32)
    obj_pos = torch.tensor([e["object_pos"] for e in entries], device=self.device, dtype=torch.float32)
    obj_ori = torch.tensor(
        [e.get("object_ori", [1.0, 0.0, 0.0, 0.0]) for e in entries],
        device=self.device, dtype=torch.float32,
    )
    arm_joints = torch.tensor([e["arm_joints"] for e in entries], device=self.device, dtype=torch.float32)
    grip_states = torch.tensor([e["gripper_state"] for e in entries], device=self.device, dtype=torch.float32)
    obj_type_indices = torch.tensor([int(e["object_type_idx"]) for e in entries], device=self.device, dtype=torch.long)

    # 상대 좌표 → 절대 좌표 변환 (destination env의 origin 기준)
    env_origins = self.scene.env_origins[env_ids]  # (num, 3)
    base_pos = base_pos + env_origins
    obj_pos = obj_pos + env_origins

    # ★ Per-load noise: 같은 handoff entry라도 매번 다른 state
    if self.cfg.handoff_arm_noise_std > 0:
        arm_joints = arm_joints + torch.randn_like(arm_joints) * self.cfg.handoff_arm_noise_std
        if self.cfg.arm_action_to_limits:
            arm_lo = self.robot.data.soft_joint_pos_limits[0, self.arm_idx[:5], 0]
            arm_hi = self.robot.data.soft_joint_pos_limits[0, self.arm_idx[:5], 1]
            arm_joints = torch.clamp(arm_joints, arm_lo, arm_hi)
    if self.cfg.handoff_base_pos_noise_std > 0:
        base_pos[:, :2] = base_pos[:, :2] + torch.randn(num, 2, device=self.device) * self.cfg.handoff_base_pos_noise_std
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

    # ★ Single-object sim write (multi_object=False일 때)
    if not self._multi_object and self.object_rigid is not None:
        pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
        pose[:, :3] = obj_pos
        pose[:, 3:7] = obj_ori
        self.object_rigid.write_root_pose_to_sim(pose, env_ids)

    if self._multi_object:
        clamped_idx = obj_type_indices.clamp(max=len(self._catalog_bbox) - 1)
        self.object_bbox[env_ids] = self._catalog_bbox[clamped_idx]
        self.object_category_id[env_ids] = self._catalog_category[clamped_idx.clamp(max=len(self._catalog_category) - 1)]

    self.robot.write_root_state_to_sim(root_states, env_ids)
    joint_vel = torch.zeros_like(joint_positions)
    self.robot.write_joint_state_to_sim(joint_positions, joint_vel, env_ids=env_ids)

    # home = destination env의 origin (env_origin XY + robot Z)
    self.home_pos_w[env_ids, 0:2] = env_origins[:, 0:2]
    self.home_pos_w[env_ids, 2] = root_states[:, 2]

    # Multi-object hide/show (v8 코드 재사용, 물체 위치만 handoff 기반)
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
    # ... random yaw + joint init + home_pos + object placement ...

    # Single-object sim write (multi_object=False일 때)
    if not self._multi_object and self.object_rigid is not None:
        pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
        pose[:, :3] = self.object_pos_w[env_ids]
        self.object_rigid.write_root_pose_to_sim(pose, env_ids)

    self.object_grasped[env_ids] = True
    self._apply_domain_randomization(env_ids)
    self._attach_grasp_fixed_joint_for_envs(env_ids)
    self._finish_reset(env_ids, num)

def _finish_reset(self, env_ids: torch.Tensor, num: int):
    """공통 리셋 후처리."""
    self.task_success[env_ids] = False
    self.just_grasped[env_ids] = False
    self.just_dropped[env_ids] = False
    self.intentional_placed[env_ids] = False
    self.prev_home_dist[env_ids] = 10.0
    self.prev_object_dist[env_ids] = 10.0
    self.grasp_entry_step[env_ids] = 0
    self.episode_reward_sum[env_ids] = 0.0
    self.actions[env_ids] = 0.0
    self.prev_actions[env_ids] = 0.0
    if self._action_delay_buf is not None:
        self._action_delay_buf[:, env_ids] = 0.0
```

### 3-6. Reward (CarryAndPlace)

```python
def _get_rewards(self) -> torch.Tensor:
    # ★ _cached_metrics 사용 — _get_dones()에서 이미 _update_grasp_state() 실행됨
    # _update_grasp_state()를 다시 호출하면 just_dropped이 리셋되어 drop_penalty가 0이 됨
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

    # ★ Drop penalty — grasp break 감지 시 즉시 발동
    reward += self.just_dropped.float() * self.cfg.rew_drop_penalty

    # Place 성공: home 근처에서 의도적으로 놓음
    place_dist = torch.norm(self.object_pos_w[:, :2] - self.home_pos_w[:, :2], dim=-1)
    near_home = place_dist < self.cfg.return_thresh
    # 의도적 place = home 근처에서 gripper를 열어서 놓은 경우
    place_success = (~self.object_grasped) & near_home & (~self.just_dropped)
    reward += place_success.float() * self.cfg.rew_place_success_bonus
    self.task_success = place_success

    # Effort
    reward += self.cfg.rew_effort_weight * (self.actions[:, 6:9] ** 2).sum(dim=-1)
    reward += self.cfg.rew_arm_move_weight * (metrics["arm_vel"] ** 2).sum(dim=-1)

    self.prev_home_dist[:] = metrics["home_dist"]
    self.episode_reward_sum += reward
    return reward
```

### 3-7. `_get_dones()` — ★ Skill-3 전용 (Skill-2와 다름)

Skill-2의 `_get_dones()`를 복사하되, **drop 시 즉시 terminated** 처리가 핵심 차이:

```python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    metrics = self._compute_metrics()
    self._update_grasp_state(metrics)  # drop 감지 포함
    self._cached_metrics = metrics

    # ★ CRITICAL-2 Fix: place_success로 task_success 오버라이드
    # 부모(Skill-2)의 _update_grasp_state()가 lifted 물체에 대해
    # task_success=True를 설정하지만, Skill-3은 handoff buffer에서
    # 이미 grasped+lifted 상태로 시작하므로 즉시 True가 된다.
    # place_success(home 근처에서 의도적으로 놓음)를 사용해야 한다.
    place_dist = torch.norm(
        self.object_pos_w[:, :2] - self.home_pos_w[:, :2], dim=-1
    )
    near_home = place_dist < self.cfg.return_thresh
    place_success = (~self.object_grasped) & near_home & (~self.just_dropped)
    self.task_success = place_success

    root_pos = metrics["root_pos_w"]
    out_of_bounds = torch.norm(
        root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1
    ) > self.cfg.max_dist_from_origin
    env_z = self.scene.env_origins[:, 2] if hasattr(self.scene, "env_origins") else 0.0
    fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)

    # ★ Skill-3 핵심: drop → terminated (에피소드 즉시 종료, 페널티 학습)
    dropped = self.just_dropped

    terminated = out_of_bounds | fell | dropped

    time_out = self.episode_length_buf >= (self.max_episode_length - 1)
    truncated = self.task_success | time_out

    # Logging
    self.extras["task_success_rate"] = self.task_success.float().mean()
    self.extras["drop_rate"] = dropped.float().mean()

    return terminated, truncated
```

> **Skill-2 vs Skill-3 `_get_dones()` 차이:**
> - Skill-2: `terminated = out_of_bounds | fell` — drop 없음 (grasp 시도 중 재시도 허용). `task_success` = lift 성공 (Skill-2 목표).
> - Skill-3: `terminated = out_of_bounds | fell | dropped` — drop 즉시 종료 (운반 실패). `task_success` = `place_success`로 오버라이드 (부모의 lift 기반 task_success가 아닌 place 기반). Handoff buffer에서 이미 lifted 상태로 시작하므로 필수.

---

## 4. `models.py` 수정

기존 `PolicyNet`, `ValueNet`에 **orthogonal initialization** 추가 (PPO 37 implementation details 기반). `CriticNet` 추가:

```python
import math

def _ortho_init(module: nn.Module, gain: float = math.sqrt(2)):
    """Orthogonal initialization (PPO 37 implementation details)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
```

- `PolicyNet`: hidden layers `gain=sqrt(2)`, `mean_layer` `gain=0.01` (작은 초기 action)
- `ValueNet`/`CriticNet`: hidden layers `gain=sqrt(2)`, output layer `gain=1.0`

```python
class CriticNet(DeterministicMixin, Model):
    """Asymmetric Critic — Actor보다 넓은 observation을 받는다."""

    def __init__(self, observation_space, action_space, device,
                 critic_obs_dim=None, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        obs_dim = critic_obs_dim if critic_obs_dim is not None else observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        # Orthogonal init: hidden=sqrt(2), value output=1.0
        for m in list(self.net.children())[:-1]:
            if isinstance(m, nn.Linear):
                _ortho_init(m, gain=math.sqrt(2))
        _ortho_init(self.net[-1], gain=1.0)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
```

---

## 5. `train_lekiwi.py` 수정

### 5-1. argparse 추가

```python
parser.add_argument("--skill", type=str, default="approach_and_grasp",
                    choices=["navigate", "approach_and_grasp", "carry_and_place", "legacy"],
                    help="학습할 skill (navigate = Skill-1 장애물 회피, legacy = v8 monolithic env)")
parser.add_argument("--handoff_buffer", type=str, default=None,
                    help="Skill-3용 handoff buffer pickle 경로")
```

### 5-2. main() 환경 생성 분기

기존 `LeKiwiNavEnv, LeKiwiNavEnvCfg` import를 조건부로 변경:

```python
if args.skill == "navigate":
    from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg
    env_cfg = Skill1EnvCfg()
elif args.skill == "approach_and_grasp":
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
    env_cfg = Skill2EnvCfg()
elif args.skill == "carry_and_place":
    from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
    if not args.handoff_buffer:
        raise ValueError("--handoff_buffer required for carry_and_place")
    env_cfg = Skill3EnvCfg()
    env_cfg.handoff_buffer_path = os.path.expanduser(args.handoff_buffer)

env_cfg.scene.num_envs = args.num_envs
# ... 기존 calibration/dynamics/arm_limit/object 설정 그대로 적용 ...
# ★ Navigate는 grasp/contact/multi_object 관련 설정 불필요 (base-only)

if args.skill == "navigate":
    raw_env = Skill1Env(cfg=env_cfg)
elif args.skill == "approach_and_grasp":
    raw_env = Skill2Env(cfg=env_cfg)
elif args.skill == "carry_and_place":
    raw_env = Skill3Env(cfg=env_cfg)
else:
    raw_env = LeKiwiNavEnv(cfg=env_cfg)  # legacy v8
```

### 5-3. BC dim 검증

```python
if mode == "bc_finetune":
    bc_obs_dim = infer_bc_obs_dim(args.bc_checkpoint)
    expected = 20 if args.skill == "navigate" else (30 if args.skill == "approach_and_grasp" else 29)
    # ... 기존 mismatch 경고 로직 그대로 ...
```

### 5-4. AAC 통합

skrl 1.4.3는 native AAC를 지원하지 않으므로, 3개 파일(`aac_wrapper.py`, `aac_ppo.py`, `aac_trainer.py`)을 사용한다. Navigate(Skill-1), ApproachAndGrasp(Skill-2), CarryAndPlace(Skill-3) 모두 AAC를 사용한다. Navigate 전용 변경: `entropy_loss_scale=0.02` (더 많은 탐색), `clip_predicted_values=False`:

```python
from aac_wrapper import wrap_env_aac
from aac_ppo import AAC_PPO
from aac_trainer import AACSequentialTrainer
from models import CriticNet

wrapped = wrap_env_aac(raw_env)  # IsaacLabWrapper + state() monkey-patch
critic_obs_dim = wrapped._aac_state_space.shape[0] if wrapped._aac_state_space is not None else None

if use_aac and wrapped._aac_state_space is not None:
    models = {
        "policy": PolicyNet(wrapped.observation_space, wrapped.action_space, device),
        "value": CriticNet(
            wrapped.observation_space, wrapped.action_space, device,
            critic_obs_dim=critic_obs_dim,
        ),
    }
else:
    models = {
        "policy": PolicyNet(wrapped.observation_space, wrapped.action_space, device),
        "value": ValueNet(wrapped.observation_space, wrapped.action_space, device),
    }

# critic_state_preprocessor 설정 (AAC 사용 시에만)
cfg_ppo["critic_state_preprocessor"] = RunningStandardScaler
cfg_ppo["critic_state_preprocessor_kwargs"] = {"size": wrapped._aac_state_space, "device": device}

agent = AAC_PPO(models=models, memory=memory, cfg=cfg_ppo,
                observation_space=wrapped.observation_space,
                action_space=wrapped.action_space, device=device,
                critic_observation_space=wrapped._aac_state_space)

trainer = AACSequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent)
trainer.train()
```

---

## 6. `collect_demos.py` 수정

### 6-1. `extract_robot_state_9d()` 변경

v8 현재:
```python
def extract_robot_state_9d(env: LeKiwiNavEnv) -> torch.Tensor:
    arm_pos = env.robot.data.joint_pos[:, env.arm_idx]   # 6D
    wheel_vel = env.robot.data.joint_vel[:, env.wheel_idx]  # 3D
    return torch.cat([arm_pos, wheel_vel], dim=-1)
```

변경 후:
```python
def extract_robot_state_9d(env) -> torch.Tensor:
    """
    VLA용 robot_state 9D:
    [arm_pos(5), gripper_pos(1), base_body_vel_x(1), base_body_vel_y(1), base_body_vel_wz(1)]

    yubinnn11/lekiwi3 (v3.0) 포맷:
    [arm_shoulder_pan.pos, arm_shoulder_lift.pos, arm_elbow_flex.pos,
     arm_wrist_flex.pos, arm_wrist_roll.pos, arm_gripper.pos,
     x.vel, y.vel, theta.vel]
    
    단위: arm=rad, gripper=rad, base=m/s(x, y), rad/s(theta)
    단위 변환 불필요: sim과 real 모두 m/s, rad/s
    """
    arm_pos = env.robot.data.joint_pos[:, env.arm_idx]   # 6D (arm5 + grip1)
    
    # ★ body-frame velocity 직접 읽기 (NOT displacement, NOT wheel_angular_vel)
    vx_body = env.robot.data.root_lin_vel_b[:, 0:1]   # x.vel (m/s)
    vy_body = env.robot.data.root_lin_vel_b[:, 1:2]   # y.vel (m/s)
    wz_body = env.robot.data.root_ang_vel_b[:, 2:3]   # theta.vel (rad/s)
    base_body_vel = torch.cat([vx_body, vy_body, wz_body], dim=-1)  # (N, 3)
    
    return torch.cat([arm_pos, base_body_vel], dim=-1)  # 9D
```

### 6-2. Action 저장 시 gripper binary 변환

수집 루프에서 action 저장 직전:
```python
# 기존:
ep_act[i].append(action[i].cpu().numpy())

# 변경:
action_to_save = action[i].clone()
action_to_save[5] = 1.0 if action_to_save[5].item() > 0.5 else 0.0  # gripper binary
ep_act[i].append(action_to_save.cpu().numpy())
```

### 6-3. `--skill` 분기 추가

train_lekiwi.py와 동일한 패턴으로 환경 import 분기. Navigate(navigate), ApproachAndGrasp(approach_and_grasp), CarryAndPlace(carry_and_place) 모두 지원. Navigate 데이터 수집 시 arm은 TUCKED_POSE로 고정, gripper=open(1.0)으로 오버라이드하여 저장한다.

### 6-4. 카메라 서브클래스

VLA 데이터 수집 시 카메라 렌더링이 필요하므로, 환경에 TiltedCamera 2대(base_cam, wrist_cam)를 추가한 서브클래스를 `collect_demos.py` 내부에 정의한다:

```python
class Skill1EnvWithCam(Skill1Env):
    """Skill1Env + base_cam + wrist_cam 카메라 추가 (Navigate 데이터 수집용)."""
    def _setup_scene(self):
        super()._setup_scene()
        # TiltedCamera 2대 추가 (base_cam, wrist_cam)

class Skill2EnvWithCam(Skill2Env):
    """Skill2Env + base_cam + wrist_cam 카메라 추가."""
    def _setup_scene(self):
        super()._setup_scene()
        # TiltedCamera 2대 추가 (base_cam, wrist_cam)
        # prim path는 2-6 Isaac Sim 5.0.0 검증 결과 참조

class Skill3EnvWithCam(Skill3Env):
    """Skill3Env + base_cam + wrist_cam 카메라 추가."""
    def _setup_scene(self):
        super()._setup_scene()
        # TiltedCamera 2대 추가 (base_cam, wrist_cam)
```

RL 학습(`train_lekiwi.py`)에서는 카메라가 불필요하므로 기본 Skill2Env/Skill3Env를 사용하고, 데이터 수집(`collect_demos.py`)에서만 WithCam 서브클래스를 사용한다. 이렇게 분리하면 RL 학습 시 2048개 병렬 환경을 유지할 수 있다.

---

## 7. `convert_hdf5_to_lerobot_v3.py` 수정

### 7-1. 채널명 업데이트 (v3.0 포맷)

```python
# === 채널명: yubinnn11/lekiwi3 (LeRobot v3.0) 호환 ===
STATE_CHANNEL_NAMES = [
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
    "x.vel",
    "y.vel",
    "theta.vel",
]
```

### 7-2. 단위 변환 제거

```python
# === 삭제: m→mm 단위 변환 ===
# 이전 displacement 방식에서 필요했던 ×1000 변환 제거.
# sim velocity(m/s, rad/s) = real velocity(m/s, rad/s) — 동일 단위이므로 변환 불필요.

# 삭제된 코드:
# def convert_to_vla_units(data_9d: np.ndarray) -> np.ndarray:
#     out = data_9d.copy()
#     out[..., 6] *= 1000.0  # x: m → mm
#     out[..., 7] *= 1000.0  # y: m → mm
#     return out
```

### 7-3. robot_state 읽기

```python
# 새 skill env에서 HDF5에 robot_state 필드를 직접 저장하므로,
# 기존 infer_robot_state_from_obs() (obs[18:24]+obs[30:33]) 불필요
def read_robot_state(hdf5_episode):
    return hdf5_episode["robot_state"][:]  # shape (T, 9), 그대로 사용 (단위 변환 불필요)

# fallback: robot_state 필드가 없는 경우 obs에서 추론
def infer_robot_state_from_obs(obs: np.ndarray) -> np.ndarray:
    dim = obs.shape[-1]
    if dim == 20:  # Skill-1 Navigate: arm(5)+grip(1)+base_vel(3)+...
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1).astype(np.float32)
    elif dim == 30:  # Skill-2 ApproachAndGrasp
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1).astype(np.float32)
    elif dim == 29:  # Skill-3 CarryAndPlace
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1).astype(np.float32)
    # ... 기존 v8(33D/37D) 로직 ...
```

### 7-4. info.json 업데이트

```python
info_json["codebase_version"] = "v3.0"
info_json["robot_type"] = "lekiwi_client"

# 카메라 키:
# v8: "observation.images.base" → v3.0: "observation.images.front"
CAMERA_KEY_MAP = {
    "base_cam": "observation.images.front",   # ← 변경
    "wrist_cam": "observation.images.wrist",  # 동일
}
```

---

## 8. `generate_handoff_buffer.py` — 신규

```python
#!/usr/bin/env python3
"""
Skill-2 Expert 실행 → 성공 에피소드 종료 상태를 Handoff Buffer로 저장.

Usage:
    python generate_handoff_buffer.py \
      --checkpoint logs/ppo_skill2/best_agent.pt \
      --num_entries 500 --num_envs 64 \
      --output handoff_buffer.pkl \
      --multi_object_json object_catalog.json \
      --gripper_contact_prim_path "..." \
      --dynamics_json calibration/tuned_dynamics.json \
      --arm_limit_json calibration/arm_limits_real2sim.json \
      --headless
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_entries", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--output", type=str, default="handoff_buffer.pkl")
parser.add_argument("--noise_arm_std", type=float, default=0.05,
                    help="Arm joint 노이즈 std (rad). VLA의 부정확한 grasp 상태 모사")
parser.add_argument("--noise_obj_xy_std", type=float, default=0.02,
                    help="Object position 노이즈 std (m)")
parser.add_argument("--noise_base_xy_std", type=float, default=0.03,
                    help="Base position 노이즈 std (m)")
parser.add_argument("--noise_base_yaw_std", type=float, default=0.1,
                    help="Base orientation 노이즈 std (rad)")
parser.add_argument("--dynamics_json", type=str, default=None)
parser.add_argument("--calibration_json", type=str, default=None)
parser.add_argument("--arm_limit_json", type=str, default=None)
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np
import torch
from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
from models import PolicyNet, ValueNet, CriticNet
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from aac_wrapper import wrap_env_aac
from aac_ppo import AAC_PPO

def main():
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.calibration_json:
        env_cfg.calibration_json = os.path.expanduser(args.calibration_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
    if args.multi_object_json:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    # Curriculum을 최대로 열어서 다양한 거리 커버
    env_cfg.object_dist_min = 0.5
    env_cfg.curriculum_current_max_dist = env_cfg.object_dist_max

    env = Skill2Env(cfg=env_cfg)
    wrapped = wrap_env_aac(env)  # ★ AAC wrapper 사용
    device = wrapped.device

    critic_obs_dim = wrapped._aac_state_space.shape[0] if wrapped._aac_state_space is not None else None
    models = {
        "policy": PolicyNet(wrapped.observation_space, wrapped.action_space, device),
        "value": CriticNet(
            wrapped.observation_space, wrapped.action_space, device,
            critic_obs_dim=critic_obs_dim,
        ) if critic_obs_dim else ValueNet(wrapped.observation_space, wrapped.action_space, device),
    }
    memory = RandomMemory(memory_size=24, num_envs=args.num_envs, device=device)
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": wrapped.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    if critic_obs_dim and wrapped._aac_state_space is not None:
        cfg_ppo["critic_state_preprocessor"] = RunningStandardScaler
        cfg_ppo["critic_state_preprocessor_kwargs"] = {"size": wrapped._aac_state_space, "device": device}

    if critic_obs_dim and wrapped._aac_state_space is not None:
        agent = AAC_PPO(models=models, memory=memory, cfg=cfg_ppo,
                        observation_space=wrapped.observation_space,
                        action_space=wrapped.action_space, device=device,
                        critic_observation_space=wrapped._aac_state_space)
    else:
        agent = PPO(models=models, memory=memory, cfg=cfg_ppo,
                    observation_space=wrapped.observation_space,
                    action_space=wrapped.action_space, device=device)
    agent.load(args.checkpoint)
    agent.set_running_mode("eval")

    entries = []
    obs, _ = env.reset()
    print(f"\n  Collecting {args.num_entries} handoff entries...")

    while len(entries) < args.num_entries:
        with torch.no_grad():
            action = agent.act({"states": obs["policy"]}, timestep=0, timesteps=1)[0]
        obs, _, terminated, truncated, _ = env.step(action)

        success = env.task_success
        if success.any():
            sids = success.nonzero(as_tuple=False).squeeze(-1)
            for sid in sids:
                i = sid.item()
                # Active object의 orientation 읽기
                oi = int(env.active_object_idx[i].item())
                if env._multi_object and oi < len(env.object_rigids):
                    obj_quat = env.object_rigids[oi].data.root_quat_w[i].cpu().tolist()
                elif env.object_rigid is not None:
                    obj_quat = env.object_rigid.data.root_quat_w[i].cpu().tolist()
                else:
                    obj_quat = [1.0, 0.0, 0.0, 0.0]

                # env_origin 기준 상대 좌표로 저장 (다른 env에 로드해도 정상 동작)
                origin = env.scene.env_origins[i]
                entry = {
                    "base_pos": (env.robot.data.root_pos_w[i] - origin).cpu().tolist(),
                    "base_ori": env.robot.data.root_quat_w[i].cpu().tolist(),
                    "arm_joints": env.robot.data.joint_pos[i, env.arm_idx[:5]].cpu().tolist(),
                    "gripper_state": env.robot.data.joint_pos[i, env.arm_idx[5]].item(),
                    "object_pos": (env.object_pos_w[i] - origin).cpu().tolist(),
                    "object_ori": obj_quat,
                    "object_type_idx": env.active_object_idx[i].item(),
                }

                # ★ Noise injection — VLA의 부정확한 Skill-2 출력을 모사
                # Skill-3가 "완벽하지 않은 grasp 상태"에서도 복구/운반할 수 있도록 학습
                if args.noise_arm_std > 0:
                    entry["arm_joints"] = [
                        v + np.random.normal(0, args.noise_arm_std) for v in entry["arm_joints"]
                    ]
                if args.noise_obj_xy_std > 0:
                    entry["object_pos"][0] += np.random.normal(0, args.noise_obj_xy_std)
                    entry["object_pos"][1] += np.random.normal(0, args.noise_obj_xy_std)
                if args.noise_base_xy_std > 0:
                    entry["base_pos"][0] += np.random.normal(0, args.noise_base_xy_std)
                    entry["base_pos"][1] += np.random.normal(0, args.noise_base_xy_std)
                if args.noise_base_yaw_std > 0:
                    # quaternion [w,x,y,z]에 yaw 노이즈 추가
                    w, x, y, z = entry["base_ori"]
                    cur_yaw = 2.0 * np.arctan2(z, w)
                    new_yaw = cur_yaw + np.random.normal(0, args.noise_base_yaw_std)
                    entry["base_ori"] = [np.cos(new_yaw/2), 0.0, 0.0, np.sin(new_yaw/2)]
                
                entries.append(entry)

        prev_milestone = (len(entries) - sids.numel()) // 50
        curr_milestone = len(entries) // 50
        if curr_milestone > prev_milestone and len(entries) > 0:
            print(f"    {len(entries)}/{args.num_entries}")

    entries = entries[:args.num_entries]
    with open(args.output, "wb") as f:
        pickle.dump(entries, f)
    print(f"\n  ✅ Saved {len(entries)} entries to {args.output}")

    env.close()
    sim_app.close()

if __name__ == "__main__":
    main()
```

---

## 9. 실행 순서

```bash
cd ~/IsaacLab/scripts/lekiwi_nav_env
conda activate env_isaaclab && source ~/isaacsim/setup_conda_env.sh

# ── Step 0: Skill-1 Navigate RL (from scratch, BC 불필요) ──
python train_lekiwi.py --num_envs 2048 --skill navigate \
  --max_iterations 3000 --headless
# obs: actor=20D, critic=25D, 장애물 회피 + 감속 정지 학습

# ── Step 0b: Skill-1 Navigate 데이터 수집 (RL Expert rollout) ──
python collect_demos.py --checkpoint logs/ppo_navigate/best_agent.pt \
  --skill navigate --num_demos 1000 --num_envs 4

# ── Step 1: Skill-2 테스트 (환경 동작 확인) ──
python -c "
from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
cfg = Skill2EnvCfg()
cfg.scene.num_envs = 4
env = Skill2Env(cfg=cfg)
obs, _ = env.reset()
print('obs shape:', obs['policy'].shape)  # 기대: (4, 30)
import torch
action = torch.zeros(4, 9)
obs, r, term, trunc, info = env.step(action)
print('step ok, reward:', r)
env.close()
"

# ── Step 2: Skill-2 텔레옵 (10~20 데모) ──
python record_teleop.py --num_demos 20 --skill approach_and_grasp \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "..." \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json

# ── Step 3: Skill-2 BC ──
python train_bc.py --demo_dir demos_skill2/ --epochs 200 --expected_obs_dim 30

# ── Step 4: Skill-2 RL ──
python train_lekiwi.py --num_envs 2048 --skill approach_and_grasp \
  --bc_checkpoint checkpoints/bc_skill2.pt \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "..." \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json --headless

# ── Step 5: Handoff Buffer 생성 ──
python generate_handoff_buffer.py \
  --checkpoint logs/ppo_skill2/best_agent.pt \
  --num_entries 500 --num_envs 64 \
  --output handoff_buffer.pkl \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "..." \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json --headless

# ── Step 6: Skill-3 (BC→RL) ──
# (Skill-2와 동일 패턴, --skill carry_and_place --handoff_buffer handoff_buffer.pkl)

# ── Step 7: VLA 데이터 수집 ──
python collect_demos.py --checkpoint logs/ppo_skill2/best_agent.pt \
  --skill approach_and_grasp --num_demos 1000 --num_envs 4 ...
```

---

## 9-A. Isaac Sim 5.0.0 환경 검증 결과 (2026-02-19)

실제 환경에서 검증 완료. 코드 작성 시 아래 사항을 전제로 한다.

**API 존재 확인:**
| API | 존재 | shape (num_envs=1) | 비고 |
|-----|------|---------------------|------|
| `robot.data.root_lin_vel_b` | ✅ | `(1, 3)` | body-frame linear velocity — fallback 불필요 |
| `robot.data.root_ang_vel_b` | ✅ | `(1, 3)` | body-frame angular velocity — fallback 불필요 |
| `robot.data.root_lin_vel_w` | ✅ | `(1, 3)` | world-frame (참고용) |
| `robot.data.body_pos_w` | ✅ | `(1, 40, 3)` | 3D tensor: `[:, body_idx]` → `(N, 3)` |

**breakForce 메커니즘:**
- `PhysxSchema.PhysxJointAPI.CreateBreakForceAttr()` ❌ — 이 API는 존재하지 않음
- `UsdPhysics.Joint(prim).CreateBreakForceAttr(30.0)` ✅ — 이것이 정상 API
- v8 L870에서 이미 이 방식 사용 중 → codefix의 구현 그대로 적용 가능

**Gripper body:**
- USD Inspector 순서: `Moving_Jaw_08d_v1` = index [6]
- `find_bodies()` 반환: `ids=[39]` — Isaac Lab이 articulation tree를 재정렬
- **반드시 `find_bodies()`로 동적 취득** (index 하드코딩 금지)

**USD Joint Limits:**
- arm joint 6개 + gripper 1개 + wheel 3개 + roller 30개 = 전부 `(-inf, inf)`
- RL: `arm_limit_write_to_sim=True`로 실측 범위 강제, 텔레옵: `False` (USD 기본 리밋 사용, 그리퍼 완전 닫힘 허용)
- 실측 완료: `calibration/arm_limits_measured.json` (2026-02-21), tucked pose 제약 적용됨

**wrist_roll 기어비:**
- 실물 wrist_roll에 약 1.8:1 기어 증폭 존재 (서보 1rad → 손목 ~1.8rad)
- USD 관절은 기어비 미모델링 → `isaac_teleop.py`의 `SIGNS[4]=1.814`로 보상
- 측정: SIGNS=1.0에서 그리퍼 좌→우(180°) 회전, leader_raw Δ=1.7321 → `π / 1.7321 = 1.814`
- wrist_roll 캘리브레이션(LEADER_REST) 미완 — 리더↔LeKiwi 페어 캘리브레이션 시 wrist_roll 미측정(자동 입력). 리더↔sim offset 보정 필요 시 `isaac_teleop.py`의 `LEADER_REST_RAD6[4]`만 조정 (리더↔LeKiwi 페어 캘리브레이션 파일은 수정 금지)

**Contact sensor:** USD에 미포함. Isaac Lab `ContactSensorCfg`로 코드에서 동적 생성 (v8 방식 유지).

**카메라 prim path:**
- base: `/World/LeKiwi/base_plate_layer1_v5/Realsense/RSD455/Camera_OmniVision_OV9782_Color`
- wrist: `/World/LeKiwi/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera`

## 10. 검증 체크리스트

- [ ] `Skill2Env` : `num_envs=4`로 reset/step 에러 없이 동작
- [ ] `obs["policy"].shape[-1] == 30` (Skill-2), `29` (Skill-3)
- [ ] `_read_base_body_vel()`이 매 step 합리적 값 반환 (이동 시 ~0.01~0.3 m/s, 정지 시 ~0)
- [ ] Action `[0:5]` = arm, `[5]` = gripper, `[6:9]` = base 확인
- [ ] grasp 판정이 정상 동작 (물체 잡히면 `object_grasped=True`)
- [ ] ★ grasp break 감지: 물체를 잡은 후 로봇을 급가속하면 `object_grasped`가 `False`로 전환되는지
- [ ] ★ `just_dropped`가 break 감지 순간에만 True, 다음 step에서 False로 리셋되는지
- [ ] ★ break_force DR: 에피소드마다 `_per_env_break_force` 값이 15~45N 범위에서 변하는지
- [ ] Curriculum: 초기 거리 0.5m → 성공률 70% 초과 시 확대
- [ ] `Skill3Env` : handoff buffer 로드 후 물체가 잡힌 상태로 시작
- [ ] ★ handoff buffer 노이즈: 동일 entry를 여러 번 로드했을 때 arm_joints 값이 매번 다른지
- [ ] ★ Skill-3에서 drop 발생 시 `rew_drop_penalty`가 정상 적용되는지
- [ ] ★ Skill-3에서 home 근처 의도적 place와 mid-carry drop이 구분되는지
- [ ] ★ RL: `arm_limit_write_to_sim: True` 상태에서 팔이 몸체를 관통하지 않는지
- [x] ★ 텔레옵: `arm_limit_write_to_sim: False` 상태에서 그리퍼 완전 닫힘 동작 확인 (2026-02-21)
- [x] ★ 텔레옵: wrist_roll SIGNS=1.814에서 리더암↔sim 1:1 회전 확인 (2026-02-21)
- [ ] `collect_demos.py`의 robot_state가 `[arm6, base_body_vel3]` = 9D
- [ ] 저장된 action의 `action[5]`가 0.0 또는 1.0 (gripper binary)
- [ ] `convert_hdf5_to_lerobot_v3.py` 출력의 채널명이 `x.vel, y.vel, theta.vel`
- [ ] `convert_hdf5_to_lerobot_v3.py`에서 base 단위 변환이 **없음** (m/s 그대로)
- [ ] `info.json`의 `codebase_version`이 `v3.0`
- [ ] 카메라 키가 `observation.images.front` (NOT `base`)
- [ ] displacement 관련 코드가 없음: `_compute_body_displacement`, `prev_root_pos_w`, `prev_root_quat_w`, `body_displacement` 버퍼, m→mm 변환
- [ ] ★ AAC: `aac_wrapper.py`의 `state()` 메서드가 `env._critic_obs`를 반환하는지
- [ ] ★ AAC: `aac_ppo.py`의 `_update()` 내에서 `critic_states` tensor가 memory에 저장되는지
- [ ] ★ AAC: `train_lekiwi.py`에서 `CriticNet` critic_obs_dim이 Skill-2=37, Skill-3=36으로 설정되는지
- [ ] ★ AAC: `generate_handoff_buffer.py`에서 AAC checkpoint 로드 후 eval 정상 동작하는지
- [ ] ★ Intentional place: Skill-3에서 home 근처 gripper open 시 `intentional_placed=True`, `just_dropped=False` 확인
- [ ] ★ Camera subclass: `collect_demos.py`의 `Skill2EnvWithCam`/`Skill3EnvWithCam`에서 카메라 2대 렌더링 정상
- [ ] ★ `deploy_vla_action_bridge.py --action_format v6`이 `[arm5,grip1,base3]` 순서로 파싱하는지
- [ ] ★ CRITICAL-1: `grasp_gripper_threshold=0.7` — 일반 물체(0.4~0.8 rad) 파지 시 `gripper_closed` 판정 통과하는지
- [ ] ★ CRITICAL-2: Skill-3 에피소드가 1 step에서 종료되지 않는지 (handoff buffer 상태에서 episode_length > 10 확인)
- [ ] ★ HIGH-1: `aac_ppo.py`의 `compute_gae()`에서 `next_values` 파라미터가 사용되는지 (`last_values` closure가 아닌)
- [ ] ★ Multi-object gripper shaping: `close_target` clamp bounds가 `min=0.0, max=1.0`인지 (이전 `min=-1.0, max=0.0` 아닌)

---

## 11. 알려진 제한사항 및 향후 개선

### 11-1. Arm Dynamics 캘리브레이션 미적용

현재 `tune_sim_dynamics.py`는 바퀴 파라미터만 CMA-ES로 튜닝한다. **팔 stiffness/damping은 DR만 걸려 있고 캘리브레이션되지 않는다.** Skill-2의 마지막 20cm grasp 구간에서 sim-real gap이 발생할 수 있다.

**권장 조치** (Skill-2 학습 후, sim2real 테스트 시):
1. `replay_in_sim.py`로 실제 로봇의 arm trajectory를 sim에서 재생
2. arm joint position/velocity 오차가 크면 `tune_sim_dynamics.py`를 확장하여 arm 파라미터도 CMA-ES 대상에 포함
3. 최소한 `compare_real_sim.py`에서 arm trajectory 비교 플롯 추가

### 11-2. Contact Sensor 한계

현재 contact sensor는 단일 스칼라를 좌/우 동일값으로 복사하여 2채널로 사용한다. 비대칭 grasp(한쪽 finger만 닿은 상태) 감지가 불가능하다. 실제 로봇에서 grasp 품질 문제가 발생하면 contact sensor를 좌/우 finger에 각각 배치하는 것을 검토한다.

### 11-3. AAC 구현 ✅ 완료

skrl 1.4.3의 native AAC 미지원 문제는 3개 파일(`aac_wrapper.py`, `aac_ppo.py`, `aac_trainer.py`)로 해결하였다. `train_lekiwi.py`와 `generate_handoff_buffer.py` 모두 AAC를 사용하도록 업데이트 완료. Skill-2 Critic 37D, Skill-3 Critic 36D.

### 11-4. Skill 전환 로직

sim에서는 각 Skill을 독립적으로 학습하지만, 실제 배포 시 Skill-1→2→3 전환 시점 판단은 VLA 또는 `deploy_vla_action_bridge.py`(`--action_format v6/legacy` 지원)가 담당한다. 전환 로직(예: "gripper가 닫히고 N step 유지되면 Skill-3로 전환")의 설계와 검증은 VLA 배포 단계에서 별도로 진행해야 한다.

---

## Audit v2 수정 사항 (2026-02-20)

### CRITICAL: Handoff Buffer 좌표계 버그 수정

**문제**: `generate_handoff_buffer.py`가 절대 world-frame 좌표로 저장하고, `Skill3Env._reset_from_handoff`가 home을 `(0,0,z)`로 하드코딩. `env_spacing=10.0`일 때 env 0 외의 entry는 `out_of_bounds`로 즉시 종료 → Skill-3 학습 사실상 불가.

**수정**:
- `generate_handoff_buffer.py`: `env.scene.env_origins[i]`를 빼서 상대 좌표로 저장
- `lekiwi_skill3_env.py _reset_from_handoff`: `self.scene.env_origins[env_ids]`를 더해서 절대 좌표로 복원, `home_pos_w`를 `env_origins[:, 0:2]`로 설정

### HIGH: Skill-3 `_action_delay_buf` 리셋 누락

**문제**: Skill-3가 부모 `_reset_idx`를 건너뛰면서 `_action_delay_buf` 초기화도 누락. 이전 에피소드의 stale action이 새 에피소드 첫 step에서 실행되어 `break_force=30N` 초과 → spurious drop.

**수정**: `_finish_reset()`에 추가:
```python
if self._action_delay_buf is not None:
    self._action_delay_buf[:, env_ids] = 0.0
```

### LOW (방어적): Skill-3 Object velocity 미초기화

**문제**: `_reset_from_handoff`에서 `write_root_velocity_to_sim` 미호출.

**판단**: FixedJoint가 constraint solver를 통해 잔여 속도를 즉시 소거하므로 실질적 영향 없음. 방어적으로 코드에 추가는 했지만 학습 결과에 영향을 주지 않음.

**수정**: Single-object/multi-object 경로 모두에 `write_root_velocity_to_sim(zero_vel, env_ids)` 추가 (방어적).

### HIGH: Curriculum window 오염

**문제**: 난이도 상승 후 `_curriculum_success_window` 미초기화. 이전 (쉬운) 난이도의 높은 성공률이 남아 즉시 연쇄 상승 → 0.5m에서 2.5m(최대)으로 직행.

**수정**: 난이도 상승 직후 window와 idx 초기화:
```python
self._curriculum_success_window.zero_()
self._curriculum_idx = 0
```

### 비적용 항목

다음 항목은 검토 결과 실제 문제가 아니므로 수정하지 않음:

- **`prev_object_dist=10.0`**: 상수 오프셋으로 모든 에피소드가 동일하게 +0.2 클램핑 → PPO advantage 정규화에 의해 상쇄. 정책 gradient에 영향 없음.
- **`clip_predicted_values=True`**: 하이퍼파라미터 선택이지 버그가 아님. PPO 37 논문이 비활성화를 권장하지만, 성능 차이는 환경에 따라 다르고 학습이 깨지지 않음.
