1. 장애물 회피 문제
이건 솔직히 현재 파이프라인의 가장 큰 구멍이다.
문서에서 이 문제를 어떻게 다루는지 보면, Safety Layer(Jetson, 100Hz)가 D455 depth로 전방 30cm 이내 장애물을 감지하면 emergency stop을 건다. 그런데 이건 **"부딪히기 직전에 멈추는 것"**이지 **"돌아가는 것"**이 아니다. 멈춘 다음에 어떻게 하느냐가 문제인데, VLM이 3초 후에 이미지를 보고 "장애물이 앞에 있으니 왼쪽으로 돌아가라" 같은 지시를 내려줄 수 있다고 가정하고 있다.
근데 문제는 VLA가 그 지시를 받아도 회피 행동을 학습한 적이 없다는 것이다. Navigate 스크립트 데이터에는 장애물 자체가 없으니까, "왼쪽으로 돌아가라"는 지시를 받으면 왼쪽으로 회전은 할 수 있겠지만, 장애물을 보면서 적절한 거리를 두고 우회하는 행동은 훈련 데이터에 한 번도 없다. VLA의 pretrained 지식에서 일반화되길 기대하는 건데, 그건 보장이 안 된다.
문서에서 Navigate 환경 요구사항으로 "바닥/벽 텍스쳐 DR + distractor 가구 배치"를 언급하긴 하는데, 이건 시각적 다양성을 위한 것이지 가구를 피해 다니는 행동을 학습시키는 게 아니다. 스크립트 정책은 가구가 있든 없든 목표 방향으로 직진한다.
2. 물체까지 적절한 거리에서 멈추는 문제
이건 실제로 설계에서 어느 정도 커버가 되어 있다. 다만 "멈춘다"는 개념 자체가 좀 다르게 동작한다.
핵심은 Navigate가 물체 앞에서 정확히 멈출 필요가 없다는 구조다. 흐름을 보면:
VLM이 이미지에서 물체의 시각적 크기를 보고 판단
  → 물체가 작게 보임 → "navigate toward the red cup" (Navigate 계속)
  → 물체가 충분히 크게 보임 → "approach and grasp the red cup" (ApproachAndGrasp로 전환)
ApproachAndGrasp는 curriculum으로 0.7m~2.5m 범위에서 base+arm 동시 접근을 학습했으니까, VLM이 대략 그 범위 안에서 instruction을 전환해주면 된다. VLM의 전환 판단이 정밀하지 않아도 되는 이유가 이거다 — ApproachAndGrasp가 커버하는 범위가 넓어서.
그런데 여기서 진짜 걱정해야 할 건 다른 거다. Navigate 스크립트 데이터에서 물체에 접근하는 행동의 "마지막 구간"이 어떤 모습이냐는 거다. 스크립트 proportional controller는 물체에 가까워질수록 속도가 줄어드는 게 아니라, gain에 따라 계속 밀고 들어간다. 실제 데이터를 보면 물체에 거의 부딪히는 수준까지 가거나, 아니면 에피소드가 끝나거나 할 텐데, 이런 데이터로 학습하면 VLA가 Navigate 모드에서 감속 없이 물체에 돌진하는 행동을 배울 수 있다.
물론 VLM이 그 전에 instruction을 ApproachAndGrasp로 바꿔줄 거라 기대하지만, VLM의 판단이 0.5초만 늦어도 로봇이 물체를 이미 밀어버릴 수 있다.
정리하면
우려현재 설계의 대응충분한가장애물 회피Safety Layer(emergency stop) + VLM 지시불충분 — 학습 데이터에 회피 행동 자체가 없음적절한 거리에서 전환VLM 시각 판단 + ApproachAndGrasp의 넓은 커버 범위(0.5~2.5m)대체로 괜찮음 — 단, Navigate의 감속 행동 부재가 리스크
현실적으로 첫 실기 테스트는 장애물 없는 열린 공간에서 할 가능성이 높고, 그 환경에서는 스크립트 Navigate로도 충분히 동작할 수 있다. 하지만 실제 가정집이나 사무실 같은 환경으로 가면 Navigate가 병목이 될 거라는 건 거의 확실하다. 그때 Navigate RL을 추가하든, 아니면 Navigate에 간단한 장애물 회피 규칙(depth 기반 reactive avoidance)을 스크립트에 넣어서 데이터를 다시 만들든 하는 후속 작업이 필요해질 거다.








# Skill-1 Navigate RL 환경 구현 가이드

> **[2026-02-22 업데이트] Direction-Conditioned RL로 전환됨**
>
> Navigate RL이 "목표물 접근" 방식에서 **"방향 명령 실행"** 방식으로 변경되었다.
> - VLM이 방향 명령(forward/backward/left/right/turn_left/turn_right) 제공
> - RL은 방향 명령을 받아 실행하면서 장애물을 회피하는 것을 학습
> - Observation [9:12]이 `rel_object_body` → `direction_cmd (cmd_vx, cmd_vy, cmd_wz)`로 변경
> - 보상: approach/arrival/heading/deceleration 제거 → direction_following(3.0) + collision(-2.0) + proximity(-0.5) + smoothness(-0.005)
> - 에피소드 종료: arrival 없음, timeout/OOB만
> - Skill-1→2 전환: VLM이 base cam으로 물체 인식 + 0.7m 이내일 때 판단 (RL이 아닌 VLM 레벨)
>
> 아래 내용은 구 설계(target-seeking)의 참고 자료로 보존됨.

> **목적**: Navigate를 Script Policy에서 RL Expert로 전환한다. 장애물 회피 + VLM 방향 명령 실행을 RL이 학습하도록 환경 `lekiwi_skill1_env.py`를 만들고, 기존 파이프라인을 업데이트한다.
>
> **수정 금지 파일**: `lekiwi_robot_cfg.py`, `spawn_manager.py`, `calibration_common.py`, 모든 calibration/comparison 스크립트, `build_object_catalog.py`, `leader_to_home_tcp_rest_matched_with_keyboard_base.py`

---

## 0. 설계 요약

### 왜 RL인가

기존 Script Policy(P-controller)는 장애물 회피를 못하고, 물체 근접 감속이 없고, 정지 행동 데이터가 없다. GT 기반 스크립트 개선으로는 VLA가 실제 카메라 이미지에서 장애물을 보고 회피하는 일반화를 보장할 수 없다. RL은 다양한 장애물 배치에서 최적의 회피+접근 전략을 학습하여 더 높은 품질의 VLA 학습 데이터를 생성한다.

### Navigate RL의 특징 (vs Skill-2/3)

| 항목 | Skill-2 ApproachAndGrasp | Skill-1 Navigate (Direction-Conditioned) |
|------|--------------------------|------------------------------------------|
| Arm | 능동 제어 (5D) | TUCKED_POSE 고정 (출력은 9D이나 arm 무시) |
| Gripper | 능동 제어 (continuous) | open 고정 (1.0) |
| 실질 action | base 3D + arm 5D + grip 1D | **base 3D만** |
| 핵심 난이도 | 접근 + 파지 | **방향 명령 추종 + 장애물 회피** |
| 장애물 | 없음 | **있음 (랜덤 cuboid 3~8개)** |
| 입력 | 물체 상대위치 (body frame) | **VLM 방향 명령 (6가지 cardinal)** |
| 성공 조건 | object grasped + lifted | **없음 (timeout까지 방향 추종 + 회피)** |
| 핵심 메트릭 | grasp 성공률 | **direction_compliance (95%+)** |
| BC warm-start | 필요 (teleop 10-20개) | **불필요** (3D action, 랜덤 탐색으로 충분) |
| 학습 예상 시간 | 1-2일 | **수 시간** (action space 작고, reward 단순) |

---

## 1. 새 파일: `lekiwi_skill1_env.py`

### 1-1. 전체 구조

`lekiwi_skill2_env.py`를 기반으로 만들되, 대폭 단순화한다.

```
lekiwi_skill1_env.py
├── Skill1EnvCfg (config)
├── Skill1Env (환경)
│   ├── __init__()
│   ├── _setup_scene()        — 로봇 + 물체 + 장애물 spawn
│   ├── _pre_physics_step()   — arm 강제 고정 + base action 적용
│   ├── _get_observations()   — Actor 20D + Critic 25D
│   ├── _get_rewards()        — direction_following + collision + proximity + smoothness
│   ├── _get_dones()          — timeout, out_of_bounds (no arrival)
│   ├── _reset_idx()          — 로봇/장애물 재배치 + 방향 명령 샘플링
│   ├── _compute_lidar_scan() — 8방향 pseudo-lidar (GT 기반)
│   └── _read_base_body_vel() — Skill-2와 동일
```

### 1-2. Config

```python
@configclass
class Skill1EnvCfg(DirectRLEnvCfg):
    """Navigate RL 환경 설정."""

    # === Simulation (Skill-2와 동일) ===
    sim: SimulationCfg = SimulationCfg(
        dt=0.02, render_interval=2,
        gravity=(0.0, 0.0, -9.81), device="cpu",
    )
    decimation: int = 2
    episode_length_s: float = 10.0  # 방향 명령 실행, 도착 조건 없음

    # === Scene (Skill-2와 동일) ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=10.0, replicate_physics=True,
    )

    # === Robot (Skill-2와 동일) ===
    robot_cfg = LEKIWI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # === Spaces ===
    observation_space: int = 20    # Actor obs
    action_space: int = 9          # 9D (arm+grip 무시, base만 사용)
    state_space: int = 25          # Critic obs (AAC)

    # === Action (Skill-2와 동일한 스케일) ===
    max_lin_vel: float = 0.5
    max_ang_vel: float = 3.0

    # === Calibration (Skill-2와 동일) ===
    calibration_json: str | None = None
    dynamics_json: str | None = None
    dynamics_apply_cmd_scale: bool = True
    arm_limit_json: str | None = None
    arm_limit_margin_rad: float = 0.0
    arm_limit_write_to_sim: bool = True

    # === Object Spawning (for camera collection; not used in RL obs/rewards) ===
    object_dist_min: float = 1.0   # 물체 최소 거리 (m, 카메라 수집용)
    object_dist_max: float = 4.0   # 물체 최대 거리 (m, 카메라 수집용)

    # === Obstacle ===
    num_obstacles_min: int = 3
    num_obstacles_max: int = 8
    obstacle_size_min: float = 0.15  # 한 변 최소 (m)
    obstacle_size_max: float = 0.5   # 한 변 최대 (m)
    obstacle_height_min: float = 0.15
    obstacle_height_max: float = 0.5
    obstacle_dist_min: float = 0.6   # 로봇으로부터 최소 거리 (m)
    obstacle_dist_max: float = 3.8   # 로봇으로부터 최대 거리 (m)
    collision_dist: float = 0.20     # 충돌 판정 거리 (m)

    # === Lidar ===
    lidar_num_rays: int = 8          # pseudo-lidar 방향 수
    lidar_max_range: float = 2.0     # 최대 감지 거리 (m)

    # === Reward ===
    rew_direction_weight: float = 3.0       # dot(cmd, vel_norm) — 방향 추종 (메인)
    rew_collision_penalty: float = -2.0     # 장애물 충돌 하드 페널티
    rew_obstacle_proximity_weight: float = -0.5  # 장애물 근접 소프트 페널티
    obstacle_proximity_safe_dist: float = 0.5    # 근접 페널티 시작 거리 (m)
    rew_action_smoothness: float = -0.005   # delta_action² 페널티

    # === Multi-Object (Skill-2와 동일) ===
    multi_object_json: str = ""
    object_usd: str = ""
    object_mass: float = 0.3
    object_scale: float = 1.0
    object_height: float = 0.03
    object_prim_path: str = "/World/envs/env_.*/Object"

    # === Domain Randomization ===
    # Skill-2와 동일한 dynamics DR 적용
    # 추가: 장애물 배치 DR (리셋마다 자동)
```

### 1-3. Observation 설계

**Actor Obs (20D):**

```
Index   Name                    Dim   Source                          비고
─────────────────────────────────────────────────────────────────────────
0-4     arm_joint_pos           5     robot.data.joint_pos[:, arm_idx[:5]]    고정값이지만 VLA 데이터 일관성
5       gripper_pos             1     robot.data.joint_pos[:, arm_idx[5]]     고정 1.0
6-8     base_body_vel           3     root_lin_vel_b[:, :2] + root_ang_vel_b[:, 2]
9-11    direction_cmd           3     VLM 방향 명령 (cmd_vx, cmd_vy, cmd_wz)
12-19   lidar_scan              8     8방향 pseudo-lidar (normalized, 0=장애물 접촉, 1=감지 범위 밖)
─────────────────────────────────────────────────────────────────────────
Total: 20D
```

**Direction Commands (6가지, +y = robot forward):**
| 명령 | cmd_vx | cmd_vy | cmd_wz |
|------|--------|--------|--------|
| forward | 0 | 1 | 0 |
| backward | 0 | -1 | 0 |
| strafe left | -1 | 0 | 0 |
| strafe right | 1 | 0 | 0 |
| turn left (CCW) | 0 | 0 | 1 |
| turn right (CW) | 0 | 0 | -1 |

**Critic Obs (25D, AAC privileged):**

```
Index   Name                    Dim   Source
─────────────────────────────────────────────────────────────────────────
0-19    actor_obs               20    위와 동일
20      speed                   1     선속도 크기 (m/s)
21      direction_compliance    1     dot(cmd, vel_normalized) — 방향 추종도
22      closest_obstacle_dist   1     가장 가까운 장애물까지 거리 (m, clamp max=5.0)
23      closest_obstacle_angle  1     가장 가까운 장애물의 body-frame 각도 (rad)
24      time_remaining          1     1.0 - (step / max_step) — 남은 시간 비율
─────────────────────────────────────────────────────────────────────────
Total: 25D
```

### 1-4. Action 처리

```python
def _pre_physics_step(self, actions: torch.Tensor):
    """
    actions: (N, 9) — [arm5, grip1, base3]
    
    Navigate에서는 arm과 gripper를 무시하고, base만 적용한다.
    """
    # Arm 강제 고정: TUCKED_POSE
    arm_target = self._tucked_pose.unsqueeze(0).expand(self.num_envs, -1)
    self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_idx[:5])
    
    # Gripper 강제 고정: open (1.0 → 실제 joint position)
    grip_target = torch.full((self.num_envs, 1), 1.0, device=self.device)
    self.robot.set_joint_position_target(grip_target, joint_ids=self.arm_idx[5:6])
    
    # Base: RL 출력의 [6:9] 사용
    base_vx = actions[:, 6] * self.cfg.max_lin_vel
    base_vy = actions[:, 7] * self.cfg.max_lin_vel
    base_wz = actions[:, 8] * self.cfg.max_ang_vel
    
    # Kiwi Drive IK (Skill-2와 동일한 함수 재사용)
    wheel_vel = self._kiwi_ik(base_vx, base_vy, base_wz)
    self.robot.set_joint_velocity_target(wheel_vel, joint_ids=self.wheel_idx)
    
    # Action delay (Skill-2와 동일한 1-step delay)
    if self._action_delay_buf is not None:
        # ... Skill-2와 동일한 delay 로직 ...
```

> **중요**: RL은 9D를 출력하지만 arm[0:5]와 grip[5]는 무시된다. RL은 빠르게 이 차원들이 영향 없음을 학습하고 base[6:9]에 집중한다. 이 방식의 장점은 VLA 데이터 수집 시 action format이 Skill-2/3과 동일하다는 것이다.

### 1-5. Pseudo-Lidar 구현

Isaac Sim의 raycast 대신 **GT 장애물 위치에서 기하학적으로 계산**한다. 2048 envs에서 빠르게 동작해야 하므로 벡터화 필수.

```python
def _compute_lidar_scan(self):
    """
    8방향 pseudo-lidar.
    각 방향으로 가장 가까운 장애물까지의 거리를 계산한다.
    
    Returns:
        scan: (N, 8) — normalized [0, 1]. 0=접촉, 1=범위 밖
    """
    N = self.num_envs
    num_rays = self.cfg.lidar_num_rays
    max_range = self.cfg.lidar_max_range
    
    # 로봇 위치/heading
    robot_xy = self.robot.data.root_pos_w[:, :2]          # (N, 2)
    robot_heading = self._get_robot_heading()               # (N,)
    
    # ray 방향 (body frame, 균등 분포)
    ray_angles = torch.linspace(0, 2 * math.pi, num_rays + 1, device=self.device)[:-1]  # (8,)
    # body → world 변환
    world_angles = ray_angles.unsqueeze(0) + robot_heading.unsqueeze(1)  # (N, 8)
    ray_dirs = torch.stack([torch.cos(world_angles), torch.sin(world_angles)], dim=-1)  # (N, 8, 2)
    
    # 모든 장애물에 대해 각 ray의 최소 거리 계산
    # obstacle_xy: (N, M, 2) — M = max_obstacles_per_env
    delta = self._obstacle_xy.unsqueeze(1) - robot_xy.unsqueeze(1).unsqueeze(2)  # (N, 1, M, 2) - broadcasting
    # 실제로는:
    delta = self._obstacle_xy - robot_xy.unsqueeze(1)  # (N, M, 2)
    
    # 각 obstacle에 대한 거리와 각도
    obs_dist = torch.norm(delta, dim=-1)  # (N, M)
    obs_angle = torch.atan2(delta[:, :, 1], delta[:, :, 0])  # (N, M)
    
    # 각 ray에 대해: ray 방향에 가까운 장애물의 최소 거리
    scan = torch.full((N, num_rays), max_range, device=self.device)
    
    for ray_idx in range(num_rays):
        ray_world_angle = world_angles[:, ray_idx]  # (N,)
        # 장애물과 ray 사이의 각도 차이
        angle_diff = obs_angle - ray_world_angle.unsqueeze(1)  # (N, M)
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        # ray 빔 폭 (±22.5° for 8 rays)
        beam_half_width = math.pi / num_rays
        in_beam = torch.abs(angle_diff) < beam_half_width  # (N, M)
        
        # valid 장애물만 고려
        valid = in_beam & self._obstacle_valid  # (N, M)
        dist_masked = torch.where(valid, obs_dist, torch.tensor(max_range, device=self.device))
        
        # 장애물 반경 보정 (cuboid 크기의 절반을 빼서 표면까지 거리)
        dist_to_surface = dist_masked - self._obstacle_radius.unsqueeze(0)  # (N, M) 대략적 보정
        
        scan[:, ray_idx] = dist_to_surface.min(dim=-1).values
    
    # normalize: [0, max_range] → [0, 1]
    scan = (scan.clamp(min=0.0) / max_range).clamp(max=1.0)
    
    return scan
```

> **최적화 노트**: 위 코드는 개념 설명용. 실제 구현에서는 for 루프를 제거하고 (N, 8, M) 텐서 연산으로 완전 벡터화해야 한다. 2048 envs × 8 rays × 8 obstacles = ~130K 연산이므로 GPU에서 충분히 빠르다.

### 1-6. 장애물 Spawn 및 관리

```python
def _setup_scene(self):
    """로봇 + 물체 + 장애물을 spawn한다."""
    # ── 로봇 (Skill-2와 동일) ──
    self.robot = Articulation(self.cfg.robot_cfg)
    self.scene.articulations["robot"] = self.robot
    
    # ── 물체 (Skill-2와 동일, spawn_manager 사용) ──
    # ... multi-object 또는 single object spawn ...
    
    # ── 장애물 (신규) ──
    # 최대 개수로 미리 spawn, 사용하지 않는 것은 범위 밖으로 이동
    self._max_obstacles = self.cfg.num_obstacles_max
    self._obstacle_prims = []
    
    for env_idx in range(self.cfg.scene.num_envs):
        for obs_idx in range(self._max_obstacles):
            prim_path = f"/World/envs/env_{env_idx}/Obstacle_{obs_idx}"
            # Cuboid 생성 (kinematic rigid body)
            # 초기 위치는 (0, 0, -10) — 화면 밖
            spawn_cuboid_obstacle(prim_path, position=(0, 0, -10))
            self._obstacle_prims.append(prim_path)
    
    # 장애물 상태 텐서
    self._obstacle_xy = torch.zeros(
        self.cfg.scene.num_envs, self._max_obstacles, 2, device=self.device
    )
    self._obstacle_radius = torch.zeros(
        self._max_obstacles, device=self.device
    )  # 대략적 반경 (충돌 판정용)
    self._obstacle_valid = torch.zeros(
        self.cfg.scene.num_envs, self._max_obstacles, 
        dtype=torch.bool, device=self.device
    )  # 활성 장애물 마스크
    self._num_active_obstacles = torch.zeros(
        self.cfg.scene.num_envs, dtype=torch.long, device=self.device
    )
```

```python
def _reset_obstacles(self, env_ids: torch.Tensor):
    """리셋된 env들의 장애물을 재배치한다."""
    for i in env_ids.cpu().tolist():
        origin = self.scene.env_origins[i].cpu()
        num_obs = random.randint(self.cfg.num_obstacles_min, self.cfg.num_obstacles_max)
        self._num_active_obstacles[i] = num_obs
        
        for j in range(self._max_obstacles):
            if j < num_obs:
                # 랜덤 위치 (polar)
                r = random.uniform(self.cfg.obstacle_dist_min, self.cfg.obstacle_dist_max)
                theta = random.uniform(0, 2 * math.pi)
                x = origin[0].item() + r * math.cos(theta)
                y = origin[1].item() + r * math.sin(theta)
                
                # 물체와 겹침 방지: 물체 위치에서 0.3m 이상
                obj_pos = self.object_pos_w[i, :2].cpu()
                while ((x - obj_pos[0].item())**2 + (y - obj_pos[1].item())**2) < 0.3**2:
                    theta = random.uniform(0, 2 * math.pi)
                    x = origin[0].item() + r * math.cos(theta)
                    y = origin[1].item() + r * math.sin(theta)
                
                sx = random.uniform(self.cfg.obstacle_size_min, self.cfg.obstacle_size_max)
                sy = random.uniform(self.cfg.obstacle_size_min, self.cfg.obstacle_size_max)
                sz = random.uniform(self.cfg.obstacle_height_min, self.cfg.obstacle_height_max)
                
                # prim 위치/크기 업데이트
                prim_idx = i * self._max_obstacles + j
                set_prim_transform(self._obstacle_prims[prim_idx], 
                                   position=(x, y, sz / 2), scale=(sx, sy, sz))
                
                self._obstacle_xy[i, j] = torch.tensor([x, y], device=self.device)
                self._obstacle_radius[j] = max(sx, sy) / 2  # 대략적 반경
                self._obstacle_valid[i, j] = True
            else:
                # 비활성: 범위 밖으로
                prim_idx = i * self._max_obstacles + j
                set_prim_transform(self._obstacle_prims[prim_idx],
                                   position=(0, 0, -10))
                self._obstacle_valid[i, j] = False
```

### 1-7. Reward 설계

```python
def _get_rewards(self) -> torch.Tensor:
    """
    Navigate reward:
    1. approach_progress: 물체에 가까워지면 + (Skill-2와 유사)
    2. arrival_bonus: 도착 시 +15
    3. collision_penalty: 장애물 충돌 시 -2
    4. speed_bonus: 목표 방향 속도 보상
    5. action_smoothness: 급격한 action 변화 패널티
    """
    # ── 1. Direction Following (메인 보상) ──
    # dot(cmd, vel_normalized): 명령 방향으로 빠르게 이동하면 +, 반대면 -
    compliance = metrics["direction_compliance"]
    rew_direction = self.cfg.rew_direction_weight * compliance

    # ── 2. Collision Penalty (하드) ──
    min_obs_dist = metrics["min_obs_dist"]
    collision = min_obs_dist < self.cfg.collision_dist
    rew_collision = torch.where(collision, self.cfg.rew_collision_penalty, 0.0)

    # ── 3. Obstacle Proximity (소프트) ──
    safe_dist = self.cfg.obstacle_proximity_safe_dist
    proximity_factor = (1.0 - min_obs_dist / safe_dist).clamp(0.0, 1.0)
    rew_proximity = self.cfg.rew_obstacle_proximity_weight * proximity_factor

    # ── 4. Action Smoothness ──
    delta_action = self.actions[:, 6:9] - self.prev_actions[:, 6:9]  # base만
    rew_smooth = self.cfg.rew_action_smoothness * (delta_action ** 2).sum(dim=-1)

    total = rew_direction + rew_collision + rew_proximity + rew_smooth

    # Logging
    self.extras["log"] = {
        "rew_direction": rew_direction.mean(),
        "rew_collision": rew_collision.mean(),
        "rew_speed": rew_speed.mean(),
        "rew_smooth": rew_smooth.mean(),
        "dist_to_target": curr_dist.mean(),
        "min_obstacle_dist": min_obs_dist.mean(),
        "direction_compliance": compliance.mean(),
        "collision_rate": collision.float().mean(),
        "avg_speed": metrics["lin_speed"].mean(),
    }

    return total
```

### 1-8. Termination

```python
def _get_dones(self):
    """
    Navigate (Direction-Conditioned) 종료 조건:
    - timeout: episode_length 초과 (10초)
    - out_of_bounds: env 범위 이탈
    - fell: 로봇 넘어짐

    NOTE: arrival 조건 없음 — 목표물이 아닌 방향 명령 실행이므로.
    collision도 terminate 하지 않음 — 패널티만 주고 계속 진행.
    """
    time_out = self.episode_length_buf >= (self.max_episode_length - 1)

    root_pos = self.robot.data.root_pos_w
    out_of_bounds = torch.norm(
        root_pos[:, :2] - self.home_pos_w[:, :2], dim=-1
    ) > self.cfg.max_dist_from_origin

    env_z = self.scene.env_origins[:, 2]
    fell = ((root_pos[:, 2] - env_z) < 0.01) | ((root_pos[:, 2] - env_z) > 0.5)

    terminated = out_of_bounds | fell
    truncated = time_out
    self.task_success[:] = False  # 방향 명령 모드에서는 task_success 없음

    return terminated, truncated
```

---

## 2. 수정 파일

### 2-1. `train_lekiwi.py` — `--skill navigate` 분기 추가

```python
# 기존 분기:
#   --skill approach_and_grasp → Skill2Env, obs=30, critic=37
#   --skill carry_and_place    → Skill3Env, obs=29, critic=36

# 추가:
#   --skill navigate           → Skill1Env, obs=20, critic=25

if args.skill == "navigate":
    from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg
    env_cfg = Skill1EnvCfg()
    env_cfg.observation_space = 20
    env_cfg.state_space = 25
    # BC warm-start 불필요 — 바로 PPO from scratch
    # 나머지는 Skill-2와 동일한 AAC + PPO 설정
```

**주의사항:**
- Navigate는 BC warm-start가 필요 없다. `--bc_checkpoint`가 없으면 from scratch 시작하도록 기존 로직에서 이미 처리됨.
- PPO 하이퍼파라미터는 Skill-2와 동일하게 시작. Action space가 작으므로 수렴이 빠를 것.
- `entropy_coef=0.005`: 6/9 action dims가 dead(arm/gripper 고정)이므로 entropy를 낮춤. dead dims (0:6) log_std는 -3.0에 고정(gradient zero).

### 2-2. `collect_demos.py` — `Skill1EnvWithCam` 추가

```python
# 기존: Skill2EnvWithCam, Skill3EnvWithCam

# 추가:
class Skill1EnvWithCam(Skill1Env):
    """Navigate RL 환경 + 카메라 2대 (base_cam + wrist_cam)."""
    
    def _setup_scene(self):
        super()._setup_scene()
        # base camera (D455) — Skill2EnvWithCam과 동일한 설정 복사
        self.base_cam = Camera(CameraCfg(
            prim_path="/World/envs/env_.*/Robot/.../Camera_OmniVision_OV9782_Color",
            width=1280, height=720, ...
        ))
        # wrist camera — Skill2EnvWithCam과 동일
        self.wrist_cam = Camera(CameraCfg(
            prim_path="/World/envs/env_.*/Robot/.../wrist_camera",
            width=640, height=480, ...
        ))
```

`--skill navigate` 분기 추가:
```python
if args.skill == "navigate":
    from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg
    EnvClass = Skill1EnvWithCam
    env_cfg = Skill1EnvCfg()
```

### 2-3. `collect_navigate_data.py` — RL Expert Rollout으로 전환

기존의 script policy 로직을 전부 제거하고, `collect_demos.py`와 동일한 RL expert rollout 구조로 교체한다.

**사실상 `collect_demos.py --skill navigate --checkpoint ...`로 대체 가능.** `collect_navigate_data.py`를 유지할지 제거할지는 선택이지만, 파이프라인 일관성을 위해 `collect_demos.py`에 통합하는 것을 권장한다.

```bash
# 기존 (script policy):
python collect_navigate_data.py --num_demos 1000 --num_envs 4 ...

# 변경 (RL expert):
python collect_demos.py --skill navigate \
    --checkpoint logs/ppo_navigate/best_agent.pt \
    --num_demos 2000 --num_envs 4 \
    --multi_object_json object_catalog.json \
    --dynamics_json calibration/tuned_dynamics.json
```

### 2-4. `models.py` — 변경 없음

PolicyNet, ValueNet, CriticNet 모두 obs_dim을 생성자에서 받으므로, Navigate의 20D/25D에 자동 대응. **수정 불필요.**

### 2-5. `convert_hdf5_to_lerobot_v3.py` — `infer_robot_state_from_obs()` 업데이트

```python
def infer_robot_state_from_obs(obs: np.ndarray) -> np.ndarray:
    dim = obs.shape[1]
    if dim == 20:  # Skill-1 Navigate: arm(0:6) + base_body_vel(6:9)
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1)  # 9D
    if dim == 30:  # Skill-2
        return np.concatenate([obs[:, 0:6], obs[:, 6:9]], axis=1)
    # ... 기존 ...
```

---

## 3. Navigate 데이터의 robot_state 및 action 저장

### 3-1. robot_state (9D, 다른 skill과 동일)

```
[arm_shoulder_pan.pos, arm_shoulder_lift.pos, arm_elbow_flex.pos,
 arm_wrist_flex.pos, arm_wrist_roll.pos, arm_gripper.pos,
 x.vel, y.vel, theta.vel]
```

- arm[0:5]: TUCKED_POSE 값 (거의 고정)
- gripper[5]: ~1.0 (open, 거의 고정)
- base[6:8]: body-frame linear velocity (m/s)
- base[8]: body-frame angular velocity (rad/s)

### 3-2. action (9D, v6 format)

```
[arm0, arm1, arm2, arm3, arm4, gripper, base_vx, base_vy, base_wz]
```

- arm[0:5]: RL 출력이지만 env에서 무시됨 → 거의 0 (학습 후)
- gripper[5]: RL 출력이지만 env에서 1.0으로 강제됨
- base[6:9]: RL이 실제로 제어하는 부분

**VLA 데이터 저장 시**: action 그대로 저장. arm 부분이 노이즈처럼 보일 수 있으므로, **저장 시 arm[0:5]를 TUCKED_POSE normalized 값으로, gripper[5]를 1.0으로 덮어쓰는 것을 권장.**

```python
# collect_demos.py에서 Navigate action 저장 시:
if skill == "navigate":
    action_to_save = action.clone()
    action_to_save[:, 0:5] = tucked_pose_normalized  # arm 고정값
    action_to_save[:, 5] = 1.0  # gripper open
    # base[6:9]는 RL 출력 그대로
```

### 3-3. Gripper Binary 변환

Navigate에서는 gripper가 항상 open이므로:
```python
action_to_save[:, 5] = 1.0  # 항상 open → binary도 1.0
```

---

## 4. Instruction 텍스트

VLA 학습 시 instruction text가 필요하다. Navigate는 direction-conditioned이므로, VLM이 생성하는 방향 명령과 일치하는 instruction을 사용한다.

```python
NAVIGATE_INSTRUCTIONS = {
    "forward":    ["move forward", "go straight ahead", "drive forward"],
    "backward":   ["move backward", "go back", "reverse"],
    "left":       ["move left", "strafe left", "go to the left"],
    "right":      ["move right", "strafe right", "go to the right"],
    "turn_left":  ["turn left", "rotate left", "turn counterclockwise"],
    "turn_right": ["turn right", "rotate right", "turn clockwise"],
}

SEARCH_INSTRUCTIONS = [
    "turn to search for the {object_name}",
    "look around for the {object_name}",
    "rotate to find the {object_name}",
]
```

에피소드 시작 시 랜덤 선택. 장애물 유무와 무관하게 동일한 instruction 사용 — VLA가 이미지에서 장애물을 보고 판단하도록.

---

## 5. 학습 파이프라인 업데이트

### 5-1. 실행 순서 변경

```
Phase 1 (RL Expert):
  [추가] Skill-1 Navigate RL → 90%+ arrival rate
  [기존] Skill-2 ApproachAndGrasp RL → 90%+ success
  [기존] Handoff Buffer 생성
  [기존] Skill-3 CarryAndPlace RL → 90%+ success

Phase 2 (VLA Data):
  [변경] Navigate RL expert rollout 1K-2K (기존: script policy)
  [기존] Skill-2 expert rollout 1K-10K
  [기존] Skill-3 expert rollout 1K-10K
```

### 5-2. 실행 명령어

```bash
# ── Navigate RL 학습 ──
python train_lekiwi.py --skill navigate --num_envs 2048 \
    --multi_object_json object_catalog.json \
    --dynamics_json calibration/tuned_dynamics.json --headless

# ── Navigate Expert 데모 수집 ──
python collect_demos.py --skill navigate \
    --checkpoint logs/ppo_navigate/best_agent.pt \
    --num_demos 2000 --num_envs 4 \
    --multi_object_json object_catalog.json \
    --dynamics_json calibration/tuned_dynamics.json

# ── LeRobot 변환 ──
python convert_hdf5_to_lerobot_v3.py \
    --input outputs/navigate_demos/*.hdf5 \
    --output_root ~/datasets/lekiwi_navigate_v3 \
    --repo_id yubinnn11/lekiwi3 --overwrite
```

---

## 6. Domain Randomization

### 6-1. 장애물 DR (리셋마다 자동)

- 장애물 수: 3~8개
- 위치: polar random (0.6~3.8m)
- 크기: 0.15~0.5m (각 변)
- 높이: 0.15~0.5m

### 6-2. Dynamics DR (Skill-2와 동일)

```python
# 리셋 시 적용:
wheel_stiffness: 0.75 ~ 1.5x
wheel_damping: 0.3 ~ 3.0x
wheel_friction: 0.7 ~ 1.3x
# arm은 고정이므로 arm DR 불필요
```

### 6-3. Observation Noise (Skill-2와 유사)

```python
base_vel_noise: ±0.02 m/s
lidar_noise: ±0.05 (normalized)
rel_object_noise: ±0.03 m
```

### 6-4. Visual DR (데이터 수집 시, Strong)

- 바닥/벽 텍스쳐 랜덤
- 조명 DR
- 장애물 색상/텍스쳐 랜덤
- distractor 물체 1~3개 (장애물과 별도)

---

## 7. 검증 체크리스트

### 환경 기본 동작

- [ ] `Skill1Env` : `num_envs=4`로 reset/step 에러 없이 동작
- [ ] `obs["policy"].shape[-1] == 20`
- [ ] `obs["critic"].shape[-1] == 25` (AAC 활성 시)
- [ ] Action `[0:5]` arm이 무시되고 TUCKED_POSE가 유지되는지
- [ ] Action `[5]` gripper가 무시되고 1.0 (open)이 유지되는지
- [ ] Action `[6:9]` base가 정상 적용되는지

### 장애물

- [ ] 장애물이 sim에서 보이고 물리 충돌이 작동하는지
- [ ] 장애물이 물체(Object)와 겹치지 않는지
- [ ] 장애물이 로봇 초기 위치와 겹치지 않는지
- [ ] 에피소드 리셋 시 장애물이 재배치되는지
- [ ] `_obstacle_valid` 마스크가 활성/비활성 장애물을 정확히 구분하는지

### Pseudo-Lidar

- [ ] `_compute_lidar_scan()` 출력 shape: `(N, 8)`
- [ ] 장애물 없는 방향은 값 ~1.0 (max range)
- [ ] 장애물 가까운 방향은 값 ~0.0
- [ ] 장애물이 로봇 뒤에 있을 때 전방 ray에 영향 없는지

### Reward

- [ ] 명령 방향으로 이동하면 `rew_direction > 0`
- [ ] 명령 반대 방향으로 이동하면 `rew_direction < 0`
- [ ] 장애물 충돌 시 `rew_collision = -2.0`
- [ ] 장애물 근접 시 `rew_proximity < 0` (soft, 0.5m 이내)
- [ ] `direction_compliance`가 학습 진행에 따라 증가하는지 (95%+ 목표)

### 데이터 저장

- [ ] `collect_demos.py --skill navigate`로 데이터 수집 가능한지
- [ ] robot_state가 9D `[arm6, base_body_vel3]`인지
- [ ] action이 9D v6 format이고, arm[0:5]가 고정값, grip[5]가 1.0인지
- [ ] 카메라 이미지가 정상 저장되는지 (base_cam, wrist_cam)
- [ ] LeRobot 변환 후 channel names가 `x.vel, y.vel, theta.vel`인지

### AAC

- [ ] `env._critic_obs`가 25D를 반환하는지
- [ ] `aac_wrapper.state()`가 정상 동작하는지
- [ ] `train_lekiwi.py --skill navigate`에서 CriticNet critic_obs_dim=25인지

---

## 8. 구현 순서

```
1단계: lekiwi_skill1_env.py 기본 골격
   - Skill1EnvCfg 작성
   - __init__, _setup_scene (로봇+물체만, 장애물 없이)
   - _pre_physics_step (arm 고정 + base 적용)
   - _get_observations (lidar 제외한 12D만)
   - _get_rewards (approach + arrival만)
   - _get_dones
   - _reset_idx
   → num_envs=4로 reset/step 동작 확인

2단계: 장애물 추가
   - _setup_scene에 장애물 spawn
   - _reset_obstacles
   - collision 감지 + penalty
   → 장애물 시각적 확인

3단계: Pseudo-Lidar
   - _compute_lidar_scan 구현
   - obs에 lidar 8D 추가 → Actor 20D 완성
   → lidar 값이 합리적인지 로그 확인

4단계: AAC Critic
   - Critic obs 25D 구현
   - _critic_obs 저장 로직
   → aac_wrapper.state() 동작 확인

5단계: train_lekiwi.py 통합
   - --skill navigate 분기
   - PPO 학습 실행
   → arrival_rate 추이 확인 (수 시간 내 50%+ 기대)

6단계: 데이터 수집 파이프라인
   - collect_demos.py에 Skill1EnvWithCam 추가
   - --skill navigate 분기
   - action 저장 시 arm/grip 덮어쓰기
   → HDF5 포맷 검증

7단계: LeRobot 변환 검증
   - convert_hdf5_to_lerobot_v3.py 업데이트
   - infer_robot_state_from_obs에 20D 케이스 추가
   → v3 dataset 구조 확인
```

---

## 9. 파일 변경 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `lekiwi_skill1_env.py` | **신규 생성** | Navigate RL 환경 (장애물 + pseudo-lidar) |
| `train_lekiwi.py` | 수정 | `--skill navigate` 분기 추가 |
| `collect_demos.py` | 수정 | `Skill1EnvWithCam` 서브클래스 + `--skill navigate` 분기 |
| `convert_hdf5_to_lerobot_v3.py` | 수정 | `infer_robot_state_from_obs()`에 20D 케이스 추가 |
| `collect_navigate_data.py` | **삭제 또는 deprecated** | `collect_demos.py --skill navigate`로 대체 |

**수정하지 않는 파일:**
- `lekiwi_skill2_env.py` — Skill-2 환경 변경 없음
- `lekiwi_skill3_env.py` — Skill-3 환경 변경 없음
- `models.py` — obs_dim 자동 대응, 변경 없음
- `aac_wrapper.py`, `aac_ppo.py`, `aac_trainer.py` — AAC 구현 변경 없음
- `train_bc.py` — Navigate는 BC 불필요
- `generate_handoff_buffer.py` — Navigate → Skill-2 handoff는 VLM이 처리
- 모든 수정 금지 파일

---

## 10. Skill-2와 코드 공유

`lekiwi_skill1_env.py`는 `lekiwi_skill2_env.py`에서 다음 함수/로직을 **그대로 복사**한다:

- `_read_base_body_vel()` — body-frame velocity 읽기
- `_kiwi_ik()` — Kiwi Drive IK
- `_apply_dynamics_dr()` — wheel dynamics DR
- `_load_calibration()` / `_load_dynamics()` — calibration 로딩
- `_apply_arm_limits()` — arm limits PhysX 반영
- `_action_delay_buf` 로직 — 1-step action delay
- Multi-object spawn 로직 (`spawn_manager.py` 호출)
- `_get_robot_heading()` — quaternion → yaw 변환

**복사하지 않는 것:**
- Grasp 관련 전체 (FixedJoint, break_force, contact sensor, object_grasped 등)
- Curriculum 관련 (Navigate에는 curriculum 불필요 — 거리 범위가 고정)
- Phase/FSM 관련 (이미 Skill-2에서도 제거됨)
- `_update_grasp_state()`, `_try_attach()`, `_release_joint()` 등

---

## 11. 핵심 설계 원칙

1. **Action space는 9D를 유지한다.** arm/grip은 무시되지만, VLA 데이터 포맷 통일을 위해 9D 출력. RL은 [6:9]만 의미있음을 빠르게 학습한다.

2. **Lidar는 GT 기반이다.** sim에서 GT 장애물 위치로 계산. VLA는 카메라 이미지에서 장애물을 보고 행동을 학습하므로, RL의 lidar obs가 GT여도 VLA 일반화에 문제 없다 (Skill-2의 rel_object obs도 GT).

3. **충돌은 terminate하지 않는다.** penalty만 줘서, 충돌 후 후진→우회하는 복구 행동도 학습하게 한다. 이 데이터가 VLA에 유용하다.

4. **BC warm-start는 불필요하다.** 3D 실질 action space에서 랜덤 탐색만으로 물체 도달 경험이 충분히 나온다. Holonomic base라서 어느 방향으로든 즉시 이동 가능.

5. **arrival_thresh = 0.7m = 핸드오프 지점 = Skill-2 curriculum 시작점.** Navigate가 여기까지 데려다주면, Skill-2가 인수받아 나머지 접근+파지를 처리한다.
