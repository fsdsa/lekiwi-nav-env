# Navigate (Skill1) 수정 사양서 v2

## 배경 및 전체 아키텍처

### 전체 파이프라인
```
Qwen VLM (0.3Hz, Task Planner)
  │
  │ 카메라 이미지 + task instruction
  │ → 상황 판단 → 이산 command 또는 스킬 전환
  │
  ├── Navigate: 이산 command (전진/후진/좌이동/우이동/좌회전/우회전)
  ├── Approach & Grasp: displacement 또는 스킬 트리거
  └── Carry & Place: displacement 또는 스킬 트리거

단일 VLA 모델, 3 스킬 파인튜닝
Action space 통일: 9D = [arm(5), gripper(1), base_vx, base_vy, base_wz]
```

### Navigate 단계 역할
- VLM이 0.3Hz로 6개 이산 command 중 하나를 선택
- VLA가 command + 카메라 + robot_state를 받아 9D 연속 action 출력
- Navigate 시 arm ≈ 0 (tucked), base만 활성
- 정지 타이밍은 VLM이 결정 (VLA가 아님)
- displacement 사용하지 않음 (탐색 중 목표를 지나칠 수 있으므로)

### VLA 학습 전략 (privileged teacher → vision student 증류)
- Sim에서 RL expert는 **lidar** (privileged info)로 장애물 감지하여 회피 학습
- RL expert로 데이터 수집 시 **카메라(RGB+D)도 함께 녹화**
- VLA는 카메라만 입력으로 받아 동일 행동을 모방 학습
- 실물에 lidar 없어도 VLA가 카메라로 장애물 보고 피하게 됨

---

## 수정 사항

### 1. Lidar FOV를 D455 depth FOV에 맞추기

**현재:**
```
360° / 8 rays = 45° 간격, 전방위
```

**변경:**
```
~87° 전방 FOV / 8 rays ≈ 11° 간격, 전방만
- D455 Depth FOV: 수평 ~87°
- robot forward = body +Y 방향
- lidar rays를 body +Y 중심으로 ±43.5° 범위에 8개 균등 배치
```

**이유:**
VLA는 전방 카메라만 사용하므로, RL expert도 전방만 감지해야 증류 시 혼란 없음.
360° lidar로 학습하면 카메라 FOV 밖 장애물 때문에 행동하는 경우가 생기고,
VLA가 이를 재현할 수 없음.

---

### 2. FOV 기반 장애물 보상 (핵심 변경)

**현재 문제:**
장애물은 360° 전방위에 스폰되는데, collision/proximity reward가 **모든 장애물** 기준으로 계산됨.
→ RL이 관측할 수 없는 뒤쪽 장애물에 대해 페널티를 받으면 학습 신호가 혼란스러움.
→ "아무것도 안 보이는데 갑자기 -2.0" → 정책 불안정 또는 보수적 수렴.

**변경: 보상은 FOV 내 장애물만, 종료는 모든 충돌**

```
FOV 내 장애물 (lidar가 감지 가능):
  - collision penalty: -2.0 (충돌 시)
  - proximity penalty: -0.5 × closeness (가까울수록)
  → RL이 "보이는 장애물은 내 책임" 학습

FOV 밖 장애물 (lidar가 감지 불가):
  - collision penalty: 없음 (보이지 않으므로 페널티 불공정)
  - proximity penalty: 없음
  - 물리적 충돌 시: episode termination만
  → "네 잘못은 아니지만 실패는 실패"
  → 간접적으로 후방 이동 최소화 학습
```

**구현 방법:**

_compute_metrics()에서 두 종류의 최소 거리를 계산:

```python
# 1. FOV 내 장애물만 필터링 (reward용)
#    각 장애물이 robot 기준 ±43.5° (FOV/2) 안에 있는지 체크
obstacle_angle_body = atan2(delta_y_body, delta_x_body)  # body frame 기준
in_fov = abs(obstacle_angle_body - π/2) < fov_half_rad   # +Y(전방) 중심
obs_dist_fov = where(in_fov & valid, surface_dist, inf)
min_obs_dist_fov = obs_dist_fov.min(dim=-1)  # → collision/proximity reward에 사용

# 2. 모든 장애물 (termination용)
obs_dist_all = where(valid, surface_dist, inf)
min_obs_dist_all = obs_dist_all.min(dim=-1)  # → termination 판단에 사용
```

**reward 함수 수정:**
```python
# collision & proximity: FOV 내만
collision_fov = (min_obs_dist_fov < collision_dist).float()
rew_collision = rew_collision_weight * collision_fov

proximity_fov = clamp(1.0 - min_obs_dist_fov / lidar_max_range, min=0.0)
rew_proximity = rew_proximity_weight * proximity_fov
```

**termination 수정:**
```python
# 어떤 장애물이든 충돌하면 에피소드 종료
any_collision = min_obs_dist_all < collision_dist
terminated = out_of_bounds | fell | any_collision
```

**이유:**
- 보상은 "관측 가능한 것"에 대해서만 → 학습 가능한 신호
- 종료는 "물리적 사실"에 대해서 → 후방 이동 자체를 간접적으로 억제
- 후진 페널티(rew_backward)와 결합하면, RL이 자연스럽게 "뒤로 갈 일 있으면 회전 후 전진" 패턴 학습

---

### 3. 후진 페널티

**설정:** `rew_backward = -0.3`

```python
# body vy < 0 = backward movement
backward_speed = clamp(-body_vel_y, min=0.0)
rew_backward = -0.3 * backward_speed
```

**이유:**
- 전방 카메라만 있으므로 후진 시 뒤를 볼 수 없음
- RL이 학습하는 행동: "뒤로 가야 하는 상황 → 회전해서 전방 확인 후 전진"
- FOV 밖 충돌 termination과 시너지: 후진 자체를 줄이면 뒤쪽 충돌도 자연히 감소

---

### 4. 장애물 환경 구성

**장애물은 360° 전방위에 스폰 (변경 없음)**

실제 집에서 장애물은 모든 방향에 있으므로 현실적.
reward만 FOV 기반으로 제한하고, 스폰은 그대로 유지.

**에피소드별 장애물 다양성:**
```
장애물 없음 (obstacle_none_prob=0.3):  ~30%
장애물 적음 (1~3개):                  ~40%
장애물 많음 (4~8개):                  ~30%
```

**장애물 배치 다양성:**
- 정면 장애물 → 크게 우회 학습
- 측면 장애물 → 살짝 피함 학습
- 좁은 통로 → 정밀 통과 학습
- 장애물 없음 → 직진 유지 학습 (BC 행동 보존)

**robot 시작 위치 30cm, 타겟 오브젝트 40cm 이내에는 장애물 미배치 (기존 유지)**

---

### 5. Navigate는 이산 command 유지 (변경 없음)

displacement 사용하지 않음.

**이유:**
- 탐색 목적은 "물체 찾기"이지 "특정 거리 이동"이 아님
- displacement 기반이면 목표 거리를 채우는 동안 탐색 대상을 지나칠 수 있음
- VLM이 0.3Hz로 "전진" 유지하다가 타겟 발견 시 "정지" 또는 다음 스킬 전환

---

### 6. Critic observation 수정

Critic extra에 FOV 기반 장애물 정보 사용:

```
Critic 25D = Actor 20D + [
  speed(1),
  direction_compliance(1),
  closest_fov_obstacle_dist(1),    ← FOV 내 최소 거리
  closest_fov_obstacle_angle(1),   ← FOV 내 최근접 장애물 각도
  time_remaining(1)
]
```

---

## 변경하지 않는 것

- obs 구조 (Actor 20D, Critic 25D 차원 동일)
- direction_cmd (6방향 이산 유지)
- action space (9D 연속 유지)
- BC 데이터/학습 코드 (기존 epoch250 그대로 사용)
- 기존 reward 중: track_lin, track_ang, action_smoothness, time_penalty (그대로)
- 장애물 스폰 로직 (360° 전방위, 랜덤)
- DR (domain randomization) 설정

---

## 학습 파이프라인

### Step 1: BC (기존 것 재사용)
- P-controller 데이터, 장애물 없는 환경
- direction_cmd → base velocity 매핑만 학습
- lidar 값 거의 1.0이었으므로 사실상 lidar 무시
- 재학습 불필요

### Step 2: Residual RL 재학습 (이것만 하면 됨)
- 기존 BC 위에서 장애물 회피 보정값 학습
- 장애물 있는 환경, 새로운 FOV-aware reward 적용
- `최종 action = BC_action + residual_action × scale`
- 장애물 없으면 residual ≈ 0 (BC 그대로)
- 장애물 보이면 살짝 우회하는 보정값 출력

### Step 3: VLA 파인튜닝용 데이터 수집
- Step 2의 expert를 다양한 장애물 환경에서 실행
- 기록 데이터: (camera_rgb, camera_depth, robot_state, command, 9D_action)
- 같은 "전진" command라도 장애물 유무에 따라 action이 다르게 기록됨

### Step 4: VLA 파인튜닝
- 카메라(RGB or RGB-D) + language + robot_state → 9D action
- sim lidar → camera 증류 완료

---

## 수정 체크리스트

### Skill1Env (lekiwi_skill1_env.py)
- [x] Lidar rays: 360° → D455 FOV 87° 전방 ±43.5°, 8 rays (이미 반영 확인)
- [x] _compute_metrics(): min_obs_dist_fov (FOV 내), min_obs_dist_all (전체) 두 개 계산
- [x] _get_rewards(): collision/proximity reward에 min_obs_dist_fov 사용
- [x] _get_dones(): termination에 min_obs_dist_all 사용 (terminate_on_any_collision)
- [x] _get_observations(): critic extra에 FOV 기반 장애물 정보 사용
- [x] 후진 페널티 rew_backward = -0.3 (이미 반영 확인)
- [x] cfg에 terminate_on_any_collision: bool = True 추가

### train_resip.py (navigate 분기)
- [x] Navigate reward에서 collision/proximity 계산 시 FOV 기반 거리 사용
- [x] env.env._compute_metrics()에서 min_obs_dist_fov 키 사용
- [x] 로깅에 collision_fov_rate, collision_any_rate 분리 추가
- [x] 기존 NAV_W_COLLISION, NAV_W_PROXIMITY, NAV_W_BACKWARD 값은 그대로

---

## 핵심 설계 원칙 요약

```
1. 관측 가능한 것만 보상한다 (FOV 내 → reward)
2. 물리적 사실은 종료로 반영한다 (모든 충돌 → termination)
3. 관측 불가능한 위험은 간접적으로 억제한다 (후진 페널티 → 뒤쪽 충돌 감소)
4. 위 3가지가 결합되면 RL이 자연스럽게 학습하는 행동:
   - 전방 장애물 → 직접 회피
   - 뒤로 갈 일 → 회전 후 전진
   - 결과: 항상 카메라가 이동 방향을 바라봄 → VLA 증류 시 카메라 입력으로 재현 가능
```
