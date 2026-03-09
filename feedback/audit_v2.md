# Navigate (Skill1) 수정 사양서 v3

## v2 → v3 변경사항

**Direction command를 6개 → 4개로 축소 (strafe 제거)**

이유:
- 좌/우 strafe는 이동 방향이 카메라 FOV(전방 87°) 밖
- strafe 중 옆쪽 장애물을 감지할 수 없어 충돌 불가피
- 페널티로 억제하면 "command는 strafe인데 action은 안 움직임" → VLA 학습 시 모순적 신호
- Navigate에서 strafe 대신 "회전 + 전진" 조합으로 동일한 이동 가능
- 순수 strafe가 필요한 미세 정렬은 Skill2(Approach) 영역

---

## 배경 및 전체 아키텍처

### 전체 파이프라인
```
Qwen VLM (0.3Hz, Task Planner)
  │
  │ 카메라 이미지 + task instruction
  │ → 상황 판단 → command 또는 스킬 전환
  │
  ├── Navigate: 이산 command (전진/후진/좌회전/우회전) ← 4개
  ├── Approach & Grasp: displacement 또는 스킬 트리거
  └── Carry & Place: displacement 또는 스킬 트리거

단일 VLA 모델, 3 스킬 파인튜닝
Action space 통일: 9D = [arm(5), gripper(1), base_vx, base_vy, base_wz]
```

### Navigate 단계 역할
- VLM이 0.3Hz로 4개 이산 command 중 하나를 선택
- VLA가 command + 카메라 + robot_state를 받아 9D 연속 action 출력
- Navigate 시 arm ≈ 0 (tucked), base만 활성
- 정지 타이밍은 VLM이 결정 (VLA가 아님)
- displacement 사용하지 않음
- strafe는 Navigate에서 제거 (Skill2에서만 사용)

### VLA 학습 전략 (privileged teacher → vision student 증류)
- Sim에서 RL expert는 lidar (privileged info)로 장애물 감지하여 회피 학습
- RL expert로 데이터 수집 시 카메라(RGB+D)도 함께 녹화
- VLA는 카메라만 입력으로 받아 동일 행동을 모방 학습
- 실물에 lidar 없어도 VLA가 카메라로 장애물 보고 피하게 됨

---

## 수정 사항

### 1. Direction command 4개로 축소 (v3 신규)

**현재 (6개):**
```python
[0.0,  1.0, 0.0],   # forward
[0.0, -1.0, 0.0],   # backward
[-1.0, 0.0, 0.0],   # strafe left     ← 제거
[1.0,  0.0, 0.0],   # strafe right    ← 제거
[0.0,  0.0, 0.33],  # turn left CCW
[0.0,  0.0,-0.33],  # turn right CW
```

**변경 (4개):**
```python
_DIRECTION_COMMANDS = torch.tensor([
    [0.0,  1.0, 0.0],   # forward     — FOV 정중앙, 안전
    [0.0, -1.0, 0.0],   # backward    — FOV 밖, 페널티로 억제
    [0.0,  0.0, 0.33],  # turn left   — 제자리 회전, 안전
    [0.0,  0.0,-0.33],  # turn right  — 제자리 회전, 안전
], dtype=torch.float32)
```

**각 command와 FOV 안전성:**
```
전진:   이동방향 = FOV 중앙     → 장애물 감지 가능 ✓
후진:   이동방향 = FOV 반대편   → 감지 불가, 페널티로 억제 △
좌회전: 제자리 회전             → 충돌 위험 없음 ✓
우회전: 제자리 회전             → 충돌 위험 없음 ✓
```

**VLM 프롬프트도 4개로 변경:**
```
사용 가능한 command: forward, backward, turn_left, turn_right
"왼쪽으로 가고 싶으면 turn_left → forward 조합 사용"
```

**Note:**
- action space(9D)는 변경 없음. base_vx(좌우)는 여전히 출력 가능
- RL이 장애물 회피 시 살짝 옆으로 틀 수 있음 → 이건 command가 아니라 reactive avoidance
- strafe command가 없을 뿐, 물리적 횡이동이 완전 차단되는 건 아님

---

### 2. Lidar FOV를 D455 depth FOV에 맞추기

**현재:** 360° / 8 rays = 45° 간격, 전방위

**변경:** ~87° 전방 FOV / 8 rays ≈ 11° 간격, 전방만
- D455 Depth FOV: 수평 ~87°
- robot forward = body +Y 방향
- lidar rays를 body +Y 중심으로 ±43.5° 범위에 8개 균등 배치

---

### 3. FOV 기반 장애물 보상 (핵심)

**원칙:** 보상은 FOV 내 장애물만, 종료는 모든 충돌

```
FOV 내 장애물 (lidar 감지 가능):
  - collision penalty: -2.0
  - proximity penalty: -0.5 × closeness
  → "보이는 장애물은 내 책임"

FOV 밖 장애물 (lidar 감지 불가):
  - collision/proximity penalty: 없음
  - 물리적 충돌 시: episode termination만
  → "네 잘못은 아니지만 실패는 실패"
```

**_compute_metrics()에서 두 종류의 최소 거리 계산:**

```python
# 1. FOV 내 장애물만 (reward용)
delta_body = quat_apply_inverse(root_quat, obstacle_delta)  # body frame 변환
obs_angle_body = atan2(delta_body_y, delta_body_x)
in_fov = abs(obs_angle_body - π/2) < fov_half_rad
obs_dist_fov = where(valid & in_fov, surface_dist, inf)
min_obs_dist_fov = obs_dist_fov.min(dim=-1)

# 2. 모든 장애물 (termination용)
obs_dist_all = where(valid, surface_dist, inf)
min_obs_dist_all = obs_dist_all.min(dim=-1)
```

**_get_rewards():** min_obs_dist_fov만 사용
**_get_dones():** min_obs_dist_all로 termination

---

### 4. 후진 페널티

```python
rew_backward = -0.3 * clamp(-body_vel_y, min=0.0)
```

- 4 command 체계에서 후진은 유지하되 페널티로 억제
- RL이 학습: "뒤로 갈 일 있으면 회전 후 전진"
- 완전히 제거하지 않는 이유: 막다른 길에서 탈출 등 드물지만 필요한 경우 존재

---

### 5. 장애물 환경 구성

- 장애물은 360° 전방위에 스폰 (현실적)
- reward만 FOV 기반으로 제한

**에피소드별 장애물 다양성:**
```
장애물 없음 (obstacle_none_prob=0.3):  ~30%
장애물 적음 (1~3개):                  ~40%
장애물 많음 (4~8개):                  ~30%
```

---

### 6. Critic observation

```
Critic 25D = Actor 20D + [
  speed(1),
  direction_compliance(1),
  closest_fov_obstacle_dist(1),
  closest_fov_obstacle_angle(1),
  time_remaining(1)
]
```

---

## 변경하지 않는 것

- Actor obs 구조 (20D 차원 동일)
- Action space (9D 연속 유지)
- BC 데이터/학습 코드 (기존 epoch250 그대로 사용)
  - Note: BC가 6 command로 학습됐더라도, strafe command의 weight가 작으면 영향 미미
  - ResiP가 4 command 환경에서 재학습하므로 보정됨
- 기존 reward: track_lin, track_ang, action_smoothness, time_penalty
- 장애물 스폰 로직 (360° 전방위, 랜덤)
- DR 설정

---

## 학습 파이프라인

### Step 1: BC (기존 것 재사용)
- 기존 epoch250 checkpoint 그대로
- strafe command가 포함된 데이터로 학습됐지만, ResiP가 위에서 보정

### Step 2: Residual RL 재학습
- 4 command 환경, FOV-aware reward
- 장애물 있는 환경에서 회피 보정값 학습
- `최종 action = BC_action + residual_action × scale`

### Step 3: VLA 파인튜닝용 데이터 수집
- Step 2의 expert를 다양한 장애물 환경에서 실행
- 기록: (camera_rgb, camera_depth, robot_state, command, 9D_action)
- 4종류 command만 포함

### Step 4: VLA 파인튜닝
- 카메라(RGB or RGB-D) + language + robot_state → 9D action

---

## 수정 체크리스트

### Skill1Env (lekiwi_skill1_env.py)
- [x] _DIRECTION_COMMANDS: 6개 → 4개 (strafe 2개 제거)
- [x] _compute_metrics(): min_obs_dist_fov + min_obs_dist_all 분리
- [x] _get_rewards(): collision/proximity에 min_obs_dist_fov 사용
- [x] _get_dones(): termination에 min_obs_dist_all 사용
- [x] _get_observations(): critic extra에 FOV 기반 장애물 정보
- [x] Lidar FOV 87° 전방 (이미 반영)
- [x] 후진 페널티 rew_backward = -0.3 (이미 반영)
- [x] cfg에 terminate_on_any_collision: bool = True

### train_resip.py (navigate 분기)
- [x] collision/proximity 계산에 min_obs_dist_fov 사용
- [x] 로깅에 collision_fov_rate, collision_any_rate 분리
- [x] 기존 reward 파라미터 값 유지

### eval_resip.py (navigate)
- [x] direction schedule: 6 → 4 command
- [x] 장애물 시각화 (render_obstacles)
- [x] collision FOV/ALL 추적 로깅

### VLM 프롬프트 (추후)
- [ ] command 목록을 4개로 변경: forward, backward, turn_left, turn_right
- [ ] "횡이동이 필요하면 회전 + 전진 조합" 안내

---

## 핵심 설계 원칙 요약

```
1. 모든 이동은 FOV가 커버하는 방향으로 (전진 + 회전 조합)
2. 관측 가능한 것만 보상 (FOV 내 → reward)
3. 물리적 사실은 종료로 반영 (모든 충돌 → termination)
4. 관측 불가능한 위험은 간접 억제 (후진 페널티)
5. command와 action의 일관성 유지 (모순적 학습 신호 방지)

결과: RL → VLA 증류 시 깨끗한 데이터
  - command "전진" → 전방 이동 + 장애물 회피 (카메라로 재현 가능)
  - command "좌회전" → 제자리 회전 (카메라 불필요)
  - command "후진" → 거의 없음 (페널티 + termination으로 자연 억제)
```

---

## 학습 실행 명령어

```bash
python train_resip.py \
    --skill navigate \
    --bc_checkpoint checkpoints/dp_bc_nav/dp_bc_epoch250.pt \
    --num_envs 1024 \
    --num_env_steps 250 \
    --total_timesteps 3000000 \
    --action_scale_base 0.25 \
    --lr_actor 3e-4 \
    --lr_critic 5e-3 \
    --warmup_steps_initial 600 \
    --warmup_steps_final 0 \
    --warmup_decay_iters 30 \
    --eval_interval 3 \
    --eval_first true \
    --save_dir checkpoints/resip_nav_v3 \
    --headless
```

### 모니터링 지표
- `ColFOV`: FOV 내 충돌률 → 낮아져야 함 (회피 학습 성공)
- `ColAll`: 전체 충돌률 → ColFOV보다 약간 높음 (후방 충돌 포함)
- `ColAll - ColFOV`: 후방/측면 충돌 → backward penalty가 이 gap을 줄여야 함
- `R_lin + R_ang`: 방향 추종 성능 → 높아져야 함
- `rew_backward`: 후진 사용량 → 0에 가까워져야 함
