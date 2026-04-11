# Skill-3 CarryAndPlace RL 학습 기록

## 개요

Skill-3는 LeKiwi 로봇이 약병(source)을 들고 목적지 컵(dest) 옆에 내려놓는 태스크.
BC (Diffusion Policy) + ResiP (Residual PPO) 방식으로 학습.

- **BC**: 55D obs, 9D action, DP (UNet)
- **RL**: combined_s2_s3 (S2 expert가 lift → S3 BC+RL이 place)
- **환경**: Isaac Lab DirectRLEnv (Skill2Env 상속)

---

## 1. BC 분석

### 1.1 원본 v6 BC (oversample 없음)

| 지표 | 값 |
|---|---|
| Phase B 진입 (base_dst<0.40) | 34% |
| arm lowered (arm1>2.0) | 13% |
| grip open (>0.50) | 7% |
| grip wide (>0.90) | 2% |

**문제**: Phase A에서 base 거의 안 움직임 (act_base 97.5%가 |<0.05|)

### 1.2 PA10 BC (Phase A 10x oversample)

| 지표 | 원본 | PA10 | 개선 |
|---|---|---|---|
| Phase B 진입 | 34% | **62%** | ×1.8 |
| arm lowered | 13% | **27%** | ×2.1 |
| grip open | 7% | **17%** | ×2.4 |
| grip wide | 2% | **10%** | ×5 |

**PA20 (20x oversample)은 PA10보다 나쁨** — Phase B 데이터 비율 과소 → arm/grip 퇴화

### 1.3 BC Action 패턴 (PA10)

- **Phase A base**: vy mean +0.024 (거의 zero), 7.5%만 motion > 0.05
- **Phase A arm**: -0.976 (tucked, 안정)
- **Phase A grip**: -0.450 (closed, 안정)
- **Phase B arm[1]**: mean -0.528, 19.3%만 lowering 시도 (>0.5)
- **Phase B grip**: mean -0.376, 5.9%만 opening 시도 (>-0.2)
- **Lowered 도달 시**: LOWER_POSE 거의 완벽 (arm1=2.99 vs target 2.844)

### 1.4 데모 기반 3-body 분석

성공 에피소드 (9개) placement moment 기준:

```
Robot-Dest distance (base_dst): mean=0.357, std=0.043, range [0.261, 0.436]
Src-Dest distance (src_dst): mean=0.161, std=0.080, range [0.075, 0.344]
Arm joints: [−0.179, +3.005, −1.519, −2.455, +0.106] (≈ LOWER_POSE)
```

**목표 상수 (데모에서 추출):**
```python
V15_TARGET_DST_B = (0.013, 0.380)   # dest body 위치 (robot 기준 1.3cm 오른쪽, 38cm 전방)
V15_TARGET_SRC_B = (-0.122, 0.380)  # bottle body 위치 (dest 왼쪽 13.5cm)
LOWER_POSE = [-0.394, 2.844, -1.489, -2.303, 0.010]
REST_POSE = [-0.070, -0.207, 0.203, 0.121, 0.024]
```

---

## 2. RL 학습 이력

### v15v1~v6 (Phase-gated reward)

**설계**: Phase A → M1(align) → Phase B → M2(lower) → Phase C(release) → Phase D(retract)

**문제점들**:
- v15v1: Drop detection이 placement-like 상태(46%) 죽임 → agent 퇴화
- v15v2: Phase A/B reward holding gating 없음 → "drop이 이득" 학습
- v15v3: PPO 후퇴 (align 줄고 drop 증가)
- v15v4: **12 successes (최고)** — lr 1e-4로 안정. 하지만 iter 37 이후 후퇴
- v15v7: PA10 BC로 개선 → **6 successes** 빠르게, 하지만 역시 후퇴
- v15v8~v10: PPO NaN (resume 문제, reward magnitude 문제)

**핵심 실패 원인**:
1. Drop penalty (-30) 너무 큼 → "arm 안 내리는 게 안전" 학습
2. Per-step reward (×4-8) 너무 약함 (Skill2는 ×200)
3. Phase gate가 BC 동시 동작과 충돌
4. 19개 reward term → 수렴 어려움

### v16 (Phase-less, Skill2 패턴)

**핵심 전환**: Phase gate 제거, proximity gating만 사용

**Skill2 성공 패턴 분석**:
```python
# Skill2: 5개 term, 64% 성공률
R4:  held × height × 200.0     # DOMINANT per-step
R4b: held × pose_sim × 160.0   # quality shaping (two-stage)
R5:  tiered sustain             # +10/30/60/150/500
R7:  -0.01                     # time
R8:  ground × -2.0             # ground contact
```

**v16 FINAL 설계 (7 terms)**:

| # | Term | Scale | Skill2 대응 | 조건 |
|---|---|---|---|---|
| 1 | ★ GOAL | ×100/step | R4 | released + placement_err<0.15 + floor + upright + arm lowered |
| 2 | LOWER_POSE quality | ×30/step | R4b | near_dest (dst_body<0.15) + holding, two-stage |
| 3 | REST_POSE quality | ×30/step | R4b (retract) | released + placed |
| 4 | APPROACH | ×5/step | breadcrumb | dst_body_dist delta (heading+position) |
| 5 | TIERED SUSTAIN | +10/30/60/150 | R5 | sustained lowered |
| 6 | Time | -0.01 | R7 | always |
| 7 | Drop | -5 | R8 | bottle unsafe 12+ steps |

---

## 3. Reward 설계 핵심 원칙

### 3.1 Goal State (데모 3-body 관계)

```python
# placement_err: dest 기준 bottle body offset (데모 분석값)
src_to_dst_body = src_body[:2] - dst_body[:2]
TARGET_OFFSET = (-0.135, 0)  # dest에서 13.5cm 왼쪽 (body frame)
placement_err = norm(src_to_dst_body - TARGET_OFFSET)

# Goal 조건: released + 정확한 위치 + floor + upright + arm lowered
in_goal = (~has_contact) & (placement_err < 0.15) & (src_h < 0.06) & (upright > 0.70) & (lower_dist < 1.5)
```

### 3.2 Approach Signal (heading + position 동시)

```python
# dst_body_dist: dest가 robot body 기준 올바른 위치에 있는지
# → position + heading 모두 반영 (데모: dest at body (0.013, 0.380))
dst_body_dist = norm(dst_body[:2] - V15_TARGET_DST_B[:2])
```

### 3.3 Proximity Gating (Phase 대신)

```python
near_dest = dst_body_dist < 0.15  # heading + position 맞아야 열림
# → near_dest일 때만 lower pose quality reward 발화
```

### 3.4 Dynamic Grip Scale

```python
arm_lowered_enough = arm1 > 2.0
grip_scale = 0.50 if arm_lowered_enough else 0.0  # per-env dynamic
# → lowering 중 grip 보호, lowered 후에만 release 허용
```

---

## 4. Action Scale 설계

| Phase | arm | grip | base | 근거 |
|---|---|---|---|---|
| A (carry) | 0.20 | 0.0 | 0.30 | arm: skill2처럼 접근 중 pre-positioning, grip: carry 보호 |
| B (lowering) | 0.30 | 0.0 (dynamic) | 0.30 | arm: BC 과적합 궤적 보정, grip: lower 중 보호 |
| B (lowered) | 0.30 | 0.15 (dynamic) | 0.30 | arm: 동일, grip: BC release 약간 보정 |

**Skill2 비교**: arm 0.20, grip 0.30, base 0.35 → v16는 arm/base 동등 이상

---

## 5. PPO Config

| 항목 | 값 | 근거 |
|---|---|---|
| lr_actor | 1e-4 | v15v4에서 97+ iter 안정 |
| lr_critic | 1e-3 | skill2 stage 2 |
| target_kl | 0.05 | conservative |
| normalize_reward | True | 안정성 |
| init_logstd | -2.0 | 작은 exploration noise |
| ent_coef | 0.001 | 기본 |
| clip_vloss | (code에 미구현) | - |

---

## 6. 주요 교훈

### BC 관련
1. **Phase A oversample이 핵심** — PA10으로 base motion ×2 개선
2. **PA20은 PA10보다 나쁨** — Phase B 학습 데이터 축소로 arm/grip 퇴화
3. **BC의 장단점 정확히 파악해야** — Phase A arm/grip은 fine, base가 문제

### Reward 관련
4. **Skill2 패턴 따라야** — per-step DOMINANT (×100-200), 소수 term, minimal penalty
5. **Phase gate는 해악** — BC 동시 동작 차단, "안전 전략" 유발
6. **Drop penalty 최소화 (-5)** — 시도를 처벌하면 안 됨
7. **Reward term 수 ≤ 7** — 많으면 수렴 불가

### Scale 관련
8. **Skill2급 scale (0.20-0.35)** — 보수적이면 RL이 BC 못 바꿈
9. **Dynamic grip scale** — lowering 중 보호, lowered 후 개방
10. **Phase A arm도 열어야** — RL이 더 나은 전략 발견

### 3-Body 관계
11. **dst_body_dist가 heading + position 동시 유도** — separate heading reward 불필요
12. **placement_err (body frame offset)이 placement 판정** — robot heading 의존적이라 heading 자동 유도
13. **base 목표 = 데모의 robot-dest 상대좌표** — 단순 거리가 아님

---

## 7. v15 → v16 Reward 전체 매핑 (19개 → 7개)

### Phase A (carry → align) — 6개 → 1개

| v15 term | 유도 목표 | v16 | 제거/통합 이유 |
|---|---|---|---|
| R1 approach delta ×6 | base를 dest 방향 이동 | **→ APPROACH ×5 (dst_body_dist delta)** | src_dst_xy → dst_body_dist로 교체 (heading 포함) |
| R2 tanh proximity ×2 | dest 가까울수록 보상 | **제거** | dst_body_dist delta가 이미 거리 유도 |
| R3 heading cos ×0.5 | dest 방향 바라보기 | **제거** | dst_body_dist가 heading 포함 |
| R4 lateral Gaussian ×0.5 | dest 정면 정렬 | **제거** | dst_body_dist가 lateral 포함 |
| R6 hold bonus 0.20 | 약병 들고 있기 | **제거** | goal/pose가 holding gate라서 자연 유도 |
| M1 +80 (base aligned) | alignment milestone | **reward 제거, logging만** | per-step dominant면 one-time 불필요 |

### Phase B (lower) — 7개 → 1개

| v15 term | 유도 목표 | v16 | 제거/통합 이유 |
|---|---|---|---|
| R7 lower delta ×4 | arm 내리기 진행 | **제거** | LOWER_POSE quality ×30이 continuous gradient 제공 |
| R8 src body delta ×3 | bottle → target body offset | **제거** | goal ×100의 placement_err가 대체 |
| R9 src prox tanh ×2 | bottle-dest 가까워지기 | **제거** | 동일 |
| R10 height_q ×1 | bottle 바닥으로 | **제거** | goal 조건 src_h<0.06이 유도 |
| R11 lower_q ×1.5 | arm LOWER_POSE 유사도 | **→ LOWER_POSE quality ×30** | ×1.5 → ×30 (20배 강화, Skill2 R4b 패턴) |
| R12b stable lowered ×1.0 | lowered 유지 | **제거** | pose quality ×30이 대체 |
| M2 +100 (lowered milestone) | lower 달성 보너스 | **reward 제거, logging만** | per-step dominant면 불필요 |

### Phase C (release) — 3개 → 0개 (Goal이 대체)

| v15 term | 유도 목표 | v16 | 제거 이유 |
|---|---|---|---|
| R13 grip open ×5 | grip 열기 | **제거** | BC가 grip 처리, goal ×100이 released 조건으로 유도 |
| R15 place quality ×3 | bottle 위치 품질 | **제거** | goal ×100의 placement_err Gaussian이 대체 |
| M3 +150 (released) | release 달성 | **reward 제거, logging만** | 동일 |

### Phase D (retract) — 3개 → 1개

| v15 term | 유도 목표 | v16 | 제거/통합 이유 |
|---|---|---|---|
| R16 rest delta ×4 | arm REST_POSE 진행 | **→ REST_POSE quality ×30** | 3개 → 1개 통합 (Skill2 R4b 패턴) |
| R17 rest_q ×2 | REST_POSE 유사도 | 통합됨 | |
| R18 grip close ×1 | grip 닫기 | **제거** | BC가 자연 처리 |
| M4 +150 (rested) | retract 달성 | **reward 제거, logging만** | |

### 기타

| v15 term | v16 | 변화 |
|---|---|---|
| Time -0.01 | **유지** | |
| Drop -10~-30 | **→ -5** | Skill2 -2 참고, 시도 처벌 방지 |
| Smoothness -0.5 | **제거** | scale 적절하면 불필요, arm 움직임 억제 부작용 |
| Final +500~1000 | **→ +500 유지** | ms_rested 도달 시 |

---

## 8. v16 Reward 충분성 검증

### 학습 단계별 signal 분석

#### Step 1: Base 이동 (position + heading)
- **Signal**: APPROACH ×5 (dst_body_dist delta) + GOAL ×100 backprop
- **BC 기여**: PA10이 63% Phase B 도달 → RL은 refine만
- **수치**: 200 step × 0.25/step = +50 누적 vs time -2. **충분**

#### Step 2: Arm 내리기
- **Signal**: POSE QUALITY ×30 (near_dest gated, two-stage)
- **BC 기여**: 27%가 arm lowering 시도
- **수치**: pre-lowered 4.5/step → post 30/step. 50 step × 30 = +1500. **충분**

#### Step 3: Grip 열기 (가장 약한 부분)
- **Signal**: 직접 reward **없음**. GOAL ×100 (released) - POSE ×30 (holding) = **+70/step 간접 advantage**
- **BC 기여**: 17% grip open
- **근거**: Skill2도 grip 직접 reward 없이 held reward만으로 학습 성공
- **리스크**: 느릴 수 있음. **안 되면 `grip_open ×5 when on_floor` 한 줄 추가 (trivial)**

#### Step 4: Arm 복귀
- **Signal**: REST_POSE quality ×30 + Final +500
- **수치**: 30/step × 30 steps = +900. **충분**

### Skill2와 비교

| 측면 | Skill2 | v16 | 비교 |
|---|---|---|---|
| Dominant per-step | ×200 (lift height) | ×100 (goal placement) | v16 낮지만 dominant |
| Pose quality | ×160 (LIFTED_POSE) | ×30 (LOWER) + ×30 (REST) | 2개 pose target |
| Approach | R2 ×30 (stage 1만) | ×5 (always) | v16 약하지만 BC가 보완 |
| Grip/release | implicit (held reward) | implicit (goal requires release) | **동일 패턴** |
| Drop penalty | -2 | -5 | 유사 |
| Term 수 | 5-6 | 7 | 유사 |
| BC baseline | ~30% grasp | 27% arm lower, 17% grip open | 유사 |
| **Result** | **64% success** | **TBD** | |

### 판단

- **모든 학습 단계에 gradient 존재** (gap 없음)
- **Skill2와 동일 구조** (per-step dominant + pose quality + minimal penalty)
- **BC PA10이 16% baseline** → RL amplify
- **가장 약한 grip도 +70/step 간접 incentive** (Skill2 동일 패턴)
- v15의 19개 term → credit assignment 혼란이 수렴 실패 원인이었음

---

## 9. 파일 구조

```
train_resip.py          — RL 학습 (main_combined → v16 reward block)
lekiwi_skill3_env.py    — Skill3 env (combined에서는 Skill2Env 사용)
skill3_bc_obs.py        — 55D BC obs 빌드
train_diffusion_bc.py   — BC 학습 (--phase_a_oversample)

checkpoints/
  dp_bc_skill3_55d_v6_pa10/  — PA10 BC (사용 중)
  resip_s3_55d_v16_2048/     — v16 RL checkpoints

logs/
  resip_s3_55d_v16_final_*.log — v16 학습 로그
```

---

## 10. 현재 상태 (2026-04-09)

- **v16 FINAL 학습 진행 중**
- BC: PA10 (epoch 300)
- Curriculum: dest spawn 0.4-0.6m
- 7 reward terms (Skill2 패턴)
- Phase-less proximity gating
- Dynamic grip scale
- Skill2급 action scales
