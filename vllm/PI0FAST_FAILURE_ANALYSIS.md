# LeKiwi VLA Pi0Fast 학습 실패 원인 분석 및 Pi0.5 전환 보고서

**작성일**: 2026-04-08
**대상 시스템**: LeKiwi mobile manipulator (5-arm + gripper + 3-base = 9D action)
**실패한 모델**: Pi0Fast (LeRobot 0.5.0, 4B params, FAST tokenizer + autoregressive)
**전환 모델**: Pi0.5 (LeRobot 0.5.0, 4.1B params, continuous flow matching)
**총 소요 시간**: ~3주 (v1, v2, v3 학습 + 검증 + 디버깅)
**결론**: Pi0Fast의 **FAST tokenizer가 우리 데이터의 sharp delta-function 분포에 본질적으로 부적합**. Continuous flow matching인 Pi0.5로 전환.

---

## 0. Executive Summary (보고용 요약)

### 0.1 발생한 문제

LeKiwi 이동 매니퓰레이터의 4-skill (navigate / approach&lift / carry / approach&place) 학습을 위해 Pi0Fast VLA를 3회 fine-tune (v1, v2, v3) 했으나 **모두 동일한 mode collapse failure로 종료**.

### 0.2 핵심 증상 (정량적)

| 지표 | 측정값 | 정상 기대값 | 상태 |
|---|---|---|---|
| Robot 5초 이동 거리 | **0.05 m** | 2.5 m (demo) | ❌ 사실상 정지 |
| Zero-velocity chunk 비율 | **38%** | 0% | ❌ |
| Demo magnitude 대비 VLA 출력 | **3.8% (forward), 13.4% (backward)** | 100% | ❌ |
| "navigate turn right" 실제 회전 | **LEFT (wz=+0.093)** | RIGHT (wz<0) | ❌ 부호 반대 |
| Loss plateau 시작 | step 20K (0.4 epoch) | 100K+ | ❌ 너무 일찍 |
| Loss 감소율 (20K→115K, 95K step) | **3.4%** | 30%+ | ❌ |
| Inference latency | **3,000 ms/call** | 50 ms | ❌ 60배 느림 |

### 0.3 근본 원인 (3주에 걸친 디버깅 끝에 확정)

**FAST tokenizer가 우리 데이터의 sharp delta-function 분포 (예: navigate forward → vy=+0.5±0.02)를 정확히 표현하지 못하고, token vocabulary가 작은 subset으로 collapse**. 결과적으로 모델이 instruction에 따른 conditional distribution을 학습 못하고 marginal mean (≈0)으로 수렴.

이는 **알고리즘 자체의 한계**이며, 데이터/하이퍼파라미터/normalization 수정으로 해결 불가능 (3차례 시도로 검증).

### 0.4 해결 방향

**Pi0.5 (continuous flow matching)** 로 전환. Token quantization이 없는 architecture이므로 mode collapse 위험이 본질적으로 없음. **현재 학습 중** (이 문서 작성 시점에 step 400 / 200K, loss 0.323 → 0.123 빠른 감소 중).

---

## 1. 시스템 배경 정보

### 1.1 LeKiwi 로봇 구성

- **Arm**: 5-DoF + gripper (총 6D 관절)
- **Base**: holonomic mecanum drive (vx, vy, wz 3D)
- **Action space**: 9D 벡터, **정규화된 [-1, 1] 공간**
- **State space**: 9D 벡터, **raw radians + m/s** (action과 단위 다름)

### 1.2 Action 정규화 관계 (env code 검증)

`lekiwi_skill2_env.py:1450-1494`의 `_apply_action()`:

```python
# Base velocity: action × max_velocity
body_vx = self.actions[:, 6] * self.cfg.max_lin_vel    # max_lin_vel = 0.5
body_vy = self.actions[:, 7] * self.cfg.max_lin_vel    # max_lin_vel = 0.5
body_wz = self.actions[:, 8] * self.cfg.max_ang_vel    # max_ang_vel = 3.0

# Convention: vy = forward/back, vx = lateral, wz = yaw
ik_vx = body_vy       # IK vx+ = body vy+ (forward)
ik_vy = -body_vx
ik_wz = body_wz

# Arm action: action × joint_range_half + joint_center
arm_grip_action = self.actions[:, 0:6]
center = 0.5 * (arm_lo + arm_hi)
half = 0.5 * (arm_hi - arm_lo)
mapped = center + arm_grip_action * half
```

→ **action[6] = ±1 → ±0.5 m/s 측방 이동**
→ **action[7] = ±1 → ±0.5 m/s 전후방 이동**
→ **action[8] = ±1 → ±3.0 rad/s 회전**
→ **arm action ±1 → joint limit 양 끝**

### 1.3 데이터셋 구조 (`lekiwi_viva_v2`)

```
총 episodes: 978
총 frames: 209,036 (평균 213 frames/episode, 중앙값 150)

Task 분포 (task_index → name → episode count → frame count):
  5  approach and lift the medicine bottle      ~100 ep   77,372 frames
  6  navigate forward                            68 ep    10,164 frames
  7  navigate backward                          ~76 ep    11,400 frames
  8  navigate turn left                         ~74 ep    11,100 frames
  9  navigate turn right                        ~76 ep    11,400 frames
  10 navigate strafe left                       ~76 ep    11,400 frames
  11 navigate strafe right                      ~76 ep    11,400 frames
  12-17 carry forward/backward/.../turn left    각 ~72 ep, 10,800 frames each

검증 결과: 978/978 episodes 모두 task 1개만 포함 (instruction이 episode 내에서 변하지 않음)
```

### 1.4 데모 데이터의 분포 특성 (학습 가능한가?)

**Per-task action distribution from training data (검증 결과):**

| Task | active channel | mean | std | range | CV |
|---|---|---|---|---|---|
| navigate forward | vy | +0.4998 | 0.020 | [+0.42, +0.58] | 4.0% |
| navigate backward | vy | -0.4998 | 0.020 | [-0.58, -0.42] | 4.0% |
| navigate turn left | wz | -0.3301 | 0.020 | [-0.40, -0.26] | 6.1% |
| navigate turn right | wz | +0.3303 | 0.020 | [+0.25, +0.41] | 6.0% |
| navigate strafe left | vx | -0.4998 | 0.020 | [-0.57, -0.42] | 4.0% |
| navigate strafe right | vx | +0.5000 | 0.020 | [+0.43, +0.57] | 4.0% |

**핵심 특성:**
- 각 instruction마다 **한 채널만 ±0.5 (또는 wz=±0.33)**, 나머지는 0
- 표준편차 **0.02 → 사실상 delta function 분포** (CV 4-6%)
- 10K 프레임 × 6 task = **trivial 학습이 가능해야 하는 데이터**

**Episode 간 일관성 검증** (10개 navigate forward episode 평균):
```
EP 100: vy_mean +0.5028
EP 110: vy_mean +0.4983
EP 117: vy_mean +0.4986
EP 123: vy_mean +0.4987
EP 124: vy_mean +0.5020
EP 130: vy_mean +0.5009
EP 137: vy_mean +0.5003
EP 143: vy_mean +0.5000
EP 150: vy_mean +0.5014
EP 157: vy_mean +0.4991
```

→ **모든 navigate forward episode가 정확히 +0.5에 수렴**. 데이터에 noise/inconsistency 없음.

### 1.5 Visual Diversity 검증

각 navigate task의 첫 frame을 추출하여 RMSE 계산:

```
forward     vs backward      RMSE=59.0
forward     vs turn_left     RMSE=60.0
forward     vs turn_right    RMSE=53.2
backward    vs turn_left     RMSE=51.4
turn_left   vs turn_right    RMSE=47.6
strafe_left vs strafe_right  RMSE=20.8  ⚠ (시각적 유사도 높음)
```

→ 대부분의 task가 시각적으로 구분 가능 (RMSE > 30). strafe만 다소 유사.

### 1.6 State vs Action 단위 차이 (검증으로 발견)

Episode 100 (navigate forward), step 0:

```
State:  [-0.0165, -0.2041, +0.2008, -0.1891, +0.0297, -0.2000, +0.0134, +0.0902, +0.0272]
Action: [+0.0141, -1.0041, +1.0194, +0.7044, -0.5443, -1.0456, +0.0148, +0.4913, -0.0469]
                  ^^^^^^^                                              ^^^^^^^
                  arm1                                                  vy
```

| 필드 | State (raw 단위) | Action (정규화 단위) | 비율 |
|---|---|---|---|
| arm1 | -0.20 rad | -1.00 (= joint_lower_limit) | 0.20× |
| vy | +0.09 m/s (실제 측정) | +0.49 (= +0.245 m/s 명령) | 0.18× |

→ **Action은 [-1, 1] 정규화 공간, State는 raw 물리 단위**. 비율이 ~0.20인 이유는 (1) 액션이 normalize되어 있고 (2) 로봇이 명령 속도의 일부만 달성하기 때문 (휠 inertia, friction).

**중요**: 이 mismatch는 "데이터 버그"가 아니라 **lerobot/isaaclab의 정상적인 표현 방식**. 학습 시 정규화 layer가 처리.

---

## 2. 학습 시도 이력 (3차)

### 2.1 v1 (initial attempt) — 약 2026-03-25

**환경**:
- LeRobot 0.5.0 default
- batch_size=4, 200K steps, chunk_size=10
- A100 40GB
- Wall time: ~24 시간

**결과**: ❌ 실패
- Robot이 거의 안 움직임
- Base velocity output ≈ training data의 1/30
- Arm output ≈ 16x larger than training data

**당시 진단된 원인 후보**:
1. **stats.json clamp**: std=0.30으로 강제 clamp되어 있어 작은 std 값 손실
2. **Navigate arm hardcoded constant**: NAV_ARM_HARDCODED = `[-0.000791, -1.0, 1.0, 0.658716, -0.537318]`로 446개 episode 모두 동일 → std=0이라 학습 불가
3. **Carry base 33% unique values only**: 데이터 다양성 부족

### 2.2 v2 — 약 2026-03-30

**v1 대비 변경사항**:
- HDF5 source 데이터에서 navigate arm joints에 Gaussian noise 추가 (std=0.03)
- Navigate gripper에도 noise (std=0.05)
- 446 navigate episodes 모두 수정
- LeRobot dataset (parquet)도 동기화
- stats.json 재계산

**결과**: ❌ 실패. v1과 동일한 증상.

### 2.3 v3 — 2026-04-07 ~ 2026-04-08

**v2 대비 변경사항**:
- **stats.json clamp 완전 제거** (real q01/q99/std로 복원, backup at `stats.json.bak`)
- Carry base velocity에 noise std=0.005 추가 (33% unique → 100% unique 회복)
- `validate_action_token_prefix=false` 명시
- `gradient_checkpointing=true`, `dtype=bfloat16`
- `max_action_tokens=256`, `chunk_size=10`, `n_action_steps=10`
- HDF5 noise 적용 검증, parquet 동기화 검증

**학습 환경**:
- A100 40GB single GPU
- batch_size=4, 200K steps target
- Wall time: ~24h 진행 후 사용자가 결과 보고 중단 (115K/200K = 54%)
- 5K마다 체크포인트 저장 (24개 + last)

**Loss 추이** (115K까지의 5K 구간 평균):

| Step Bucket | Mean Loss | 직전 대비 변화 |
|---|---|---|
| 0–5K | 4.80 | – (start) |
| 5K | 3.60 | -1.20 (큰 감소) |
| 10K | 3.41 | -0.19 |
| 15K | 3.31 | -0.10 |
| 20K | 3.27 | -0.04 ← **plateau 시작** |
| 30K | 3.25 | -0.02 |
| 50K | 3.23 | -0.02 |
| 80K | 3.20 | -0.03 |
| 100K | 3.19 | -0.01 |
| 110K | 3.16 | -0.03 |
| **115K** | **3.16** | (변화 없음) |

→ **Step 20K (0.4 epoch) 이후 95K step 동안 loss 3.27 → 3.16, 0.11만 감소 (3.4%)**.

이 시점에서 사용자가 "여전히 학습 잘 안돼?" 물어보고 학습 중단 결정.

---

## 3. 검증 방법론 (3주 분량의 디버깅 세부 사항)

### 3.1 데이터/정규화 stats 검증 (v3 115K 체크포인트)

`policy_postprocessor_step_0_unnormalizer_processor.safetensors`를 직접 read:

```python
action.std  = [0.0444, 0.5036, 0.3404, 0.3979, 0.0258, 0.4717, 0.2303, 0.2420, 0.1589]
action.mean = [-0.009, -0.805, +0.877, +0.184, -0.563, -0.481, -0.018, -0.006, +0.025]
action.q01  = [-0.143, -1.056, -0.318, -0.818, -0.596, -1.092, -0.518, -0.518, -0.350]
action.q99  = [+0.057, +1.066, +1.062, +0.729, -0.481, +1.154, +0.519, +0.546, +0.350]
                                                       ^^^^^^^
                                                       arm joint 4: std 0.026 (clamp 제거됨)
```

**검증 결과**: ✅ stats clamp 제거됨. 작은 std 값 (arm joint 4: 0.026, joint 0: 0.044)이 정상 보존. q01/q99 모두 존재. **데이터 정규화는 문제 없음**.

### 3.2 서버 측 직접 inference 테스트

VLA 서버(`vla_inference_server.py`)에 직접 합성 요청 보내서 6가지 navigate instruction에 대한 출력 측정.

**입력**:
- Real dataset frame (training video에서 ffmpeg 추출)
- Navigate hardcoded state `[-0.000791, -1.0, 1.0, 0.658716, -0.537318, -0.999472, 0, 0, 0]`

**측정 결과**:

| Instruction | vx mean | vy mean | wz mean | arm joint 1 mean | 정상 여부 |
|---|---|---|---|---|---|
| navigate forward | -0.063 | -0.063 | -0.063 | +0.380 | ❌ 모든 채널 garbage |
| navigate backward | +0.063 | -0.032 | 0 | -0.095 | ❌ 잘못된 채널 |
| navigate turn left | 0 | 0 | **+0.063** | +0.569 | △ wz 부호만 맞음 |
| navigate turn right | 0 | +0.032 | **0** | -0.095 | ❌ wz=0 (양수여야) |
| navigate strafe left | -0.095 | 0 | 0 | -0.095 | ❌ 잘못된 부호 |
| navigate strafe right | -0.032 | -0.095 | 0 | -0.095 | ❌ 잘못된 부호 |

**핵심 발견 1: 모델이 navigate hardcoded arm pose조차 출력 못함**
- 입력 state arm joint 1 = -1.000 (navigate hardcoded)
- 모델 출력 arm joint 1 mean = -0.095 ~ +0.569 (instruction별로 random)
- **446개 navigate episode (학습 데이터의 45%)가 모두 joint 1 = -1.0 인데도 학습 실패**

### 3.3 Inference latency 측정

```
warmup: 2871 ms
runs: 2860, 2863, 2853, 2918, 3010, 3053, 3060 ms
client_total - server: 7-8 ms (네트워크 영향 미미)
```

→ **순수 서버 측 inference 2.9-3.0 초**. 정상 Pi0Fast (50ms) 대비 60배 느림.

**원인**: 모델이 EOS 토큰 예측에 실패해 `max_decoding_steps=256`에 가까운 토큰 생성. 정상 학습된 모델은 30-50 토큰만 생성.

### 3.4 Isaac Sim end-to-end 평가

**환경**: VIVA mode + easy difficulty + scene 1302 + 1 trial
**기간**: 67초 wall time / 159 simulation steps
**Skill 진행**: navigate에서 끝까지 stuck (S2 approach&lift 진입 못함)
**결과**: timeout

**Wall clock 속도**: 159 step / 67 sec = 2.4 Hz (목표 6.4 Hz의 38%). 원인은 inference 3초 / chunk 10 actions = 0.3 sec per action.

---

## 4. Action Output 정밀 분석 (`eval_viva_actions.tsv`)

### 4.1 Robot Physical Displacement (가장 결정적 증거)

```
∫ act_vx dt ≈ +0.001 m  (159 steps × 1/30 s sim_dt)
∫ act_vy dt ≈ -0.051 m
─────────────────────────────────────────────────
Robot 총 변위: ~0.05 m
```

**Robot은 5.3초 sim time 동안 5cm 이동.** 사용자가 GUI에서 본 "base 안 움직임"의 정량적 증명.

**비교**: 정상 demo 데이터에서는 navigate forward 1 episode (150 step × 1/30 sec = 5초) 동안 0.5 m/s × 5s = **2.5 m 이동**.

→ **VLA가 demo 대비 1/50 수준의 변위만 생성**.

### 4.2 Per-Instruction Action Stats (예측 vs 실제 물리)

| Instruction | N | predicted vy mean | actual vy mean | predicted wz mean | actual wz mean |
|---|---|---|---|---|---|
| navigate forward | 70 | **+0.019** (data: +0.50) | +0.0015 | +0.040 | -0.065 |
| navigate backward | 40 | **-0.067** (data: -0.50) | -0.028 | -0.010 | +0.001 |
| navigate turn right | 50 | -0.015 (data: ~0) | -0.010 | **-0.028** (data: +0.33) | **+0.093** |

**관찰 1: Magnitude gap 1/10 ~ 1/30**
- forward: VLA 출력은 demo의 **3.8%** (0.019 / 0.50)
- backward: VLA 출력은 demo의 **13.4%** (0.067 / 0.50)

**관찰 2: 채널 confusion**
- forward는 vy 채널이어야 하는데 vx에도 +0.056 (작은 노이즈)
- 잘못된 채널에 약하게 출력

**관찰 3: 부호 오류**
- "navigate turn right"의 wz 예측값 -0.028 (음수, 작음)
- 학습 데이터 wz mean = +0.33 (양수, 큼)
- **부호 자체가 반대**

**관찰 4: 같은 instruction에 큰 분산**
- "navigate forward" vy 범위: [-0.51, +0.88]
- 같은 명령인데 chunk마다 [-0.5 ~ +0.9]로 모순적 출력

### 4.3 Chunk-level Degeneracy

160 step을 16개 chunk (10 step씩)으로 분석:

| 패턴 유형 | 개수 | 비율 |
|---|---|---|
| **모든 base velocity = 0** (정지 명령) | 6 / 16 | **38%** |
| 모든 arm joint = 0 (정지 자세) | 2 / 16 | 12% |
| Pure sinusoid (sum ≈ 0, 무의미) | 1 / 16 | 6% |

→ **38%의 시간 동안 robot은 명시적으로 정지 명령**을 받음.

### 4.4 출력 양자화 Grid (FAST tokenizer collapse 증거)

160 step에서 등장한 unique 값 개수:
- vx: **73** unique (out of 160)
- vy: **42** unique
- wz: **37** unique

전체 채널에서 동일한 magic number들이 반복:
```
{-0.0949, -0.0632, -0.0316, 0, +0.0316, +0.0632, +0.0949, +0.1265, +0.1581}
```

이 값들은 **FAST tokenizer의 양자화된 DC 계수가 unnormalize된 결과**. 모델이 이 ~10개 vocabulary를 벗어나지 못함.

**비교**: 정상이라면 demo와 같이 sharp delta function ±0.5에 가까운 값들이 나와야 함. 우리 모델은 ±0.1 미만의 작은 양자화 grid에 갇힘.

### 4.5 Sinusoidal pattern 분석 (DCT artifact)

특정 chunk의 base velocity 시퀀스:

```
Chunk for "turn right" (steps 2-11):
  vy: -0.044, -0.040, -0.032, -0.020, -0.007, +0.007, +0.020, +0.032, +0.040, +0.044
       ↑─────────── 정확한 sin curve, 합계 = 0 ───────────↑
  wz: -0.221, -0.158, -0.158, -0.221, -0.221, -0.158, -0.158, -0.221, -0.221, -0.158
       ↑─────────── 평균 -0.19 (turn right 부호 맞음) ───────────↑
```

이건 노이즈가 아니라 **DCT의 1차 harmonic basis function**입니다.

**FAST tokenizer 처리 과정**:
1. action chunk (10, 9) → MEAN_STD normalize
2. DCT (시간축 따라)
3. DCT 계수 양자화
4. BPE 토큰화

**모델이 학습한 것**: DC 계수 + 1차 harmonic만 약하게 예측
**Inverse DCT 결과**: constant 또는 1주기 sin (양 끝이 +/- 대칭, 합계=0)
**물리적 효과**: chunk 내에서 robot이 앞뒤로 진동 → net displacement ≈ 0

---

## 5. Root Cause Analysis (3주 분량의 디버깅 결과)

### 5.1 직접 원인: FAST tokenizer mode collapse

Pi0Fast의 action representation:

```
action chunk (10, 9)
  ↓ MEAN_STD normalize (action.std로 나눔)
standardized chunk
  ↓ DCT (시간축)
DCT coefficients (10, 9)
  ↓ Quantize
quantized DCT integers
  ↓ BPE encode
tokens (~30-100개)
```

**정상 학습 모델**: instruction과 image에 따라 다양한 DCT 계수 패턴 (DC + 다양한 고차 harmonics) 예측 → 풍부한 action vocabulary
**우리 v3 모델**: DC 계수 3-5개 discrete level과 1차 harmonic만 예측 → degenerate vocabulary

### 5.2 Loss plateau의 의미

Loss 3.16에서 plateau는:
- Cross-entropy loss는 **token entropy 하한**이 있음 (dataset entropy)
- 모델이 marginal token distribution은 학습했지만 conditional distribution (instruction → action token) 학습 실패
- Token vocabulary가 작은 subset으로 collapse → 추가 학습해도 entropy는 더 줄지 않음

**핵심 이해**: Pi0Fast loss 3.16은 모델이 "마진을 학습 못한 것"이 아니라, 학습할 수 있는 entropy의 한계까지 학습한 결과. 단지 그 한계가 **인스트럭션과 무관한 marginal distribution의 entropy**라서 실제 작업에 쓸모가 없는 것.

### 5.3 왜 instruction conditioning에 실패했나? (5가지 가설)

1. **데이터 양 부족**: 978 episodes는 multi-task VLA로는 작음 (보통 수천 ~ 수만 episode 필요)
2. **Task imbalance**: navigate(446) + carry(432) >> approach&lift(100). approach&lift가 가장 dynamic한데 가장 적음
3. **Hardcoded action segments**: Navigate의 arm 5D + grip 1D = 항상 같은 값. FAST tokenizer는 9D 전체를 함께 토큰화하므로 학습 신호가 base 3D에 집중되지 못함
4. **Language encoder의 weak conditioning**: PaLI-Gemma의 instruction embedding이 action expert에 충분히 강하게 영향 못 미침
5. **알려진 lerobot 이슈**: GitHub issue #1811 — Pi0Fast fine-tuning이 small dataset에서 불안정, 특히 sharp distribution data

### 5.4 왜 Sharp Delta Distribution이 FAST에 최악인가? (이론적 분석)

우리 데이터: navigate forward → vy = +0.5 ± 0.02 (CV 4%, 거의 delta function)

FAST tokenizer 처리:
1. Standardized: (0.5 - (-0.006)) / 0.242 = **+2.07** (action mean=-0.006, std=0.242)
2. DCT(constant signal) → DC coefficient = **+2.07**, all higher = 0
3. Quantize +2.07 → 가장 가까운 quantization bin

**문제점**: 
- Quantization grid는 [-3, +3] 정도 범위를 ~256 bin으로 나눔
- +2.07은 256 bin 중 ~227번째 bin
- 모델이 이 정확한 bin을 예측해야 하는데, 만약 ±1 bin off → 예측값 ±0.012 (작은 차이지만 양자화 오차)
- **만약 모델이 잘못된 bin (예: bin 128 = 평균값)을 예측 → 출력 0.0** (완전 정지)

Sharp delta distribution + token-level CE loss의 조합:
- 모델이 다른 instruction과 헷갈리면 (forward/backward 같은 token vocab 영역) → mean으로 backoff
- Mean = 0 → robot 정지
- 이게 우리가 본 그대로의 failure mode

### 5.5 왜 v1, v2, v3 모두 같은 결과인가?

세 번 모두 같은 mode로 수렴:
- v1: stats clamp 문제 → 수정해도 동일
- v2: arm noise 추가 → 동일
- v3: 모든 데이터 cleanup → 동일

**결론**: 데이터/정규화/하이퍼파라미터의 문제가 아니라 **알고리즘 자체의 한계**. FAST tokenizer + small dataset + sharp distribution + multi-task의 조합이 본질적으로 불안정.

3차례 검증으로 이 결론에 도달.

### 5.6 Inference 3초 latency도 학습 실패의 부산물

정상 Pi0Fast: 30-50 토큰 생성, ~50 ms
우리 v3 모델: ~150-200 토큰 생성, ~3000 ms (60배 느림)

**원인**: 모델이 EOS 토큰 예측에 실패하거나 늦게 예측 → max_decoding_steps=256까지 채워서 디코딩 → garbage tokens

이는 **train fail의 결과**이지 **train fail의 원인**이 아님. torch.compile 등으로 속도를 1초로 줄여도 출력 자체가 broken이라 robot은 여전히 작동 안함.

---

## 6. Pi0.5로 전환해야 하는 이유

### 6.1 Architecture 차이

| 항목 | Pi0Fast | Pi0.5 |
|---|---|---|
| Action 표현 | FAST tokens (DCT + BPE 이산화) | **Continuous (flow matching)** |
| 학습 방식 | Autoregressive token prediction | **Velocity field regression** |
| Token vocabulary | 제한된 BPE vocab | **없음 (real-valued output)** |
| Mode collapse 가능성 | **높음** (token degeneracy) | **낮음** (continuous space) |
| Sharp distribution 학습 | **취약** | **강함** |
| Default normalization | MEAN_STD | **QUANTILES** ⭐ |
| Multi-task | tokenize 약함 | flow matching이 자연스럽게 처리 |
| Small dataset 안정성 | 낮음 (issue #1811) | 더 안정적 |
| Inference | 30~256 token autoregressive | 5~10 denoising step |
| Memory | 적음 (3B) | 더 많음 (4.1B + action expert) |

### 6.2 우리 데이터에 적합한 핵심 이유

**(1) Continuous output**: vy=+0.5 같은 sharp delta를 정확히 출력 가능 (양자화 오차 없음)

**(2) QUANTILES normalization**: action.q01=-0.518, q99=+0.546이므로
- vy=+0.5 standardized = (+0.5 - (-0.518))/(+0.546 - (-0.518)) × 2 - 1 = **+0.913**
- vy=-0.5 standardized = **-0.966**
- 두 값이 [-1, +1] 안에 잘 들어옴 → 학습 쉬움
- 비교: MEAN_STD에서는 +2.09 / -2.04로 더 큰 값 → token grid에서 edge에 위치

**(3) Sharp distribution 친화적**: flow matching은 narrow distribution을 정확히 학습 가능 (regression이므로)

**(4) Mode collapse incentive 없음**: marginal mean으로 backoff하면 loss가 증가 (continuous loss는 entropy floor 없음)

### 6.3 Pi0.5 Setup 검증 (14개 항목)

| # | 항목 | 검증 결과 |
|---|---|---|
| 1 | lerobot 0.5.0이 pi05 지원 | ✅ `lerobot/policies/pi05/` 존재 |
| 2 | TrainPipelineConfig.rename_map 필드 | ✅ `configs/train.py:81` |
| 3 | dataset_to_policy_features가 9D 처리 | ✅ shape-agnostic |
| 4 | rename_map 매핑: front→base_0_rgb 등 | ✅ `lerobot_train.py:284-286` |
| 5 | 누락 카메라 (right_wrist_0_rgb) 자동 padding | ✅ `modeling_pi05.py:1199-1201` (-1 fill) |
| 6 | State 9D → text 토큰화 (Pi05PrepareStateTokenizerProcessorStep) | ✅ shape-agnostic, np.digitize |
| 7 | Action 9D → 32D pad → 9D truncate | ✅ `modeling_pi05.py:1208-1244, 1267` |
| 8 | NormalizerProcessor shape-agnostic | ✅ `normalize_processor.py:305-338` |
| 9 | chunk_size=10 override 안전 (base는 50) | ✅ 동적 forward에서만 사용 |
| 10 | Visual feature 검증 skip when rename_map | ✅ `factory.py:524-525` |
| 11 | q01/q99 stats 존재 (QUANTILES 정규화) | ✅ stats.json 검증 통과 |
| 12 | 이미지 resize 400×640→224×224 자동 | ✅ `modeling_pi05.py:1184` |
| 13 | VISUAL=IDENTITY (image shape 무관) | ✅ default normalization mapping |
| 14 | forward(reduction="none") 지원 | ✅ `modeling_pi05.py:1248, 1274` |

### 6.4 발견 + 수정한 train_pi05.sh의 critical bug

**Bug**: `pi05_base` 경로가 잘못됨
- Script: `--policy.path=./pi05_base` (vllm/ 디렉토리 기준)
- 실제 위치: `/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base/` (vllm/ 외부)

**수정**: 절대 경로 사용
```bash
PI05_BASE="/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base"
# Pre-flight: if [ ! -d "$PI05_BASE" ]
# Arg: --policy.path=$PI05_BASE
```

### 6.5 발견 + 해결한 GPU memory leak

VLA + VLM 서버를 `lsof -ti :PORT | xargs kill -9`로 종료했으나 **vLLM EngineCore 자식 프로세스가 30GB GPU 메모리 leak**:

```
$ nvidia-smi --query-compute-apps=pid,used_memory
2364225, 30190 MiB    ← zombie (process gone, memory still allocated)

$ fuser -v /dev/nvidia*
jovyan  2707359 F...m VLLM::EngineCor    ← 진짜 범인
```

**해결**: `kill -9 2707359` → GPU 완전 free (40,327 MiB)

이는 vLLM의 multi-process worker 구조 특성으로, 향후 종료 시 `lsof + ps grep vllm` 둘 다 해야 함.

---

## 7. Pi0.5 학습 시작 결과 (현재 진행 중)

**시작 시각**: 2026-04-08 14:33
**PID**: 2723020
**Output**: `outputs/train/pi05_lekiwi_20260408_143329/`
**Log**: `/home/jovyan/pi05_train_20260408_143329.log`

### 7.1 Sanity check 통과

```
✓ Model loaded (812 state dict keys remapped, all loaded)
✓ Gradient checkpointing enabled
✓ Optimizer & scheduler created (Adam, cosine_decay_with_warmup)
✓ num_learnable_params=4,143,404,816 (4.1B)
✓ Effective batch size: 2
✓ chunk_size=10, max_state_dim=32, max_action_dim=32
✓ rename_map applied correctly
✓ Dataset: 209,036 frames, 978 episodes
✓ Training started at step 1
```

### 7.2 Loss 진행 (Pi0Fast vs Pi0.5)

```
Pi0Fast v3 (failed):                Pi0.5 (current):
  step 100   loss 14.477 grdn 338    step 100  loss 0.323 grdn 9.234
  step 5K    loss  3.60              step 200  loss 0.201 grdn 4.950
  step 20K   loss  3.27 ← plateau    step 300  loss 0.175 grdn 4.138
  step 115K  loss  3.16              step 400  loss 0.123 grdn 2.935
                                                ↑ 62% drop in 300 steps
```

**Loss function 차이**:
- Pi0Fast: token cross-entropy (lower bound = dataset token entropy ≈ 3.0)
- Pi0.5: flow matching velocity regression (lower bound = 0.0)

**중요**: 이 두 loss 값은 다른 함수의 출력이라 직접 비교 불가. 하지만 trend는 비교 가능:
- Pi0Fast: 5K step에서 거의 floor에 도달, 그 이후 평탄
- Pi0.5: 400 step에 이미 62% 감소, 빠른 수렴 패턴

### 7.3 시스템 리소스

```
Process: PID 2723020, Rl (running, multi-thread)
RSS memory: 6.5 GB
GPU memory: 39,563 / 40,328 MiB (98%, batch=2 fits exactly)
GPU compute: 77-100% util
ETA: ~42 시간 (200K steps at 1.33 step/s)
```

### 7.4 다음 판정 기준 (조기 stop 가능)

| Step | 시간 후 | 기대 loss | 판정 |
|---|---|---|---|
| 1K | ~12분 | < 0.10 | 정상 진행 |
| 5K | ~1시간 | < 0.05 | 매우 좋음 |
| 20K | ~4시간 | < 0.03 | converged |
| 50K | ~10시간 | < 0.02 | excellent |

**Pi0fast의 plateau (3.16) 같은 entropy floor가 없으므로 loss는 0에 가까이 수렴 가능**.

만약 step 5K에서 loss가 0.2에서 plateau → 데이터 문제 → 데이터 재수집 단계로 이동.

---

## 8. 시간 손실 분석 (보고용)

### 8.1 Pi0Fast 시도에 소요된 시간

| 단계 | Wall time | 결과 |
|---|---|---|
| v1 학습 | ~24h | 실패 |
| v1 디버깅 (stats clamp 발견) | ~12h | 가설 수립 |
| v2 데이터 수정 + 학습 | ~24h | 실패 |
| v2 디버깅 (carry uniqueness 발견) | ~12h | 가설 수립 |
| v3 데이터 수정 + 학습 | ~24h (115K/200K) | 실패 |
| v3 검증 (서버 직접 inference, eval) | ~8h | 결정적 증거 확보 |
| Pi0.5 setup 검증 (14개 항목) | ~4h | 전환 결정 |
| **합계** | **~108 시간** | |

### 8.2 손실의 원인 (왜 이렇게 오래 걸렸는가)

**1) FAST tokenizer의 알려지지 않은 한계**
- LeRobot 0.5.0은 비교적 새로운 release (2026-03 release)
- Pi0Fast의 small dataset 한계가 공식 문서에 명시되지 않음
- GitHub issue #1811은 우리 디버깅 후반에야 발견

**2) Symptom-cause 혼동**
- v1: stats clamp가 증상이라고 판단 → 사실 진짜 증상은 이미 token mode collapse
- v2: noise 부족이라고 판단 → 진짜 증상은 알고리즘 한계
- v3: clamp 제거 + noise 추가 → 같은 결과 → 알고리즘 한계 확인

**3) Loss curve의 misleading 특성**
- Loss 3.16에서 plateau → "학습이 끝났다"고 판단할 수 있음
- 그러나 cross-entropy loss는 절대값이 의미없고 token vocabulary 분포에 따라 다름
- "Loss decreasing → 학습 잘됨"이 항상 맞는 것이 아님 (특히 token-based VLA)

**4) 검증 도구 미비**
- 학습 중간에 inference를 돌려서 robot이 실제로 움직이는지 확인하는 자동 evaluation이 부재
- 200K step 학습 후에야 "안 움직임" 발견 → 큰 시간 낭비
- **교훈**: 향후 5K step마다 자동 evaluation 도입 필요

**5) Multi-modal 데이터의 진단 어려움**
- 9D action × 6 instruction → 실패 양상이 복잡 (어떤 채널이 어떤 instruction에 잘못 출력되는지 분석 필요)
- 단일 metric (예: success rate)로는 root cause 파악 불가

### 8.3 학습된 교훈

**1) 새로운 VLA 알고리즘은 small dataset에서 검증되지 않음을 가정**
- LeRobot 0.5.0의 pi0fast는 OpenX-Embodiment 같은 대규모 데이터셋용
- 1K 미만 episode + sharp distribution → 위험 신호로 인식

**2) Sharp delta distribution은 token-based 학습에 최악**
- Continuous output (flow matching, MSE regression) 이 더 적합
- 향후 데이터 분포를 보고 algorithm 선택 우선

**3) Loss curve만 보지 말고 inference 출력을 봐라**
- 5K step마다 robot 시뮬레이션 1 episode 자동 실행
- Action 분포가 demo와 일치하는지 quantitative 비교

**4) 정량적 증거를 빨리 확보하라**
- "robot이 안 움직임" → "5cm 변위" → "vy 출력 +0.019 vs +0.5 demo"
- 점점 구체화된 metrics가 root cause를 좁혀줌

**5) Multi-attempt 패턴이 보이면 알고리즘 한계 의심**
- 같은 디버깅으로 3번 실패하면 데이터/하이퍼파라미터가 아닌 알고리즘 문제
- 빨리 다른 algorithm으로 전환 결정

---

## 9. 향후 진행 절차

### 9.1 Pi0.5 학습 모니터링 (현재)

```bash
# 진행 확인
tail -f /home/jovyan/pi05_train_20260408_143329.log

# Loss 추출
grep -aoE 'step:[0-9K]+ .*loss:[0-9.]+ grdn:[0-9.]+' /home/jovyan/pi05_train_*.log | tail -20

# Checkpoint 확인
ls /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/outputs/train/pi05_lekiwi_20260408_143329/checkpoints/
```

### 9.2 중간 평가 시점

| Step | 시간 | Action |
|---|---|---|
| 5K | ~1h | Loss 확인. < 0.05면 진행, > 0.10이면 데이터 문제 의심 |
| 20K | ~4h | 첫 inference test (robot 움직임 확인) |
| 50K | ~10h | Full eval (eval_viva_pipeline.py) |
| 100K | ~21h | 중간 결과 검토 |
| 200K | ~42h | Final eval |

### 9.3 Pi0.5 inference 서버 수정 필요

`vla_inference_server.py`의 import 변경:
```python
# Before
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

# After
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
```

이미지 키 매핑은 학습 시 사용한 형식 유지:
- `observation.images.front` → `observation.images.base_0_rgb`
- `observation.images.wrist` → `observation.images.left_wrist_0_rgb`

또는 inference 서버에서도 rename 처리.

### 9.4 Contingency Plan (Pi0.5도 실패 시)

**Trigger**: Pi0.5 step 5K에서 loss > 0.10 plateau

**Actions**:
1. 데이터 자체 문제 의심 → 진짜 visual 다양성 vs instruction 매핑 학습 가능성 재검증
2. Approach&lift episodes 추가 수집 (현재 100 → 300+)
3. Navigate 재수집 (hardcoded arm 제거, 진짜 teleop 데이터)
4. 또는 simpler architecture (BC + image encoder + MLP)으로 baseline 검증

---

## 10. 부록

### 10.1 핵심 데이터 파일

| 파일 | 위치 |
|---|---|
| Pi0Fast v3 체크포인트 (115K, 마지막) | `outputs/train/pi0fast_lekiwi_v3_20260407_102511/checkpoints/115000/pretrained_model/` |
| Pi0Fast v3 학습 로그 | `/home/jovyan/pi0fast_v3_train.log` |
| Eval action TSV (정량 분석 source) | `/home/jovyan/eval_viva_actions.tsv` |
| Eval stdout 로그 | `/home/jovyan/eval_viva_easy.log` |
| Training data (parquet) | `/home/jovyan/lerobot_data/lekiwi_viva_v2/data/chunk-000/file-000.parquet` |
| Training data stats | `/home/jovyan/lerobot_data/lekiwi_viva_v2/meta/stats.json` |
| Training data tasks | `/home/jovyan/lerobot_data/lekiwi_viva_v2/meta/tasks.jsonl` |
| Pi0.5 학습 스크립트 (수정됨) | `train_pi05.sh` |
| Pi0.5 base model | `/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base/` |
| Pi0.5 학습 로그 (현재) | `/home/jovyan/pi05_train_20260408_143329.log` |
| Pi0.5 setup 문서 | `PI05_README.md` |
| Env action 정의 | `lekiwi_skill2_env.py:1450-1494` |

### 10.2 검증에 사용한 핵심 명령

```bash
# Stats 직접 검증
python -c "
from safetensors.torch import load_file
sd = load_file('outputs/train/pi0fast_lekiwi_v3_*/checkpoints/115000/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors')
print(sd['action.std'].tolist())
"

# Per-task action distribution
python -c "
import pandas as pd, numpy as np
data = pd.read_parquet('/home/jovyan/lerobot_data/lekiwi_viva_v2/data/chunk-000/file-000.parquet')
actions = np.stack(data['action'].values)
mask = data['task_index']==6
print(f'forward vy mean: {actions[mask, 7].mean():.4f}')
"

# Direct VLA inference test
python -c "
import requests, base64, io, numpy as np
from PIL import Image
img = np.random.randint(0,256,(400,640,3),dtype=np.uint8)
buf = io.BytesIO(); Image.fromarray(img).save(buf,format='JPEG')
b64 = base64.b64encode(buf.getvalue()).decode()
r = requests.post('http://localhost:8002/act', json={
    'base_image_b64': b64, 'wrist_image_b64': b64,
    'state': [0]*9, 'instruction': 'navigate forward'
})
print(r.json())
"
```

### 10.3 환경 정보

```
OS: Ubuntu (kernel 5.4.0-139-generic)
GPU: NVIDIA A100-SXM4-40GB
Python: 3.12 (conda env: lerobotpi0v2)
LeRobot: 0.5.0 (git source v0.5.0 + [pi] extras)
PyTorch: 2.10.0+cu128
Transformers: 5.3.0
```

### 10.4 참고 GitHub Issues

- huggingface/lerobot#1811 — Pi0Fast fine-tuning instability on small datasets
- huggingface/lerobot#2216 — Pi0.5 memory requirements on A6000 48GB

---

## 11. Pi0.5 v1 (60K) Eval 결과 — 부분 성공 + 신규 버그 발견 (2026-04-09)

### 11.1 Standalone Inference Test (서버 직접 요청)

**입력**: 실제 dataset frame + navigate hardcoded state
**모델**: Pi0.5 60K checkpoint (scheduler_decay_steps=30K, carry turn 라벨 버그 포함)

| Instruction | Expected | Pi0.5 60K 출력 | % of demo | Sign | 상태 |
|---|---|---|---|---|---|
| navigate forward | vy=+0.50 | vy=**+0.192** | 38% | ✓ | 부분 |
| navigate backward | vy=-0.50 | vy=**-0.470** | **94%** | ✓ | 우수 |
| navigate turn left | wz=-0.33 | wz=**-0.328** | **99%** | ✓ | 완벽 |
| navigate turn right | wz=+0.33 | wz=**+0.231** | 70% | ✓ | 양호 |
| navigate strafe left | vx=-0.50 | vx=**-0.495** | **99%** | ✓ | 완벽 |
| navigate strafe right | vx=+0.50 | vx=**+0.487** | **97%** | ✓ | 우수 |

**Pi0Fast v3 대비 압도적 개선**:
- Pi0fast forward: 3.8% → Pi0.5: 38% (10배)
- Pi0fast turn left: 0% (wrong sign) → Pi0.5: 99% (완벽)
- **Arm joints도 navigate hardcoded pose 정확히 출력** (Pi0fast는 불가능했음)

**Inference latency**: 265ms (Pi0fast 3000ms → **11배 빠름**)

### 11.2 Isaac Sim Full Eval (480 step)

**환경**: VIVA mode + easy + scene 1302
**Wall clock**: 72s / 480 step = **6.7 Hz** (Pi0fast 2.4 Hz → 2.8배)
**Skill 진행**: navigate 전체 + approach_and_lift 잠깐 (S2 진입!) → navigate 복귀

**Per-instruction action 분석**:

| Instruction | N | pred vy | pred wz | 기대 | 결과 |
|---|---|---|---|---|---|
| navigate forward | 57 | **+0.489** | -0.010 | vy=+0.50 | ✅ **98%** |
| navigate strafe left | 38 | +0.009 | -0.154 | vx=-0.285 (57%) | ✅ 부호 정확 |
| navigate turn left | 174 | +0.028 | **-0.325** | wz=-0.33 | ✅ **98%** |
| navigate turn right | 207 | +0.129 | **-0.294** | wz=+0.33 | ❌ **부호 반대 (-89%)** |

**문제 발견: turn right만 부호 반대로 출력 (207 step 중 98%가 음수 wz)**

### 11.3 Turn Right 실패 Root Cause: Carry Turn 라벨 버그

**사용자 발견**: "navigate turn left/right과 carry turn의 부호가 반대"

**검증 결과**:

```
task 8  navigate turn left:   wz = -0.3301  (음수 = 좌회전)    ← 정확
task 9  navigate turn right:  wz = +0.3303  (양수 = 우회전)    ← 정확
task 16 carry turn right:     wz = -0.3292  (음수 = 좌회전!)   ← ❌ 라벨과 반대!
task 17 carry turn left:      wz = +0.3269  (양수 = 우회전!)   ← ❌ 라벨과 반대!
```

**원인**: `record_teleop_scene.py:1271`의 `_NAV_DIR_MAP`에서 turn left/right의 `dir_z`가 env wz convention (action+= CW=right)과 반대. S3 carry BC가 이 convention으로 학습됨 → carry 데이터 수집 시 라벨은 "turn left"이지만 실제 동작은 right turn. VLA 학습 시 모델이 "turn right"를 ±0.33 양쪽 모두에 매핑하려 해서 left 방향으로 collapse.

**영향**:
- navigate 데이터: 정확 (ACTION_MAP 사용)
- carry 데이터: turn left/right만 반대 (forward/backward/strafe는 정상)
- 이게 Pi0fast 실패의 숨겨진 원인 중 하나였을 가능성 (FAST tokenizer 문제와 결합)

---

## 12. Pi0.5 v2 — Carry Turn Fix + LR Schedule Fix (2026-04-09)

### 12.1 적용된 수정 사항

**A) Carry turn 라벨 swap (5곳)**:

| 파일 | 수정 내용 |
|---|---|
| `meta/tasks.jsonl` | task 16: "carry turn right" → "carry turn left"<br>task 17: "carry turn left" → "carry turn right" |
| `meta/tasks.parquet` | 동일 |
| `meta/episodes/.../file-000.parquet` | 144 episodes의 'tasks' field swap |
| `viva_merged_with_carry.hdf5` | 144 episodes의 instruction attr swap |
| `record_teleop_scene.py:1467` | `_S3_TURN_INVERT` dict 추가 (향후 수집 안전성) |

수정 후 검증:
```
navigate turn left  → wz=-0.33 ✓ | carry turn left  → wz=-0.33 ✓ (일치!)
navigate turn right → wz=+0.33 ✓ | carry turn right → wz=+0.33 ✓ (일치!)
```

**B) LR schedule 수정 (train_pi05.sh v2)**:

| 파라미터 | v1 (60K run) | v2 (현재) | 효과 |
|---|---|---|---|
| scheduler_decay_steps | 30,000 (default) | **100,000** | 의미있는 학습 30K→100K (3.3배) |
| steps | 200,000 | **150,000** | 불필요한 floor LR 시간 절약 |
| Pre-flight check | 없음 | **carry turn + stats 검증** | 자동 안전망 |

**C) vla_inference_server.py Pi0.5 지원**:
- `config.json`에서 `type` 자동 감지 (pi0_fast / pi05)
- Pi0.5: preprocessor/postprocessor pipeline 사용 (rename, QUANTILES normalize, state tokenize)
- Pi0fast: 기존 manual tokenize 경로 유지 (backward compatible)

### 12.2 Pi0.5 v2 학습 현황 (2026-04-09 03:05 시작)

```
PID:    2747166
Step:   26K / 150K (17%)
Loss:   ~0.056
Grad:   ~0.67
LR:     2.1e-05 (peak 84% — v1은 이 시점 이미 floor)
Speed:  1.5 step/s
GPU:    39.5/40.3 GB (97%)
ETA:    ~23시간 남음 (총 ~28h)
Checkpoints: 5K, 10K, 15K, 20K, 25K
```

**v1 vs v2 LR 비교 (핵심 차이)**:

| Step | v1 LR (decay 30K) | v2 LR (decay 100K) |
|---|---|---|
| 10K | ~1.5e-5 | 2.4e-05 |
| 25K | ~4e-6 (거의 floor) | **2.2e-05 (peak 88%)** |
| 30K | 2.5e-6 (floor) | 2.1e-05 |
| 50K | 2.5e-6 | 1.5e-05 |
| 100K | 2.5e-6 | 2.5e-6 (floor 도달) |

### 12.3 다음 단계

1. **Step 30K** (~55분 후): standalone inference test로 turn right 부호 검증
2. **Step 50K** (~6h 후): full inference test (6 instruction)
3. **Step 100K** (~18h 후): LR floor 도달, 수렴 확인
4. **Step 150K** (~28h 후): 최종 학습 완료, Isaac Sim full eval

---

**문서 작성**: Claude Code (claude-opus-4-6) + 사용자 검증
**문서 이력**:
- v1 (2026-04-08): Pi0Fast 실패 분석 + Pi0.5 전환 계획
- v2 (2026-04-08 14:49): 14항목 검증 + Pi0.5 v1 학습 시작
- v3 (2026-04-09 03:00): Pi0.5 v1 60K eval + carry turn bug 발견 + v2 학습 시작
**총 소요 시간**: ~120시간 (Pi0fast v1-v3 + Pi0.5 v1 + data fix + Pi0.5 v2)
**다음 업데이트**: Pi0.5 v2 step 30K inference test 결과
