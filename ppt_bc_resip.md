# Diffusion Policy BC + Residual RL (ResiP) 기술 요약

## 1. 전체 파이프라인 개요

```
텔레옵 데이터 (HDF5)
    │
    ▼
┌─────────────────────────────┐
│  Stage 1: Diffusion Policy  │  ← Behavioral Cloning (DDPM)
│  ConditionalUnet1D (~5.3M)  │     obs → 16-step action chunk
└──────────┬──────────────────┘
           │ freeze
           ▼
┌─────────────────────────────┐
│  Stage 2: Residual Policy   │  ← PPO (Reinforcement Learning)
│  MLP (~154K params)         │     [norm_obs, base_action] → residual correction
└─────────────────────────────┘

최종 action = BC_action + residual × per_dim_scale
```

**핵심 아이디어**: 대규모 Diffusion Model이 행동 선행지식(behavioral prior)을 제공하고, 경량 MLP가 RL로 세부 교정만 수행.

---

## 2. Stage 1 — Diffusion Policy BC

### 2.1 모델 아키텍처: ConditionalUnet1D

```
입력: noisy_action (B, pred_horizon, act_dim) + timestep + obs
                    ↓
         ┌──────────────────┐
         │  Encoder          │
         │  [64] → [128] → [256]  각 레벨 2×ResBlock + Downsample
         ├──────────────────┤
         │  Mid block        │  2×ResBlock
         ├──────────────────┤
         │  Decoder          │  skip connections
         │  [256] → [128] → [64]  각 레벨 2×ResBlock + Upsample
         └──────────────────┘
                    ↓
         denoised_action (B, pred_horizon, act_dim)
```

- **1D Convolution**: 시간축(action sequence)을 따라 convolution
- **FiLM Conditioning**: diffusion timestep embedding(256D) + obs를 concat → scale/bias로 feature 변조
- **ResBlock**: Conv1d → GroupNorm(8) → Mish → Conv1d → GroupNorm → Mish + skip connection
- **파라미터 수**: down_dims=[64, 128, 256] 기준 약 5.3M

### 2.2 Diffusion Process

| 항목 | Training (DDPM) | Inference (DDIM) |
|------|-----------------|------------------|
| 스케줄러 | DDPMScheduler | DDIMScheduler |
| Beta schedule | squaredcos_cap_v2 | 동일 |
| Diffusion steps | 100 | **16** (가속) |
| Prediction type | epsilon (노이즈 예측) | 동일 |
| Clip sample | True | True |

**학습 손실함수**: 표준 DDPM epsilon-prediction loss
```
L = MSE(ε_predicted, ε_true)

1. 클린 액션 시퀀스에 랜덤 timestep t의 노이즈 추가
2. U-Net이 추가된 노이즈를 예측
3. 예측 노이즈와 실제 노이즈의 MSE 최소화
```

### 2.3 Receding Horizon 실행

```
pred_horizon = 16  (예측 길이)
action_horizon = 8  (실행 길이)

t=0: 예측 [a0..a15], 실행 [a0..a7]
t=8: 예측 [a8..a23], 실행 [a8..a15]  ← warm-start
...
```

- **Warm-start**: 이전 예측 결과를 시간 이동 후 부분 노이즈 추가 → 시간적 일관성 대폭 향상
- `warmstart_timestep = 50` (100 중 50 — 절반 수준 노이즈)

### 2.4 데이터 정규화: LinearNormalizer

- 학습 데이터 전체에서 per-dimension min/max 계산
- `[-1, 1]` 범위로 정규화/역정규화
- obs와 action 각각 별도 정규화 파라미터 유지

### 2.5 학습 설정

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW (lr=1e-4, betas=(0.95, 0.999), wd=1e-6) |
| Scheduler | CosineAnnealingLR (eta_min = lr × 0.01) |
| Batch size | 256 |
| Gradient clipping | 1.0 |
| EMA decay | 0.995 (SwitchEMA) |
| Epochs | 150~500 (skill에 따라) |

### 2.6 데이터 Augmentation (Skill-3 한정)

| Augmentation | 확률/강도 | 대상 obs 차원 | 목적 |
|---|---|---|---|
| vel_dropout | p=0.5 | obs[6:15] (base vel) | 제로 속도 초기화 mismatch 해소 |
| grip_noise | std=0.25 | obs[5] (gripper) | 그리퍼 상태 변동 robust화 |
| armvel_dropout | p=0.3 | obs[15:21] (arm vel) | arm velocity 노이즈 robust화 |

### 2.7 입출력 공간

| 항목 | Skill-2 (Approach&Grasp) | Skill-3 (Carry&Place) |
|------|---|---|
| Obs dim | 30D | 29D (33D with augmented) |
| Action dim | 9D | 9D |
| Action 순서 | [arm0-4, gripper, base_vx, base_vy, base_wz] | 동일 |

---

## 3. Stage 2 — Residual RL (ResiP)

### 3.1 핵심 원리

```python
# 1. Frozen BC가 base action 생성
base_naction = frozen_DP.base_action_normalized(obs)     # 정규화된 BC 출력

# 2. Residual policy가 보정량 생성
nobs = frozen_DP.normalizer(obs, "obs", forward=True)     # obs 정규화
residual_input = concat([nobs, base_naction])              # 결합
residual_action = ResidualPolicy(residual_input)           # 보정량

# 3. 최종 액션 = BC + 보정 (per-dim scale 적용)
final_naction = base_naction + clamp(residual, -1, 1) * per_dim_scale
final_action = frozen_DP.normalizer(final_naction, "action", forward=False)  # 역정규화
```

**BC는 완전히 frozen** — gradient가 BC로 흐르지 않음. RL은 오직 residual MLP만 학습.

### 3.2 Residual Policy 아키텍처

```
입력: [normalized_obs, base_action]  (obs_dim + act_dim)
         │
    ┌────┴────────────────────┐
    │  Actor MLP              │  Critic MLP (동일 구조, 별도 파라미터)
    │  Linear(in, 256) + ReLU │  Linear(in, 256) + ReLU
    │  Linear(256, 256) + ReLU│  Linear(256, 256) + ReLU
    │  Linear(256, act_dim)   │  Linear(256, 1)
    └────┬────────────────────┘
         │
    residual_mean (+ Gaussian noise for exploration)
```

- **총 파라미터**: ~154K (BC 5.3M의 3%)
- **초기화**: Hidden layer = Kaiming, Output layer = Orthogonal(std=0) → **초기 출력 ≈ 0** (identity start)
- **Gaussian 정책**: 학습 가능한 log_std (초기값 -1.0)

### 3.3 Per-Dimension Action Scale

BC의 강점과 약점에 맞춰 차원별로 보정 크기를 제한:

| 차원 | Scale | 근거 |
|------|-------|------|
| arm[0:5] | 0.20 | BC의 arm trajectory는 이미 양호 → 작은 보정 |
| gripper[5] | 0.30 | BC의 gripper timing은 대체로 양호하나 미세 조정 필요 |
| base[6:9] | 0.35 | BC의 가장 큰 약점 — base positioning → 큰 보정 |

### 3.4 PPO 학습 설정

| 항목 | 값 | 비고 |
|------|-----|------|
| Discount (γ) | 0.999 | 긴 horizon |
| GAE λ | 0.95 | |
| Clip coefficient | 0.2 | |
| Target KL | 0.1 | Early stopping per epoch |
| Entropy coef | 0.001 | 탐색 유지 |
| Value function coef | 1.0 | |
| Max grad norm | 1.0 | |
| Actor lr | 3e-4 (AdamW) | |
| Critic lr | 5e-3 (AdamW) | Actor의 ~17× |
| LR scheduler | CosineAnnealing | |
| Update epochs | 50 (max) | KL early stop로 실제 더 적게 |
| Envs | 64 (parallel) | Isaac Sim vectorized |
| Rollout steps | 700 | |
| Num minibatches | 1 (full-batch) | |

### 3.5 Warmup 메커니즘

```
초기 500 steps: BC만 실행 (residual 꺼짐)
  ↓ 30 iterations에 걸쳐 선형 감소
이후: BC + residual 정상 작동
```

- 주기적 리셋: warmup 중 500 steps마다 env 리셋 → BC에게 짧은 시도 여러 번 제공
- 목적: BC가 합리적인 기본 궤적을 만들어 놓은 후 residual이 보정 시작

### 3.6 보상 구조 (Skill-2: Approach & Grasp, v6.4)

#### Milestone 보상 (일회성)

| 보상 | 값 | 조건 |
|------|-----|------|
| R1: Gripper Open | +10 | grip > 1.0 (R2/R3의 gate) |
| R3: Verified Grasp | +25 | 5-step 연속 grasp 유지 + dual contact + EE in bbox |
| R3b: Verified Lift | +100 | 25-step 연속 hold + height > 5cm + EE < 10cm |
| R4c: Pose Return | +300 | LIFTED_POSE similarity > 0.80 × 8 연속 step |
| R6: Soft Lift | +100 | 15-step 연속 + grip quality > 0.3 |

#### Per-step 보상

| 보상 | 가중치 | 내용 |
|------|--------|------|
| R2: Arm Approach (XY) | ×30 | **base-subtracted** EE 접근 (base rushing exploit 방지) |
| R4: Lift Height | ×200 | 0.04~0.12m 높이 선형 보상 |
| R4b: Lifted Pose | ×160 | LIFTED_POSE까지의 Gaussian similarity |
| R5: Sustained Lift | ×50 | 15+ step hold × grip quality |
| R7: Time Penalty | -0.01 | 항상 |
| R8: Ground Contact | -2.0 | 그리퍼 접지 + 물체 낮음 |

#### 핵심 설계 원칙

1. **Milestone-only grasp reward**: Per-step grasp shaping은 "don't grasp" exploit 유발 → milestone 방식이 정답
2. **Base-subtracted approach (R2)**: `(prev_ee - cur_ee) - (prev_base - cur_base)` — residual이 base만 전진해서 EE 접근 위장하는 exploit 차단
3. **Verified grasp (R3)**: 5-step 지속 + dual contact + EE-bbox 확인 — phantom grasp 방지
4. **Pre-grasp grip shaping 금지**: BC의 gripper timing 시퀀스를 residual이 덮어쓰면 diverge

### 3.7 LIFTED_POSE Target

```python
LIFTED_POSE = [-0.045, -0.194, 0.277, -0.908, 0.020]  # (action space)
# joint1 ≈ -1.0, joint2 ≈ +1.0 = 팔을 몸 쪽으로 완전히 접는 자세
# 22개 텔레옵 에피소드 분석에서 도출
```

---

## 4. 성능 비교

### 4.1 BC-only vs BC+ResiP

| 메트릭 | BC-only | BC+ResiP (v57) |
|--------|---------|----------------|
| 물체까지 도달 | ~60% | ~90%+ |
| 파지(grasp) 성공 | ~15% | ~35% |
| 들어올림(lift) 유지 | 거의 0% | ~35% |

- BC-only의 한계: obs distribution mismatch (학습 시 데모 obs vs 실행 시 env obs), base positioning 부정확
- ResiP가 보완하는 것: base 위치 미세 조정, 그리퍼 타이밍 교정, 자세 안정화

### 4.2 학습 진행 히스토리 (주요 버전)

| 버전 | 성공률 | 핵심 변경 |
|------|--------|-----------|
| v52c | 18.75% | Base-subtracted R2 도입 |
| v53f | 25.49% | Verified grasp (5-step) |
| v54b | 28.81% | Verified lift + pose similarity |
| v56b | 31.84% | Sustained lift + grip quality |
| v57 | **35.25%** | Pose return milestone (+300) |
| v6.4b | 28.91% | Phantom grasp fix |
| v7_GTO | 25.29% | GTO (Grasp Timeout) 도입 |

### 4.3 발견된 주요 Exploit 패턴

| Exploit | 메커니즘 | 해결 |
|---------|----------|------|
| Base rushing | residual이 base만 전진 → EE approach 위장 | R2 base-subtracted |
| Phantom grasp | contact spike → env_grasped 오판 | R3 5-step + dual contact + EE-bbox 검증 |
| Don't grasp | per-step grasp reward → 파지 안 하고 open 유지가 더 유리 | milestone-only |
| Grip override | residual이 BC gripper timing 덮어씀 → diverge | pre-grasp grip shaping 완전 제거 |

---

## 5. 추론 파이프라인 (Inference)

```
매 step:
  1. env에서 obs 수신 (30D)
  2. frozen_DP에 obs 전달 → action queue 확인
     - queue 비어있으면: DDIM 16-step inference → 8개 action 생성
     - queue에 action 있으면: dequeue
  3. base_naction (정규화된 BC 출력) 획득
  4. obs 정규화 → [nobs, base_naction] concat → ResidualPolicy forward
  5. residual_mean × per_dim_scale → base_naction에 더함
  6. 역정규화 → 최종 9D action
  7. env.step(action)
```

- BC의 DDIM inference: 16 denoising steps (학습 시 100 → 추론 16으로 가속)
- Residual 추론: 단순 MLP forward (< 1ms)
- 전체 추론 시간: DDIM 지배적 (~50ms on GPU), 8 step에 한 번만 호출

---

## 6. 체크포인트 구조

### BC 체크포인트 (`dp_bc_epoch150.pt`)
```python
{
    "config": {
        "obs_dim": 30,
        "act_dim": 9,
        "pred_horizon": 16,
        "action_horizon": 8,
        "num_diffusion_iters": 100,
        "inference_steps": 4,        # RL에서 속도를 위해 4로 줄임
        "down_dims": [64, 128, 256],
    },
    "model_state_dict": {
        "model.*": ...,              # ConditionalUnet1D 가중치
        "normalizer.*": ...,         # LinearNormalizer 통계
    },
}
```

### ResiP 체크포인트 (`resip_best.pt`)
```python
{
    "residual_policy_state_dict": ...,   # ResidualPolicy (actor + critic)
    "args": {                             # 학습 설정 복원용
        "hidden_sizes": [256, 256],
        "action_scale": 0.1,
        "logstd_init": -1.0,
        "learn_std": True,
    },
    "iteration": 96,
    "best_soft_lift": 42,
    "metrics": {...},
}
```
