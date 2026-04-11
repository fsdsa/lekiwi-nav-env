# Residual RL (ResiP) 상세 기술 문서

## 1. 개요: Residual RL이란?

### 1.1 동기
Behavioral Cloning(BC)은 텔레옵 데이터에서 정책을 모방 학습하지만 두 가지 근본적 한계가 있다:
- **Distribution shift**: 학습 시 관측(demo obs)과 실행 시 관측(env obs) 분포 불일치
- **Compounding error**: 작은 오차가 시간에 따라 누적

BC만으로는 성공률 ~15%에 머무른다. Residual RL은 이 gap을 RL로 메운다.

### 1.2 핵심 아이디어
```
최종 action = frozen_BC_action + residual_correction × per_dim_scale
```
- **대규모 Diffusion Model (5.3M)**: 행동 선행지식 제공 (궤적 형태, 그리퍼 타이밍)
- **경량 MLP (154K)**: RL(PPO)로 학습, base 위치/그리퍼 미세 교정만 담당
- BC 파라미터는 **완전 frozen** — gradient가 BC로 흐르지 않음

### 1.3 왜 처음부터 RL만 안 쓰나?
- 9D action space × 30D obs = 탐색 공간이 너무 넓음
- BC가 "대략 맞는 궤적"을 제공 → RL의 탐색 범위를 좁힘
- Per-dim scale로 BC가 잘하는 부분(arm trajectory)은 보정량 제한, 못하는 부분(base positioning)은 보정량 확대

---

## 2. 아키텍처

### 2.1 전체 추론 흐름 (매 step)

```
env.obs (30D)
    │
    ├──→ frozen_DP.base_action_normalized(obs)
    │         │  DDIM 4-step denoising (8 step마다 1번)
    │         │  action queue에서 dequeue
    │         ▼
    │    base_naction (9D, 정규화)
    │
    ├──→ frozen_DP.normalizer(obs, "obs") → clamp([-3,3]) → nan_to_num
    │         │
    │         ▼
    │    normalized_obs (30D)
    │
    └────────────┐
                 ▼
    ┌───────────────────────┐
    │ concat([norm_obs(30D), │
    │         base_action(9D)]) = 39D │
    └──────────┬────────────┘
               │
    ┌──────────▼────────────┐
    │   Residual Policy MLP  │  (154K params)
    │   [256, ReLU, 256, ReLU] │
    │   → residual_mean (9D) │
    │   + Gaussian noise     │
    │     (학습 시만)          │
    └──────────┬────────────┘
               │
    residual = clamp(sampled_action, -1, 1)
               │
    final_naction = base_naction + residual × per_dim_scale
               │
    action = denormalize(final_naction) → env.step(action)
```

### 2.2 Residual Policy 네트워크

```
Actor MLP:
  Input(39D) → Linear(39, 256) → ReLU
             → Linear(256, 256) → ReLU
             → Linear(256, 9)        ← orthogonal init (std=0.0)
                                       → 초기 출력 ≈ 0 (BC와 동일하게 시작)

Critic MLP (별도 파라미터):
  Input(39D) → Linear(39, 256) → ReLU
             → Linear(256, 256) → ReLU
             → Linear(256, 1)        ← orthogonal init (std=0.25)
                                       → bias=0.25

Log-std: 학습 가능 파라미터, 초기값 -1.0 (std ≈ 0.368)
```

| 항목 | 값 |
|------|-----|
| 총 파라미터 | ~154K |
| Hidden dims | [256, 256] |
| Activation | ReLU |
| Actor init | Hidden: Kaiming, Output: Orthogonal(std=0) |
| Critic init | Hidden: Kaiming, Output: Orthogonal(std=0.25, bias=0.25) |
| Action distribution | Gaussian (학습 가능 log_std) |
| NaN 안전장치 | mean: nan→0, std: nan→1, clamp(min=1e-6) |

### 2.3 Per-Dimension Action Scale

BC의 차원별 능력에 맞춰 residual 보정량 제한:

```python
per_dim_scale = [
    0.20, 0.20, 0.20, 0.20, 0.20,  # arm[0:5]   — BC가 잘함, 작은 보정
    0.30,                            # gripper[5]  — 중간 보정
    0.35, 0.35, 0.35,               # base[6:9]   — BC가 가장 약함, 큰 보정
]
```

**효과**: residual 출력 범위 [-1, 1] × scale → 실제 보정 범위
- arm: ±0.20 (정규화 공간)
- gripper: ±0.30
- base: ±0.35

BC가 만든 행동의 **구조(temporal shape)**를 유지하면서 **정밀도(precision)**만 개선.

---

## 3. PPO 학습

### 3.1 학습 루프 구조

```
for iteration in range(total_iterations):
    ┌─────────────────────────────────┐
    │  (1) Warmup Phase               │
    │      BC만 실행, env 리셋         │
    │      (iter 0~30, 점진적 감소)     │
    └─────────────┬───────────────────┘
                  ▼
    ┌─────────────────────────────────┐
    │  (2) Rollout Collection          │
    │      700 steps × 64 envs        │
    │      BC + residual → env.step   │
    │      reward 계산, 버퍼 저장       │
    └─────────────┬───────────────────┘
                  ▼
    ┌─────────────────────────────────┐
    │  (3) GAE Computation             │
    │      advantages, returns 계산    │
    └─────────────┬───────────────────┘
                  ▼
    ┌─────────────────────────────────┐
    │  (4) PPO Update                  │
    │      최대 50 epochs              │
    │      KL > 0.1이면 early stop    │
    └─────────────┬───────────────────┘
                  ▼
    ┌─────────────────────────────────┐
    │  (5) Eval & Checkpoint           │
    │      5 iteration마다 평가        │
    │      best SR → resip_best.pt    │
    └─────────────────────────────────┘
```

### 3.2 하이퍼파라미터

| 범주 | 항목 | 값 |
|------|------|-----|
| **환경** | num_envs | 64 |
| | rollout_steps | 700 |
| | device | cuda:0 |
| **PPO** | discount (γ) | 0.999 |
| | GAE λ | 0.95 |
| | clip coefficient | 0.2 |
| | target KL | 0.1 |
| | entropy coef | 0.001 |
| | value function coef | 1.0 |
| | max grad norm | 1.0 |
| | normalize advantages | True |
| | clip value loss | False |
| | update epochs (max) | 50 |
| | num minibatches | 1 (full batch) |
| **옵티마이저** | actor | AdamW, lr=3e-4, wd=1e-6 |
| | critic | AdamW, lr=5e-3, wd=1e-6 |
| | LR scheduler | CosineAnnealing (eta_min=lr×0.01) |
| **Warmup** | initial steps | 500 |
| | decay iterations | 30 |
| | periodic reset | 매 500 steps |

### 3.3 PPO 손실 함수

**Policy Loss (Clipped Surrogate)**:
```
ratio = exp(new_log_prob - old_log_prob)
L_clip = max(-A × ratio, -A × clamp(ratio, 1-ε, 1+ε))
```
- ε = 0.2 (clip_coef)
- A = normalized advantage

**Value Loss**:
```
L_value = 0.5 × MSE(V_pred, returns)
```

**Entropy Bonus**:
```
L_entropy = -0.001 × H(π)
```

**Total**:
```
L = L_clip - L_entropy + L_value × 1.0
```

**KL Early Stopping**: 매 update epoch 후 KL divergence 계산, KL > 0.1이면 나머지 epoch 건너뜀.

### 3.4 GAE (Generalized Advantage Estimation)

```
δ_t = r_t + γ × V(s_{t+1}) × (1 - done_{t+1}) - V(s_t)
A_t = Σ_{l=0}^{T-t} (γλ)^l × δ_{t+l}
returns = A + V    (value function target)
```
- γ = 0.999 (장기 미래 보상 중시)
- λ = 0.95 (bias-variance tradeoff)

### 3.5 Clamp-Logprob Fix

```python
# 1. 원래 sampling
ra_sampled = Normal(mean, std).sample()

# 2. Clamp (action 범위 제한)
ra_clamped = clamp(ra_sampled, -1.0, 1.0)

# 3. Clamped action으로 log_prob 재계산 (핵심!)
_, log_prob, _, _, _ = policy.get_action_and_value(obs, ra_clamped)
```

**왜 필요한가**: clamp 전의 log_prob를 사용하면 clamp 경계에서 ratio가 부정확 → PPO 학습 불안정. Clamped action 기준으로 재계산해야 정확한 importance sampling ratio.

---

## 4. Warmup 메커니즘

### 4.1 목적
BC가 합리적인 기본 궤적을 만들어 놓은 후 residual이 보정 시작.
BC 없이 residual만으로는 의미 있는 탐색 불가 (action space 너무 넓음).

### 4.2 동작

```
Iteration 0:   warmup = 500 steps (BC만 실행)
Iteration 1:   warmup = 483 steps
...
Iteration 15:  warmup = 250 steps
...
Iteration 30:  warmup = 0 steps (이후 warmup 비활성)
```

- 선형 감소: `warmup_steps = 500 × (1 - iter/30)`
- Warmup 중 residual 출력 = 0 (BC action 그대로 사용)

### 4.3 주기적 리셋 (WU_RESET)

```
Warmup 500 steps 내에서:
  step 0~500: BC 첫 번째 시도
  step 500: env 리셋 → BC deque 클리어
  step 500~1000: BC 두 번째 시도
  ...
```

**왜**: BC가 한 번에 500 step 실행하면 실패 시 나머지 7000+ step을 허비. 주기적 리셋으로 BC에게 짧은 시도 여러 번 제공 → 더 나은 초기 상태에서 RL 시작.

---

## 5. 보상 설계 (Skill-2: Approach & Grasp)

### 5.1 보상 설계 원칙

1. **Milestone 기반**: per-step shaping보다 milestone 보상이 exploit-resistant
2. **Base-subtracted**: arm 접근 보상에서 base 이동분을 차감 → "base rushing" exploit 차단
3. **Verified detection**: grasp/lift 판정을 단일 step이 아닌 N-step 지속으로 → phantom grasp 방지
4. **Pre-grasp grip shaping 금지**: BC의 gripper timing 시퀀스를 residual이 덮어쓰면 diverge

### 5.2 보상 구성 요소 (8개)

#### Milestone 보상 (일회성, 조건 달성 시 한 번만 지급)

```
R1: Gripper Open        +10     grip > 1.0 (BC가 그리퍼를 열었는지 확인)
                                 → R2, R3의 gate 역할

R3: Verified Grasp      +25     5 step 연속으로:
                                 - env가 grasp 감지
                                 - dual contact (양쪽 jaw)
                                 - EE가 물체 bbox 내부
                                 - 물체 서있음
                                 - grip < 0.65

R3b: Verified Lift      +100    25 step 연속으로:
                                 - R3 조건 + height > 5cm
                                 - EE 거리 < 10cm

R4c: Pose Return        +300    8 step 연속 LIFTED_POSE similarity > 0.80
                                 (verified lift 이후에만)

R6: Soft Lift           +100    15 step 연속 hold + grip quality > 0.3
```

#### Per-Step 보상 (매 step 지급)

```
R2: Arm Approach (XY)    ×30    (prev_ee - cur_ee) - (prev_base - cur_base)
                                 → base 이동분 제거, 순수 arm 접근만 보상
                                 → clamp(-1.0, 3.0)
                                 → grip open(R1) 이후, grasp(R3) 이전에만 활성

R4: Lift Height          ×200   clamp((objZ - 0.04) / 0.08, 0, 1)
                                 → 4cm~12cm 범위에서 선형 보상
                                 → 3 step 이상 지속 필요

R4b: Lifted Pose         ×160   exp(-||joints - LIFTED_POSE||² / (2×0.35²))
                                 → Gaussian similarity to target pose
                                 → verified lift 전: 15% 강도
                                 → verified lift 후: 100% 강도

R5: Sustained Lift       ×50    exp(-((grip - 0.50) / 0.20)²) × 50
                                 → grip quality Gaussian (최적: grip=0.50)
                                 → 15 step 이상 hold 유지 시

R7: Time Penalty         -0.01  매 step 무조건

R8: Ground Contact       -2.0   그리퍼 접지 force > 1N + objZ < 5cm
```

### 5.3 보상 흐름도

```
Episode 시작
    │
    ▼
[R7: time -0.01] ← 매 step
    │
    ▼ grip > 1.0?
    ├─ No → 대기 (arm folded)
    └─ Yes → R1: +10 (gripper open milestone)
              │
              ▼
         [R2: arm approach ×30] ← 매 step, base-subtracted
              │
              ▼ 5-step verified grasp?
              ├─ No → R2 계속
              └─ Yes → R3: +25 (verified grasp)
                        │
                        ▼
                   [R4: lift height ×200] ← 매 step
                   [R4b: lifted pose ×160 × 0.15] ← pre-verified-lift
                        │
                        ▼ 25-step verified lift?
                        ├─ No → R4, R4b 계속
                        └─ Yes → R3b: +100 (verified lift)
                                  │
                                  ▼
                             [R4b: ×160 × 1.0] ← post-verified-lift (full)
                             [R5: sustained lift ×50]
                                  │
                                  ▼ pose similarity > 0.80 × 8 steps?
                                  ├─ No → R4b, R5 계속
                                  └─ Yes → R4c: +300 (pose return)
                                            │
                                            ▼ 15-step hold + gq > 0.3?
                                            └─ Yes → R6: +100 (soft lift)
```

### 5.4 LIFTED_POSE (목표 자세)

```python
LIFTED_POSE = [-0.045, -0.194, 0.277, -0.908, 0.020]
# action space 값 (정규화됨)
# 22개 텔레옵 에피소드에서 공통적으로 관찰된 "들어올린 자세"
# joint1 ≈ -1.0, joint2 ≈ +1.0 = 팔을 몸 쪽으로 완전히 접음
```

### 5.5 발견된 Exploit과 대응

| Exploit | 메커니즘 | 결과 | 대응 |
|---------|----------|------|------|
| **Base rushing** | residual이 base만 전진 → EE가 물체에 가까워진 것처럼 보임 | R2에서 가짜 보상 수령 | R2에 base-subtracted 적용 |
| **Phantom grasp** | 순간적 contact spike → env가 grasp 오판 | 가짜 grasp milestone 달성 | R3에 5-step 연속 조건 + dual contact + bbox 검증 |
| **Don't grasp** | per-step grasp reward → grip open 유지가 더 유리 | 절대 잡으려 하지 않음 | milestone-only grasp reward |
| **Grip override** | residual이 BC의 gripper timing을 덮어씀 | 잡을 타이밍에 그리퍼가 열림 | pre-grasp grip shaping 완전 제거 |
| **GTO (Grasp Timeout)** | phantom grasp → ms_gr=True 영구 유지 → lift 없이도 R4/R5 보상 | 가짜 hold 보상 | 30 step 연속 hold 실패 시 ms_gr 리셋 |
| **v_loss 폭발** | 5% 에피소드만 R4×200 → return 분산 극대 | critic 학습 실패 → 정책 사망 | normalize_reward=True |

---

## 6. 환경 설정

### 6.1 Env 구성

| 항목 | 값 |
|------|-----|
| Env class | Skill2Env (lekiwi_skill2_eval.py) |
| Num envs | 64 (parallel, Isaac Sim vectorized) |
| Device | cuda:0 |
| Obs dim | 30D (actor), 37D (critic) |
| Action dim | 9D [arm5, grip1, base3] |
| Episode length | 300s (env step에 의한 자동 종료) |
| Action body→IK | ik_vx=body_vy, ik_vy=-body_vx, ik_wz=body_wz |

### 6.2 핵심 Env 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| grasp_contact_threshold | 0.55 | contact force 기반 grasp 판정 |
| grasp_gripper_threshold | 0.65 | grip 위치 기반 grasp 판정 |
| grasp_max_object_dist | 0.50 | EE-물체 최대 거리 |
| max_lin_vel | 0.5 m/s | base 선속도 제한 |
| max_ang_vel | 3.0 rad/s | base 각속도 제한 |
| arm_action_to_limits | True | action [-1,1] → joint limits 매핑 |
| spawn_heading_noise_std | 0.3 | 초기 heading 노이즈 |
| spawn_heading_max_rad | 0.5 | 초기 heading 범위 |

### 6.3 Domain Randomization (선택적)

| 파라미터 | 범위 | 목적 |
|----------|------|------|
| Object static friction | ×1.0~1.5 | 물체 마찰 변동 |
| Object dynamic friction | ×1.0~1.5 | 동적 마찰 변동 |
| Wheel damping | 0.3~3.0 | 가장 중요한 DR |
| Object mass | 0.5~2.0 | 실제 편차 큼 |
| Obs noise (joint) | 0.01 rad | 센서 노이즈 |
| Obs noise (base_vel) | 0.02 m/s | 센서 노이즈 |
| Action delay | 1 step | SSH/ZMQ 지연 |

---

## 7. 학습 결과

### 7.1 버전별 성공률 추이

```
v52c  ██████████░░░░░░░░░░  18.75%  — base-subtracted R2 도입
v53f  ████████████░░░░░░░░  25.49%  — reward normalization
v54b  ██████████████░░░░░░  28.81%  — stronger grasp verification
v55d  ████████████░░░░░░░░  24.41%  — ground contact sensor
v56   ████████████████░░░░  31.64%  — reward tuning
v56b  ████████████████░░░░  31.84%  — fine-tuning
v57   █████████████████░░░  35.25%  — pose return milestone (+300)
v6.4b ██████████████░░░░░░  28.91%  — phantom grasp fix
v7    ████████████░░░░░░░░  25.29%  — GTO 도입
```

### 7.2 BC-only vs ResiP 비교

| 메트릭 | BC-only | ResiP (best) | 개선 |
|--------|---------|--------------|------|
| 물체 도달 | ~60% | ~90%+ | +30%p |
| 파지 성공 | ~15% | ~35% | +20%p |
| Lift 유지 | ~0% | ~35% | **0→35%** |
| Base 정밀도 | 0.4m 오차 | 0.1m 이내 | 4× 개선 |

### 7.3 체크포인트 구조

```python
{
    "residual_policy_state_dict": ...,        # actor + critic weights
    "optimizer_actor_state_dict": ...,         # 학습 재개용
    "optimizer_critic_state_dict": ...,
    "dp_checkpoint": "path/to/bc.pt",          # 사용한 BC 경로
    "dp_config": {...},                         # BC config 복원용
    "iteration": 96,
    "global_step": 4300000,
    "success_rate": 0.3525,
    "best_soft_lifts": 42,
    "args": {                                   # CLI 인자 전체
        "hidden_sizes": [256, 256],
        "action_scale": 0.1,
        "logstd_init": -1.0,
        "learn_std": True,
        "action_scale_arm": 0.20,
        "action_scale_gripper": 0.30,
        "action_scale_base": 0.35,
    },
}
```

---

## 8. 멀티스킬 확장

### 8.1 Navigate (Skill-1)

| 항목 | Approach&Grasp 대비 차이 |
|------|--------------------------|
| Obs dim | 26D (arm5+grip1+base3+dir3+lidar8+init6) |
| Per-dim scale | arm=0.05, grip=0.05, **base=0.0** |
| Warmup | 없음 |
| 보상 | velocity tracking (lin+ang) + tucked pose + smoothness |
| 목적 | BC base action은 그대로 쓰고, arm pose drift만 교정 |

### 8.2 Carry & Place (Skill-3)

| 항목 | 내용 |
|------|------|
| 전제 | Skill-2 expert가 먼저 물체를 lift |
| Phase A | 운반 (base 이동, arm 고정, grip 유지) |
| Phase B | 배치 (arm 펼치기, grip 열기, 목표 위치) |
| Per-dim scale (A) | arm=0.0, grip=0.05, base=0.10 |
| Per-dim scale (B) | arm 점진 증가, grip gated by arm1>2.0 |
| 핵심 보상 | 목표 도달 + placement quality (Gaussian) + release + success |

### 8.3 Combined S2→S3 학습

```
Phase 0 (S2 Frozen Expert):
  Frozen Skill-2 BC+ResiP가 자율 실행
  물체 grasp + lift (400 step 유지 시 전환)
        │
        ▼ lift 성공
Phase 1 (S3 Trainable):
  Dest object 스폰
  Skill-3 BC + Residual(학습 대상)
  carry → approach dest → place
```

- S2 실행 중 step은 PPO update에서 제외 (done=1.0 마스킹)
- S3 phase만 GAE/PPO 학습에 반영
- Curriculum: place_hold → release → carry_stabilize → full

---

## 9. 핵심 교훈 요약

### 9.1 보상 설계

| 교훈 | 상세 |
|------|------|
| Milestone > Shaping | per-step reward는 exploit 유발, milestone이 안전 |
| Base-subtracted | arm approach에서 base 이동분 제거 필수 |
| Multi-step 검증 | 단일 step 감지는 phantom 유발, N-step 지속 조건 필요 |
| Pre-grasp grip 금지 | BC의 gripper 시퀀스를 보존해야 grasp 성공 |

### 9.2 학습 안정성

| 교훈 | 상세 |
|------|------|
| v_loss 폭발 | R4×200 보상이 5% 에피소드에만 발생 → return 분산 극대 → critic 학습 실패 |
| Warmup 필수 | BC 없이 시작하면 의미 있는 탐색 불가 |
| WU_RESET | BC에게 짧은 시도 여러 번 > 긴 시도 한 번 |
| KL early stop | target_kl=0.1로 정책 붕괴 방지 |
| Critic lr >> Actor lr | critic 5e-3 vs actor 3e-4 (17×) — value 학습이 더 어려움 |

### 9.3 시스템 설계

| 교훈 | 상세 |
|------|------|
| Per-dim scale이 핵심 | BC가 잘하는 차원 보호 + 못하는 차원 집중 교정 |
| Identity init | residual 초기 출력 ≈ 0 → 학습 초기 = 순수 BC |
| DDIM 4-step | 추론 속도 가속 (16→4), 품질 열화 미미 |
| Clamp-logprob fix | 경계값에서 PPO ratio 정확도 보장 |
| Frozen BC | BC gradient 차단으로 behavioral prior 보존 |
