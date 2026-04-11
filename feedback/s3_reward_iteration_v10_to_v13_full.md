# LeKiwi Skill-3 BC→RL Reward Iteration: v10→v13

## 1. Task 개요

**로봇:** LeKiwi 모바일 매니퓰레이터 (3-wheel omni base + 5DOF arm + 1DOF gripper)
**Task:** 약병을 들고 0.6~0.9m 거리의 컵 옆에 place하고 rest pose
**파이프라인:** 텔레옵 데모(20개) → BC(Diffusion Policy, 36D obs) → Residual RL(PPO, 2048 envs)

### 구조: Combined S2→S3

S2 expert(frozen ResiP)가 물체를 들어올림 → lift 400step 유지 → dest 컵 스폰 → S3 phase 시작.

S3는 두 sub-phase로 나뉨:
- **Phase A (carry):** base가 dest까지 이동. arm/grip에 RL 개입 없음 (BC 그대로). base만 약간 보정.
- **Phase B (place):** base→dest XY < 0.42m이면 진입 (latch). arm을 내리고 물체를 놓음.

### Action space

9D: arm(5) + gripper(1) + base(3). Residual RL이 BC action 위에 scale된 residual을 더함.

---

## 2. 데모 데이터 특징

### arm1 (shoulder pitch)
- 20개 **전부** arm1_max ≥ 2.67, 평균 ~3.02
- arm descent가 place의 핵심 메커니즘. -0.2 → 3.0+ (약 3.21 radian, ~583 step)

### grip
- **action[5]은 항상 -0.68 (닫기 명령).** grip_pos가 높은 건 물체가 jaw를 물리적으로 밀기 때문
- **12/20 (60%) grip을 의도적으로 안 열고 arm descent만으로 place**
- 나머지 8개도 objZ가 이미 바닥(0.0017)인 상태에서 grip open = "이미 놓은 후 손 떼기"
- **데모의 place = "arm을 깊이 내려서 물체를 바닥에 놓기". grip open은 원인이 아니라 결과.**

### place 시 src_dst
- 데모 place 시 src_dst(source↔dest 거리) = 0.119 평균

---

## 3. BC Epoch 비교 (핵심 발견)

### BC v5 (epoch 150) — 20ep eval

| 지표 | 값 |
|---|---|
| Place REAL | **7/20 (35%)** |
| Phase B 진입 | 17/20 |
| Arm descent 성공 (arm1>2.5) | **11/17 (65%)** |
| Drop | 6 (ghost 2, PhB 3, topple 1) |
| Timeout | 6 |
| Place src_dst mean | 0.129 |

**패턴 분류:**
- arm descent + grip open + PLACE: 5건 (EP3,6,9,12,20) — 가장 robust
- arm descent + grip 닫힘 + PLACE: 2건 (EP8,15) — 물리적 release
- arm descent + drop: 3건 — 물리 불안정
- arm descent + timeout: 4건 — src_dst 미충족
- arm 안 내림: 2건
- ghost drop: 2건

### BC v6 (epoch 300) — 20ep eval

| 지표 | 값 |
|---|---|
| Place REAL | **1/20 (5%)** |
| Phase B 진입 | 16/20 |
| Arm descent 성공 | **7/16 (44%)** |
| Drop | 5 |
| Timeout | **12** |
| PhaseA timeout (4000step) | **3** |

**epoch 300은 overfitting.** arm descent 능력 퇴화 (65%→44%), PhaseA에서 base 접근 자체를 못 하는 에피소드 3건 발생. arm1_max가 0~2.0에서 멈추는 에피소드가 6건.

### 결론: **BC epoch 150이 압도적으로 우수.** RL 학습에는 epoch 150을 사용해야 함.

---

## 4. 물리 설정 변경

```python
cfg.object_contact_offset = 0.01    # 기존 0.005 → 2배
cfg.object_rest_offset = 0.004      # 기존 0.002 → 2배
cfg.sim.physx.enable_ccd = True     # 기존 False
```

**효과:** BC eval에서 src_dst 0.150→0.115, base drift +0.10→+0.05. grip 자연 release 메커니즘 작동 확인.
**부작용:** 일부 iter에서 objZ 폭발 (물체가 튕겨나감). 간헐적이며 학습에 치명적이지는 않음.

---

## 5. 버전 매트릭스

| 버전 | BC epoch | PLACE_RADIUS | 물리 | grip gate | R_release src | arm/grip scale | 핵심 변경 |
|---|---|---|---|---|---|---|---|
| v10 | 300 | 0.14 | 기존 | 없음 | <0.20 | —/1.20 | baseline |
| v11 | 300 | 0.14 | 기존 | 없음 | <0.20 | 0.30/0.80 | 보상 전면 재설계 |
| v11c | 300 | 0.18 | 기존 | 없음 | <0.20 | 0.30/0.80 | 새 BC + 반경 완화 |
| v12 | 300 | 0.18 | **새** | 없음 | <0.20 | 0.30/0.80 | 물리 설정 추가 |
| v12c | 300 | 0.18 | 새 | **>0.35** | <0.25 | 0.30/0.80 | grip gate + release 완화 |
| v12d | **150** | 0.18 | 새 | >0.35 | <0.25 | 0.30/0.80 | BC epoch 150 |
| **v13** | **150** | 0.18 | 새 | >0.35 | <0.25 | **0.20/0.50** | scale 축소 |

---

## 6. v10 근본 원인 (42 iter 분석)

### 원인 1: Phase B 진입 시 grip scale 0→1.20 즉시 점프
BC grip action = -0.573. RL noise 0.3이면: combined = -0.573 + 0.3×1.20 = -0.213 → grip_pos ≈ 0.35 → drop.
**결과: Phase B drop 88%가 arm1 < 0.5에서 발생.**

### 원인 2: R_release(8.0/step) > R_arm_lower(1.5/step)
RL이 arm1=2.0 도달 → 즉시 grip open → R_release 수령. arm을 더 내리면 1.5/step + 98.7% drop 리스크.
**결과: place 68.6%가 arm1_max [2.0, 2.2)에 수렴 (exploit).**

---

## 7. v11 보상 재설계 핵심

| 항목 | v10 | v11 | 근거 |
|---|---|---|---|
| Phase B scale | 즉시 점프 | **100step ramp + grip gate** | early drop 88%→27% |
| R_arm_lower | ×30 | **×80** | 데모 핵심 동작 |
| R_release | ×8.0, arm1>2.0 | **×3.0, arm1>2.5** | exploit 방지 |
| R_grip_penalty | -0.5 | **-1.0** | 조기 grip open 억제 |
| Place gate | arm1_max>2.0 | **arm1_max>2.5** | 데모 최소 2.67 |
| Place reward | 100+200×q | **200+300×q** | arm descent 유인 강화 |
| Drop 패널티 | 없음 | **-5.0** | drop 방지 |
| R_base_approach | ×15 | **×30** | base drift 억제 |

---

## 8. grip gate 도입 배경 (v12→v12c)

### v12 PLACE grip 분석 (1,667건)

```
Train iter 5: active(grip≥0.35) = 56%, grip_mean=0.445
Train iter 7: active(grip≥0.35) = 65%, grip_mean=0.533

Eval iter 6:  active(grip≥0.35) = 15%, grip_mean=0.216  ← 문제
```

**RL이 train에서는 exploration noise로 grip을 열어서 place하지만, eval의 mean action은 grip을 안 열음.** place_cond에 grip 요구가 없어서 "arm만 내리면 45~85% 확률로 place 보상"을 받을 수 있었기 때문.

### 수정: `place_cond`에 `grip_pos > 0.35` 추가

```python
place_cond = (
    ...기존 조건...
    & (grip_pos > 0.35)  # grip이 실제로 열려야 place 인정
)
```

### 효과 (v12c): 100% active place 달성. g<.35=0 전 iter.

---

## 9. Scale 문제 발견 (v12d 36 iter 분석)

### v12d 설정: BC ep150, grip>0.35 gate, arm=0.30, grip=0.80

### EVAL 추이 (36 iter)

```
Iter  1: PLACE=18, a>2=170, src=0.213
Iter  6: PLACE=24, a>2=143, src=0.220
Iter 11: PLACE=21, a>2=125, src=0.239
Iter 16: PLACE=19, a>2=119, src=0.261
Iter 21: PLACE=12, a>2= 85, src=0.289
Iter 26: PLACE=11, a>2= 83, src=0.308
Iter 31: PLACE=16, a>2= 82, src=0.338
```

**3가지 심각한 문제:**

1. **EVAL arm>2 하락:** 170→82. RL mean action이 BC arm descent를 파괴.
2. **src_dst 계속 악화:** 0.213→0.347. base positioning이 나빠짐.
3. **Train rate도 하락:** 20.4%→12.3%. policy 발산.

### TRAIN grip 분석

```
Train grip_mean: 0.49→0.77 (학습 중 상승)
Eval  grip_mean: 0.48~0.55 (고정)
```

RL이 exploration noise에서만 grip을 열고, mean action은 수렴 안 함. scale이 너무 커서 noise와 mean의 행동 차이가 큼.

### Train/Eval 격차

```
v12d iter 6:  train ~17% → eval 2.2%  (8배)
v12d iter 31: train ~12% → eval 1.5%  (8배)
```

### 원인: arm=0.30, grip=0.80이 과도

- grip 0.80: BC action -0.573 + RL(+1.0)×0.80 = +0.227. stochastic에서는 극단값으로 열리지만 mean≈0은 안 열림.
- arm 0.30: RL residual이 BC arm action을 30% 변형. 학습 과정에서 BC arm descent를 방해.

### Drop 분석

```
EVAL: PhaseA 80~84%, PhaseB 16~21%  (BC 순수에 가까움)
TRAIN: PhaseA 44~49%, PhaseB 51~56% (RL noise로 PhB drop 증가)
```

---

## 10. v13: Scale 축소

### 변경

```python
arm:  0.30 → 0.20
grip: 0.80 → 0.50
base: 0.20 유지
```

나머지 전부 동일 (BC ep150, grip>0.35 gate, R_release src<0.25, 새 물리).

### v13 결과 (9 iter)

```
Iter  mode | place   Δpl | drop   Δdr | rate  | a>2  g>.4 | src_avg
  1     E  |    16   +16 |  1253 +1253 |  1.3% |  174   34 |   0.211
  2     T  |    58   +42 |  2531 +1278 |  3.2% |  147   31 |   0.213
  3     T  |   118   +60 |  3985 +1454 |  4.0% |  144   27 |   0.224
  4     T  |   183   +65 |  5529 +1544 |  4.0% |  111   20 |   0.225
  5     T  |   579  +396 |  7568 +2039 | 16.3% |  328   59 |   0.237
  6     E  |   596   +17 |  8615 +1047 |  1.6% |  167   22 |   0.223
  7     T  |  1158  +562 | 10640 +2025 | 21.7% |  398  125 |   0.262
  8     T  |  1538  +380 | 12374 +1734 | 18.0% |  330  139 |   0.262
```

### PLACE grip 분포 (100% active, grip gate 작동)

```
Iter  1 E: mean=0.532 (g.35-.5=9, g.5-.7=5, g>.7=2)
Iter  7 T: mean=0.555 (g.35-.5=275, g.5-.7=169, g>.7=118)
Iter  9 T: mean=0.614 (g.35-.5=190, g.5-.7=105, g>.7=117)
```

### v12d vs v13 비교 (같은 iter)

| 지표 | v12d (0.30/0.80) | v13 (0.20/0.50) | 판단 |
|---|---|---|---|
| **EVAL iter 6 a>2** | 143 | **167** | v13 BC 보존 ↑ |
| EVAL iter 6 PLACE | 24 | 17 | v13 낮지만 이른 판단 |
| EVAL iter 6 src_dst | 0.220 | 0.223 | 유사 |
| Train iter 7 rate | 20.4% | **21.7%** | v13 유사~약간 ↑ |
| Train iter 8 rate | 15.4% | **18.0%** | v13 ↑ |
| Train iter 8 src_dst | 0.283 | **0.262** | v13 악화 느림 ↑ |
| v12d iter 35 a>2 | 82 (52% 하락) | — | v13 TBD |
| v12d iter 35 src_dst | 0.347 | — | v13 TBD |

### v13 초기 판단

**긍정적 신호:**
- EVAL a>2 174→167: BC arm descent 능력 보존 (v12d는 170→82)
- src_dst 악화 속도 감소: iter 8에서 0.262 (v12d: 0.283)
- Train rate 후반 유지: iter 8에서 18.0% (v12d: 15.4%)
- grip_mean 상승 중: 0.53→0.61 (train)

**남은 우려:**
- EVAL PLACE 17건으로 여전히 낮음 (train 562건 대비 33배)
- 학습 초기 느림 (iter 3-4에서 4.0%, v12d는 9.5~17.8%)
- 20+ iter 후 EVAL 추이 확인 필요

---

## 11. 현재 보상 구조 전체 (v13)

### Phase A (carry)
- **R_hold:** +0.05/step (contact + objZ > 0.033)
- **R_arm_maintain:** init_arm_pose 유지 (Gaussian, arm 0.10 + grip 0.05)
- **R1_approach:** base→dest delta × 30 (hold일 때만)

### Phase B (place)
- **R_hold_b:** +0.05/step (contact 유지)
- **R_arm:** src_dst → 0.12 수렴 delta × 80
- **R_base_approach:** base→dest delta × 30
- **R_lower:** objZ 하강 delta × 80 (src_dst < 0.30)
- **R_arm_lower:** arm1 하강 delta × 80 (src_dst < 0.30)
- **R_release:** grip_open × proximity × 3.0 (arm1>2.5, **src_dst<0.25**)
- **R_grip_penalty:** -1.0 (arm1<2.0, grip>0.50)

### Place / Drop / Timeout
- **Place:** 200 + 300×quality (**grip_pos > 0.35 필수**, arm1_max > 2.5)
- **Drop:** -5.0
- **Timeout:** -5.0

### Phase B scale (v13)
```python
arm_ramp = (phb_elapsed / 100.0).clamp(0, 1)
grip_gate = (arm1 > 2.0).float()
scale_arm  = 0.20 * arm_ramp           # v12d: 0.30
scale_grip = 0.50 * grip_gate * arm_ramp  # v12d: 0.80
scale_base = 0.20
```

---

## 12. 전체 RL 학습 결과 비교

| 버전 | iter | Train best | EVAL best | EVAL a>2 trend | src_dst trend | 핵심 |
|---|---|---|---|---|---|---|
| v10 | 42 | 1.5% | 13 place | — | — | exploit, 88% early drop |
| v11 | 10 | 3.9% | — | — | 0.22→0.27 ↑ | exploit 제거, early drop 27% |
| v11c | 6 | 18.5% | 54 place | — | 0.23→0.29 ↑ | 새 BC + 반경 완화 |
| v12 | 7 | 18.1% | 55 place | — | 0.24→0.26 ↑ | 물리 변경 |
| v12c | 7 | 17.2% | 13 place | — | — | grip gate (BC ep300 문제) |
| v12d | 36 | 20.4%→12.3% ↓ | 24→11 ↓ | 170→82 **↓↓** | 0.21→0.35 **↓↓** | **scale 과대 → 발산** |
| **v13** | **9** | **21.7%** | **17** | **174→167 ✓** | **0.21→0.26** | **scale 축소, 안정** |

---

## 13. 남은 문제 / 모니터링 항목

### v13에서 20+ iter 후 확인할 것

1. **EVAL a>2가 유지되는지** — v12d처럼 82까지 떨어지면 arm scale 0.20도 과도
2. **src_dst mean 추이** — 0.30 이상으로 가면 R_arm/R_base가 부족
3. **Train/Eval 격차** — 현재 21.7% vs 1.6%. 격차가 좁혀지는지
4. **EVAL PLACE 추이** — iter 11, 16에서 20건 이상 유지되는지

### 구조적 한계

- **Phase A drop ~50% (train), ~80% (eval):** S2→S3 handoff + BC carry 품질. 보상으로 해결 불가.
- **objZ 폭발:** 간헐적 물리 불안정. 학습에 치명적이지는 않지만 로깅 오염.

### 다음 실험 후보 (v13 결과 확인 후)

- scale 추가 조정: eval 격차가 안 좁혀지면 grip 0.50→0.35
- R_release 조건 추가 완화: eval에서 grip이 여전히 안 열리면
- BC ensemble: epoch 100~200 사이 여러 체크포인트 평균

---

## 14. 토론 포인트

1. **Scale vs 학습 속도 트레이드오프:** v12d는 빠르게 peak(20%) 찍고 발산. v13은 느리게 올라가지만 안정적. 어느 쪽이 최종 성능이 높을까?

2. **EVAL 격차의 본질:** train 21% vs eval 1.6%. 이게 scale 문제인지, 아니면 residual RL + BC 구조 자체의 한계인지? BC가 deterministic이면 mean residual도 deterministic인데, 왜 이렇게 다른가?

3. **grip gate 0.35의 적정성:** eval에서 grip_mean=0.54. gate를 0.40~0.45로 올려도 될까? 아니면 현재로 충분한가?

4. **Phase A drop 40~80%:** 이걸 줄이면 place 기회가 2배. carry BC 개선? arm interpolation? Phase A에서도 RL base를 더 강하게?

5. **src_dst 악화 원인:** arm descent가 물체를 dest에서 밀어내는 물리적 메커니즘? 아니면 base positioning이 학습 과정에서 나빠지는 것?
