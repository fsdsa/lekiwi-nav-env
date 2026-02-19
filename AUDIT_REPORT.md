# Codebase Audit Report: lekiwi-nav-env

전체 코드, 설정 파일, 문서를 전수 검토하고 파이프라인 흐름을 end-to-end로 추적한 결과입니다.

---

## CRITICAL-1: `grasp_gripper_threshold = -0.3`은 도달 불가능한 값

**파일:** `lekiwi_skill2_env.py:119`, `lekiwi_skill2_env.py:1271`

**문제:** Physics grasp 모드에서 그리퍼 "닫힘" 판정 조건이 절대로 참이 될 수 없습니다.

```python
# lekiwi_skill2_env.py:119
grasp_gripper_threshold: float = -0.3

# lekiwi_skill2_env.py:1271
gripper_closed = gripper_pos < float(self.cfg.grasp_gripper_threshold)  # < -0.3
```

**원인:** `lekiwi_robot_cfg.py:68`에서 그리퍼 조인트의 baked limit은:

```python
"STS3215_03a_v1_4_Revolute_57": (0.006566325252047892, 1.7453292519943295)
```

즉, 그리퍼의 물리적 최솟값은 **0.007 rad**입니다.

`arm_action_to_limits=True` (기본값, `lekiwi_skill2_env.py:90`)이므로 action 공간 [-1, 1]이
[0.007, 1.745] rad으로 매핑됩니다. 따라서 `gripper_pos < -0.3`은 **항상 False**입니다.

**영향:**
- `gripper_closed = False` → `can_grasp = False` → 물체를 절대 잡을 수 없음
- Physics grasp 모드에서 Skill-2 (ApproachAndGrasp) 학습이 완전히 불가능
- 해당 값은 `train_lekiwi.py:76`, `collect_demos.py:81`, `record_teleop.py:107`에서도 default=-0.3

**수정 방안:** `grasp_gripper_threshold`을 양수 값(예: 0.3)으로 변경. 그리퍼가 0.3 rad 이하로
닫히면 "closed" 판정이 되어야 합니다.

---

## CRITICAL-2: Skill-3 에피소드가 1 step 만에 종료

**파일:** `lekiwi_skill3_env.py:254-277`, `lekiwi_skill2_env.py:1261-1299`

**문제:** DirectRLEnv의 step 순서는 `_get_dones()` → `_get_rewards()` → `_reset_idx()` →
`_get_observations()` 입니다. Skill-3에서 `_get_dones()`가 부모(Skill-2)의 lift 기반
`task_success`를 그대로 사용하여 에피소드가 즉시 종료됩니다.

**실행 흐름:**

1. **`_get_dones()` 호출** (`lekiwi_skill3_env.py:254-277`):
   ```python
   self._update_grasp_state(metrics)  # line 256
   ```

2. **`_update_grasp_state()` → `super()._update_grasp_state()`** (`lekiwi_skill2_env.py:1261-1299`):
   ```python
   self.task_success[:] = False         # line 1261: 초기화
   # ... grasp 판정 ...
   lifted = self.object_grasped & ((obj_z - env_z) > self.cfg.grasp_attach_height)
   self.task_success[lifted] = True     # line 1299
   ```

   Skill-3에서 물체는 handoff buffer로부터 이미 **grasped + lifted** 상태로 시작
   (`lekiwi_skill3_env.py:358`: `self.object_grasped[env_ids] = True`)
   → `lifted = True` → `task_success = True`

3. **`_get_dones()` 에서 truncation 결정** (`lekiwi_skill3_env.py:271`):
   ```python
   truncated = self.task_success | time_out  # True | False = True
   ```

4. **`_get_rewards()` 는 `_get_dones()` 이후 호출됨** (`lekiwi_skill3_env.py:240`):
   ```python
   self.task_success = place_success  # 너무 늦음! truncated는 이미 True
   ```

**영향:** 모든 Skill-3 에피소드가 step 1에서 truncated → RL 학습 불가능

**수정 방안:** Skill-3의 `_get_dones()`에서 부모의 lift 기반 `task_success`가 아닌
`place_success`를 사용해야 합니다. 예를 들어:
- `_get_dones()` 내에서 `_update_grasp_state()` 호출 후 `task_success`를 place 조건으로 재계산
- 또는 `_update_grasp_state()` 오버라이드에서 부모 호출 후 `task_success`를 즉시 리셋

---

## HIGH-1: `compute_gae()`에서 `next_values` 파라미터 무시

**파일:** `aac_ppo.py:174-189`

**문제:**

```python
def compute_gae(rewards, dones, values, next_values,   # next_values 파라미터 존재
                discount_factor=0.99, lambda_coefficient=0.95):
    ...
    for i in reversed(range(memory_size)):
        nv = values[i + 1] if i < memory_size - 1 else last_values  # closure의 last_values 사용!
```

`next_values` 파라미터를 받지만 함수 본체에서 사용하지 않고, enclosing scope의 `last_values`를
직접 참조합니다.

**영향:** 현재는 호출부에서 `next_values=last_values`를 전달하므로 동작에 문제가 없지만,
코드가 fragile하고 의도가 불명확합니다. 향후 `next_values`에 다른 값을 전달하면 무시되어
silent bug가 됩니다.

**수정 방안:** `last_values` 대신 `next_values` 파라미터를 사용:
```python
nv = values[i + 1] if i < memory_size - 1 else next_values
```

---

## HIGH-2: 문서-코드 불일치 (`place_gripper_threshold`)

**파일:** `pipeline_new/1_전체_파이프라인.md:273`, `lekiwi_skill3_env.py:51`

**문제:**
- 문서: "gripper가 `place_gripper_threshold(-0.3)` 이상으로 열리고"
- 코드: `place_gripper_threshold: float = 0.3` (양수)

CRITICAL-1과 같은 맥락에서, -0.3은 그리퍼 baked limit (min=0.007) 아래의 도달 불가능한 값이므로
**코드의 0.3이 정확하고 문서가 잘못**된 것입니다.

`pipeline_new/codefix.md:878`에도 `-0.3`으로 기재되어 있어 문서 전반적으로 수정이 필요합니다.

---

## MEDIUM-1: `base_body_vel` (3D)은 기존 obs와 완전 중복

**파일:** `lekiwi_skill2_env.py` — `_get_observations()`

**현상:** Observation에 포함된 `base_body_vel = [vx, vy, wz]` (3D)는
`root_lin_vel_b[:, :2]` + `root_ang_vel_b[:, 2]`와 동일한 값입니다.
해당 obs가 별도의 3D 슬롯을 차지하면서 정보 중복이 발생합니다.

**영향:** 학습에 해를 끼치지는 않지만 observation 차원이 불필요하게 커집니다.
문서(`3_코드_현황_정리.md`)에서 "M1: 의도적 skip"으로 표기.

---

## MEDIUM-2: Domain Randomization이 전체 env에 적용

**파일:** `lekiwi_skill2_env.py` — `_apply_domain_randomization()`

**현상:** `write_joint_stiffness_to_sim`/`write_joint_damping_to_sim` 호출 시 env_ids를
필터링하지 않고 전체 env의 joint 속성을 덮어씁니다. 리셋된 env만 DR이 필요하지만
모든 env에 적용됩니다.

**영향:** 기능적 오류는 아니지만 (매 리셋마다 어차피 새로 랜덤화), 불필요한 GPU 연산이
발생합니다. 대규모 병렬 env (1024+) 에서 성능에 영향이 있을 수 있습니다.

---

## MEDIUM-3: `contact_lr`이 fake L/R

**파일:** `lekiwi_skill2_env.py` — contact force 계산

**현상:** Left/Right contact force를 구분하는 것처럼 보이지만, 실제로 두 채널이 동일한 값을
사용합니다. 문서에서 "M2: 의도적 skip"으로 표기.

---

## LOW-1: `generate_handoff_buffer.py` — raw env 직접 접근

**파일:** `generate_handoff_buffer.py`

**현상:** Wrapped env가 아닌 raw env를 직접 접근하여 `obs["policy"]`를 읽습니다. 현재는
동작하지만 wrapper 스택이 변경되면 깨질 수 있습니다.

---

## 파이프라인 흐름 검증 요약

| 구간 | 상태 |
|------|------|
| Skill-1 (Navigate) → script policy | OK (proportional controller K_LIN=0.8, K_ANG=1.5) |
| Skill-2 obs 차원 (30D actor / 37D critic) | OK — 코드-문서 일치 |
| Skill-3 obs 차원 (29D actor / 36D critic) | OK — 코드-문서 일치 |
| Action format v6 [arm5, grip1, base3] | OK — 전 skill 공통 |
| BC → RL weight 전달 (PolicyNet ↔ BCPolicy) | OK — 아키텍처 일치 |
| AAC wrapper/trainer/ppo 통합 | OK (HIGH-1 제외) |
| Kiwi IK (3-wheel omni) | OK — 기하 상수 일치 |
| Handoff buffer 생성 → Skill-3 로드 | OK (CRITICAL-2의 task_success 문제 제외) |
| Sim-Real calibration 상수 | OK — LIN_SCALE=1.0166, ANG_SCALE=1.2360, WZ_SIGN=-1.0 |
| Curriculum learning | OK — 0.25m increment @ 70% success |

---

## 최종 요약

| 심각도 | 개수 | 핵심 |
|--------|------|------|
| **CRITICAL** | 2 | Physics grasp 불가 + Skill-3 즉시 종료 |
| **HIGH** | 2 | GAE next_values 무시 + 문서 불일치 |
| **MEDIUM** | 3 | Obs 중복, DR 범위, contact_lr |
| **LOW** | 1 | Handoff buffer raw env 접근 |

**두 CRITICAL 버그는 Physics grasp 모드에서 Skill-2와 Skill-3 학습을 모두 불가능하게 만들므로
즉시 수정이 필요합니다.**
