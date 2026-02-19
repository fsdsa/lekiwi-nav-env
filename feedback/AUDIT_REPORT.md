# 남은 이슈 정리 — 전부 지금 수정

수정 완료된 항목: CRITICAL-1, CRITICAL-2, HIGH-1, HIGH-2, NEW-1, NEW-3, NEW-5  
**아래 3개를 모두 수정한 뒤 학습 진행.**

---

## 1. `grasp_gripper_threshold` 값 조정

**현재값:** `0.3` rad → **변경값:** `0.7` rad

### 문제

그리퍼 조인트 범위는 `[0.007, 1.745]` rad이다.
물체를 쥐면 그리퍼가 완전히 닫히지 않고 물체 폭만큼 벌어진 상태에서 멈춘다.

```
그리퍼 범위: 0.007 ──────────────────────────── 1.745 rad
              (완전닫힘)                          (완전열림)

threshold=0.3:  0.007 ═══ 0.3 |
                 ↑ 이 좁은 구간에서만 "잡음" 판정
                 → 거의 빈 손으로 닫아야만 통과

threshold=0.7:  0.007 ═══════════ 0.7 |
                 ↑ 물체를 쥔 상태에서도 판정 통과 가능
```

일반적 잡기 대상(음료캔, 작은 박스 등)을 쥐면 `gripper_pos`가 `0.4~0.8` rad 부근에서 멈춘다. `threshold=0.3`이면 아주 작은 물체만 `gripper_closed = True`를 통과하고, 나머지는 `can_grasp = False`가 된다.

### 수정

| 파일 | 위치 | 변경 |
|------|------|------|
| `lekiwi_skill2_env.py` | `Skill2EnvCfg.grasp_gripper_threshold` | `0.3` → `0.7` |
| `train_lekiwi.py` | CLI default | `0.3` → `0.7` |
| `collect_demos.py` | CLI default | `0.3` → `0.7` |
| `record_teleop.py` | CLI default | `0.3` → `0.7` |
| `lekiwi_nav_env.py` | CLI default (해당 시) | `0.3` → `0.7` |
| `pipeline_new/*.md` | 문서 내 모든 threshold 언급 | `0.3` → `0.7` |

> **참고:** `place_gripper_threshold = 0.3`은 Skill-3에서 "그리퍼가 0.3 이상 열리면 놓은 것"이라는 **반대 방향** 판정이므로 현재 값이 적절하다. 이건 수정하지 않는다.

### 검증

Skill-2 학습 초반에 로깅:
- `gripper_closed` 비율 — 0이면 threshold이 여전히 너무 낮음
- `can_grasp` 비율 — contact + gripper_closed 동시 충족 빈도
- 최대 bbox 물체를 쥐었을 때 `gripper_pos` 값 직접 확인

---

## 2. Single-Object 모드 물체 Sim 위치 미반영

### 문제

`_reset_from_handoff()`에서 tracking 텐서 `self.object_pos_w`는 업데이트하지만, 실제 sim에 물체 pose를 쓰는 코드가 multi-object 분기 안에만 있다:

```python
# object_pos_w는 갱신됨
self.object_pos_w[env_ids] = obj_pos

# 하지만 sim write는 multi-object일 때만 실행
if self._multi_object and len(self.object_rigids) > 0:
    # ... rigid.write_root_pose_to_sim() ...
```

`_multi_object=False`이면 물리 오브젝트가 default 위치에 그대로 있는 상태에서 fixed joint가 걸린다. tracking 텐서와 실제 sim 물체 위치가 불일치.

### 수정

`_reset_from_handoff()`의 multi-object 분기 앞에 single-object 처리를 추가:

```python
self.object_pos_w[env_ids] = obj_pos
self._handoff_object_ori[env_ids] = obj_ori
self.active_object_idx[env_ids] = obj_type_indices

# ── 추가: Single-object sim write ──
if not self._multi_object and hasattr(self, 'object_rigid') and self.object_rigid is not None:
    pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = obj_pos
    pose[:, 3:7] = obj_ori
    self.object_rigid.write_root_pose_to_sim(pose, env_ids)

# Multi-object hide/show (기존 코드 유지)
if self._multi_object and len(self.object_rigids) > 0:
    ...
```

`_reset_fallback()`에서도 동일 패턴 적용:

```python
self.object_pos_w[env_ids, :2] = default_root_state[:, :2]
self.object_pos_w[env_ids, 2] = default_root_state[:, 2] + float(self.cfg.grasp_attach_height)

# ── 추가: Single-object sim write ──
if not self._multi_object and hasattr(self, 'object_rigid') and self.object_rigid is not None:
    pose = self.object_rigid.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = self.object_pos_w[env_ids]
    self.object_rigid.write_root_pose_to_sim(pose, env_ids)
```

### 검증

- `_multi_object=False`로 Skill-3를 실행
- 1번째 step에서 `self.object_pos_w`와 `self.object_rigid.data.root_pos_w`가 일치하는지 확인
- Fixed joint attach 후 물체가 그리퍼에 실제로 붙어 있는지 시각적 확인

---

## 3. Handoff Buffer Noise Per-Load 적용

### 문제

`generate_handoff_buffer.py`에서 noise가 buffer 생성 시 1회만 적용된다. 같은 entry를 여러 번 load해도 동일한 noisy state가 나온다.

### 왜 지금 잡아야 하는가

- 500개 entry, 2048개 env → reset 1회당 평균 4번씩 같은 entry가 쓰임
- 동일 state 반복은 Skill-3 초기 state diversity를 떨어뜨리고, 특정 entry에 overfitting 유발
- 나중에 바꾸면 이전 학습 결과와 비교 불가 — 처음부터 맞춰놓는 게 맞음

### 수정

**Skill3EnvCfg에 추가:**

```python
@configclass
class Skill3EnvCfg(Skill2EnvCfg):
    ...
    # Handoff noise (per-load)
    handoff_arm_noise_std: float = 0.02       # ~1° joint noise
    handoff_base_pos_noise_std: float = 0.01  # 1cm position noise
    handoff_base_yaw_noise_std: float = 0.02  # ~1° heading noise
```

**`_reset_from_handoff()`에서 batch tensor 생성 직후:**

```python
# Batch tensor construction
base_pos = torch.tensor([e["base_pos"] for e in entries], ...)
base_ori = torch.tensor([e["base_ori"] for e in entries], ...)
arm_joints = torch.tensor([e["arm_joints"] for e in entries], ...)
grip_states = torch.tensor([e["gripper_state"] for e in entries], ...)
...

# ── 추가: Per-load noise ──
if self.cfg.handoff_arm_noise_std > 0:
    arm_joints += torch.randn_like(arm_joints) * self.cfg.handoff_arm_noise_std
    # arm limits 내로 clamp
    if self.cfg.arm_action_to_limits:
        arm_lo = self.robot.data.soft_joint_pos_limits[0, self.arm_idx[:5], 0]
        arm_hi = self.robot.data.soft_joint_pos_limits[0, self.arm_idx[:5], 1]
        arm_joints = torch.clamp(arm_joints, arm_lo, arm_hi)

if self.cfg.handoff_base_pos_noise_std > 0:
    base_pos[:, :2] += torch.randn(num, 2, device=self.device) * self.cfg.handoff_base_pos_noise_std

if self.cfg.handoff_base_yaw_noise_std > 0:
    # quaternion에 작은 yaw 회전 추가
    dyaw = torch.randn(num, device=self.device) * self.cfg.handoff_base_yaw_noise_std
    half = dyaw * 0.5
    dq = torch.zeros(num, 4, device=self.device)
    dq[:, 0] = torch.cos(half)
    dq[:, 3] = torch.sin(half)
    from isaaclab.utils.math import quat_mul
    base_ori = quat_mul(dq, base_ori)
```

### 검증

- 같은 handoff entry index를 2번 연속 로드해서 `arm_joints`, `base_pos` 값이 다른지 확인
- Skill-3 학습 초반 `_reset_from_handoff()` 호출 시 state 분포 히스토그램 로깅

---

## 수정 체크리스트

- [ ] `grasp_gripper_threshold`: 5개 파일 + 문서에서 `0.3` → `0.7`
- [ ] `_reset_from_handoff()`: single-object sim write 추가
- [ ] `_reset_fallback()`: single-object sim write 추가
- [ ] `Skill3EnvCfg`: handoff noise config 3개 추가
- [ ] `_reset_from_handoff()`: per-load noise 적용 코드 추가
- [ ] 검증: Skill-2 `gripper_closed` 비율 > 0 확인
- [ ] 검증: single-object 모드에서 물체 위치 일치 확인
- [ ] 검증: 같은 buffer entry 2회 로드 시 state 값 차이 확인

**전부 완료 후 학습 시작.**
