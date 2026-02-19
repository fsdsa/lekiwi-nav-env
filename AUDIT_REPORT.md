# 코드베이스 감사 보고서

**날짜**: 2026-02-19
**범위**: 전체 파일, 파이프라인 문서, 코드 구현 교차 검증

---

## 요약

전체 코드베이스의 모든 소스 파일, 파이프라인 문서, 설정 파일을 검토했다. 파이프라인 설계와 실제 구현 사이에 **치명적 이슈 2건**, **높은 심각도 3건**, **중간 심각도 5건**, **낮은 심각도 3건**을 발견했다.

| 심각도 | 건수 | 설명 |
|--------|------|------|
| CRITICAL | 2 | 파이프라인이 의도대로 동작 불가 |
| HIGH | 3 | 핵심 기능 누락 또는 dead code |
| MEDIUM | 5 | 문서-코드 불일치, 낭비, 혼동 유발 |
| LOW | 3 | 비효율, 미완성 |

---

## CRITICAL — 파이프라인 동작 불가

### C1. Skill-3 place_success 도달 불가능

**파일**: `lekiwi_skill3_env.py:196-201`
**문서**: `2_Sim_데이터_수집_파이프라인.md` §3-4-3 "place → +20, 의도적으로 놓음"

**현상**: `place_success` 조건이 코드상 절대 True가 될 수 없다.

**분석**:
```python
# lekiwi_skill3_env.py:196-201
place_success = (~self.object_grasped) & near_home & (~self.just_dropped)
```

이 조건이 True가 되려면 `object_grasped=False`이면서 `just_dropped=False`여야 한다. 그런데:

1. **FixedJoint가 중력만으로는 안 깨진다**: 물체 무게 ≈ 3N (0.3kg × 9.81), break_force = 15~45N (DR 범위). 중력의 3~15배에 해당하므로 물체가 자연 낙하로 joint를 깰 수 없다.

2. **agent가 그리퍼를 열어도 FixedJoint는 유지된다**: FixedJoint는 gripper body와 object body를 물리적으로 결합한다. 그리퍼 관절을 열어도 FixedJoint 자체는 파손되지 않는다 (접촉력이 아닌 관성력만이 break_force를 초과시킨다).

3. **급격한 움직임으로 FixedJoint를 깨면** → `_update_grasp_state()`에서 `grip_obj_dist > 0.15m` 감지 → `just_dropped=True` → `place_success`의 `~just_dropped` 조건 실패 → 동시에 `_get_dones()`에서 `terminated=True`로 에피소드 즉시 종료.

**결론**: agent가 물체를 "의도적으로 놓을" 수 있는 메커니즘이 존재하지 않는다. `rew_place_success_bonus=20`의 보상은 절대 수령되지 않으며, Skill-3 학습은 "물체를 들고 home으로 이동"만 가능하고 "내려놓기"는 불가능하다.

**수정 방향**: place 메커니즘 추가가 필요하다. 예: home 근처에서 gripper가 open이면 FixedJoint를 disable하고 `object_grasped=False`, `just_dropped=False`로 설정하여 intentional release와 accidental drop을 구분.

```python
# 제안 코드 (lekiwi_skill3_env.py의 _update_grasp_state 내부)
if self.object_grasped.any():
    gripper_pos = self.robot.data.joint_pos[:, self.gripper_idx]
    gripper_open = gripper_pos > float(self.cfg.place_gripper_threshold)  # 예: 0.3
    near_home = metrics["home_dist"] < self.cfg.return_thresh
    intentional_place = self.object_grasped & gripper_open & near_home
    if intentional_place.any():
        place_ids = intentional_place.nonzero(as_tuple=False).squeeze(-1)
        self._disable_grasp_fixed_joint_for_envs(place_ids)
        self.object_grasped[intentional_place] = False
        # just_dropped는 False 유지 → place_success 조건 충족
```

---

### C2. AAC (Asymmetric Actor-Critic)가 실제로 작동하지 않음

**파일**: `train_lekiwi.py` (ValueNet 사용), `lekiwi_skill2_env.py:1415` (critic obs 반환)
**문서**: `1_전체_파이프라인.md` §7-1 "Actor와 Critic이 서로 다른 observation을 받는다"

**현상**: 환경 코드는 `{"policy": 30D, "critic": 37D}`를 반환하지만, 학습 코드는 `ValueNet`을 사용하고 skrl 1.4.3 PPO는 critic obs를 별도 저장하지 않는다.

**분석**:
- `lekiwi_skill2_env.py:1415`: `return {"policy": actor_obs, "critic": critic_obs}` — 환경은 별도 critic obs를 생성
- `train_lekiwi.py:97-98`: `PolicyNet` + `ValueNet`을 사용. `CriticNet`은 import만 되고 사용되지 않음
- skrl 1.4.3의 `wrap_env(env, wrapper="isaaclab")`는 `"policy"` 키만 observation으로 전달
- `ValueNet.compute()`는 `inputs["states"]`를 받으며, 이것은 30D policy obs와 동일한 텐서

**결과**: Critic은 30D policy obs만 받고, 37D critic obs(추가 bbox, mass, dist, heading, vel)는 **완전히 버려진다**. 파이프라인 문서가 핵심 설계로 강조하는 AAC가 실제로는 symmetric AC로 동작 중이다.

**영향**: Critic이 privileged 정보 없이 value estimation을 하므로, reward에 물체 거리/방향/속도가 반영됨에도 Critic이 이를 모르는 상태다. 이는 value estimation 부정확 → GAE 품질 저하 → 학습 효율 하락으로 이어진다.

**수정 방향**:
1. skrl PPO를 서브클래싱하여 critic obs를 별도 memory buffer에 저장
2. 또는 `CriticNet(critic_obs_dim=37)`을 사용하되 PPO의 `_update()` 메서드에서 critic obs를 전달하도록 수정

---

## HIGH — 핵심 기능 누락 또는 Dead Code

### H1. Skill-2/3 카메라 수집 미구현

**파일**: `collect_demos.py:564-568`
**문서**: `2_Sim_데이터_수집_파이프라인.md` §4-4 "카메라 이미지 저장"

```python
# collect_demos.py:563-568
if args.skill == "approach_and_grasp":
    if use_camera:
        # TODO: Camera subclass for Skill2Env
        env = Skill2Env(cfg=env_cfg)  # 카메라 없이 생성
    else:
        env = Skill2Env(cfg=env_cfg)
```

VLA 학습 데이터에는 카메라 이미지가 필수(`observation.images.base`, `observation.images.wrist`)인데, Skill-2/3 환경에는 `LeKiwiNavEnvWithCam`에 해당하는 카메라 서브클래스가 없다. legacy 모드(`lekiwi_nav_env.py`)만 카메라를 지원한다.

**영향**: Phase 2 VLA 데이터 수집이 불가능. state-only 데이터만 수집 가능하며, 이는 VLA 학습에 사용할 수 없다.

---

### H2. `curriculum_current_max_dist` 설정이 dead code

**파일**: `generate_handoff_buffer.py:71`, `collect_navigate_data.py:379`, `lekiwi_skill2_env.py:110,232`

```python
# generate_handoff_buffer.py:71
env_cfg.curriculum_current_max_dist = env_cfg.object_dist_max  # 이 필드는 런타임에 읽히지 않음

# lekiwi_skill2_env.py:232 (실제 초기화)
self._curriculum_dist = float(self.cfg.object_dist_min)  # object_dist_min(0.5)에서 시작
```

`curriculum_current_max_dist` config 필드를 설정해도 런타임 변수 `_curriculum_dist`는 항상 `object_dist_min`(0.5m)에서 시작한다. `generate_handoff_buffer.py`의 의도는 "curriculum을 최대로 열어서 다양한 거리 커버"였지만, 실제로는 0.5m에서 시작하여 성공률에 따라 점진적으로만 확장된다. `collect_navigate_data.py:379`도 동일한 문제.

**영향**: Handoff Buffer의 다양성 부족. 대부분의 entry가 짧은 거리(0.5~1.0m)에 집중되어, Skill-3가 먼 거리에서 home으로 돌아오는 상황을 충분히 경험하지 못한다.

**수정**: `__init__`에서 config 값을 읽도록 변경:
```python
# lekiwi_skill2_env.py __init__
if hasattr(self.cfg, 'curriculum_current_max_dist') and self.cfg.curriculum_current_max_dist > 0:
    self._curriculum_dist = float(self.cfg.curriculum_current_max_dist)
else:
    self._curriculum_dist = float(self.cfg.object_dist_min)
```

---

### H3. `deploy_vla_action_bridge.py` 액션 순서가 legacy 전용

**파일**: `deploy_vla_action_bridge.py:161-174`
**문서**: `1_전체_파이프라인.md` §3-1 "새 skill 환경에서는 [arm5, grip1, base3] 순서로 통일"

```python
# deploy_vla_action_bridge.py:161-163
action[0:2]  # vx, vy  ← legacy: base가 앞
action[3:9]  # arm     ← legacy: arm이 뒤
```

실배포 시 VLA가 출력하는 action은 `[arm5, grip1, base3]` 순서인데, `deploy_vla_action_bridge.py`는 `[base3, arm6]` (legacy) 순서를 가정한다. 이대로 배포하면 arm 명령이 base로, base 명령이 arm으로 전달되어 로봇이 잘못 동작한다.

**수정**: 액션 파싱 로직에 `--action_format` 플래그 추가하거나, 새 순서를 기본값으로 변경.

---

## MEDIUM — 문서-코드 불일치, 낭비

### M1. Observation에 3D 중복 존재

**파일**: `lekiwi_skill2_env.py:1386-1398` (30D obs 구성)

```
[6:9]   base_body_vel = [root_lin_vel_b[0], root_lin_vel_b[1], root_ang_vel_b[2]]
[9:12]  lin_vel_b     = root_lin_vel_b = [vx, vy, vz]
[12:15] ang_vel_b     = root_ang_vel_b = [wx, wy, wz]
```

Isaac Lab에서 `root_lin_vel_b`는 body-frame 선속도이고, `_read_base_body_vel()`도 동일한 텐서에서 읽는다. 따라서:
- `obs[6]` = `obs[9]` (vx)
- `obs[7]` = `obs[10]` (vy)
- `obs[8]` = `obs[14]` (wz)

3차원이 완전히 중복된다. 30D 중 실질 정보량은 27D뿐이다.

**영향**: 학습은 가능하나 obs space가 불필요하게 크고, BC→RL weight transfer 시 입력 차원을 맞추는 데 혼란을 줄 수 있다.

---

### M2. `contact_lr`이 동일한 값의 복제

**파일**: `lekiwi_skill2_env.py:1378-1380`

```python
contact_binary = (contact_force > float(self.cfg.grasp_contact_threshold)).float()
contact_lr = torch.stack([contact_binary, contact_binary], dim=-1)  # 같은 값 2번
```

문서에는 "contact L/R (2D)"로 좌/우 그리퍼 접촉을 나타낸다고 하지만, 실제로는 단일 contact sensor의 값을 복제한 것이다. 2D이지만 정보는 1D.

**영향**: 학습에 해는 없으나, obs 구조가 문서와 다르며 1D를 낭비한다.

---

### M3. Critic obs 구성이 문서와 다름

**파일**: `lekiwi_skill2_env.py:1402-1413` vs `2_Sim_데이터_수집_파이프라인.md` §3-2-3

**문서**: "Critic 37D = Actor 30D + obj_bbox_full(6D, AABB min/max) + obj_mass(1D)"
**코드**: Critic 37D = Actor 30D + obj_bbox(3D) + obj_mass(1D) + object_dist(1D) + heading_object(1D) + vel_toward_object(1D)

6D AABB min/max가 아니라 3D bbox + 3개의 스칼라 metric이다. 합계는 동일하게 7D extra이지만 구성이 다르다. 문서 업데이트 필요.

---

### M4. `train_bc.py` 평가 섹션의 action 이름이 legacy 순서

**파일**: `train_bc.py:266`

```python
names = ["vx", "vy", "wz", "arm0", "arm1", "arm2", "arm3", "arm4", "arm5"]
```

새 skill 환경의 action 순서는 `[arm0~4, gripper, vx, vy, wz]`이므로 올바른 이름은:
```python
names = ["arm0", "arm1", "arm2", "arm3", "arm4", "gripper", "vx", "vy", "wz"]
```

**영향**: 평가 결과 출력 시 혼란 유발. 학습 자체에는 영향 없음.

---

### M5. `lekiwi_nav_env.py`(v8)의 `grasp_joint_break_force`가 여전히 1e8

**파일**: `lekiwi_nav_env.py` (cfg 기본값)
**문서**: `1_전체_파이프라인.md` §7-3 "break_force 기본값 30N"

v8 환경(`lekiwi_nav_env.py`)은 `grasp_joint_break_force=1e8`(사실상 영구 결합)을 유지 중이다. 파이프라인 문서에서 30N으로 변경하라고 기술하지만, v8 코드 자체는 수정되지 않았다. Skill-2 환경(`lekiwi_skill2_env.py`)은 30N으로 올바르게 설정되어 있다.

**영향**: legacy 모드 사용 시 조심스러운 carry 행동을 학습하지 않는다. Skill-2/3에는 영향 없음.

---

## LOW — 비효율, 미완성

### L1. `test.py`가 빈 파일

**파일**: `test.py` — 1줄 docstring만 있고 구현이 없다.

---

### L2. `test_env.py`가 v8 FSM 전용

**파일**: `test_env.py:94` — `env.phase`, `env.object_visible` 등 v8 전용 속성에 의존. Skill-2/3 환경에서는 이 속성이 없어서 검증 도구로 사용 불가.

---

### L3. `generate_handoff_buffer.py`에서 성공 진행 로그가 부정확

**파일**: `generate_handoff_buffer.py:146`

```python
if len(entries) % 50 == 0 and len(entries) > 0:
```

한 step에서 여러 환경이 동시 성공하면 len(entries)가 50의 배수를 건너뛸 수 있어서 진행 로그가 출력되지 않을 수 있다. 기능에는 영향 없음.

---

## 파이프라인 정합성 확인 — 정상 항목

| 항목 | 상태 |
|------|------|
| Action 순서 Skill-2/3: [arm5, grip1, base3] | ✅ `_apply_action()` 인덱싱 일치 |
| base velocity 단위 (m/s, rad/s) | ✅ sim/real 동일, 변환 불필요 |
| `extract_robot_state_9d()` — body-frame velocity | ✅ `root_lin_vel_b`, `root_ang_vel_b` 직접 사용 |
| BC-RL 네트워크 구조 동일 | ✅ `BCPolicy` = `PolicyNet.net + mean_layer` |
| Kiwi IK matrix | ✅ dynamics calibration RMSE 통과 (0.117) |
| arm_limit_write_to_sim | ✅ USD inf limit 덮어쓰기 구현됨 |
| DR break_force가 attach 전 적용 | ✅ Skill-3 `_reset_from_handoff()`에서 순서 정확 |
| Handoff Buffer noise injection | ✅ arm/base/object 노이즈 주입 구현 |
| Multi-object hide/show | ✅ z=-10 숨김 + 선택 물체만 배치 |
| Curriculum learning | ✅ 성공률 기반 거리 점진 확대 (dead code 초기화 제외) |
| convert_hdf5_to_lerobot_v3.py 단위 변환 | ✅ v3.0에서 m→mm 변환 삭제 확인 |
| Gripper binary 변환 | ✅ 수집 시 0.5 threshold로 변환 |

---

## 수정 우선순위 권장

1. **C1** (Skill-3 place 메커니즘 추가) — Skill-3 학습 전 반드시 수정
2. **C2** (AAC 실제 구현) — RL 학습 효율에 직접 영향
3. **H1** (카메라 서브클래스) — VLA 데이터 수집 전 반드시 구현
4. **H2** (curriculum dead code 수정) — Handoff Buffer 다양성에 영향
5. **H3** (deploy action 순서) — 실배포 전 반드시 수정
6. **M1~M5** — 코드 정리/문서 업데이트 (학습 진행과 병행 가능)
