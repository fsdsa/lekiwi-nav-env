# LeKiwi 파이프라인 감사 보고서 (온라인 검증 완료)

> **작성일**: 2026-02-19  
> **범위**: 3-Skill RL → VLA 시뮬-투-리얼 전체 파이프라인  
> **상태**: Phase 0 (캘리브레이션) 완료, Phase 1 (RL 학습) 진행 전 사전 점검  

---

## 요약

총 **13개 이슈** 발견 (Critical 2 / High 3 / Medium 5 / Low 3)

| 등급 | 개수 | 핵심 |
|------|------|------|
| **CRITICAL** | 2 | Skill-3 place 불가능, AAC 미구현 |
| **HIGH** | 3 | 카메라 수집 누락, 커리큘럼 dead code, 배포 action 순서 |
| **MEDIUM** | 5 | obs 중복, contact 가짜 2D, critic obs 문서 불일치, BC 이름 오류, 레거시 break_force |
| **LOW** | 3 | 빈 테스트파일, v8 전용 테스트, 로그 누락 |

---

## CRITICAL — 파이프라인 블로커

### C1. Skill-3 place_success 달성 불가능 🔴

**파일**: `lekiwi_skill3_env.py:196-201`

**문제**: `place_success = (~object_grasped) & near_home & (~just_dropped)` 조건이 **절대 True가 될 수 없음**

**원인 분석**:
- FixedJoint `break_force` (15~45N) >> 물체 무게 (3N) → 중력만으로는 joint가 깨지지 않음
- 그리퍼를 열어도 FixedJoint는 물리 constraint이므로 자동 해제 안 됨
- 급격한 움직임으로 joint를 깨면 `just_dropped=True` → `~just_dropped` 실패 → 에피소드 즉시 종료

**결과**: Skill-3는 "집으로 운반"은 학습하지만 "내려놓기"는 **절대 학습 불가** → +20 place 보상 수령 불가

**수정안**:
```python
# _update_grasp_state()에 의도적 배치 로직 추가:
if self.object_grasped.any():
    gripper_open = self.robot.data.joint_pos[:, gripper_idx] > 0.3
    near_home = metrics["home_dist"] < self.cfg.return_thresh
    intentional_place = self.object_grasped & gripper_open & near_home

    if intentional_place.any():
        place_ids = intentional_place.nonzero(as_tuple=False).squeeze(-1)
        self._disable_grasp_fixed_joint_for_envs(place_ids)
        self.object_grasped[intentional_place] = False
        # just_dropped는 False 유지 → place_success 조건 충족 가능
```

---

### C2. AAC (Asymmetric Actor-Critic) 미구현 🔴

**파일**: `train_lekiwi.py:97-98`, `lekiwi_skill2_env.py:1415`

**문제**: 환경은 `{"policy": 30D, "critic": 37D}` dict obs를 반환하지만:
1. `train_lekiwi.py`는 `ValueNet` 사용 (AAC용 `CriticNet` 아님)
2. skrl 1.4.3 `wrap_env(wrapper="isaaclab")`는 **policy 키만** actor와 critic 모두에 전달
3. Critic은 30D policy obs를 받고, 37D critic obs는 **버려짐**

**온라인 검증 결과** ✅ **확인됨**:
- **GitHub Discussion #180** (skrl): "Currently, there is necessary to modify several components in skrl to support asymmetric learning" — 공식 미지원
- 작업 브랜치 존재 (`toni/agents_observations_spaces` → `toni/develop_observation_states`) but **1.4.3에 미병합**
- **Isaac Lab Issue #2712** (2025년 6월): 사용자들이 AAC 구현법 질문 → 공식 예제 없음
- **arXiv 2509.26000**, **OpenReview 2025**: AAC 이론적 유효성은 검증되었으나 skrl 1.4.3 구현 갭 존재

**결과**: Symmetric AC로 동작 → Critic이 privileged info(bbox, mass, distances) 활용 못함 → 가치 추정 저하 → GAE 품질 저하 → 학습 속도 저하

**수정 옵션 (우선순위순)**:
1. **rsl_rl 전환** (추천) — 이미 AAC 네이티브 지원 (별도 obs 공간)
2. **skrl PPO 서브클래스** — dict obs 처리 + 별도 critic 버퍼 구현
3. **수동 메모리 관리** — critic obs를 별도 저장 후 `critic.compute()`에 전달

---

## HIGH — 핵심 기능 누락

### H1. Skill-2/3 카메라 데이터 수집 누락 🟠

**파일**: `collect_demos.py:564-568`
```python
if args.skill == "approach_and_grasp":
    if use_camera:
        # TODO: Camera subclass for Skill2Env
        env = Skill2Env(cfg=env_cfg)  # 카메라 없음!
```

**영향**: Phase 2 VLA 데이터 수집 **차단** — (image, state, action) 튜플 생성 불가

**필요 작업**: `Skill2EnvWithCam`, `Skill3EnvWithCam` 클래스 생성 (`LeKiwiNavEnvWithCam` 참조)

---

### H2. curriculum_current_max_dist Dead Code 🟠

**파일**: `generate_handoff_buffer.py:71`, `lekiwi_skill2_env.py:232`

**문제**: Config에서 `curriculum_current_max_dist = object_dist_max` 설정하지만, 런타임은 항상 `object_dist_min`(0.5m)에서 시작

**영향**: Handoff Buffer 다양성 부족 — 대부분 0.5~1.0m 항목 → Skill-3 장거리 귀환 학습 부족

**수정안**:
```python
# __init__에서 config 읽기:
if hasattr(self.cfg, 'curriculum_current_max_dist') and self.cfg.curriculum_current_max_dist > 0:
    self._curriculum_dist = float(self.cfg.curriculum_current_max_dist)
```

---

### H3. deploy_vla_action_bridge.py Action 순서 레거시 🟠

**파일**: `deploy_vla_action_bridge.py:161-174`

**문제**: `[base3, arm6]` (v8 레거시) 형식으로 파싱하지만, VLA 출력은 `[arm5, grip1, base3]` (v3.0 형식)

**영향**: 실제 배포 시 arm 명령이 base로, base 명령이 arm으로 전달 → **로봇 오작동**

**수정안**: `--action_format` 플래그 추가 또는 v3.0을 기본값으로 변경

---

## MEDIUM — 문서/효율성

### M1. Observation 3D 중복

**파일**: `lekiwi_skill2_env.py:1386-1398`

`obs[6:9] = base_body_vel` ≈ `obs[9:12] = lin_vel_b` + `obs[12:15] = ang_vel_b` (vx, vy, wz 중복)

**영향**: 30D obs 중 실제 유니크 정보는 27D → 네트워크 용량 낭비

---

### M2. contact_lr 가짜 2D

**파일**: `lekiwi_skill2_env.py:1378-1380`
```python
contact_lr = torch.stack([contact_binary, contact_binary], dim=-1)  # 같은 값 2번
```

단일 센서를 복제 → 2D이지만 1D 정보

---

### M3. Critic Obs 문서 불일치

- **문서**: "Critic 37D = Actor 30D + obj_bbox_full(6D AABB) + mass(1D)"
- **코드**: Actor 30D + obj_bbox(3D) + mass(1D) + object_dist(1D) + heading(1D) + vel_toward(1D)

총합은 동일(+7D)하지만 구성이 다름

---

### M4. train_bc.py Action 이름 오류

**파일**: `train_bc.py:266`
```python
names = ["vx", "vy", "wz", "arm0", ...]  # 레거시 순서
```

v3.0 형식: `["arm0", "arm1", ..., "gripper", "vx", "vy", "wz"]`이어야 함

---

### M5. lekiwi_nav_env.py break_force 레거시

v8 환경의 `grasp_joint_break_force=1e8` (영구) → 30N으로 변경 필요

**영향**: 레거시 모드에서 조심스러운 운반 행동 학습 불가

---

## LOW

| ID | 설명 |
|----|------|
| L1 | `test.py` 비어있음 (1줄 docstring만) |
| L2 | `test_env.py` v8 전용 (`env.phase`, `env.object_visible` 사용) |
| L3 | `generate_handoff_buffer.py:146` 진행 로그가 50의 배수를 건너뛸 수 있음 |

---

## 검증 완료 항목 ✅

| 항목 | 상태 |
|------|------|
| Action 순서 Skill-2/3: `[arm5, grip1, base3]` | ✅ 정확 |
| Body velocity 단위: m/s, rad/s (변환 불필요) | ✅ 정확 |
| `extract_robot_state_9d()`: `root_lin_vel_b`, `root_ang_vel_b` 사용 | ✅ 정확 |
| BC-RL 네트워크 구조 동일 | ✅ 정확 |
| Kiwi IK 캘리브레이션: RMSE 0.117 < 0.15 | ✅ PASS |
| `arm_limit_write_to_sim=True`: USD inf limits 덮어쓰기 | ✅ 정확 |
| DR break_force: attach 전 적용 | ✅ 정확 |
| Handoff Buffer 노이즈 주입 | ✅ 정확 |
| Multi-object hide/show (z=-10) | ✅ 정확 |
| 커리큘럼 학습 로직 | ✅ 정확 |
| `convert_hdf5_to_lerobot_v3.py`: m→mm 변환 없음 | ✅ 정확 |
| Gripper binary 변환 (0.5 임계값) | ✅ 정확 |

---

## 환경 의존성 검증

### A100 Setup (`setup_env.sh` v3.1)
- ✅ PyTorch 2.7.1+cu126 통합
- ✅ flash-attn 2.7.4.post1 (prebuilt wheels)
- ✅ LeRobot 0.4.3 (네이티브 π0-FAST 지원, openpi 제거)
- ✅ Patch 5: `bertwarper.py` get_extended_attention_mask device arg 제거 (transformers 5.x 호환)
- ⚠️ SmolVLM-2 + VLA = ~13GB VRAM (40GB 예산 내)

### RTX 3090 Setup (`setup_bc_rl.sh`)
- ✅ Isaac Sim 5.0 + Isaac Lab 0.44.9
- ✅ Python 3.11, PyTorch 2.7.0+cu128
- ✅ skrl 1.4.3, rsl_rl, robomimic
- ⚠️ AAC 구현을 위해 rsl_rl 전환 또는 skrl 수정 필요

---

## 우선순위 로드맵

### 🔴 Phase 1 (RL 학습) 전 필수
1. **C1** — Skill-3 intentional place 로직 추가
2. **C2** — rsl_rl 전환 또는 skrl PPO 서브클래스로 AAC 구현
3. **H2** — 커리큘럼 초기화 코드 수정

### 🟠 Phase 2 (데이터 수집) 전 필수
4. **H1** — Skill-2/3 카메라 환경 클래스 생성

### 🟡 병렬 진행 가능
5. **M1~M5** — 코드 정리 (학습과 병행)
6. **L1~L3** — 경미한 이슈

### ⚪ Phase 5 (배포) 전 필수
7. **H3** — deploy action 순서 v3.0으로 업데이트
