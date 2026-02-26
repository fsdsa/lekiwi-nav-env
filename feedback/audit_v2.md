# Task 변경 사양서: "약병 찾아서 빨간 컵 옆에 놔"

> 이 문서는 기존 파이프라인의 task를 변경하기 위해 수정이 필요한 모든 파일과 변경 내용을 정의한다.
> Claude Code에게 이 문서를 주고 수정을 요청할 것.

---

## 1. Task 변경 요약

### 기존 Task
```
"빨간 컵 바구니에 넣어"
Navigate(컵 탐색) → Grasp(컵) → Navigate(바구니로 이동) → Place(바구니 안에)
목적지: 바구니 (basket) — 랜덤 스폰 4.0~5.0m
```

### 새 Task
```
"약병 찾아서 빨간 컵 옆에 놓아"
Navigate(약병 탐색) → Grasp(약병) → Navigate(빨간 컵 탐색) → Place(빨간 컵 옆에)
source_object: 약병 (grasp 대상)
destination_object: 빨간 컵 (place 기준점)
```

### 핵심 변경점
1. **목적지가 바구니 → 두 번째 물체(빨간 컵)**로 변경
2. **VLM이 2-object 순차 추적** 필요 (잡은 후 추적 대상 전환)
3. **Success criteria**: "바구니 안" → "빨간 컵으로부터 XY 20cm 이내"
4. **물체 2개가 환경에 동시 존재**: 약병 + 빨간 컵

### 물체 USD 경로
- **약병 (source, grasp 대상)**: `/home/yubin11/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd`
- **빨간 컵 (destination, place 기준점)**: `/home/yubin11/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd`

---

## 2. 수정 대상 파일 목록

| 파일 | 수정 범위 | 우선도 |
|------|----------|--------|
| `lekiwi_skill3_env.py` | 대규모 — 바구니 → 두 번째 물체 목적지 | 최우선 |
| `lekiwi_skill2_env.py` | 소규모 — 두 번째 물체(빨간 컵) 동시 스폰 | 높음 |
| `vlm_orchestrator.py` | 대규모 — 2-object 추적 오케스트레이션 | 높음 |
| `vlm_prompts.py` | 대규모 — system prompt 전면 재작성 | 높음 |
| `record_teleop.py` | 중규모 — combined 모드 목적지 변경 | 중간 |
| `collect_demos.py` | 소규모 — instruction 템플릿 변경 | 중간 |
| `eval_full_system.py` | 중규모 — success criteria 변경 | 중간 |
| `generate_handoff_buffer.py` | 소규모 — 두 번째 물체 상태 포함 | 중간 |
| `convert_hdf5_to_lerobot_v3.py` | 변경 없음 — 9D state/action 구조 불변 | — |
| `train_lekiwi.py` | 변경 없음 — RL 학습 구조 불변 | — |
| `train_bc.py` | 변경 없음 — BC 학습 구조 불변 | — |
| `lekiwi_skill1_env.py` | 변경 없음 — Navigate는 방향 추종 전용 | — |

---

## 3. 파일별 상세 변경 사항

### 3-1. `lekiwi_skill3_env.py` (대규모)

이 파일이 가장 큰 변경. 바구니(basket)를 두 번째 물체(destination object)로 대체.

#### 3-1-1. Config 변경

```python
# 기존
class Skill3EnvCfg:
    basket_spawn_dist_min: float = 4.0       # 바구니 스폰 최소 거리
    basket_spawn_dist_max: float = 5.0       # 바구니 스폰 최대 거리
    basket_radius: float = 0.20              # 바구니 안 판정 반경
    basket_height: float = 0.25              # 바구니 높이 판정

# 변경 →
class Skill3EnvCfg:
    dest_object_usd: str = "/home/yubin11/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd"
    dest_spawn_dist_min: float = 2.0         # 빨간 컵 스폰 최소 거리 (source object 기준)
    dest_spawn_dist_max: float = 4.0         # 빨간 컵 스폰 최대 거리
    dest_spawn_min_separation: float = 1.0   # source와 dest 최소 이격 거리
    place_radius: float = 0.20               # "옆에" 판정 반경 (XY 20cm 이내)
    place_height_tolerance: float = 0.10     # 높이 허용 오차 (바닥 물체이므로 작게)
```

#### 3-1-2. 바구니 스폰 → 빨간 컵 스폰

```python
# 기존: _spawn_basket() — 바구니 USD를 env origin에서 4.0~5.0m 거리에 스폰
# 변경: _spawn_dest_object() — 빨간 컵 USD를 스폰

# 빨간 컵은 일반 rigid body로 스폰 (바구니와 달리 고정 아님)
# 바닥에 놓이므로 z = bbox_z * 0.5
# source object(약병)과 최소 dest_spawn_min_separation(1.0m) 이격
# 랜덤 yaw 회전 적용
```

#### 3-1-3. `basket_pos_w` → `dest_object_pos_w`

모든 코드에서 `basket_pos_w` 변수명을 `dest_object_pos_w`로 변경. 이 위치는 빨간 컵의 현재 world position.

```python
# 기존
self.basket_pos_w = ...  # 바구니 위치

# 변경 →
self.dest_object_pos_w = ...  # 빨간 컵 위치 (매 step 갱신 — rigid body이므로 밀릴 수 있음)
```

**주의**: 바구니는 고정 물체였지만 빨간 컵은 **rigid body**이므로, 로봇이 접근하다가 밀 수 있음. 매 step에서 `dest_object_pos_w`를 갱신해야 함. 또는 빨간 컵을 kinematic body로 고정하는 옵션 추가 (`dest_object_fixed: bool = True`).

#### 3-1-4. Success Criteria 변경

```python
# 기존: _check_place_success()
# 조건: XY radius(0.20m) + Z height(0.25m) 이내 + gripper open + not dropped
def _check_place_success(self):
    obj_pos = self.object_pos_w
    basket_pos = self.basket_pos_w
    xy_dist = torch.norm(obj_pos[:, :2] - basket_pos[:, :2], dim=1)
    z_ok = obj_pos[:, 2] < basket_pos[:, 2] + self.cfg.basket_height
    return (xy_dist < self.cfg.basket_radius) & z_ok & gripper_open & ~dropped

# 변경 →
def _check_place_success(self):
    obj_pos = self.source_object_pos_w       # 약병 (놓인 후 위치)
    dest_pos = self.dest_object_pos_w        # 빨간 컵 위치
    xy_dist = torch.norm(obj_pos[:, :2] - dest_pos[:, :2], dim=1)
    z_diff = torch.abs(obj_pos[:, 2] - dest_pos[:, 2])
    return (xy_dist < self.cfg.place_radius) & (z_diff < self.cfg.place_height_tolerance) & gripper_open & ~dropped
```

#### 3-1-5. Actor Observation 변경

```python
# 기존 Skill-3 Actor obs (29D):
# 9D + base_vel(6) + arm_vel(6) + basket_rel(3) + grip_force(1) + obj_bbox(3) + obj_category(1)

# 변경 →
# 9D + base_vel(6) + arm_vel(6) + dest_object_rel(3) + grip_force(1) + obj_bbox(3) + obj_category(1)
# = 여전히 29D, basket_rel → dest_object_rel로만 바뀜

# basket_rel 계산:
# 기존: basket_pos - robot_pos (world frame → body frame 변환)
# 변경: dest_object_pos - robot_pos (동일 변환)
```

obs 차원(29D)은 변하지 않음. `basket_rel`의 의미가 "바구니까지 상대 위치"에서 "빨간 컵까지 상대 위치"로 바뀔 뿐.

#### 3-1-6. Intentional Place 메커니즘 변경

```python
# 기존: gripper_open + near_basket(basket_radius × 3.0 = 0.6m 이내) → FixedJoint 해제
# 변경: gripper_open + near_dest_object(place_radius × 3.0 = 0.6m 이내) → FixedJoint 해제
# 로직 동일, 거리 기준만 dest_object_pos_w로 변경
```

#### 3-1-7. 리셋 로직

```python
# _reset_idx() 또는 _reset_fallback()에서:
# 기존: 바구니 위치 재스폰
# 변경: 빨간 컵 위치 재스폰 (source object와 충분히 이격)

# _reset_from_handoff()에서:
# 기존: handoff buffer에서 로봇+source 물체 상태 복원, 바구니 스폰
# 변경: handoff buffer에서 로봇+source 물체 상태 복원, 빨간 컵 스폰
# handoff buffer에 dest_object 정보는 불필요 (매 에피소드 새로 스폰)
```

#### 3-1-8. 변수명 일괄 치환 요약

| 기존 | 변경 | 비고 |
|------|------|------|
| `basket_pos_w` | `dest_object_pos_w` | 빨간 컵 world position |
| `basket_rel` | `dest_object_rel` | body-frame 상대 위치 (obs) |
| `basket_radius` | `place_radius` | "옆에" 판정 반경 |
| `basket_height` | `place_height_tolerance` | 높이 판정 |
| `basket_spawn_dist_min/max` | `dest_spawn_dist_min/max` | 스폰 거리 |
| `_spawn_basket()` | `_spawn_dest_object()` | 스폰 함수 |
| `_check_place_success()` | 시그니처 유지, 내부 로직 변경 | 위 참조 |
| `parse_basket_detection()` | 삭제 또는 `parse_dest_object_detection()`으로 변경 | VLM orchestrator |

---

### 3-2. `lekiwi_skill2_env.py` (소규모)

Skill-2 환경에 **빨간 컵(destination object)을 배경 물체로 동시 스폰**해야 함. 이유: VLA 데이터 수집 시 빨간 컵이 이미지에 보여야 VLA가 "이건 잡지 않는다"를 학습할 수 있음.

#### 변경 사항

```python
# 에피소드 리셋 시:
# 기존: source object 1개만 스폰
# 변경: source object(약병) + dest object(빨간 컵) 2개 스폰
#        dest object는 잡기 대상이 아님 (RL reward에 영향 없음)
#        source와 dest는 최소 1.0m 이격

# dest object는 rigid body로 스폰하되:
# - contact sensor 불필요
# - FixedJoint 대상 아님
# - reward 계산에 포함 안 됨
# - 카메라에 보이기만 하면 됨
```

**RL 학습 시**: dest object가 있어도 RL 보상 구조는 불변. Actor obs에 dest object 정보 포함 안 됨. 단, 물리 시뮬레이션에 두 물체가 존재하므로 충돌은 자연스럽게 처리됨.

**Phase 2 데이터 수집 시**: 카메라 이미지에 빨간 컵이 보이므로, VLA가 "빨간 컵은 잡지 않고 약병을 잡는다"를 instruction으로 학습 가능.

---

### 3-3. `vlm_orchestrator.py` (대규모)

VLM 오케스트레이터를 2-object 순차 추적 구조로 변경.

#### 3-3-1. `/classify` 변경

```python
# 기존: 사용자 명령 → {mode: "single", target_object: "빨간 컵"}
# 변경: 사용자 명령 → {mode: "relative_placement", source_object: "약병", destination_object: "빨간 컵"}

# VLM /classify 프롬프트:
# "사용자 명령을 분석하여 source_object(잡을 물체)와 destination_object(놓을 기준 물체)를 추출하라."
# 예: "약병 찾아서 빨간 컵 옆에 놓아" → {source: "약병", destination: "빨간 컵"}
```

#### 3-3-2. 오케스트레이터 Phase 구조

```python
# 기존 SingleObjectOrchestrator phases:
# SEARCH_OBJECT → APPROACH → GRASP → SEARCH_BASKET → NAVIGATE_TO_BASKET → PLACE → DONE

# 변경 → RelativePlacementOrchestrator phases:
# SEARCH_SOURCE → APPROACH_SOURCE → GRASP → SEARCH_DESTINATION → APPROACH_DESTINATION → PLACE → DONE

class RelativePlacementOrchestrator:
    """
    Phase 전환 로직:
    
    1. SEARCH_SOURCE: 약병을 탐색 (제자리 회전, 전진 탐색)
       → 약병이 이미지에 보이면 → APPROACH_SOURCE
    
    2. APPROACH_SOURCE: 약병에 접근
       → 약병이 팔 닿을 거리에 보이면 → GRASP
    
    3. GRASP: 약병을 잡음
       → 물체를 잡은 것을 확인 → SEARCH_DESTINATION
       ★ 여기서 VLM의 추적 대상이 약병 → 빨간 컵으로 전환
    
    4. SEARCH_DESTINATION: 빨간 컵을 탐색 (물체를 든 채로)
       → 빨간 컵이 이미지에 보이면 → APPROACH_DESTINATION
    
    5. APPROACH_DESTINATION: 빨간 컵 근처로 접근
       → 빨간 컵이 가까이 보이면 → PLACE
    
    6. PLACE: 빨간 컵 옆에 약병을 내려놓음
       → 놓은 것을 확인 → DONE
    """
```

#### 3-3-3. VLM `/infer` 호출 시 instruction 전환

```python
# Phase별 VLM에게 주는 맥락:
# SEARCH_SOURCE ~ GRASP: "너의 목표는 약병을 찾아서 잡는 것이다"
# SEARCH_DESTINATION ~ PLACE: "약병을 들고 있다. 이제 빨간 컵을 찾아서 그 옆에 놓아라"
#
# VLM이 VLA에게 생성하는 instruction 예시:
# SEARCH_SOURCE: "turn right slowly to search for the medicine bottle"
# APPROACH_SOURCE: "move toward the medicine bottle and grasp it"
# SEARCH_DESTINATION: "turn left to find the red cup"
# APPROACH_DESTINATION: "move toward the red cup"
# PLACE: "place the medicine bottle next to the red cup"
```

---

### 3-4. `vlm_prompts.py` (대규모)

기존 SINGLE_OBJECT_SYSTEM_PROMPT, MULTI_CLEANUP_SYSTEM_PROMPT를 삭제/대체.

#### 새 Prompt 구조

```python
CLASSIFY_SYSTEM_PROMPT = """
사용자 명령을 분석하여 JSON으로 반환하라.
- source_object: 잡아야 할 물체
- destination_object: 놓을 기준 물체
예: "약병 찾아서 빨간 컵 옆에 놓아" → {"source_object": "medicine bottle", "destination_object": "red cup"}
JSON만 출력하라.
"""

RELATIVE_PLACEMENT_SYSTEM_PROMPT = """
너는 모바일 매니퓰레이터의 지휘자다.

현재 임무: {source_object}를 찾아서 잡고, {destination_object} 옆에 놓아라.
현재 Phase: {current_phase}

카메라 이미지를 보고, 로봇이 다음에 해야 할 행동 하나를 자연어로 지시해라.

Phase별 판단 기준:
- SEARCH_SOURCE: {source_object}가 이미지에 보이지 않으면 → 탐색 지시 (회전/전진)
                  {source_object}가 보이면 → "approach the {source_object}"
- APPROACH_SOURCE: {source_object}가 멀면 → 접근 지시
                    {source_object}가 팔 닿을 거리면 → "grasp the {source_object}"
- GRASP: 잡는 중 → 기다림 / 잡았으면 → phase를 SEARCH_DESTINATION으로 전환
- SEARCH_DESTINATION: {destination_object}가 보이지 않으면 → 탐색 지시
                       {destination_object}가 보이면 → "move toward the {destination_object}"
- APPROACH_DESTINATION: {destination_object}가 멀면 → 접근 지시
                         가까우면 → "place the {source_object} next to the {destination_object}"
- PLACE: 놓는 중 → 기다림 / 놓았으면 → "done"

출력 형식:
{{"instruction": "...", "phase": "SEARCH_SOURCE|APPROACH_SOURCE|GRASP|SEARCH_DESTINATION|APPROACH_DESTINATION|PLACE|DONE"}}
"""
```

**기존 BASKET_DETECTION_PROMPT 삭제** — 바구니 감지가 불필요해짐.

---

### 3-5. `record_teleop.py` (중규모)

#### Combined 모드 변경

```python
# 기존 Combined 3-Phase:
# Phase 1 (Skill-2 기록): 접근+파지
# Phase 2 (Transit, 미기록): 바구니 근처까지 이동
# Phase 3 (Skill-3 기록): 바구니 접근+place

# 변경 →
# Phase 1 (Skill-2 기록): 약병 접근+파지
# Phase 2 (Transit, 미기록): 빨간 컵 근처까지 이동
# Phase 3 (Skill-3 기록): 빨간 컵 접근 + 옆에 놓기

# Phase 2→3 전환 조건 변경:
# 기존: dest_dist < 0.7m (바구니까지 거리) AND |heading_to_dest| < 0.76rad
# 변경: dest_dist < 0.7m (빨간 컵까지 거리) AND |heading_to_dest| < 0.76rad

# 목적지 마커 변경:
# 기존: 초록 구체가 바구니 위치에 표시
# 변경: 초록 구체가 빨간 컵 위치에 표시 (또는 빨간 컵 자체가 보이므로 마커 불필요)
```

#### 환경 스폰 변경

```python
# 텔레옵 시 환경에 두 물체가 동시에 존재해야 함:
# - 약병: grasp 대상 (기존 source object 스폰 로직 그대로)
# - 빨간 컵: place 기준점 (기존 basket 스폰 위치에 빨간 컵 스폰)
```

---

### 3-6. `collect_demos.py` (소규모)

#### Instruction 템플릿 변경

```python
# 기존 (Skill-2):
# "approach the {object} and grasp it"
# "pick up the {object}"

# 변경 (Skill-2) — 동일:
# "approach the medicine bottle and grasp it"
# "pick up the medicine bottle"

# 기존 (Skill-3):
# "carry the {object} to the basket and place it inside"
# "bring the {object} to the basket"

# 변경 (Skill-3):
# "carry the medicine bottle to the red cup and place it next to it"
# "place the medicine bottle next to the red cup"
```

#### Skill-3 데이터 수집 시 환경

```python
# 기존: handoff buffer에서 복원 + 바구니 스폰
# 변경: handoff buffer에서 복원 + 빨간 컵 스폰
# collect_demos.py의 Skill3EnvWithCam이 lekiwi_skill3_env.py를 상속하므로,
# env 변경이 자동 반영됨
```

---

### 3-7. `eval_full_system.py` (중규모)

#### 환경 설정

```python
# 기존: 물체 랜덤 배치 + 바구니 env origin 근처 스폰
# 변경: 약병 랜덤 배치 + 빨간 컵 별도 랜덤 배치 (최소 1.0m 이격)
```

#### Success 판정

```python
# 기존: 물체가 바구니 안(XY 0.20m + Z 0.25m) 안착 + gripper open
# 변경: 약병이 빨간 컵으로부터 XY 0.20m 이내 + Z 높이 차이 0.10m 이내 + gripper open
```

#### 오케스트레이터

```python
# 기존: SingleObjectOrchestrator / MultiCleanupOrchestrator 선택
# 변경: RelativePlacementOrchestrator 사용
# /classify 결과로 source_object, destination_object 전달
```

---

### 3-8. `generate_handoff_buffer.py` (소규모)

변경 최소. Handoff buffer는 Skill-2 성공 상태(로봇 + source 물체)를 저장하므로 destination object 정보는 포함하지 않음. Skill-3가 리셋할 때 빨간 컵을 새로 스폰함.

```python
# 기존: Skill-2 성공 시 (robot_state, source_object_state) 저장
# 변경: 동일 — destination object는 Skill-3 리셋 시 새로 스폰되므로 buffer에 불필요
```

---

## 4. 변경하지 않는 것

다음은 **명시적으로 변경하지 않는** 항목:

| 항목 | 이유 |
|------|------|
| 9D state/action 구조 | observation.state[9], action[9] 불변 |
| Skill-1 Navigate 환경 | 방향 추종 전용, task 무관 |
| RL 학습 구조 (train_lekiwi.py) | PPO/AAC 구조 불변 |
| BC 학습 (train_bc.py) | obs → action 매핑 불변 |
| 데이터 변환 (convert_hdf5_to_lerobot_v3.py) | 9D 포맷 불변 |
| Skill-2 RL 보상 구조 | 접근+파지 보상 불변, dest object는 배경 |
| deploy_vla_action_bridge.py | sim→real 변환 불변 |
| 캘리브레이션 파일 | 물리 캘리브레이션과 무관 |

---

## 5. Skill-3 Actor Observation (29D) 대응표

obs 차원과 구조는 동일. 의미만 변경.

| Index | 기존 | 변경 | 비고 |
|-------|------|------|------|
| 0:5 | arm_joint_pos (5D) | 동일 | |
| 5 | gripper_pos (1D) | 동일 | |
| 6:9 | base_body_vel (3D) | 동일 | |
| 9:15 | base_vel (6D) | 동일 | lin+ang world vel |
| 15:21 | arm_vel (6D) | 동일 | |
| 21:24 | **basket_rel (3D)** | **dest_object_rel (3D)** | 빨간 컵까지 body-frame 상대 위치 |
| 24 | grip_force (1D) | 동일 | |
| 25:28 | obj_bbox (3D) | 동일 | source object(약병) bbox |
| 28 | obj_category (1D) | 동일 | source object category |

Critic obs도 동일한 치환 (36D 중 해당 인덱스만 변경).

---

## 6. 실험 설계 변경 (기존 C1~C7 → 새 A/B/C 구조)

기존 C1~C7을 폐기하고, **비교 축을 명확히 분리한 A/B/C 구조**로 재설계한다.

### 6-1. 비교 축

| 비교 | 무엇을 보여주는가 |
|------|------------------|
| A1 vs B1 | VLM 오케스트레이션 + skill 분리의 효과 |
| B1 vs B2 | Mimic 데이터 증강의 효과 (VLM+VLA 구조 내에서) |
| B2 vs C1 | RL expert 데이터의 효과 (증강 텔레옵 vs RL rollout) |
| C1 vs C2 | RL + Mimic 합산의 효과 |

### 6-2. 실험 조건 (5개)

모든 조건에서 task는 동일: **"약병 찾아서 빨간 컵 옆에 놓아"**
모든 조건에서 VLA 모델은 동일: **π0-FAST**

#### Group A: E2E 베이스라인 (단일 VLA, VLM 없음)

| 조건 | VLM | RL | 데이터 | 설명 |
|------|-----|----|--------|------|
| **A1** | ✗ | ✗ | 전체 task 텔레옵 50개 | E2E 베이스라인 |

- 사람이 전체 task(약병 탐색 → 잡기 → 빨간 컵 탐색 → 옆에 놓기)를 처음부터 끝까지 시범
- 에피소드당 30~60초, 성공 에피소드만 저장
- **고정 instruction 하나**로 VLA fine-tune: "find the medicine bottle and place it next to the red cup"
- VLM 없음, skill 분리 없음, 중간 phase 전환을 VLA가 implicit하게 해야 함

#### Group B: VLM + VLA, 텔레옵만 (RL 없음)

| 조건 | VLM | RL | 데이터 | 설명 |
|------|-----|----|--------|------|
| **B1** | ✓ | ✗ | skill별 텔레옵 20~40개 | VLM+VLA, 텔레옵 only |
| **B2** | ✓ | ✗ | skill별 텔레옵 + Mimic 1K | VLM+VLA, 텔레옵+증강 |

- 우리 파이프라인과 동일한 VLM+VLA 구조 (VLM 오케스트레이션 + skill별 VLA)
- 단, **RL을 거치지 않음** — 텔레옵 데이터를 바로 VLA fine-tune에 사용
- B1: Skill-2 텔레옵 10~20개 + Skill-3 텔레옵 10~20개 → VLA fine-tune
- B2: B1 텔레옵을 Isaac Lab Mimic으로 skill별 1K개로 증강 → VLA fine-tune
- B1의 텔레옵 데이터는 **우리 파이프라인 Phase 1에서 수집하는 것과 동일** (추가 수집 불필요)

#### Group C: VLM + VLA + RL Expert (우리 최종 파이프라인)

| 조건 | VLM | RL | 데이터 | 설명 |
|------|-----|----|--------|------|
| **C1** | ✓ | ✓ | RL rollout 1K | 우리 파이프라인 (1K) |
| **C2** | ✓ | ✓ | RL rollout 1K + Mimic 1K | 우리 파이프라인 (RL+Mimic) |

- 우리 최종 파이프라인: 텔레옵 → BC → RL → Expert Rollout → VLA fine-tune
- VLM 오케스트레이션 동일
- C1: RL Expert rollout 1K개 (성공 에피소드, DR 적용)
- C2: RL rollout 1K + Mimic 1K 합산 (데이터 소스 다양성)

### 6-3. 데이터 수집 요약

| 데이터 | 수집 방법 | 사용 조건 | 수집 비용 |
|--------|----------|----------|----------|
| 전체 task 텔레옵 50개 | 사람이 E2E 시범 (30~60초/개) | A1 | 높음 (사람 ~50분) |
| skill별 텔레옵 20~40개 | 사람이 skill별 시범 (5~10초/개) | B1, B2 | 낮음 (사람 ~10분) |
| skill별 Mimic 1K | 텔레옵에서 keypoint 보간 증강 | B2, C2 | 자동 |
| RL rollout 1K | RL Expert 자동 실행 | C1, C2 | 자동 (sim) |

### 6-4. 예상 스토리라인

- **A1 성공률이 매우 낮음** (5~15%) → 단일 VLA는 phase 전환을 implicit하게 처리할 수 없다
- **A1 vs B1**: B1이 크게 우세 → VLM 오케스트레이션의 핵심 기여 증명
- **B1 vs B2**: B2가 약간 우세 → Mimic 증강이 소규모 텔레옵에서 유효
- **B2 vs C1**: C1이 우세 → RL Expert 데이터가 사람 텔레옵+증강보다 품질 높음
- **C1 vs C2**: C2가 약간 우세 → 데이터 소스 다양성이 VLA 일반화에 도움

핵심 메시지: **"데이터 양(A1: 50개 E2E)보다 구조(B1: VLM+skill 20~40개)가 중요하고, 구조 위에 RL 데이터(C1: 1K rollout)를 올리면 더 좋아진다."**

### 6-5. 평가 프로토콜

**Stage A — Sim (3090 Desktop + A100 서버)**:
- 5개 조건 전부 sim에서 각 30회+
- 5 × 30 = 150회
- 하위 조건 탈락, 상위 2~3개만 Stage B로

**Stage B — Real Robot**:
- 선별된 조건만 실기 각 30회+
- 95% binomial CI

**성공 판정**:
```
약병이 빨간 컵으로부터 XY 20cm 이내 + Z 높이 차이 10cm 이내 + gripper open
```

**실패 유형 분류**:

| 실패 유형 | 의미 | 주로 어디서 발생 |
|-----------|------|----------------|
| navigation miss (source) | 약병을 못 찾음 | A1, B1 |
| grasp fail | 약병 파지 실패 | 전 조건 |
| navigation miss (dest) | 빨간 컵을 못 찾음 | A1 (★ E2E 최대 실패 지점) |
| drop during carry | 운반 중 낙하 | 전 조건 |
| placement fail | 빨간 컵 옆에 못 놓음 | 전 조건 |

`navigation miss (dest)` — 물체를 잡은 후 빨간 컵을 탐색하지 못하는 실패. E2E(A1)에서 가장 많이 발생할 것으로 예상. VLM+VLA(B/C)에서는 VLM이 추적 대상을 명시적으로 전환하므로 발생률이 크게 낮아짐. 이 실패 유형의 비율 차이가 파이프라인의 핵심 contribution을 보여줌.

**Go 기준**:
- Skill별 단독 sim 평가: 70%+ 성공률
- VLM+VLA 통합 sim 평가: 50%+ 성공률 (C1 또는 C2에서)
- E2E 대비 우위: C 그룹 성공률 > A1 성공률 (통계적 유의)

### 6-6. 평가 시 물체 배치

```python
# 약병: env origin에서 1.5~3.0m, 랜덤 방향, 바닥
# 빨간 컵: env origin에서 2.0~4.0m, 랜덤 방향, 바닥
# 약병과 빨간 컵 최소 이격: 1.0m
# 실내 환경: 가구/장애물 배치 (cluttered indoor)
# 5개 조건 모두 동일한 물체 배치 세트에서 평가 (공정 비교)
```

---

## 7. 문서 업데이트

다음 문서의 해당 섹션을 새 task에 맞게 갱신해야 함:

| 문서 | 갱신 섹션 |
|------|----------|
| `1_전체_파이프라인.md` | §1(문제 정의), §4(VLM 루프), §5(3-Skill 설명), §14(실험 설계), §16(체크리스트) |
| `2_Sim_데이터_수집_파이프라인.md` | §3-3(Skill-3 학습), §4-5(Skill-3 데이터 수집), §7(Phase 4.5 평가) |
| `3_코드_현황_정리.md` | §1(아키텍처), §1-4(핵심 설계 결정), §7(비교 실험), §8(남은 작업) |

---

## 8. 물체 사용 전략 (Phase별)

### 8-1. 텔레옵 (Phase 1, BC warm-start용)

**약병 + 빨간 컵만 사용한다.**

텔레옵 데모는 Skill-2 10~20개, Skill-3 10~20개로 총 20~40개뿐이다. 여기서 다른 물체를 섞으면 물체당 데모 수가 너무 적어져서 BC가 의미 있는 행동 패턴을 학습하지 못한다.

```bash
# Skill-2 텔레옵: 약병을 잡는 시범
# source_object = 약병 (5_HTP)
# dest_object = 빨간 컵 (배경으로 존재, 잡지 않음)

# Skill-3 텔레옵 (combined 모드): 약병을 빨간 컵 옆에 놓는 시범
# source_object = 약병
# dest_object = 빨간 컵 (place 기준점)
```

`--multi_object_json` 플래그는 사용하지 않는다. 대신 약병 USD를 직접 지정:
```bash
python record_teleop.py --num_demos 20 \
  --skill combined \
  --object_usd /home/yubin11/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd /home/yubin11/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1" \
  --arm_limit_json calibration/arm_limits_measured.json
```

### 8-2. RL 학습 (Phase 1, Expert 정책 학습)

**Skill-2 (ApproachAndGrasp)**: 약병 50% + 나머지 22종 50% 혼합.

약병만으로 학습하면 약병 형태에 과적합되어 sim2real 전이 시 깨질 수 있다. 다중 물체를 섞으면 다양한 형태/크기에 대한 robust한 grasp 전략이 나온다. 기존 `obj_bbox(3D)` + `obj_category(1D)`가 Actor obs에 있으므로, 물체별 차별화 학습이 가능하다.

구현: `object_catalog.json`에서 약병의 sampling weight를 50%로, 나머지 21종을 균등하게 50%로 설정.

```python
# object_catalog.json에 weight 필드 추가 또는 train_lekiwi.py에서 sampling 로직:
# 약병(5_HTP): weight = 0.5
# 나머지 21종: weight = 0.5 / 21 ≈ 0.024 each
```

**Skill-3 (CarryAndPlace)**: 약병만 사용.

source object = 약병 고정, destination object = 빨간 컵 고정. Handoff buffer가 Skill-2의 약병 성공 상태에서 샘플링되므로, Skill-3에서 다른 물체가 올 수 없다. 다중 물체가 필요하면 Skill-2의 handoff buffer를 다중 물체로 생성해야 하지만, 최종 평가가 약병 고정이므로 불필요하다.

**Skill-1 (Navigate)**: 물체 무관. 방향 추종 전용이므로 기존 그대로.

### 8-3. VLA 데이터 수집 (Phase 2, Expert Rollout)

**Skill-2**: RL Expert가 다중 물체(22종)로 학습되었으므로, rollout도 다중 물체로 수행하여 VLA 학습 데이터의 다양성을 확보한다. 단, instruction에 물체명을 반영:
- 약병 에피소드: "approach the medicine bottle and grasp it"
- 다른 물체 에피소드: "approach the {object_name} and grasp it"

**Skill-3**: 약병 + 빨간 컵 고정. Instruction: "place the medicine bottle next to the red cup"

**Navigate**: 기존과 동일 (방향 명령 + 장애물 회피).

### 8-4. 요약

| Phase | Skill-2 물체 | Skill-3 source | Skill-3 dest | 이유 |
|-------|-------------|----------------|--------------|------|
| 텔레옵 (BC) | 약병만 | 약병 | 빨간 컵 | 데모 20개 → 물체 분산 불가 |
| RL 학습 | 약병 50% + 22종 50% | 약병 | 빨간 컵 | 다양성으로 robust grasp |
| VLA 데이터 수집 | 22종 전체 | 약병 | 빨간 컵 | VLA 일반화 |
| 평가 | 약병 | 약병 | 빨간 컵 | 최종 task 고정 |

---

## 9. 구현 순서 권장

```
1. lekiwi_skill3_env.py 수정 (가장 큰 변경, 핵심)
   → 빨간 컵 스폰, success criteria, obs 변경
   → 단독 테스트: python train_lekiwi.py --skill carry_and_place --num_envs 4 로 env 로드 확인

2. lekiwi_skill2_env.py 수정 (빨간 컵 배경 스폰 추가)
   → 기존 RL 학습에 영향 없음을 확인

3. record_teleop.py 수정 (combined 모드 목적지 변경)
   → 텔레옵 테스트 가능

4. vlm_prompts.py + vlm_orchestrator.py 수정 (VLM 오케스트레이션)
   → Phase 4.5에서 필요, 당장은 아님

5. collect_demos.py 수정 (instruction 템플릿)
   → Phase 2에서 필요

6. eval_full_system.py 수정 (평가 스크립트)
   → Phase 4.5에서 필요

7. 문서 갱신
```
