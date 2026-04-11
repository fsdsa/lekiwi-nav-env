# VIVA S1~S4 수정 지시서

## 문제 요약

현재 VIVA의 VLM-VLA 상호작용에 다음 문제들이 있다:

1. **S2/S4**: VLM이 instruction을 계속 갈아끼움 → VLA가 학습한 skill trajectory와 충돌
2. **S2/S4**: LIFTED_COMPLETE, PLACE_COMPLETE를 VLM이 판별 → 코드에서 판별 가능한데 불필요한 VLM 호출
3. **S1/S3**: VLM이 자유 텍스트로 instruction 생성 → VLA 학습 데이터와 불일치 가능
4. **S1/S3**: VLM이 거의 매 스텝(~2스텝마다) 호출됨 → instruction이 너무 자주 바뀌어 로봇이 한 방향으로 충분히 이동하지 못함
5. **S1→S2, S3→S4 전환**: 이미지에서 target이 보이기만 하면 전환 → S2/S4는 0.6~0.9m 거리에서 시작하도록 학습됐으므로 target이 충분히 가까워진 후 전환해야 함
6. **S2 장애물 회피**: OBSTACLE 시 무조건 S1(navigate)로 전환 → 물체를 이미 잡은 상태라면 S3(carry)로 가야 함
7. **S2 + OBSTACLE + contact 후 복귀 로직 모순**: 물체를 잡은 상태에서 OBSTACLE → S3(carry) 복귀 후 TARGET_FOUND → `_interrupted_skill`이 S2라서 다시 "approach and lift"로 재진입 → 이미 물체를 잡고 있으므로 S2는 무의미
8. **S1/S3 VLM 응답 validation 없음**: VLM이 7개 고정 커맨드 외의 텍스트를 반환해도 그대로 VLA에 전달됨
9. **초기 instruction 불일치**: `_latest_instruction = "move forward"` → S1 커맨드 목록에 없음
10. **S2/S4 프롬프트 format 불필요 파라미터**: `prev_instruction` 전달하지만 프롬프트에 해당 placeholder 없음
11. **LIFTED_POSE_RANGE 중복 정의**: `run_full_task.py`와 `vlm_orchestrator.py`에 둘 다 정의
12. **check_place_complete 오탐**: grip open + no contact만으로 판별 → 일시적 contact 끊김 시 오탐
13. **S2/S4 depth warning 시 VLM 반복 호출**: 목표물에 접근하면 depth가 계속 threshold 이하 → CONTINUE 판정 후에도 매 스텝 VLM 호출

---

## 수정 후 설계

### Skill별 역할 분담

| Skill | VLA instruction | VLM 역할 | VLM 호출 빈도 | 전환 조건 |
|-------|-----------------|----------|--------------|----------|
| S1 navigate | 6개 고정 커맨드 중 택 1 | 방향 선택 + TARGET_FOUND | **50스텝마다** | VLM이 TARGET_FOUND 출력 |
| S2 approach & lift | `"approach and lift the {source}"` (고정) | depth warning 시 OBSTACLE 판별만 | depth warning 최초 1회만 | **코드** (joint + contact) |
| S3 carry | 6개 고정 커맨드 중 택 1 | 방향 선택 + TARGET_FOUND | **50스텝마다** | VLM이 TARGET_FOUND 출력 |
| S4 approach & place | `"place the {source} next to the {dest}"` (고정) | depth warning 시 OBSTACLE 판별만 | depth warning 최초 1회만 | **코드** (gripper + contact, 연속 N스텝) |

### S1 커맨드 목록

```
navigate forward
navigate backward
navigate left
navigate right
navigate turn left
navigate turn right
TARGET_FOUND
```

### S3 커맨드 목록

```
carry forward
carry backward
carry left
carry right
carry turn left
carry turn right
TARGET_FOUND
```

### S1/S3 호출 주기

- **50스텝마다 VLM 호출** (6.4Hz 기준 약 7.8초)
- VLM 응답이 올 때까지 이전 instruction 유지
- 장애물 회피는 safety layer가 매 스텝 코드로 처리 (depth < 0.3m → base vx, vy 정지, wz 회전 유지)

### TARGET_FOUND의 의미

**"target이 보이고, 충분히 가까워서 S2/S4를 시작할 수 있다"**를 의미한다.
- S2/S4는 0.6~0.9m 거리에서 시작하도록 학습됨
- target이 보이지만 멀면 → target 방향으로 접근하는 커맨드를 계속 출력

### S2/S4 VLM 호출 조건

VLM은 **depth_min < threshold이고, 아직 CONTINUE 판정을 받지 않았을 때만** 호출된다.
한 번 CONTINUE (= 목표물로 확인) 판정을 받으면 이후 depth가 계속 낮아져도 VLM을 재호출하지 않는다.

```
S2/S4 진입 시: _obstacle_cleared = False

매 스텝:
  1. 코드 전환 판별 (LIFTED_COMPLETE / PLACE_COMPLETE) → 해당 시 전환
  2. depth_min < threshold AND _obstacle_cleared == False:
       → VLM 호출 → OBSTACLE / CONTINUE
       ├─ CONTINUE → _obstacle_cleared = True (이후 재호출 안 함)
       └─ OBSTACLE → 이전 스킬로 복귀 (_obstacle_cleared는 재진입 시 다시 False)
  3. 그 외 → VLM 호출 안 함, VLA 자율 수행
```

### S2 OBSTACLE + contact 시 복귀 로직

S2에서 물체를 이미 잡은 상태(contact=True)에서 OBSTACLE이 발생하면:
- S2("approach and lift")는 이미 완료된 것이나 마찬가지
- `_interrupted_skill`을 S2가 아닌 **없음(None)**으로 설정
- S3(carry)로 전환하되 **정상 흐름으로 진행** (장애물 회피 복귀가 아님)
- S3에서 dest_object를 찾아 TARGET_FOUND → **S4로 정상 전환**

```
S2 + OBSTACLE + contact 없음:
  → _interrupted_skill = S2
  → S1(navigate)로 복귀
  → TARGET_FOUND → S2로 재진입 (정상 복귀)

S2 + OBSTACLE + contact 있음:
  → _interrupted_skill = None (복귀 아님, 정상 진행)
  → S3(carry)로 전환
  → TARGET_FOUND → S4로 정상 전환

S4 + OBSTACLE:
  → _interrupted_skill = S4
  → S3(carry)로 복귀
  → TARGET_FOUND → S4로 재진입 (정상 복귀)
```

---

## 수정 대상 파일

1. `vlm_prompts.py` — S1/S2/S3/S4 프롬프트 전체 수정
2. `vlm_orchestrator.py` — 대부분의 로직 수정
3. `run_full_task.py` — VLM 호출 조건 분기, LIFTED_POSE_RANGE 중복 제거

---

## 1. `vlm_prompts.py` 수정

### S1 프롬프트 (VIVA_NAVIGATE_SYSTEM_PROMPT) — 전체 교체

```python
VIVA_NAVIGATE_SYSTEM_PROMPT = """You are the navigation commander of a mobile manipulator robot.
You see through a front-facing camera (RealSense D455, 87° FOV).

Current task: find the {target_object}.
The robot has NOT found the {target_object} yet. Guide the robot to search for it.

{robot_status}

Look at the camera image and output ONE of the following commands:

- "navigate forward" — path ahead is clear, go straight
- "navigate backward" — need to back up (only if stuck or dead end)
- "navigate left" — strafe left
- "navigate right" — strafe right
- "navigate turn left" — rotate left to explore or avoid obstacle
- "navigate turn right" — rotate right to explore or avoid obstacle
- "TARGET_FOUND" — the {target_object} is visible AND close enough to reach (the object occupies a large portion of the frame, roughly within 1 meter). Do NOT output this if the target is far away or small in the image.

Decision rules:
1. If you see the {target_object} close and large in the frame → "TARGET_FOUND"
2. If you see the {target_object} but it is far away or small → navigate toward it (e.g. "navigate forward")
3. If path ahead is clear and no target visible → "navigate forward"
4. If wall or furniture blocks the path → turn toward the side with more open space
5. If you see a doorway or corridor → turn toward it
6. If the robot seems stuck (seeing the same wall up close) → "navigate backward"
7. Prefer exploring new areas over revisiting the same space

Previous command: "{prev_command}"

Output ONLY the command, nothing else."""
```

### S3 프롬프트 (VIVA_CARRY_SYSTEM_PROMPT) — 전체 교체

```python
VIVA_CARRY_SYSTEM_PROMPT = """You are the navigation commander of a mobile manipulator robot.
You see through a front-facing camera (RealSense D455, 87° FOV).

The robot is currently CARRYING the {source_object}.
Current task: carry the {source_object} and navigate to find the {dest_object}.

{robot_status}

Look at the camera image and output ONE of the following commands:

- "carry forward" — path ahead is clear, go straight
- "carry backward" — need to back up (only if stuck or dead end)
- "carry left" — strafe left
- "carry right" — strafe right
- "carry turn left" — rotate left to explore or avoid obstacle
- "carry turn right" — rotate right to explore or avoid obstacle
- "TARGET_FOUND" — the {dest_object} is visible AND close enough to reach (the object occupies a large portion of the frame, roughly within 1 meter). Do NOT output this if the target is far away or small in the image.

Decision rules:
1. If you see the {dest_object} close and large in the frame → "TARGET_FOUND"
2. If you see the {dest_object} but it is far away or small → navigate toward it (e.g. "carry forward")
3. If path ahead is clear → "carry forward"
4. If obstacle blocks the path → turn toward open space
5. If you see a doorway or corridor → turn toward it
6. If the robot seems stuck → "carry backward"

Previous command: "{prev_command}"

Output ONLY the command, nothing else."""
```

### S2 프롬프트 (VIVA_APPROACH_LIFT_SYSTEM_PROMPT) — 전체 교체

```python
VIVA_APPROACH_LIFT_SYSTEM_PROMPT = """You are the obstacle monitor of a mobile manipulator robot.
You see through a front-facing camera (RealSense D455, 87° FOV).

The robot is currently executing the "approach and lift" skill for the {source_object}.
The depth sensor has detected a close object.

{robot_status}

Your job: determine if the close object is the target ({source_object}) or an unexpected obstacle.

Look at the camera image and output ONE of the following:
- "CONTINUE" — the close object is the {source_object} (expected, VLA should keep going)
- "OBSTACLE" — the close object is NOT the {source_object} (unexpected obstacle, need to retreat and reroute)

Output ONLY one word, nothing else."""
```

### S4 프롬프트 (VIVA_APPROACH_PLACE_SYSTEM_PROMPT) — 전체 교체

```python
VIVA_APPROACH_PLACE_SYSTEM_PROMPT = """You are the obstacle monitor of a mobile manipulator robot.
You see through a front-facing camera (RealSense D455, 87° FOV).

The robot is currently executing the "approach and place" skill.
It is holding the {source_object} and placing it next to the {dest_object}.
The depth sensor has detected a close object.

{robot_status}

Your job: determine if the close object is the target ({dest_object}) or an unexpected obstacle.

Look at the camera image and output ONE of the following:
- "CONTINUE" — the close object is the {dest_object} (expected, VLA should keep going)
- "OBSTACLE" — the close object is NOT the {dest_object} (unexpected obstacle, need to retreat and reroute)

Output ONLY one word, nothing else."""
```

**주의**: S2/S4 프롬프트에 `{prev_instruction}` placeholder가 없으므로, `_build_vlm_payload()`에서 `prev_instruction` 인자를 전달하지 않도록 수정해야 한다 (문제 10번).

---

## 2. `vlm_orchestrator.py` 수정

### (a) LIFTED_POSE_RANGE 중복 제거 (문제 11번)

`run_full_task.py`에서 `LIFTED_POSE_RANGE`를 제거하고, `vlm_orchestrator.py`에만 정의한다.
`run_full_task.py`의 `check_lifted_pose()`도 제거하고 `vlm_orchestrator.py`의 `check_lifted_complete()`만 사용한다.

`vlm_orchestrator.py`에 이미 정의되어 있으므로 유지:
```python
LIFTED_POSE_RANGE = {
    "arm0": (-0.09, +0.16),
    "arm1": (-0.20, -0.19),
    "arm2": (+0.23, +0.31),
    "arm3": (-1.52, -0.98),
    "arm4": (-0.06, +0.01),
    "grip": (0.13, 0.55),
}
```

`run_full_task.py`에서:
- `LIFTED_POSE_RANGE` 정의 삭제
- `check_lifted_pose()` 함수 삭제
- `build_robot_status()` 내에서 `check_lifted_pose()` 호출 → `from vlm_orchestrator import LIFTED_POSE_RANGE` 후 직접 체크하거나, orchestrator의 메서드를 사용

### (b) 초기 instruction 수정 (문제 9번)

```python
# 기존
self._latest_instruction = "move forward"

# 수정
self._latest_instruction = "navigate forward"
```

### (c) `_obstacle_cleared` 플래그 추가 (문제 13번)

`__init__()`에 추가:
```python
self._obstacle_cleared = False  # S2/S4에서 CONTINUE 판정 후 VLM 재호출 방지
```

`_transition_to()`에서 S2/S4 진입 시 리셋:
```python
def _transition_to(self, next_skill: SkillState):
    prev = self._current_skill
    self._current_skill = next_skill
    self._skill_step_count = 0
    if next_skill == SkillState.DONE:
        self._done = True

    # S2/S4 진입 시 VLA instruction 고정 + obstacle 플래그 리셋
    if next_skill == SkillState.APPROACH_AND_LIFT:
        self._latest_instruction = f"approach and lift the {self.source_object}"
        self._obstacle_cleared = False
    elif next_skill == SkillState.APPROACH_AND_PLACE:
        self._latest_instruction = f"place the {self.source_object} next to the {self.dest_object}"
        self._obstacle_cleared = False
    elif next_skill == SkillState.CARRY:
        self._latest_instruction = "carry forward"

    print(f"  [SKILL] {prev.value} → {next_skill.value}")
```

`_process_vlm_response()`에서 CONTINUE 시 플래그 설정:
```python
def _process_vlm_response(self, raw: str) -> str:
    cleaned = raw.strip().strip('"').strip("'")

    if cleaned == "TARGET_FOUND":
        self._handle_target_found()
        return self._latest_instruction

    if cleaned == "OBSTACLE":
        self._handle_obstacle()
        return self._latest_instruction

    # S2/S4에서 CONTINUE → obstacle_cleared 설정, instruction 변경 안 함
    if self._current_skill in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
        if cleaned == "CONTINUE":
            self._obstacle_cleared = True
        return self._latest_instruction

    # S1/S3 VLM 응답 validation (문제 8번)
    valid_commands = self._get_valid_commands()
    if cleaned in valid_commands:
        return cleaned
    else:
        print(f"  [VLM] Invalid response: \"{cleaned}\", keeping previous instruction")
        return self._latest_instruction
```

`obstacle_cleared` 프로퍼티 추가:
```python
@property
def obstacle_cleared(self) -> bool:
    return self._obstacle_cleared
```

### (d) S1/S3 VLM 응답 validation (문제 8번)

valid commands 목록 메서드 추가:

```python
NAVIGATE_COMMANDS = {
    "navigate forward", "navigate backward",
    "navigate left", "navigate right",
    "navigate turn left", "navigate turn right",
    "TARGET_FOUND",
}

CARRY_COMMANDS = {
    "carry forward", "carry backward",
    "carry left", "carry right",
    "carry turn left", "carry turn right",
    "TARGET_FOUND",
}

def _get_valid_commands(self) -> set:
    if self._current_skill == SkillState.NAVIGATE:
        return NAVIGATE_COMMANDS
    elif self._current_skill == SkillState.CARRY:
        return CARRY_COMMANDS
    return set()
```

### (e) `_handle_obstacle()` — contact 기반 분기 + 복귀 로직 수정 (문제 6, 7번)

```python
def _handle_obstacle(self):
    """S2/S4에서 VLM이 "OBSTACLE" 판단 시 호출.
    contact 여부에 따라 복귀 스킬 결정:
      - S2 + contact 없음 → S1(navigate)로 복귀 (장애물 회피 후 S2 재진입)
      - S2 + contact 있음 → S3(carry)로 정상 전환 (물체 잡았으므로 S2 완료 취급)
      - S4 → S3(carry)로 복귀 (장애물 회피 후 S4 재진입)
    """
    if self._current_skill == SkillState.APPROACH_AND_LIFT:
        if self._latest_contact:
            # 물체를 이미 잡은 상태 → S2는 사실상 완료
            # 정상 S3→S4 흐름으로 진행 (_interrupted_skill 설정 안 함)
            self._interrupted_skill = None
            self._transition_to(SkillState.CARRY)
        else:
            # 물체 안 잡음 → S1으로 복귀, 회피 후 S2 재진입
            self._interrupted_skill = SkillState.APPROACH_AND_LIFT
            self._transition_to(SkillState.NAVIGATE)
    elif self._current_skill == SkillState.APPROACH_AND_PLACE:
        # S4는 항상 물체 들고 있음 → S3으로 복귀, 회피 후 S4 재진입
        self._interrupted_skill = SkillState.APPROACH_AND_PLACE
        self._transition_to(SkillState.CARRY)
```

### (f) `_build_vlm_payload()` — S2/S4 불필요 파라미터 제거 (문제 10번)

S2/S4 프롬프트에 `{prev_instruction}` placeholder가 없으므로 format 인자에서 제거:

```python
elif skill == SkillState.APPROACH_AND_LIFT:
    system_prompt = VIVA_APPROACH_LIFT_SYSTEM_PROMPT.format(
        source_object=self.source_object,
        robot_status=rs,
        # prev_instruction 제거
    )
    user_text = VIVA_APPROACH_LIFT_USER_TEMPLATE

elif skill == SkillState.APPROACH_AND_PLACE:
    system_prompt = VIVA_APPROACH_PLACE_SYSTEM_PROMPT.format(
        source_object=self.source_object,
        dest_object=self.dest_object,
        robot_status=rs,
        # prev_instruction 제거
    )
    user_text = VIVA_APPROACH_PLACE_USER_TEMPLATE
```

### (g) `check_place_complete` — 연속 N스텝 유지 조건 추가 (문제 12번)

일시적 contact 끊김으로 인한 오탐 방지. 연속 10스텝 이상 gripper open + no contact일 때만 전환.

`__init__()`에 추가:
```python
self._place_complete_count = 0
self._place_complete_threshold = 10  # 연속 10스텝 유지 필요
```

`_transition_to()`에서 S4 진입 시 리셋:
```python
elif next_skill == SkillState.APPROACH_AND_PLACE:
    self._latest_instruction = f"place the {self.source_object} next to the {self.dest_object}"
    self._obstacle_cleared = False
    self._place_complete_count = 0
```

`check_place_complete` 수정:
```python
def check_place_complete(self, grip_pos: float, contact: bool) -> bool:
    """S4 → DONE 전환: gripper open + no contact가 연속 N스텝 유지."""
    if self._current_skill != SkillState.APPROACH_AND_PLACE:
        return False
    if grip_pos > 0.5 and not contact:
        self._place_complete_count += 1
        if self._place_complete_count >= self._place_complete_threshold:
            self._transition_to(SkillState.DONE)
            return True
    else:
        self._place_complete_count = 0  # 조건 불충족 시 카운터 리셋
    return False
```

---

## 3. `run_full_task.py` 수정

### (a) LIFTED_POSE_RANGE, check_lifted_pose 제거 (문제 11번)

`run_full_task.py`에서 다음을 삭제:
- `LIFTED_POSE_RANGE` 딕셔너리 정의 (line 215~222)
- `check_lifted_pose()` 함수 (line 225~234)

`build_robot_status()`의 lifted 판정은 `vlm_orchestrator.py`의 `LIFTED_POSE_RANGE`를 import해서 사용:
```python
from vlm_orchestrator import LIFTED_POSE_RANGE

def build_robot_status(env, contact: bool, depth_min: float | None) -> str:
    jp = env.robot.data.joint_pos[0]
    arm_joints = jp[env.arm_idx[:5]].tolist()
    grip_pos = jp[env.gripper_idx].item()

    # lifted 판정 (LIFTED_POSE_RANGE 사용)
    lifted = False
    if contact:
        joints_with_grip = arm_joints + [grip_pos]
        lifted = all(
            low <= val <= high
            for val, (low, high) in zip(joints_with_grip, LIFTED_POSE_RANGE.values())
        )

    # ... 나머지 동일
```

### (b) vlm_interval 기본값

이미 50으로 변경되어 있음. 유지.

### (c) S2/S4 VLM 호출 조건에 obstacle_cleared 반영 (문제 13번)

```python
# (d) VLM 호출 — 스킬별 분기
if args.mode == "viva":
    if orch.current_skill in (SkillState.NAVIGATE, SkillState.CARRY):
        # S1/S3: 50스텝마다 VLM 호출 (방향 지시)
        if total_steps % args.vlm_interval == 0:
            orch.query_async(base_rgb)
    elif orch.current_skill in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
        # S2/S4: depth warning + 아직 CONTINUE 미판정 시에만 VLM 호출
        if (depth_min is not None
                and depth_min < args.safety_dist
                and not orch.obstacle_cleared):
            orch.query_obstacle_check_async(base_rgb)
else:
    if total_steps % args.vlm_interval == 0:
        orch.query_async(base_rgb)
```

---

## 수정 후 전체 동작 흐름

```
1. 유저 지시어: "약병을 찾아서 빨간 컵 옆에 놓아"

2. VLM /classify → source="medicine bottle", dest="red cup"

3. S1 (navigate):
   VLM: 50스텝마다 호출
     → "navigate forward" / "navigate turn left" / ...
     → target 보이지만 멀면 → target 방향으로 접근 커맨드
     → target 보이고 충분히 가까움 → "TARGET_FOUND"
     → 7개 커맨드 외 응답 → 이전 instruction 유지
   VLA: VLM 커맨드(6개 중 1개)로 action 생성
   Safety: 매 스텝 depth < 0.3m → base vx, vy 정지 (코드)
   전환: VLM이 "TARGET_FOUND" 출력 → S2로 전환

4. S2 (approach & lift):
   VLA instruction 고정: "approach and lift the medicine bottle"
   _obstacle_cleared = False
   VLM: 호출 안 함 (VLA 자율 수행)
   예외: depth_min < threshold + _obstacle_cleared == False:
     → VLM 호출 → OBSTACLE / CONTINUE
     ├─ CONTINUE → _obstacle_cleared = True (이후 재호출 안 함)
     ├─ OBSTACLE + contact 없음 → S1로 복귀 → 회피 → TARGET_FOUND → S2 재진입
     └─ OBSTACLE + contact 있음 → S3로 정상 전환 (물체 잡았으므로 S2 완료 취급)
   전환: 코드에서 check_lifted_pose() == True → S3로 전환

5. S3 (carry):
   VLM: 50스텝마다 호출
     → "carry forward" / "carry turn right" / ...
     → dest 보이지만 멀면 → dest 방향으로 접근 커맨드
     → dest 보이고 충분히 가까움 → "TARGET_FOUND"
     → 7개 커맨드 외 응답 → 이전 instruction 유지
   VLA: VLM 커맨드(6개 중 1개)로 action 생성
   Safety: 매 스텝 depth < 0.3m → base vx, vy 정지 (코드)
   전환: VLM이 "TARGET_FOUND" 출력 → S4로 전환

6. S4 (approach & place):
   VLA instruction 고정: "place the medicine bottle next to the red cup"
   _obstacle_cleared = False
   VLM: 호출 안 함 (VLA 자율 수행)
   예외: depth_min < threshold + _obstacle_cleared == False:
     → VLM 호출 → OBSTACLE / CONTINUE
     ├─ CONTINUE → _obstacle_cleared = True (이후 재호출 안 함)
     └─ OBSTACLE → S3(carry)로 복귀 → 회피 → TARGET_FOUND → S4 재진입
   전환: 코드에서 gripper open + no contact (연속 10스텝) → DONE
```

---

## 검증 포인트

### S1/S3 검증
1. VLM이 50스텝 간격으로 호출되는지 확인
2. VLM 응답이 7개 커맨드 중 하나인지 확인
3. VLM 응답이 예상 외 텍스트일 때 이전 instruction이 유지되는지 확인
4. TARGET_FOUND가 target이 충분히 가까울 때만 출력되는지 확인
5. safety layer가 VLM 호출 빈도와 무관하게 매 스텝 동작하는지 확인

### S2/S4 검증
1. 진입 시 instruction이 고정값인지 확인
2. 진입 시 `_obstacle_cleared = False`인지 확인
3. depth warning 없을 때 VLM이 호출되지 않는지 확인
4. depth warning 시 VLM 호출 → CONTINUE → `_obstacle_cleared = True` → 이후 VLM 재호출 안 되는지 확인
5. depth warning 시 VLM 호출 → OBSTACLE → 이전 스킬로 복귀 → 재진입 시 `_obstacle_cleared`가 다시 False인지 확인
6. LIFTED_COMPLETE: `check_lifted_complete()` 정상 판별 → S3 전환
7. PLACE_COMPLETE: 연속 10스텝 gripper open + no contact → DONE 전환
8. PLACE_COMPLETE 오탐 방지: 일시적 contact 끊김 시 카운터 리셋되는지 확인

### OBSTACLE + contact 분기 검증
1. S2 + OBSTACLE + contact 없음 → S1(navigate)로 복귀 → TARGET_FOUND → S2 재진입
2. S2 + OBSTACLE + contact 있음 → S3(carry)로 정상 전환 → TARGET_FOUND → **S4로 전환** (S2로 재진입 아님)
3. S4 + OBSTACLE → S3(carry)로 복귀 → TARGET_FOUND → S4 재진입

### 코드 정합성 검증
1. `LIFTED_POSE_RANGE`가 `vlm_orchestrator.py`에만 정의되어 있는지 확인
2. `run_full_task.py`에서 `LIFTED_POSE_RANGE`를 import하여 `build_robot_status()`에서 사용하는지 확인
3. S2/S4 프롬프트 format에 `prev_instruction`이 포함되지 않는지 확인
4. 초기 instruction이 `"navigate forward"`인지 확인
