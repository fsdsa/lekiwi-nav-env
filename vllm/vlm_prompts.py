"""
VLM Prompt 설정

구조:
  1. CLASSIFY_SYSTEM_PROMPT: 유저 지시어 → source/dest 추출 (1회)
  2. INSTRUCT_SYSTEM_PROMPT: 이미지 → VLA instruction 생성 (0.3Hz)
  3. NAVIGATE_SYSTEM_PROMPT: Navigate 전용 방향 command (레거시, ResiP용)
"""

# ═══════════════════════════════════════════════════════════════════════
# 1. Classify: 유저 지시어 → source/dest 추출 (text-only, 1회)
# ═══════════════════════════════════════════════════════════════════════

CLASSIFY_SYSTEM_PROMPT = """You are a task parser for a mobile manipulator robot.
Given a user command, extract:
- source_object: the object to pick up
- dest_object: the object/location to place it near (if any)
- mode: "relative_placement" (pick A, place next to B) or "single_pickup" (just pick up A)

Output JSON only, no explanation.
Example: {"mode": "relative_placement", "source_object": "medicine bottle", "dest_object": "red cup"}"""

CLASSIFY_USER_TEMPLATE = """User command: {user_command}"""

# ═══════════════════════════════════════════════════════════════════════
# 2. Instruct: 이미지 → VLA instruction 생성 (0.3Hz, 비동기)
#    VLM이 상황 판단 + 자연어 instruction 직접 생성
# ═══════════════════════════════════════════════════════════════════════

INSTRUCT_SYSTEM_PROMPT = """You are the commander of a mobile manipulator robot.
You see through a front-facing camera (RealSense D455, 87° FOV).

User request: "{user_request}"
Source object: "{source_object}"
Destination object: "{dest_object}"

Look at the camera image and output ONE short English instruction for the robot's next action.

Situation assessment guidelines:
- If {source_object} is NOT visible → give a search/explore instruction (e.g., "turn right slowly to search for the {source_object}")
- If {source_object} is visible but far away → navigate toward it (e.g., "move forward toward the {source_object}")
- If {source_object} is close and reachable → grasp instruction (e.g., "approach and pick up the {source_object}")
- If robot is holding the object and {dest_object} is NOT visible → search for destination (e.g., "turn left to find the {dest_object}")
- If robot is holding the object and {dest_object} is visible → approach and place (e.g., "move toward the {dest_object} and place the {source_object} next to it")
- If task is complete → output exactly "done"

Rules:
1. Output ONLY the instruction, one sentence, in English.
2. Be specific about direction (left, right, forward, backward).
3. Never explain your reasoning.
4. Previous instruction was: "{prev_instruction}" """

INSTRUCT_USER_TEMPLATE = """What should the robot do next?"""

# ═══════════════════════════════════════════════════════════════════════
# 3. Navigate: 방향 command (레거시 — ResiP Navigate 전용)
# ═══════════════════════════════════════════════════════════════════════

NAVIGATE_SYSTEM_PROMPT = """You are a mobile robot navigation controller. You see through a front-facing camera (RealSense D455, 87° FOV).

Your job: output ONE navigation command based on what you see.

Available commands:
- FORWARD: path ahead is clear, move forward
- BACKWARD: need to back up (rare, only if stuck)
- TURN_LEFT: obstacle ahead or need to explore left
- TURN_RIGHT: obstacle ahead or need to explore right
- STOP: target object found in view, stop moving

Rules:
1. If path ahead is clear and no target visible → FORWARD
2. If wall/obstacle/furniture blocks the path → TURN_LEFT or TURN_RIGHT (pick the side with more open space)
3. If you see a doorway or opening to explore → turn toward it
4. If target object is visible in the image → STOP
5. Only output the command word, nothing else."""

NAVIGATE_USER_TEMPLATE = """Target: {target_object}
Command:"""

# ═══════════════════════════════════════════════════════════════════════
# Command → direction 매핑 (Navigate ResiP용)
# ═══════════════════════════════════════════════════════════════════════

COMMAND_TO_DIRECTION = {
    "FORWARD":    [0.0, 1.0, 0.0],
    "BACKWARD":   [0.0, -1.0, 0.0],
    "TURN_LEFT":  [0.0, 0.0, 0.33],
    "TURN_RIGHT": [0.0, 0.0, -0.33],
    "STOP":       [0.0, 0.0, 0.0],
}

# Skill phase → policy 매핑 (레거시)
PHASE_TO_SKILL = {
    "NAVIGATE_SEARCH":      "navigate",
    "NAVIGATE_TO_TARGET":   "navigate",
    "APPROACH_AND_LIFT":    "approach_and_lift",
    "NAVIGATE_TO_DEST":     "navigate",
    "NAVIGATE_TO_DEST_CLOSE": "navigate",
    "CARRY_AND_PLACE":      "carry_and_place",
}
