"""
VLM Prompt 설정 — Navigate / Skill 전환 판단
"""

# ═══════════════════════════════════════════════════════════════════════
# Navigate: 방향 command 결정
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
# Skill 전환: 현재 상황 판단
# ═══════════════════════════════════════════════════════════════════════

SKILL_TRANSITION_SYSTEM_PROMPT = """You are a mobile robot task coordinator. You see through a front-facing camera.

Current task: Find {target_object}, pick it up, then find {dest_object} and place the {target_object} next to it.

Based on what you see, output the current skill phase:
- NAVIGATE_SEARCH: target object not visible, keep exploring
- NAVIGATE_TO_TARGET: target object visible but far away, move toward it
- APPROACH_AND_LIFT: target object is close and centered, ready to approach and grasp
- NAVIGATE_TO_DEST: holding object, destination not visible, keep exploring
- NAVIGATE_TO_DEST_CLOSE: holding object, destination visible, move toward it
- CARRY_AND_PLACE: holding object, destination is close, ready to place

Only output the phase name, nothing else."""

SKILL_TRANSITION_USER_TEMPLATE = """Phase:"""

# ═══════════════════════════════════════════════════════════════════════
# Command → obs direction_cmd 매핑
# ═══════════════════════════════════════════════════════════════════════

COMMAND_TO_DIRECTION = {
    "FORWARD":    [0.0, 1.0, 0.0],
    "BACKWARD":   [0.0, -1.0, 0.0],
    "TURN_LEFT":  [0.0, 0.0, 0.33],
    "TURN_RIGHT": [0.0, 0.0, -0.33],
    "STOP":       [0.0, 0.0, 0.0],
}

# Skill phase → 어떤 policy를 실행할지
PHASE_TO_SKILL = {
    "NAVIGATE_SEARCH":      "navigate",
    "NAVIGATE_TO_TARGET":   "navigate",
    "APPROACH_AND_LIFT":    "approach_and_lift",
    "NAVIGATE_TO_DEST":     "navigate",
    "NAVIGATE_TO_DEST_CLOSE": "navigate",
    "CARRY_AND_PLACE":      "carry_and_place",
}
