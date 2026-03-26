"""
VLM Orchestrator — 유저 지시어 분류 + VLA instruction 생성

구조:
  1. classify_user_request(): 유저 텍스트 → source/dest 추출 (VLM 1회 호출)
  2. RelativePlacementOrchestrator: 0.3Hz로 VLM 호출 → VLA instruction 생성 (레거시/비교군 ①-B)
  3. VIVAOrchestrator: VIVA 4-skill 상태 머신 + 스킬별 VLM 프롬프트 (S1→S2→S3→S4→DONE)
"""
from __future__ import annotations

import json
import threading
import time
from enum import Enum

import numpy as np
import requests
from PIL import Image
import base64
import io

from vlm_prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_TEMPLATE,
    INSTRUCT_SYSTEM_PROMPT,
    INSTRUCT_USER_TEMPLATE,
    VIVA_NAVIGATE_SYSTEM_PROMPT,
    VIVA_NAVIGATE_USER_TEMPLATE,
    VIVA_CARRY_SYSTEM_PROMPT,
    VIVA_CARRY_USER_TEMPLATE,
    VIVA_APPROACH_LIFT_SYSTEM_PROMPT,
    VIVA_APPROACH_LIFT_USER_TEMPLATE,
    VIVA_APPROACH_PLACE_SYSTEM_PROMPT,
    VIVA_APPROACH_PLACE_USER_TEMPLATE,
)


def classify_user_request(
    vlm_server: str,
    vlm_model: str,
    user_command: str,
    timeout: float = 10.0,
) -> dict:
    """
    유저 지시어 → VLM /classify (text-only, 1회).

    Returns:
        {"mode": "relative_placement", "source_object": "...", "dest_object": "..."}
        or {"mode": "single_pickup", "source_object": "..."}
    """
    payload = {
        "model": vlm_model,
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": CLASSIFY_USER_TEMPLATE.format(
                user_command=user_command
            )},
        ],
        "max_tokens": 100,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            f"{vlm_server}/v1/chat/completions",
            json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # JSON 파싱 시도
        # VLM이 ```json ... ``` 으로 감쌀 수 있음
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        # 필수 필드 검증
        if "source_object" not in result:
            raise ValueError(f"Missing source_object in VLM response: {raw}")
        if "mode" not in result:
            result["mode"] = "relative_placement" if "dest_object" in result else "single_pickup"
        if "dest_object" not in result:
            result["dest_object"] = ""

        return result

    except Exception as e:
        print(f"  [VLM classify] Parse failed: {e}")
        print(f"  [VLM classify] Falling back to simple extraction")
        # Fallback: 간단 추출
        return {
            "mode": "relative_placement",
            "source_object": "medicine bottle",
            "dest_object": "red cup",
        }


class RelativePlacementOrchestrator:
    """
    VLM 기반 오케스트레이터.

    - 0.3Hz로 VLM 호출 (비동기): base cam → 자연어 instruction 생성
    - VLA는 최신 instruction을 계속 사용 (5-10Hz)
    - VLM이 "done" 출력 시 태스크 완료
    """

    def __init__(
        self,
        vlm_server: str,
        vlm_model: str,
        source_object: str,
        dest_object: str,
        user_request: str,
        jpeg_quality: int = 80,
    ):
        self.vlm_server = vlm_server
        self.vlm_model = vlm_model
        self.source_object = source_object
        self.dest_object = dest_object
        self.user_request = user_request
        self.jpeg_quality = jpeg_quality

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # 비동기 상태
        self._lock = threading.Lock()
        self._latest_instruction = f"explore the room to find the {source_object}"
        self._pending = False
        self._done = False
        self._last_latency = 0.0
        self._call_count = 0
        self._total_latency = 0.0

    def encode_image(self, rgb_array: np.ndarray) -> str:
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_instruction(self, rgb_array: np.ndarray) -> str:
        """동기 VLM 호출: 이미지 → VLA instruction."""
        t0 = time.time()
        b64_img = self.encode_image(rgb_array)

        system_prompt = INSTRUCT_SYSTEM_PROMPT.format(
            user_request=self.user_request,
            source_object=self.source_object,
            dest_object=self.dest_object,
            prev_instruction=self._latest_instruction,
        )

        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                    }},
                    {"type": "text", "text": INSTRUCT_USER_TEMPLATE},
                ]},
            ],
            "max_tokens": 50,
            "temperature": 0.0,
        }

        try:
            resp = self._session.post(
                f"{self.vlm_server}/v1/chat/completions",
                json=payload, timeout=8.0,
            )
            resp.raise_for_status()
            instruction = resp.json()["choices"][0]["message"]["content"].strip()

            self._last_latency = time.time() - t0
            self._call_count += 1
            self._total_latency += self._last_latency

            # "done" 체크
            if instruction.lower().strip().strip('"').strip("'") == "done":
                self._done = True
                return "done"

            return instruction

        except Exception as e:
            print(f"  [VLM] Error: {e}")
            self._last_latency = time.time() - t0
            return self._latest_instruction

    def query_async(self, rgb_array: np.ndarray):
        """비동기 VLM 호출 (별도 스레드). VLA는 이전 instruction 유지."""
        if self._pending or self._done:
            return

        self._pending = True

        def _worker():
            try:
                instruction = self.query_instruction(rgb_array)
                with self._lock:
                    if instruction != "done":
                        self._latest_instruction = instruction
            finally:
                self._pending = False

        threading.Thread(target=_worker, daemon=True).start()

    @property
    def instruction(self) -> str:
        with self._lock:
            return self._latest_instruction

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_pending(self) -> bool:
        return self._pending

    @property
    def latency(self) -> float:
        return self._last_latency

    @property
    def avg_latency(self) -> float:
        return self._total_latency / max(self._call_count, 1)

    @property
    def call_count(self) -> int:
        return self._call_count


# ═══════════════════════════════════════════════════════════════════════
#  VIVA 4-Skill Orchestrator
# ═══════════════════════════════════════════════════════════════════════

LIFTED_POSE_RANGE = {
    "arm0": (-0.09, +0.16),
    "arm1": (-0.20, -0.19),
    "arm2": (+0.23, +0.31),
    "arm3": (-1.52, -0.98),
    "arm4": (-0.06, +0.01),
    "grip": (0.13, 0.55),
}


NAVIGATE_COMMANDS = {
    "navigate forward", "navigate backward",
    "navigate strafe left", "navigate strafe right",
    "navigate turn left", "navigate turn right",
    "TARGET_FOUND",
}

CARRY_COMMANDS = {
    "carry forward", "carry backward",
    "carry strafe left", "carry strafe right",
    "carry turn left", "carry turn right",
    "TARGET_FOUND",
}


class SkillState(Enum):
    NAVIGATE = "navigate"
    APPROACH_AND_LIFT = "approach_and_lift"
    CARRY = "carry"
    APPROACH_AND_PLACE = "approach_and_place"
    DONE = "done"


class VIVAOrchestrator:
    """
    VIVA 스킬 상태 머신 + VLM 오케스트레이터.

    정상 흐름:
      navigate → approach_and_lift → carry → approach_and_place → done

    장애물 회피:
      S2 중 OBSTACLE → navigate 전환 → TARGET_FOUND → S2 복귀
      S4 중 OBSTACLE → carry 전환 → TARGET_FOUND → S4 복귀

    VLM에 전달하는 정보:
      - base cam 이미지 (기존)
      - 로봇 상태 텍스트 (joint position, gripper, contact, depth warning)
        → 매 스텝 update_robot_status()로 갱신
    """

    def __init__(
        self,
        vlm_server: str,
        vlm_model: str,
        source_object: str,
        dest_object: str,
        user_request: str,
        jpeg_quality: int = 80,
        navigate_timeout: int = 2000,
        approach_lift_timeout: int = 1000,
        carry_timeout: int = 2000,
        approach_place_timeout: int = 1000,
    ):
        self.vlm_server = vlm_server
        self.vlm_model = vlm_model
        self.source_object = source_object
        self.dest_object = dest_object
        self.user_request = user_request
        self.jpeg_quality = jpeg_quality

        self.timeouts = {
            SkillState.NAVIGATE: navigate_timeout,
            SkillState.APPROACH_AND_LIFT: approach_lift_timeout,
            SkillState.CARRY: carry_timeout,
            SkillState.APPROACH_AND_PLACE: approach_place_timeout,
        }

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # 스킬 상태
        self._current_skill = SkillState.NAVIGATE
        self._interrupted_skill = None  # S2/S4 장애물 회피 복귀용
        self._skill_step_count = 0

        # VLM 비동기 상태
        self._lock = threading.Lock()
        self._latest_instruction = "navigate forward"
        self._pending = False
        self._done = False
        self._timed_out = False

        # 로봇 상태 텍스트 (매 스텝 update_robot_status()로 갱신)
        self._robot_status_text = ""
        self._latest_contact = False  # contact sensor (OBSTACLE 복귀 스킬 분기용)

        # S2/S4 장애물 판별 완료 플래그 (VLM이 CONTINUE 응답 시 True)
        self._obstacle_cleared = False

        # S4 place complete 연속 카운터
        self._place_complete_count = 0
        self._place_complete_threshold = 10

        # 통계
        self._last_latency = 0.0
        self._call_count = 0
        self._total_latency = 0.0

    # ══════════ 프로퍼티 ══════════

    @property
    def current_skill(self) -> SkillState:
        return self._current_skill

    @property
    def instruction(self) -> str:
        with self._lock:
            return self._latest_instruction

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_timed_out(self) -> bool:
        return self._timed_out

    @property
    def is_pending(self) -> bool:
        return self._pending

    @property
    def safety_layer_active(self) -> bool:
        """S1(navigate), S3(carry)에서만 True.
        S2/S4에서는 False — depth_warning을 VLM 텍스트로 전달."""
        return self._current_skill in (SkillState.NAVIGATE, SkillState.CARRY)

    @property
    def obstacle_cleared(self) -> bool:
        """S2/S4에서 VLM이 CONTINUE 응답 시 True. 이후 VLM 재호출 억제."""
        return self._obstacle_cleared

    @property
    def latency(self) -> float:
        return self._last_latency

    @property
    def avg_latency(self) -> float:
        return self._total_latency / max(self._call_count, 1)

    @property
    def call_count(self) -> int:
        return self._call_count

    # ══════════ 로봇 상태 갱신 ══════════

    def update_robot_status(self, robot_status_text: str):
        """run_full_task.py에서 매 스텝 호출. build_robot_status()의 결과를 전달."""
        self._robot_status_text = robot_status_text

    def update_contact(self, contact: bool):
        """run_full_task.py에서 매 스텝 호출. contact sensor 결과 전달."""
        self._latest_contact = contact

    def _get_valid_commands(self) -> set[str] | None:
        """현재 스킬에 해당하는 유효 명령어 집합. S2/S4는 None (자유 텍스트 사용 안 함)."""
        if self._current_skill == SkillState.NAVIGATE:
            return NAVIGATE_COMMANDS
        elif self._current_skill == SkillState.CARRY:
            return CARRY_COMMANDS
        return None

    # ══════════ 스킬 전환 ══════════

    def _transition_to(self, next_skill: SkillState):
        """스킬 전환. 호출 측(run_full_task.py)에서 vla.reset_buffer() 필요."""
        prev = self._current_skill
        self._current_skill = next_skill
        self._skill_step_count = 0
        if next_skill == SkillState.DONE:
            self._done = True

        # S2/S4 진입 시 VLA instruction 고정 + obstacle_cleared 리셋
        if next_skill == SkillState.APPROACH_AND_LIFT:
            self._latest_instruction = f"approach and lift the {self.source_object}"
            self._obstacle_cleared = False
        elif next_skill == SkillState.APPROACH_AND_PLACE:
            self._latest_instruction = f"place the {self.source_object} next to the {self.dest_object}"
            self._obstacle_cleared = False
            self._place_complete_count = 0
        # S3 진입 시 초기 instruction
        elif next_skill == SkillState.CARRY:
            self._latest_instruction = "carry forward"

        print(f"  [SKILL] {prev.value} → {next_skill.value}")

    def _handle_obstacle(self):
        """S2/S4에서 VLM이 "OBSTACLE" 판단 시 호출.
        contact 여부에 따라 복귀 스킬 결정:
          - S2 + contact 없음 → S1(navigate): 빈 손으로 회피
          - S2 + contact 있음 → S3(carry): 물체 들고 회피
          - S4 → S3(carry): 항상 물체 들고 있음"""
        if self._current_skill == SkillState.APPROACH_AND_LIFT:
            if self._latest_contact:
                # 물체를 잡고 있으면 carry로 전환 (S2로 복귀할 필요 없음)
                self._interrupted_skill = None
                self._transition_to(SkillState.CARRY)
            else:
                self._interrupted_skill = SkillState.APPROACH_AND_LIFT
                self._transition_to(SkillState.NAVIGATE)
        elif self._current_skill == SkillState.APPROACH_AND_PLACE:
            self._interrupted_skill = SkillState.APPROACH_AND_PLACE
            self._transition_to(SkillState.CARRY)

    def _handle_target_found(self):
        """navigate/carry에서 VLM이 "TARGET_FOUND" 판단 시 호출.
        장애물 회피 중이었으면 원래 스킬로 복귀, 아니면 다음 스킬로 전환."""
        if self._interrupted_skill is not None:
            # 장애물 회피 후 원래 스킬로 복귀
            restored = self._interrupted_skill
            self._interrupted_skill = None
            self._transition_to(restored)
        elif self._current_skill == SkillState.NAVIGATE:
            self._transition_to(SkillState.APPROACH_AND_LIFT)
        elif self._current_skill == SkillState.CARRY:
            self._transition_to(SkillState.APPROACH_AND_PLACE)

    def tick(self):
        """매 스텝 호출. timeout 체크."""
        self._skill_step_count += 1
        timeout = self.timeouts.get(self._current_skill, 9999)
        if self._skill_step_count >= timeout:
            self._timed_out = True
            print(f"  [TIMEOUT] {self._current_skill.value} at {self._skill_step_count} steps")

    # ══════════ VLM 호출 ══════════

    def _build_vlm_payload(self, b64_img: str) -> dict | None:
        """현재 스킬에 맞는 VLM 프롬프트 구성."""
        skill = self._current_skill
        rs = self._robot_status_text

        if skill == SkillState.NAVIGATE:
            target = self.source_object
            system_prompt = VIVA_NAVIGATE_SYSTEM_PROMPT.format(
                target_object=target, robot_status=rs,
                prev_command=self._latest_instruction,
            )
            user_text = VIVA_NAVIGATE_USER_TEMPLATE

        elif skill == SkillState.CARRY:
            system_prompt = VIVA_CARRY_SYSTEM_PROMPT.format(
                source_object=self.source_object,
                dest_object=self.dest_object,
                robot_status=rs,
                prev_command=self._latest_instruction,
            )
            user_text = VIVA_CARRY_USER_TEMPLATE

        elif skill == SkillState.APPROACH_AND_LIFT:
            system_prompt = VIVA_APPROACH_LIFT_SYSTEM_PROMPT.format(
                source_object=self.source_object, robot_status=rs,
            )
            user_text = VIVA_APPROACH_LIFT_USER_TEMPLATE

        elif skill == SkillState.APPROACH_AND_PLACE:
            system_prompt = VIVA_APPROACH_PLACE_SYSTEM_PROMPT.format(
                source_object=self.source_object,
                dest_object=self.dest_object,
                robot_status=rs,
            )
            user_text = VIVA_APPROACH_PLACE_USER_TEMPLATE

        else:
            return None

        return {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                    }},
                    {"type": "text", "text": user_text},
                ]},
            ],
            "max_tokens": 50,
            "temperature": 0.0,
        }

    def _process_vlm_response(self, raw: str) -> str:
        """VLM 응답 파싱 + 스킬 전환 트리거 처리."""
        cleaned = raw.strip().strip('"').strip("'")

        if cleaned == "TARGET_FOUND":
            self._handle_target_found()
            return self._latest_instruction

        if cleaned == "OBSTACLE":
            self._handle_obstacle()
            return self._latest_instruction

        # S2/S4에서는 CONTINUE → obstacle_cleared 설정, instruction 변경 안 함
        if self._current_skill in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
            if cleaned == "CONTINUE":
                self._obstacle_cleared = True
            return self._latest_instruction

        # S1/S3에서는 유효 명령어 검증 후 instruction으로 사용
        valid = self._get_valid_commands()
        if valid is not None and cleaned not in valid:
            print(f"  [VLM] Invalid command '{cleaned}' for {self._current_skill.value}, keeping previous")
            return self._latest_instruction
        return cleaned

    def query_instruction(self, rgb_array: np.ndarray) -> str:
        """동기 VLM 호출."""
        t0 = time.time()
        b64_img = self.encode_image(rgb_array)
        payload = self._build_vlm_payload(b64_img)
        if payload is None:
            return self._latest_instruction

        try:
            resp = self._session.post(
                f"{self.vlm_server}/v1/chat/completions",
                json=payload, timeout=8.0,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            self._last_latency = time.time() - t0
            self._call_count += 1
            self._total_latency += self._last_latency
            return self._process_vlm_response(raw)

        except Exception as e:
            print(f"  [VLM] Error: {e}")
            self._last_latency = time.time() - t0
            return self._latest_instruction

    def query_async(self, rgb_array: np.ndarray):
        """비동기 VLM 호출. VLA 루프를 블로킹하지 않음."""
        if self._pending or self._done:
            return
        self._pending = True

        def _worker():
            try:
                instruction = self.query_instruction(rgb_array)
                with self._lock:
                    if instruction != "done":
                        self._latest_instruction = instruction
            finally:
                self._pending = False

        threading.Thread(target=_worker, daemon=True).start()

    def query_obstacle_check_async(self, rgb_array: np.ndarray):
        """S2/S4 전용: depth warning 발생 시에만 호출.
        장애물인지 목표물인지 VLM에게 판별 요청."""
        if self._pending or self._done:
            return
        if self._current_skill not in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
            return

        self._pending = True

        def _worker():
            try:
                instruction = self.query_instruction(rgb_array)
                # S2/S4에서는 instruction을 변경하지 않음 (OBSTACLE 시 스킬 전환만 발생)
            finally:
                self._pending = False

        threading.Thread(target=_worker, daemon=True).start()

    def check_lifted_complete(self, arm_joints: list, grip_pos: float, contact: bool) -> bool:
        """S2 → S3 전환: joint position + contact로 판별."""
        if self._current_skill != SkillState.APPROACH_AND_LIFT:
            return False
        if not contact:
            return False
        joints_with_grip = arm_joints + [grip_pos]
        for val, (low, high) in zip(joints_with_grip, LIFTED_POSE_RANGE.values()):
            if not (low <= val <= high):
                return False
        self._transition_to(SkillState.CARRY)
        return True

    def check_place_complete(self, grip_pos: float, contact: bool) -> bool:
        """S4 → DONE 전환: gripper open + no contact가 N스텝 연속이면 전환."""
        if self._current_skill != SkillState.APPROACH_AND_PLACE:
            return False
        # gripper가 열려있고 contact가 없으면 카운터 증가
        if grip_pos > 0.5 and not contact:
            self._place_complete_count += 1
            if self._place_complete_count >= self._place_complete_threshold:
                self._transition_to(SkillState.DONE)
                return True
        else:
            self._place_complete_count = 0
        return False

    def encode_image(self, rgb_array: np.ndarray) -> str:
        """numpy RGB → base64 JPEG."""
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
