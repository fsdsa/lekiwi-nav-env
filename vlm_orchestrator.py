"""
VLM Orchestrators for LeKiwi 3-Skill pipeline.

RelativePlacementOrchestrator: "약병 찾아서 빨간 컵 옆에 놓아"
— source 물체를 잡아서 destination 물체 옆에 놓기 (6-phase)
"""
from __future__ import annotations

import json
import logging
import time

import requests

from vlm_prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    RELATIVE_PLACEMENT_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

TASK_TIMEOUT_SEC = 600


def classify_user_request(user_input: str, vlm_url: str) -> dict:
    """
    VLM 서버를 호출하여 사용자 입력에서 source/destination 물체를 추출한다.
    JSON 파싱 실패 시 기본값으로 fallback.

    Args:
        user_input: 사용자 명령 텍스트
        vlm_url: VLM 서버 URL (e.g. "http://218.148.55.186:8001")

    Returns:
        {
            "mode": "relative_placement",
            "source_object": str,
            "destination_object": str,
        }
    """
    try:
        resp = requests.post(
            f"{vlm_url}/classify",
            json={"user_input": user_input, "system_prompt": CLASSIFY_SYSTEM_PROMPT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        source = data.get("source_object")
        destination = data.get("destination_object")
    except Exception as e:
        log.warning(f"VLM classify failed, using defaults: {e}")
        source = "medicine bottle"
        destination = "red cup"

    return {
        "mode": "relative_placement",
        "source_object": source or "medicine bottle",
        "destination_object": destination or "red cup",
    }


# ─── Relative Placement Orchestrator ──────────────────────────

class RelativePlacementOrchestrator:
    """Source → Destination 상대 배치 오케스트레이터 (6-phase).

    Phase 흐름:
        search_source → approach_source → grasp
        → search_destination → approach_destination → place → done
    """

    PHASES = [
        "search_source", "approach_source", "grasp",
        "search_destination", "approach_destination", "place", "done",
    ]

    def __init__(self, source_object: str, destination_object: str, user_request: str):
        self.source_object = source_object
        self.destination_object = destination_object
        self.user_request = user_request
        self.phase = "search_source"
        self.start_time = time.time()

    def get_system_prompt(self) -> str:
        return RELATIVE_PLACEMENT_SYSTEM_PROMPT.format(
            source_object=self.source_object,
            destination_object=self.destination_object,
            current_phase=self.phase,
        )

    def process_vlm_response(self, vlm_result: dict) -> dict:
        """VLM 응답을 파싱하고 상태를 업데이트한다.

        Returns:
            {
                "instruction": str,
                "phase": str,
                "done": bool,
                "reasoning": str,
            }
        """
        # 타임아웃 안전장치
        elapsed = time.time() - self.start_time
        if elapsed > TASK_TIMEOUT_SEC:
            log.info(f"[RelPlace] Timeout ({TASK_TIMEOUT_SEC}s), finishing.")
            return {
                "instruction": "task complete",
                "phase": "done",
                "done": True,
                "reasoning": "timeout",
            }

        instruction = vlm_result.get("instruction", "stop")
        phase = vlm_result.get("phase", self.phase)
        reasoning = vlm_result.get("reasoning", "")
        done = phase == "done"

        if phase != self.phase:
            log.info(f"[RelPlace] Phase: {self.phase} -> {phase}")
            self.phase = phase

        return {
            "instruction": instruction,
            "phase": self.phase,
            "done": done,
            "reasoning": reasoning,
        }
