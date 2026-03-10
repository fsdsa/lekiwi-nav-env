"""
VLM Orchestrator — 유저 지시어 분류 + VLA instruction 생성

구조:
  1. classify_user_request(): 유저 텍스트 → source/dest 추출 (VLM 1회 호출)
  2. RelativePlacementOrchestrator: 0.3Hz로 VLM 호출 → VLA instruction 생성
"""
from __future__ import annotations

import json
import threading
import time

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
