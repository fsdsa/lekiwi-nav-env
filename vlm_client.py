#!/usr/bin/env python3
"""
VLM Client — vLLM OpenAI-compatible API로 Navigate direction command 요청.

D455 base_cam 이미지를 640×400으로 다운스케일하여 전송.
vLLM 서버(A100)에서 Qwen3-VL-8B-Instruct로 추론 후 direction command 반환.

Usage:
    from vlm_client import VLMClient
    client = VLMClient(server_url="http://218.148.55.186:30180")
    cmd = client.get_direction(image_rgb)  # "forward" | "backward" | "turn_left" | "turn_right"
"""

from __future__ import annotations

import base64
import io
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image

# D455 base_cam downscale target
TARGET_W, TARGET_H = 640, 400

SYSTEM_PROMPT = """\
You are the navigation controller for a LeKiwi mobile robot.
Look at the robot's front camera image and choose ONE direction command.

Available commands: forward, backward, turn_left, turn_right

Rules:
- If the path ahead is clear, say "forward"
- If there's an obstacle ahead, say "turn_left" or "turn_right" to avoid it
- Only use "backward" if completely stuck
- If you see the target object, move toward it

Respond with ONLY the command name, nothing else."""


class VLMClient:
    def __init__(
        self,
        server_url: str = "http://218.148.55.186:8000",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        timeout: float = 5.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._last_latency = 0.0

    def _encode_image(self, image: np.ndarray) -> str:
        """Downscale to 640×400 and encode as base64 JPEG."""
        pil = Image.fromarray(image)
        if pil.size != (TARGET_W, TARGET_H):
            pil = pil.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def get_direction(
        self,
        image_rgb: np.ndarray,
        task: str = "",
        extra_context: str = "",
    ) -> Optional[str]:
        """
        Send image to vLLM server, return direction command.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image from D455 base_cam
            task: optional task description (e.g., "find the medicine bottle")
            extra_context: optional extra context for the VLM

        Returns:
            One of: "forward", "backward", "turn_left", "turn_right", or None on error
        """
        b64 = self._encode_image(image_rgb)

        user_content = []
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

        text = "What direction should the robot move?"
        if task:
            text = f"Task: {task}. {text}"
        if extra_context:
            text += f" {extra_context}"
        user_content.append({"type": "text", "text": text})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 16,
            "temperature": 0.0,
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._last_latency = time.time() - t0

            result = resp.json()
            text = result["choices"][0]["message"]["content"].strip().lower()

            # Parse direction
            for cmd in ["forward", "backward", "turn_left", "turn_right"]:
                if cmd in text:
                    return cmd

            # Fallback: try partial matches
            if "left" in text:
                return "turn_left"
            if "right" in text:
                return "turn_right"
            if "back" in text:
                return "backward"
            return "forward"  # default

        except Exception as e:
            self._last_latency = time.time() - t0
            print(f"[VLM] Error: {e} ({self._last_latency:.2f}s)")
            return None

    @property
    def latency(self) -> float:
        return self._last_latency

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=3.0)
            return resp.status_code == 200
        except:
            return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://218.148.55.186:8000")
    parser.add_argument("--image", type=str, default=None, help="Test with image file")
    args = parser.parse_args()

    client = VLMClient(server_url=args.server)

    print(f"Health check: {client.health_check()}")

    if args.image:
        img = np.array(Image.open(args.image).convert("RGB"))
        cmd = client.get_direction(img, task="find the target object")
        print(f"Direction: {cmd} (latency: {client.latency:.3f}s)")
    else:
        # Dummy test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cmd = client.get_direction(img)
        print(f"Direction: {cmd} (latency: {client.latency:.3f}s)")
