#!/usr/bin/env python3
"""
VLM Inference Server — Qwen3-VL-8B-Instruct (Phase 4.5 / Phase 5).

LeKiwi 3-Skill 파이프라인의 VLM 오케스트레이터.
base_cam 이미지를 보고 현재 상황을 판단하여 VLA에 전달할 instruction과
skill phase를 결정한다.

Usage (A100 Server):
    conda activate inference
    python vlm_inference_server.py --port 8001

API:
    POST /infer
        body: {"image": "<base64 JPEG>", "task": "bring the red cup"}
        response: {
            "instruction": "move forward",
            "phase": "navigate",
            "done": false,
            "reasoning": "..."
        }

    GET /health
        response: {"status": "ok", "model": "Qwen3-VL-8B-Instruct"}
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import time

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── System prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the vision controller for a LeKiwi omnidirectional mobile manipulator robot.
Your job: look at the robot's front camera image and decide what the robot should do next.

The robot executes a pick-and-place task in 3 phases:
  Phase 1 — NAVIGATE: Move toward the target object. The robot can move forward, backward, strafe left/right, or turn left/right.
  Phase 2 — APPROACH_AND_GRASP: Fine-approach the object and grasp it with the gripper.
  Phase 3 — CARRY_AND_PLACE: Carry the grasped object back to the home position and place it down.

Transition rules:
  NAVIGATE → APPROACH_AND_GRASP: when the target object is clearly visible and close (roughly within 0.7m, occupying >15% of the image).
  APPROACH_AND_GRASP → CARRY_AND_PLACE: when the gripper has closed on the object (you will see the object between the gripper fingers).
  CARRY_AND_PLACE → DONE: when the robot has returned to home and placed the object.

You must respond with EXACTLY this JSON format (no markdown, no extra text):
{"instruction": "<short action instruction for VLA>", "phase": "<navigate|approach_and_grasp|carry_and_place|done>", "reasoning": "<one sentence why>"}

Phase-specific instructions you can give:
  navigate: "move forward", "move backward", "turn left", "turn right", "strafe left", "strafe right"
  approach_and_grasp: "approach and pick up the <object>"
  carry_and_place: "carry the object to home and place it down"
  done: "task complete"

Be concise. Respond ONLY with the JSON object.
"""


# ─── Request / Response models ────────────────────────────────────

class InferRequest(BaseModel):
    image: str  # base64 encoded JPEG
    task: str = "bring the red cup"
    prev_phase: str = "navigate"
    system_prompt: str | None = None  # 오케스트레이터가 구성한 커스텀 프롬프트


class InferResponse(BaseModel):
    instruction: str
    phase: str
    done: bool
    reasoning: str
    elapsed_ms: float


class ClassifyRequest(BaseModel):
    user_input: str


class ClassifyResponse(BaseModel):
    mode: str  # "single" | "multi_cleanup"
    target_object: str | None
    elapsed_ms: float


# ─── Server ───────────────────────────────────────────────────────

app = FastAPI(title="LeKiwi VLM Inference Server")


def load_model(model_name: str, device: str = "cuda"):
    """Load Qwen3-VL model and processor."""
    from transformers import AutoProcessor, AutoModelForVision2Seq

    log.info(f"Loading {model_name} ...")
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    return model, processor


_model = None
_processor = None


@app.on_event("startup")
def startup():
    global _model, _processor
    _model, _processor = load_model(app.state.model_name, app.state.device)


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.model_name}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    import json as _json

    t0 = time.time()

    # Decode image
    img_bytes = base64.b64decode(req.image)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Build conversation
    active_prompt = req.system_prompt if req.system_prompt else SYSTEM_PROMPT

    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": (
            f'Task: "{req.task}". Previous phase: {req.prev_phase}.\n'
            f"Look at the camera image and decide the next instruction and phase."
        )},
    ]
    messages = [
        {"role": "system", "content": active_prompt},
        {"role": "user", "content": user_content},
    ]

    # Process and generate
    text_prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=200, do_sample=False)

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = _processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()

    # Parse JSON response
    try:
        parsed = _json.loads(generated)
        instruction = str(parsed.get("instruction", "stop"))
        phase = str(parsed.get("phase", req.prev_phase))
        reasoning = str(parsed.get("reasoning", ""))
    except (_json.JSONDecodeError, Exception):
        log.warning(f"VLM output not valid JSON, raw: {generated[:200]}")
        instruction = "stop"
        phase = req.prev_phase
        reasoning = f"parse error: {generated[:100]}"

    done = phase == "done"
    elapsed = (time.time() - t0) * 1000

    log.info(f"[{elapsed:.0f}ms] phase={phase} inst=\"{instruction}\" reason=\"{reasoning[:60]}\"")

    return InferResponse(
        instruction=instruction, phase=phase, done=done,
        reasoning=reasoning, elapsed_ms=round(elapsed, 1),
    )


CLASSIFY_PROMPT_TEMPLATE = """\
사용자의 로봇 명령을 분석해라.
single: 특정 물체 하나를 가져오거나 집는 명령
multi_cleanup: 여러 물체를 치우거나 정리하는 명령

JSON only로 반환:
{{"mode": "single" | "multi_cleanup", "target_object": "물체명" | null}}

사용자 입력: {user_input}"""


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    import json as _json

    t0 = time.time()

    prompt_text = CLASSIFY_PROMPT_TEMPLATE.format(user_input=req.user_input)
    messages = [
        {"role": "user", "content": prompt_text},
    ]

    text_prompt = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _processor(
        text=[text_prompt], images=None, padding=True, return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=100, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated = _processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True,
    )[0].strip()

    # Parse JSON — fallback to single on failure
    try:
        parsed = _json.loads(generated)
        mode = str(parsed.get("mode", "single"))
        if mode not in ("single", "multi_cleanup"):
            mode = "single"
        target_object = parsed.get("target_object")
        if target_object is not None:
            target_object = str(target_object)
    except (_json.JSONDecodeError, Exception):
        log.warning(f"Classify parse error, raw: {generated[:200]}")
        mode = "single"
        target_object = None

    elapsed = (time.time() - t0) * 1000
    log.info(f"[classify {elapsed:.0f}ms] mode={mode} target={target_object} input=\"{req.user_input[:40]}\"")

    return ClassifyResponse(
        mode=mode, target_object=target_object, elapsed_ms=round(elapsed, 1),
    )


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeKiwi VLM Inference Server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    app.state.model_name = args.model
    app.state.device = args.device

    log.info(f"Starting VLM server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
