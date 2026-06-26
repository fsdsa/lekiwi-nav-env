#!/usr/bin/env python3
"""
VLA Inference Server — π0.5 via LeRobot 0.5.

base_cam + wrist_cam 이미지와 robot_state 9D, instruction을 받아
action chunk (9D × chunk_size)를 반환한다.

Usage (A100 Server):
    conda activate lerobotpi0v2
    python vla_inference_server.py \
        --checkpoint outputs/train/pi05_lekiwi_v2_3epoch/checkpoints/060000/pretrained_model \
        --port 8002

API:
    POST /infer
        body: {
            "base_image": "<base64 JPEG>",
            "wrist_image": "<base64 JPEG>",
            "state": [9 floats],         # arm_pos(5) + gripper(1) + vx + vy + wz
            "instruction": "move forward"
        }
        response: {
            "actions": [[9 floats], ...],  # action chunk
            "chunk_size": int
        }

    GET /health
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── Request / Response ───────────────────────────────────────────

class InferRequest(BaseModel):
    base_image: str = ""  # base64 JPEG
    wrist_image: str = ""  # base64 JPEG
    base_image_b64: str = ""  # alias (run_full_task.py uses this)
    wrist_image_b64: str = ""  # alias
    state: list[float] = []  # 9D robot state
    instruction: str = "move forward"

    def get_base_image(self) -> str:
        return self.base_image_b64 or self.base_image

    def get_wrist_image(self) -> str:
        return self.wrist_image_b64 or self.wrist_image


class InferResponse(BaseModel):
    actions: list[list[float]]  # (chunk_size, 9)
    chunk_size: int = 0
    elapsed_ms: float = 0.0
    inference_time_ms: float = 0.0  # alias for run_full_task.py


# ─── Server ───────────────────────────────────────────────────────

app = FastAPI(title="LeKiwi VLA Inference Server (Pi0.5)")

_policy = None
_preprocessor = None
_postprocessor = None
_device = None


def decode_image(b64: str) -> Image.Image:
    """Decode base64 JPEG to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load π0.5 policy + preprocessor/postprocessor from LeRobot checkpoint."""
    global _preprocessor, _postprocessor

    import lerobot.policies.pi05.processor_pi05  # register processor steps
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor.pipeline import DataProcessorPipeline

    log.info(f"Loading PI0.5 from {checkpoint_path} ...")
    t0 = time.time()

    ckpt = Path(checkpoint_path)
    ckpt_str = str(ckpt)

    # Load preprocessor & postprocessor
    _preprocessor = DataProcessorPipeline.from_pretrained(
        ckpt_str, config_filename='policy_preprocessor.json')
    _postprocessor = DataProcessorPipeline.from_pretrained(
        ckpt_str, config_filename='policy_postprocessor.json')
    log.info(f"Preprocessor steps: {[type(s).__name__ for s in _preprocessor.steps]}")

    # Load policy
    policy = PI05Policy.from_pretrained(ckpt_str)
    policy = policy.to(device)
    policy.eval()

    log.info(f"Policy loaded in {time.time() - t0:.1f}s, device={device}, "
             f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return policy


@app.on_event("startup")
def startup():
    global _policy, _device
    _device = app.state.device
    _policy = load_policy(app.state.checkpoint, _device)


@app.get("/health")
def health():
    mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "model": "pi05-lekiwi-60k",
        "checkpoint": app.state.checkpoint,
        "device": str(_device),
        "gpu_memory_mb": mem,
    }


def _do_infer(req: InferRequest) -> InferResponse:
    t0 = time.time()

    # Decode images (support both field name variants)
    base_img = decode_image(req.get_base_image())
    wrist_img = decode_image(req.get_wrist_image())

    # Convert to tensors (C, H, W), normalized to [0, 1]
    base_tensor = torch.from_numpy(np.array(base_img)).permute(2, 0, 1).float() / 255.0
    wrist_tensor = torch.from_numpy(np.array(wrist_img)).permute(2, 0, 1).float() / 255.0
    state_tensor = torch.tensor(req.state, dtype=torch.float32)

    # Build sample dict (mimics lerobot dataset output, pre-rename)
    sample = {
        "observation.images.front": base_tensor,
        "observation.images.wrist": wrist_tensor,
        "observation.state": state_tensor,
        "task": req.instruction,
    }

    # Run preprocessor (rename → batch → normalize → tokenize → to device)
    processed = _preprocessor(sample)

    # Inference
    with torch.inference_mode():
        _policy._action_queue.clear()
        action_norm = _policy.select_action(processed)

    # Postprocess (unnormalize → cpu)
    action_post = _postprocessor({"action": action_norm})
    action_chunk = action_post["action"].cpu()

    if action_chunk.dim() == 1:
        action_chunk = action_chunk.unsqueeze(0)
    if action_chunk.dim() == 3:
        action_chunk = action_chunk.squeeze(0)

    actions = action_chunk.numpy().tolist()
    elapsed = (time.time() - t0) * 1000

    log.info(f"[{elapsed:.0f}ms] chunk={len(actions)} inst=\"{req.instruction[:40]}\"")

    return InferResponse(
        actions=actions, chunk_size=len(actions),
        elapsed_ms=round(elapsed, 1), inference_time_ms=round(elapsed, 1),
    )


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    return _do_infer(req)


@app.post("/act", response_model=InferResponse)
def act(req: InferRequest):
    return _do_infer(req)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeKiwi VLA Inference Server (Pi0.5)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to π0.5 checkpoint (pretrained_model dir)")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    app.state.checkpoint = args.checkpoint
    app.state.device = args.device

    log.info(f"Starting VLA server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
