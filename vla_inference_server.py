#!/usr/bin/env python3
"""
VLA Inference Server — π0-FAST via LeRobot (Phase 4.5 / Phase 5).

base_cam + wrist_cam 이미지와 robot_state 9D, instruction을 받아
action chunk (9D × chunk_size)를 반환한다.

Usage (A100 Server):
    conda activate lerobotpi0
    python vla_inference_server.py \
        --checkpoint ~/datasets/lekiwi_vla/best_model/ \
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
    base_image: str  # base64 JPEG
    wrist_image: str  # base64 JPEG
    state: list[float]  # 9D robot state
    instruction: str = "move forward"


class InferResponse(BaseModel):
    actions: list[list[float]]  # (chunk_size, 9)
    chunk_size: int
    elapsed_ms: float


# ─── Server ───────────────────────────────────────────────────────

app = FastAPI(title="LeKiwi VLA Inference Server")

_policy = None
_device = None


def decode_image(b64: str) -> Image.Image:
    """Decode base64 JPEG to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load π0-FAST policy from LeRobot checkpoint."""
    from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

    log.info(f"Loading PI0-FAST from {checkpoint_path} ...")
    t0 = time.time()

    ckpt = Path(checkpoint_path)
    if ckpt.is_dir():
        policy = PI0FASTPolicy.from_pretrained(str(ckpt))
    else:
        policy = PI0FASTPolicy.from_pretrained(str(ckpt.parent))

    policy = policy.to(device)
    policy.eval()
    log.info(f"Policy loaded in {time.time() - t0:.1f}s, device={device}")
    return policy


@app.on_event("startup")
def startup():
    global _policy, _device
    _device = app.state.device
    _policy = load_policy(app.state.checkpoint, _device)


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": app.state.checkpoint, "device": str(_device)}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    t0 = time.time()

    # Decode images
    base_img = decode_image(req.base_image)
    wrist_img = decode_image(req.wrist_image)

    # Convert to tensors (C, H, W), normalized to [0, 1]
    base_tensor = torch.from_numpy(np.array(base_img)).permute(2, 0, 1).float() / 255.0
    wrist_tensor = torch.from_numpy(np.array(wrist_img)).permute(2, 0, 1).float() / 255.0

    state_tensor = torch.tensor(req.state, dtype=torch.float32)

    # Build observation dict for LeRobot policy
    observation = {
        "observation.images.base_cam": base_tensor.unsqueeze(0).to(_device),
        "observation.images.wrist_cam": wrist_tensor.unsqueeze(0).to(_device),
        "observation.state": state_tensor.unsqueeze(0).to(_device),
    }

    # Add instruction (language-conditioned policy)
    if hasattr(_policy, "language_tokenizer") or hasattr(_policy.config, "use_language"):
        observation["task.language"] = req.instruction

    # Inference
    with torch.no_grad():
        action_chunk = _policy.select_action(observation)

    # action_chunk: (chunk_size, action_dim) or (action_dim,)
    if action_chunk.dim() == 1:
        action_chunk = action_chunk.unsqueeze(0)

    actions = action_chunk.cpu().numpy().tolist()
    elapsed = (time.time() - t0) * 1000

    log.info(f"[{elapsed:.0f}ms] chunk={len(actions)} inst=\"{req.instruction[:40]}\"")

    return InferResponse(
        actions=actions, chunk_size=len(actions), elapsed_ms=round(elapsed, 1),
    )


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeKiwi VLA Inference Server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to π0-FAST checkpoint directory")
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
