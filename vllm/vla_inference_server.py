"""
VLA Inference Server — Pi0-FAST via LeRobot 0.4.4
FastAPI server that accepts images + state → returns 9D action chunk.

Usage:
    conda activate lerobotpi0
    python vla_inference_server.py --port 8002
"""

import argparse
import base64
import io
import time
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ── Pi0-FAST import (lerobot 0.4.x) ──
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy, PI0FastPytorch

# Monkey-patch: _prepare_attention_masks_4d needs bool input
_orig_prepare_att = PI0FastPytorch._prepare_attention_masks_4d
def _patched_prepare_att(self, att_2d_masks, dtype=None):
    return _orig_prepare_att(self, att_2d_masks.bool(), dtype=dtype)
PI0FastPytorch._prepare_attention_masks_4d = _patched_prepare_att

app = FastAPI(title="VLA Inference Server (Pi0-FAST)")

# Global model reference
_policy: Optional[PI0FastPolicy] = None
_tokenizer = None  # PaliGemma tokenizer for language
_state_dim: int = 32  # model's expected state dim (set at load time)
_action_dim: int = 32  # model's expected action dim
_device: str = "cuda"


# ── Request / Response schemas ──
class VLARequest(BaseModel):
    """Single inference request."""
    base_image_b64: str          # base camera RGB, JPEG base64
    wrist_image_b64: Optional[str] = None  # wrist camera RGB (optional)
    state: list[float]           # robot state (9D: arm5 + grip1 + base3)
    instruction: str             # language instruction
    action_horizon: int = 10     # how many future steps to predict


class VLAResponse(BaseModel):
    actions: list[list[float]]   # (horizon, 9) action chunk
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_memory_mb: float


# ── Helpers ──
def b64_to_tensor(b64_str: str, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Decode base64 JPEG → (3, H, W) float32 tensor in [0, 1]."""
    from PIL import Image
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0       # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)       # (3, H, W)


# ── Endpoints ──
@app.get("/health")
async def health() -> HealthResponse:
    mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    return HealthResponse(
        status="ok" if _policy is not None else "no_model",
        model="pi0fast-base",
        device=_device,
        gpu_memory_mb=mem,
    )


@app.post("/act")
async def act(req: VLARequest) -> VLAResponse:
    assert _policy is not None, "Model not loaded"

    try:
        return _do_inference(req)
    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


def _do_inference(req: VLARequest) -> VLAResponse:
    t0 = time.perf_counter()

    # Build observation dict — use model's expected image feature keys
    base_img = b64_to_tensor(req.base_image_b64).unsqueeze(0).to(_device)  # (1,3,H,W)

    # Detect expected image keys from model config
    img_features = {
        k: v for k, v in _policy.config.input_features.items()
        if hasattr(v, 'type') and str(v.type) == "FeatureType.VISUAL"
    }

    # Pad state to model's expected dim (zero-pad if shorter)
    state = req.state[:_state_dim]  # truncate if longer
    state = state + [0.0] * (_state_dim - len(state))  # pad if shorter

    obs = {
        "observation.state": torch.tensor([state], dtype=torch.float32, device=_device),
    }

    # Map: base_image → first image key, wrist_image → second (if exists)
    img_keys = sorted(img_features.keys())
    if img_keys:
        obs[img_keys[0]] = base_img
    if req.wrist_image_b64 and len(img_keys) > 1:
        wrist_img = b64_to_tensor(req.wrist_image_b64).unsqueeze(0).to(_device)
        obs[img_keys[1]] = wrist_img

    # Fill missing image keys with zeros (model expects all)
    for k, feat in img_features.items():
        if k not in obs:
            obs[k] = torch.zeros(1, *feat.shape, dtype=torch.float32, device=_device)

    # Tokenize language instruction (no padding — model handles variable length)
    tok_out = _tokenizer(
        req.instruction,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    obs["observation.language.tokens"] = tok_out["input_ids"].to(_device)
    obs["observation.language.attention_mask"] = tok_out["attention_mask"].bool().to(_device)

    with torch.inference_mode():
        action = _policy.select_action(obs)  # (horizon, action_dim) or (action_dim,)

    if action.dim() == 1:
        action = action.unsqueeze(0)

    actions_np = action.cpu().numpy().tolist()

    dt_ms = (time.perf_counter() - t0) * 1000
    return VLAResponse(actions=actions_np, inference_time_ms=dt_ms)


def main():
    global _policy, _tokenizer, _device

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", type=str, default="lerobot/pi0fast-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    _device = args.device

    print(f"Loading {args.model} ...")
    _policy = PI0FastPolicy.from_pretrained(args.model)
    # Disable action token prefix validation (fails on random/garbage input)
    _policy.config.validate_action_token_prefix = False
    _policy = _policy.to(_device)
    _policy.eval()

    # Expose the tokenizer for language preprocessing
    _tokenizer = _policy._paligemma_tokenizer

    # Read model's expected dims
    state_feat = _policy.config.input_features.get("observation.state")
    _state_dim = state_feat.shape[0] if state_feat else 32
    action_feat = _policy.config.output_features.get("action")
    _action_dim = action_feat.shape[0] if action_feat else 32

    print(f"Model loaded on {_device}, state_dim={_state_dim}, action_dim={_action_dim}")
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
