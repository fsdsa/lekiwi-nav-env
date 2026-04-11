"""
VLA Inference Server — Pi0-FAST or Pi0.5 via LeRobot 0.5.0
FastAPI server that accepts images + state → returns 9D action chunk.

Auto-detects model type (pi0_fast or pi05) from checkpoint config.json.

Usage:
    conda activate lerobotpi0v2
    python vla_inference_server.py --port 8002 --model <checkpoint_path>
"""

import argparse
import base64
import io
import json
import time
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="VLA Inference Server")

# Global model reference
_policy = None
_preprocessor = None  # for pi05 (uses pipeline)
_postprocessor = None  # for pi05 (uses pipeline)
_tokenizer = None  # for pi0fast (manual tokenization)
_policy_type: str = "unknown"
_state_dim: int = 32
_action_dim: int = 32
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
def b64_to_tensor_resized(b64_str: str, size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Decode base64 JPEG → (3, H, W) float32 tensor in [0, 1], resized to (224,224).
    Used by pi0fast which expects fixed input size."""
    from PIL import Image
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0       # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)       # (3, H, W)


def b64_to_tensor_raw(b64_str: str) -> torch.Tensor:
    """Decode base64 JPEG → (3, H, W) float32 tensor in [0, 1] at original resolution.
    Used by pi05 which has internal resize_with_pad to image_resolution."""
    from PIL import Image
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0       # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)       # (3, H, W)


# ── Endpoints ──
@app.get("/health")
async def health() -> HealthResponse:
    mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    return HealthResponse(
        status="ok" if _policy is not None else "no_model",
        model=_policy_type,
        device=_device,
        gpu_memory_mb=mem,
    )


@app.post("/act")
async def act(req: VLARequest) -> VLAResponse:
    assert _policy is not None, "Model not loaded"

    try:
        if _policy_type == "pi05":
            return _do_inference_pi05(req)
        else:
            return _do_inference_pi0fast(req)
    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": str(e)})


def _do_inference_pi05(req: VLARequest) -> VLAResponse:
    """Pi0.5 inference using saved preprocessor/postprocessor pipeline.

    Pipeline (preprocessor):
      rename → batch → normalize (QUANTILES) → state_tokenize → text_tokenize → device
    Pipeline (postprocessor):
      unnormalize (QUANTILES) → device(cpu)
    """
    t0 = time.perf_counter()

    # Decode at raw resolution; pi05 internal _preprocess_images resizes to (224,224)
    base_img = b64_to_tensor_raw(req.base_image_b64)  # (3, H, W) in [0,1]

    # Build raw batch using ORIGINAL dataset key names
    # The preprocessor's rename step will map: front → base_0_rgb, wrist → left_wrist_0_rgb
    # The right_wrist_0_rgb is missing → pi05's _preprocess_images auto-pads with -1
    batch = {
        "observation.images.front": base_img,
        "observation.state": torch.tensor(req.state, dtype=torch.float32),
        "task": req.instruction,
    }
    if req.wrist_image_b64:
        batch["observation.images.wrist"] = b64_to_tensor_raw(req.wrist_image_b64)

    # Apply preprocessor pipeline (rename, batch, normalize, state_tokenize, text_tokenize, device)
    batch = _preprocessor(batch)

    # Inference: predict_action_chunk returns (B=1, T=chunk_size, action_dim=9)
    with torch.inference_mode():
        action = _policy.predict_action_chunk(batch)

    # Apply postprocessor (unnormalize via QUANTILES, move to cpu)
    action = _postprocessor(action)

    # Strip batch dim
    if action.dim() == 3:
        action = action[0]   # (T, action_dim)
    elif action.dim() == 1:
        action = action.unsqueeze(0)

    actions_np = action.float().cpu().numpy().tolist()
    dt_ms = (time.perf_counter() - t0) * 1000
    return VLAResponse(actions=actions_np, inference_time_ms=dt_ms)


def _do_inference_pi0fast(req: VLARequest) -> VLAResponse:
    """Pi0-FAST inference using manual obs construction (legacy path)."""
    t0 = time.perf_counter()

    # Build observation dict — use model's expected image feature keys
    base_img = b64_to_tensor_resized(req.base_image_b64).unsqueeze(0).to(_device)  # (1,3,H,W)

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
        wrist_img = b64_to_tensor_resized(req.wrist_image_b64).unsqueeze(0).to(_device)
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
        if hasattr(_policy, "predict_action_chunk"):
            action = _policy.predict_action_chunk(obs)  # (B, horizon, action_dim)
            if action.dim() == 3:
                action = action[0]  # (horizon, action_dim)
        else:
            action = _policy.select_action(obs)
            if action.dim() == 1:
                action = action.unsqueeze(0)

    if action.dim() == 1:
        action = action.unsqueeze(0)

    actions_np = action.float().cpu().numpy().tolist()
    dt_ms = (time.perf_counter() - t0) * 1000
    return VLAResponse(actions=actions_np, inference_time_ms=dt_ms)


def _detect_policy_type(model_path: str) -> str:
    """Read config.json to detect 'pi05' vs 'pi0_fast'."""
    config_path = f"{model_path}/config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("type", "unknown")
    except Exception as e:
        print(f"Warning: failed to read config.json at {config_path}: {e}")
        return "unknown"


def main():
    global _policy, _preprocessor, _postprocessor, _tokenizer
    global _policy_type, _state_dim, _action_dim, _device

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", type=str, required=True,
                        help="Path to checkpoint dir (must contain config.json)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    _device = args.device
    _policy_type = _detect_policy_type(args.model)

    print(f"=" * 60)
    print(f"  VLA Inference Server")
    print(f"  Model: {args.model}")
    print(f"  Detected type: {_policy_type}")
    print(f"  Device: {_device}")
    print(f"=" * 60)

    if _policy_type == "pi05":
        # Pi0.5 path: load policy + saved preprocessor/postprocessor pipeline
        from lerobot.policies.pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        print(f"Loading PI05Policy from {args.model} ...")
        _policy = PI05Policy.from_pretrained(args.model)
        _policy = _policy.to(_device).eval()

        print(f"Loading preprocessor/postprocessor pipelines ...")
        _preprocessor, _postprocessor = make_pre_post_processors(
            policy_cfg=_policy.config,
            pretrained_path=args.model,
            preprocessor_overrides={
                "device_processor": {"device": _device}
            },
        )
        print(f"  ✓ Pi0.5 model + processors loaded")

    elif _policy_type == "pi0_fast":
        # Pi0-FAST path: existing manual inference
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

        print(f"Loading PI0FastPolicy from {args.model} ...")
        _policy = PI0FastPolicy.from_pretrained(args.model)
        # Disable action token prefix validation (fails on random/garbage input)
        _policy.config.validate_action_token_prefix = False
        _policy = _policy.to(_device).eval()

        # Expose the tokenizer for language preprocessing
        _tokenizer = _policy._paligemma_tokenizer
        print(f"  ✓ Pi0-FAST model + tokenizer loaded")

    else:
        raise ValueError(
            f"Unsupported policy type: {_policy_type}. "
            f"Supported types: 'pi0_fast', 'pi05'."
        )

    # Read model's expected dims (from config, may be 32 padded for pi05)
    state_feat = _policy.config.input_features.get("observation.state")
    _state_dim = state_feat.shape[0] if state_feat else 32
    action_feat = _policy.config.output_features.get("action")
    _action_dim = action_feat.shape[0] if action_feat else 32

    print(f"Model loaded:")
    print(f"  type={_policy_type}")
    print(f"  state_dim={_state_dim} (config), action_dim={_action_dim}")
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"  Listening on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
