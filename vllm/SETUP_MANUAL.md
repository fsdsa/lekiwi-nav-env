# VIVA Pi0.5 Setup Manual

> Complete reproduction manual from a fresh environment to the current working state.
> Prerequisites: lerobot 0.5.0 installed, Qwen3-VL-8B-Instruct downloaded, Pi0.5 base model at pi05_base/.

---

## 1. Environment Setup

### 1.1 Conda Environments

Two separate conda environments are required. They share a GPU but use different Python/library versions.

```bash
# VLA training + inference server (Pi0.5 policy)
conda activate lerobotpi0v2
# Python 3.12, lerobot 0.5.0, torch 2.10.0+cu128, transformers 5.3.0

# VLM server (vLLM serving Qwen3-VL)
conda activate vllm
# Python 3.11, vLLM 0.17.0
```

Verification:

```bash
conda activate lerobotpi0v2
python -c "import lerobot; print(lerobot.__version__)"
# Expected: 0.5.0

conda activate vllm
python -c "import vllm; print(vllm.__version__)"
# Expected: 0.17.0
```

### 1.2 Base Model Locations

```
Pi0.5 base:  /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base/  (14GB)
Qwen3-VL:   ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/
```

Verify Pi0.5 base model files exist:

```bash
ls /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base/
# Expected: config.json  model-00001-of-*.safetensors ...  tokenizer.json  etc.
# Total directory size should be ~14GB
du -sh /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base/
```

### 1.3 Dataset Location

```
/home/jovyan/lerobot_data/lekiwi_viva_v2/
  data/chunk-000/file-000.parquet   (209,036 frames)
  meta/stats.json                   (normalization stats -- MUST be modified, see Section 2)
  meta/stats_backup.json            (original backup)
  meta/tasks.parquet                (18 tasks)
  meta/episodes/                    (978 episodes)
  videos/                           (MP4)
```

Dataset composition:

| Task | Episodes | Frames | Frames/ep |
|---|---|---|---|
| approach and lift | 100 | 77,372 | ~774 |
| navigate (6 variants) | 446 | 65,700 | ~150 |
| carry (6 variants) | 432 | 65,964 | ~150 |
| **Total** | **978** | **209,036** | -- |

Approach demos collected with: object 0.6-0.9m away, within +/-45 degrees of robot front.

---

## 2. stats.json Modification (REQUIRED before training)

### 2.1 Problem

The original stats.json has q01/q99 biased toward navigate/carry data (63% of frames).
When the approach_and_lift arm states go through quantile normalization, they explode:

```
Example: arm4 state q01=-0.113, q99=+0.083 (range 0.196)
         approach arm4 actual value +0.534 --> normalized to +5.60 (!!)
```

This makes the model unable to learn approach_and_lift actions properly.

### 2.2 Modification Script

Run this ONCE before training. It expands q01/q99 to cover the full data range with 5% margin.

```python
import json
import shutil
import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = '/home/jovyan/lerobot_data/lekiwi_viva_v2'

# Load current stats and all data
stats = json.load(open(f'{DATA_ROOT}/meta/stats.json'))
data = pq.read_table(f'{DATA_ROOT}/data/chunk-000/file-000.parquet')
states = np.array(data.column('observation.state').to_pylist())
actions = np.array(data.column('action').to_pylist())

# Expand q01/q99 to full min/max + 5% margin
for feat_key, feat_data in [('observation.state', states), ('action', actions)]:
    for i in range(9):
        lo = float(feat_data[:, i].min())
        hi = float(feat_data[:, i].max())
        margin = (hi - lo) * 0.05
        stats[feat_key]['q01'][i] = lo - margin
        stats[feat_key]['q99'][i] = hi + margin

# Backup original, then save modified
shutil.copy(f'{DATA_ROOT}/meta/stats.json', f'{DATA_ROOT}/meta/stats_backup.json')
with open(f'{DATA_ROOT}/meta/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("stats.json updated. Backup at stats_backup.json.")
```

### 2.3 Verification Script

Run this after modifying stats.json to confirm normalization ranges are sane:

```python
import json
import numpy as np
import pyarrow.parquet as pq

DATA_ROOT = '/home/jovyan/lerobot_data/lekiwi_viva_v2'
stats = json.load(open(f'{DATA_ROOT}/meta/stats.json'))
data = pq.read_table(f'{DATA_ROOT}/data/chunk-000/file-000.parquet')
states = np.array(data.column('observation.state').to_pylist())
actions = np.array(data.column('action').to_pylist())

for feat_key, feat_data in [('observation.state', states), ('action', actions)]:
    q01 = np.array(stats[feat_key]['q01'])
    q99 = np.array(stats[feat_key]['q99'])
    normalized = (feat_data - q01) / (q99 - q01 + 1e-8)
    print(f"\n{feat_key}:")
    for i in range(9):
        col = normalized[:, i]
        print(f"  dim{i}: min={col.min():.3f}  max={col.max():.3f}")
        # ALL values should be within roughly [-0.1, 1.1]
        # Before fix: arm4 had max=+5.60
        # After fix:  arm4 should have max ~0.91
```

Expected output after fix -- every dimension min/max within approximately [-0.3, 1.3]:

```
observation.state:
  dim0: min=-0.022  max=0.913   # arm4 fixed from +5.60
  ...
```

IMPORTANT: After modifying stats.json, any existing checkpoints are INCOMPATIBLE. You must retrain from the base model.

---

## 3. VLA Training

### 3.1 Full Training Command

```bash
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm

nohup /home/jovyan/yes/envs/lerobotpi0v2/bin/lerobot-train \
    --dataset.repo_id=local/lekiwi_fetch_v6 \
    --dataset.root=/home/jovyan/lerobot_data/lekiwi_viva_v2 \
    --policy.path=/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base \
    --policy.repo_id=local/pi05_lekiwi_fixed \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.scheduler_decay_steps=200000 \
    --batch_size=2 \
    --steps=10000000 \
    --save_freq=20000 \
    --log_freq=100 \
    --eval_freq=0 \
    --num_workers=4 \
    --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --output_dir=outputs/train/pi05_fixed_stats \
    > /home/jovyan/pi05_fixed_stats.log 2>&1 &
```

### 3.2 Flag-by-Flag Explanation

| Flag | Value | Why |
|---|---|---|
| `dataset.repo_id` | `local/lekiwi_fetch_v6` | Local dataset identifier (not pushed to HF) |
| `dataset.root` | `/home/jovyan/lerobot_data/lekiwi_viva_v2` | Physical path to the parquet/videos dataset |
| `policy.path` | `pi05_base` | 14GB pretrained Pi0.5 base model to fine-tune from |
| `policy.repo_id` | `local/pi05_lekiwi_fixed` | Name for the output policy |
| `policy.compile_model` | `false` | torch.compile causes issues with Pi0.5 architecture |
| `policy.gradient_checkpointing` | `true` | Required: 14GB model + 2 images would exceed 40GB VRAM without this |
| `policy.dtype` | `bfloat16` | Mixed precision; bf16 is numerically stable for Pi0.5 |
| `policy.chunk_size` | `10` | Action chunk length (10 steps predicted at once) |
| `policy.n_action_steps` | `10` | How many of the 10 predicted steps to actually execute |
| `policy.max_state_dim` | `32` | Pi0.5 state token padding dimension |
| `policy.max_action_dim` | `32` | Pi0.5 action token padding dimension |
| `policy.scheduler_decay_steps` | `200000` | Cosine LR decay period |
| `batch_size` | `2` | Max that fits A100 40GB with gradient checkpointing |
| `steps` | `10000000` | Set very high; stop manually when loss converges |
| `save_freq` | `20000` | Checkpoint every 20k steps (~23GB per checkpoint) |
| `log_freq` | `100` | Print loss every 100 steps |
| `eval_freq` | `0` | Disable built-in eval (we use live eval instead) |
| `num_workers` | `4` | Dataloader workers |
| `rename_map` | `front->base_0_rgb, wrist->left_wrist_0_rgb` | Map dataset image keys to Pi0.5's expected camera names |
| `output_dir` | `outputs/train/pi05_fixed_stats` | Where checkpoints land |

### 3.3 Monitoring Training

```bash
# Watch loss in real-time
tail -f /home/jovyan/pi05_fixed_stats.log | grep "loss"

# Expected pattern:
#   step=100   loss=0.3024    (initial -- high)
#   step=1000  loss=0.0821    (rapid drop)
#   step=5000  loss=0.0512    (converging)
#   step=20000 loss=0.0390    (slowly decreasing)

# Check GPU usage
nvidia-smi
# Expected: ~35-38GB VRAM used on the training GPU
```

### 3.4 Training Duration

- 1 epoch = 104,518 steps (~19 hours on A100)
- 3 epochs = 313,554 steps (~57 hours)
- Loss pattern: 0.3 -> 0.05 (rapid drop in first 5K steps, then slow convergence)
- Recommended: train at least 60K steps (checkpoint at `060000/pretrained_model`)

### 3.5 Approach-Only Fine-Tune (Optional, after 3 epochs)

If approach_and_lift performance needs improvement after full training:

```bash
nohup lerobot-train \
    --dataset.repo_id=local/lekiwi_fetch_v6 \
    --dataset.root=/home/jovyan/lerobot_data/lekiwi_viva_v2 \
    --dataset.episodes="[$(seq -s, 0 99)]" \
    --policy.path=outputs/train/pi05_fixed_stats/checkpoints/300000/pretrained_model \
    --policy.optimizer_lr=5e-06 \
    --policy.scheduler_decay_steps=200000 \
    --batch_size=2 --steps=10000000 --save_freq=20000 \
    --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --output_dir=outputs/train/pi05_approach_finetune \
    > pi05_approach_finetune.log 2>&1 &
```

Episodes 0-99 = approach_and_lift only. Lower LR (5e-6) to avoid catastrophic forgetting of navigate/carry skills.

---

## 4. Code Modifications

### 4.1 vla_inference_server.py -- Complete Rewrite

**File:** `/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vla_inference_server.py`

The original file used PI0FASTPolicy. It was entirely replaced with a Pi0.5 server. The full current file is 216 lines. Here are the key sections that differ from a stock Pi0-FAST server.

#### 4.1.1 Imports (lines 30-45)

The critical imports are standard FastAPI + torch + PIL. No lerobot imports at module level -- they are imported inside `load_policy()` to avoid loading CUDA at import time.

#### 4.1.2 InferRequest with Aliases (lines 52-65)

```python
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
```

WHY: `run_full_task.py` sends `base_image_b64` / `wrist_image_b64`, but other test scripts may send `base_image` / `wrist_image`. The `get_*` methods accept either.

#### 4.1.3 InferResponse (lines 67-72)

```python
class InferResponse(BaseModel):
    actions: list[list[float]]  # (chunk_size, 9)
    chunk_size: int = 0
    elapsed_ms: float = 0.0
    inference_time_ms: float = 0.0  # alias for run_full_task.py
```

#### 4.1.4 load_policy() with Preprocessor/Postprocessor (lines 89-117)

```python
def load_policy(checkpoint_path: str, device: str = "cuda"):
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
```

KEY POINTS:
- `import lerobot.policies.pi05.processor_pi05` -- this import has a side effect: it registers the Pi0.5-specific processor steps (quantile normalization, image resize, etc). Without it, `DataProcessorPipeline.from_pretrained()` will fail with "unknown step type" errors.
- `policy_preprocessor.json` / `policy_postprocessor.json` are saved alongside the model checkpoint by lerobot-train. They contain the normalization parameters (q01/q99 from stats.json at training time).
- The preprocessor handles: image key rename (front->base_0_rgb), batching, quantile normalization, tokenization, device transfer.
- The postprocessor handles: un-normalization of actions back to raw space.

#### 4.1.5 Health Endpoint (lines 127-136)

```python
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
```

`run_full_task.py` reads `model` and `gpu_memory_mb` from this response to verify the server is running the expected model.

#### 4.1.6 _do_infer() -- Core Inference (lines 139-184)

```python
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

    # Run preprocessor (rename -> batch -> normalize -> tokenize -> to device)
    processed = _preprocessor(sample)

    # Inference
    with torch.inference_mode():
        _policy._action_queue.clear()
        action_norm = _policy.select_action(processed)

    # Postprocess (unnormalize -> cpu)
    action_post = _postprocessor({"action": action_norm})
    action_chunk = action_post["action"].cpu()

    if action_chunk.dim() == 1:
        action_chunk = action_chunk.unsqueeze(0)
    if action_chunk.dim() == 3:
        action_chunk = action_chunk.squeeze(0)

    actions = action_chunk.numpy().tolist()
    elapsed = (time.time() - t0) * 1000

    return InferResponse(
        actions=actions, chunk_size=len(actions),
        elapsed_ms=round(elapsed, 1), inference_time_ms=round(elapsed, 1),
    )
```

KEY POINTS:
- Image keys in the sample dict use the DATASET names (`observation.images.front`, `observation.images.wrist`), NOT the Pi0.5 names. The preprocessor's rename step handles the mapping.
- `_policy._action_queue.clear()` -- Pi0.5 internally buffers actions. We clear the queue to force a fresh prediction every time (no stale cached actions from previous calls).
- The dim() checks handle edge cases: single-step predictions (1D), or extra batch dimensions (3D).

#### 4.1.7 Dual Endpoints /infer and /act (lines 187-194)

```python
@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    return _do_infer(req)

@app.post("/act", response_model=InferResponse)
def act(req: InferRequest):
    return _do_infer(req)
```

Both call the same `_do_infer()`. `/act` is used by `run_full_task.py` (VLAClient sends to `/act`). `/infer` exists for backward compatibility with test scripts.

---

### 4.2 vlm_prompts.py -- Memory Placeholders and Scene Summary

**File:** `/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/vlm_prompts.py`

The original file (written by the user) contained CLASSIFY, INSTRUCT, NAVIGATE_LEGACY, and the four VIVA skill prompts. The following additions were made.

#### 4.2.1 VIVA_NAVIGATE_SYSTEM_PROMPT Additions

In the NAVIGATE system prompt (starts at line 115), the following were added to the user's original template:

**A. Placeholders in the header block** (after "Current task: find the {target_object}." line):

```
{bypass_hint}
=== EXPLORATION MEMORY ===
Areas the robot has already visited during this search. Use this to avoid revisiting
already-checked areas and to prioritize unexplored directions.
{explored_memory}
==========================
```

`{bypass_hint}` is empty string normally, filled with obstacle bypass text when recovering from OBSTACLE.
`{explored_memory}` is populated by ExplorationMemory.format_explored().

**B. TARGET_FOUND criteria** (the command list, after navigate turn right):

```
- "TARGET_FOUND" -- The {target_object} must look BIG in the image: it should fill at least 1/4 of the image height and be centered. The object should look like you could reach out and grab it immediately. Keep using "navigate forward" until the object looks truly LARGE. Getting too close is better than triggering too early.
```

**C. Priority rules block** (after the command list):

```
PRIORITY: Rules 1-3 (target object visible) ALWAYS override rules 10-12 (exploration memory).
If you can SEE the {target_object} in the image, IGNORE exploration memory and focus on approaching/centering it.
Exploration memory is ONLY for deciding where to search when the target is NOT visible.
```

**D. Decision rules with memory and centering** (rules 1-12):

```
1. If the {target_object} is at least 1/4 of the image height AND centered -> "TARGET_FOUND". Anything smaller = too far, keep approaching.
2. If you see the {target_object} but it is off to one SIDE -> steer toward it (e.g. "navigate strafe right" if on right). Do NOT output TARGET_FOUND until centered. IGNORE exploration memory.
3. If you see the {target_object} but it is SMALL or far away -> "navigate forward" to get closer. A small object means you are still too far -- keep approaching. IGNORE exploration memory.
...
10. Prefer exploring new areas over revisiting the same space
11. If the target is NOT visible AND current area is in EXPLORATION MEMORY -> turn toward unexplored direction.
12. Strongly prefer exploring new areas (not in memory) over revisiting visited ones.
```

Rules 1-3 explicitly say "IGNORE exploration memory" to prevent memory from overriding target-visible decisions.

#### 4.2.2 VIVA_CARRY_SYSTEM_PROMPT Additions

Same pattern as NAVIGATE. In the CARRY prompt (starts at line 179):

**A. Placeholders** (after "Current task: carry the {source_object}..." line):

```
{bypass_hint}
=== EXPLORATION MEMORY ===
Areas the robot has already visited (includes areas checked during the earlier search
for {source_object}). Use this to avoid revisiting already-checked areas and to find
the {dest_object} efficiently. Entries marked with "!" already contain an object of interest.
{explored_memory}
==========================
```

**B. TARGET_FOUND criteria** (same 1/4 height + centered rule).

**C. Priority rules** (same pattern: rules 1-3 override 10-12).

**D. Decision rules 10-11:**

```
10. If target NOT visible AND current area in EXPLORATION MEMORY -> turn toward unexplored direction.
11. If memory shows an area containing "{dest_object}", navigate back toward it.
```

#### 4.2.3 VIVA_SCENE_SUMMARY (New -- lines 301-322)

This is an entirely new prompt pair added at the end of the file:

```python
VIVA_SCENE_SUMMARY_SYSTEM_PROMPT = """You are a scene descriptor for a mobile robot.

Look at the camera image and describe the current area in ONE short line.
ONLY describe what you can clearly see. Do NOT guess or assume objects exist.

FORMAT (strict):
<area/room type>. <up to 3 visible furniture/objects>

EXAMPLES:
kitchen. sink, fridge, microwave
bedroom. bed, nightstand, lamp
hallway. closed door, brick wall
living room. sofa, tv, coffee table
corridor. doors, ceiling lights

Output ONLY the single-line description. No explanation, no guessing."""

VIVA_SCENE_SUMMARY_USER_TEMPLATE = """Describe this scene in one line:"""
```

CRITICAL DESIGN DECISION: The summary prompt does NOT mention the target object name. If it did, the VLM would hallucinate seeing the target in scenes where it does not exist, poisoning the memory with false positives.

#### 4.2.4 Import Update Required in vlm_orchestrator.py

The orchestrator imports the new prompt. Search for the existing import block and confirm it includes:

```python
from vlm_prompts import (
    ...
    VIVA_SCENE_SUMMARY_SYSTEM_PROMPT,
    VIVA_SCENE_SUMMARY_USER_TEMPLATE,
)
```

---

### 4.3 vlm_orchestrator.py -- ExplorationMemory and Orchestrator Extensions

**File:** `/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/vlm_orchestrator.py`

This file was originally written by the user with classify_user_request(), RelativePlacementOrchestrator, and a basic VIVAOrchestrator. The following additions were made ON TOP of the user's code.

#### 4.3.1 Import Addition (line 36)

In the import block starting at line 22, add the scene summary prompts:

```python
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
    VIVA_SCENE_SUMMARY_SYSTEM_PROMPT,      # <-- NEW
    VIVA_SCENE_SUMMARY_USER_TEMPLATE,      # <-- NEW
)
```

#### 4.3.2 ExplorationMemory Class (lines 277-392)

This entire class is new. Insert it after the CARRY_COMMANDS set definition (line 266) and before the VIVAOrchestrator class.

```python
class ExplorationMemory:
    """Rolling exploration history for VIVA navigate/carry skills."""
    def __init__(self, vlm_server, vlm_model, source_object, dest_object,
                 jpeg_quality=80, max_entries=12, update_interval=30):
        self.vlm_server = vlm_server
        self.vlm_model = vlm_model
        self.source_object = source_object
        self.dest_object = dest_object
        self.jpeg_quality = jpeg_quality
        self.max_entries = max_entries
        self.update_interval = update_interval
        self.entries: list[str] = []
        self.step = 0
        self.last_write_step = 0
        self._generation = 0
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._pending = False
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.entries.clear()
            self.step = 0
            self.last_write_step = 0
            self._generation += 1

    def should_summarize(self, skill) -> bool:
        if skill not in (SkillState.NAVIGATE, SkillState.CARRY):
            return False
        if self._pending:
            return False
        return (self.step - self.last_write_step) >= self.update_interval

    def encode_image(self, rgb_array):
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate_summary_async(self, rgb_array):
        if self._pending:
            return
        self._pending = True
        dispatch_step = self.step
        dispatch_gen = self._generation
        def _worker():
            try:
                b64_img = self.encode_image(rgb_array)
                payload = {
                    "model": self.vlm_model,
                    "messages": [
                        {"role": "system", "content": VIVA_SCENE_SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                            {"type": "text", "text": VIVA_SCENE_SUMMARY_USER_TEMPLATE},
                        ]},
                    ],
                    "max_tokens": 60, "temperature": 0.0,
                }
                resp = self._session.post(
                    f"{self.vlm_server}/v1/chat/completions",
                    json=payload, timeout=8.0,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                with self._lock:
                    if dispatch_gen != self._generation:
                        return
                    added = self._add(raw)
                    self.last_write_step = dispatch_step
                if added:
                    print(f"  [MEMORY t={dispatch_step}] +{raw}")
            except Exception as e:
                print(f"  [MEMORY] Error: {e}")
            finally:
                self._pending = False
        threading.Thread(target=_worker, daemon=True).start()

    def _add(self, summary):
        summary = summary.strip().strip('"').strip("'")
        first_line = next((ln.strip() for ln in summary.splitlines() if ln.strip()), "")
        if not first_line:
            return False
        area = first_line.split(".")[0].strip().lower()
        if not area:
            return False
        # Area-level deduplication: same area type -> update in place
        for i, entry in enumerate(self.entries):
            if entry.split(".")[0].strip().lower() == area:
                if entry == first_line:
                    return False  # exact duplicate
                self.entries[i] = first_line  # update with new details
                return True
        self.entries.append(first_line)
        # Eviction: drop oldest non-"!" entry if over max
        while len(self.entries) > self.max_entries:
            dropped = False
            for i, entry in enumerate(self.entries[:-1]):
                if "!" not in entry:
                    self.entries.pop(i)
                    dropped = True
                    break
            if not dropped:
                self.entries.pop(0)
        return True

    def format_explored(self):
        with self._lock:
            if not self.entries:
                return "(no areas explored yet)"
            return "\n".join(f"- {e}" for e in self.entries)

    def snapshot_entries(self):
        with self._lock:
            return list(self.entries)

    @property
    def is_pending(self):
        return self._pending
```

KEY DESIGN DECISIONS:
- **Area dedup**: Entries with the same area prefix (e.g., "kitchen") are merged (newer replaces older). This prevents memory from filling up with 12 "kitchen" entries when the robot circles a room.
- **"!" preservation**: Entries containing "!" (manually marked as containing an object of interest) are preserved during eviction. Only non-"!" entries get dropped when capacity is exceeded.
- **Generation counter**: When reset() is called (new trial), the generation increments. Any in-flight async summary response from the old trial checks `dispatch_gen != self._generation` and discards itself. This prevents stale data from leaking across trials.
- **Independent pending flag**: Memory has its own `_pending` separate from VIVAOrchestrator's `_pending`. This allows memory summaries and navigation VLM queries to run in parallel.

#### 4.3.3 VIVAOrchestrator.__init__ Additions (lines 411-442)

Two new parameters were added to `__init__`:

```python
def __init__(
    self,
    ...
    memory_max_entries: int = 12,       # <-- NEW
    memory_update_interval: int = 30,   # <-- NEW
):
```

And the memory instance is created in the constructor body:

```python
    # After self.jpeg_quality = jpeg_quality:

    # NEW: Exploration memory
    self._memory = ExplorationMemory(
        vlm_server=vlm_server, vlm_model=vlm_model,
        source_object=source_object, dest_object=dest_object,
        jpeg_quality=jpeg_quality, max_entries=memory_max_entries,
        update_interval=memory_update_interval,
    )
```

#### 4.3.4 Memory Properties (lines 550-556)

Add these two properties to VIVAOrchestrator (after the existing property block):

```python
    @property
    def memory_entries(self) -> list[str]:
        return self._memory.snapshot_entries()

    @property
    def memory_step(self) -> int:
        return self._memory.step
```

These are read by run_full_task.py for memory logging.

#### 4.3.5 reset_for_new_trial() Addition (line 694)

In the existing `reset_for_new_trial()` method, add this line at the end:

```python
    def reset_for_new_trial(self):
        # ... existing reset code ...
        self._generation += 1  # stale async response discard
        self._memory.reset()  # <-- NEW: clear exploration memory
```

#### 4.3.6 tick() Addition (lines 697-701)

In the existing `tick()` method, add the memory step increment:

```python
    def tick(self):
        self._skill_step_count += 1
        # NEW: increment memory step counter during navigate/carry
        if self._current_skill in (SkillState.NAVIGATE, SkillState.CARRY):
            self._memory.step += 1
        timeout = self.timeouts.get(self._current_skill, 9999)
        if self._skill_step_count >= timeout:
            self._timed_out = True
```

Memory step only increments during navigate/carry (not during approach/place), because memory is only relevant for exploration.

#### 4.3.7 _build_vlm_payload() -- bypass_hint and explored_memory (lines 709-744)

The method was extended to:

1. Build `bypass_hint` string when `_interrupted_skill` is set:

```python
    def _build_vlm_payload(self, b64_img: str) -> dict | None:
        skill = self._current_skill
        rs = self._robot_status_text

        # NEW: bypass hint for obstacle recovery
        bypass_hint = ""
        if self._interrupted_skill is not None:
            bypass_hint = (
                "\n!!! OBSTACLE BYPASS MODE !!!\n"
                "The previous approach/place attempt failed due to an obstacle. "
                "DO NOT proceed directly forward toward the same direction. "
                "Turn away (left or right) and find an alternative path.\n"
            )
```

2. Pass `explored_memory` and `bypass_hint` to format() for NAVIGATE and CARRY:

```python
        if skill == SkillState.NAVIGATE:
            target = self.source_object
            system_prompt = VIVA_NAVIGATE_SYSTEM_PROMPT.format(
                target_object=target, robot_status=rs,
                prev_command=self._latest_instruction,
                explored_memory=self._memory.format_explored(),   # <-- NEW
                bypass_hint=bypass_hint,                           # <-- NEW
            )
```

```python
        elif skill == SkillState.CARRY:
            system_prompt = VIVA_CARRY_SYSTEM_PROMPT.format(
                source_object=self.source_object,
                dest_object=self.dest_object,
                robot_status=rs,
                prev_command=self._latest_instruction,
                explored_memory=self._memory.format_explored(),   # <-- NEW
                bypass_hint=bypass_hint,                           # <-- NEW
            )
```

#### 4.3.8 query_async() -- Memory Summarize Trigger (lines 867-891)

At the top of `query_async()`, before the main VLM query, add the memory trigger:

```python
    def query_async(self, rgb_array: np.ndarray):
        # NEW: Memory update (parallel, independent pending flag)
        if self._memory.should_summarize(self._current_skill):
            self._memory.generate_summary_async(rgb_array)

        if self._pending or self._done:
            return
        # ... rest of existing code ...
```

This fires a memory summary VLM call in parallel with the navigation VLM call. They use separate `_pending` flags so they do not block each other.

#### 4.3.9 _process_vlm_response() -- Bracket Fallback Parser (lines 820-834)

In the existing `_process_vlm_response()` method, after the initial bracket parse check, add a fallback for malformed bracket output:

```python
        # S1/S3: validate command
        valid = self._get_valid_commands()
        if valid is not None and cleaned not in valid:
            # NEW: Bracket extraction failed (e.g. "[command] navigate strafe right")
            # Retry from raw lines
            lines = [l.strip().strip('"').strip("'") for l in raw.split('\n') if l.strip()]
            for line in reversed(lines):
                candidate = re.sub(r'\[.*?\]\s*', '', line).strip().strip('"').strip("'")
                if candidate in valid:
                    print(f"  [VLM] Bracket parse failed, recovered from line: '{candidate}'")
                    return candidate
            print(f"  [VLM] Invalid command '{cleaned}' for {self._current_skill.value}, "
                  f"raw='{raw[:80]}', keeping previous")
            return self._latest_instruction
        return cleaned
```

This handles the case where the VLM outputs `[command] navigate strafe right` -- the bracket extractor gets "command" instead of "navigate strafe right". The fallback strips bracket prefixes and checks each line against valid commands.

---

### 4.4 run_full_task.py -- Memory, BRAKE, State Override, Arm Clamp, Frame Save

**File:** `/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/run_full_task.py`

This file was originally written by the user. The following additions were made on top.

#### 4.4.1 New CLI Arguments (lines 82-92)

After the existing `--max_total_steps` argument, add:

```python
# After --max_total_steps:

parser.add_argument("--memory_update_interval", type=int, default=0,
                    help="scene summary generation interval (steps). 0 = use vlm_interval value")
parser.add_argument("--memory_max_entries", type=int, default=12)
parser.add_argument("--memory_log", type=str, default="",
                    help="memory TSV log path for data collection")
```

After the existing `--action_log` argument (line 62), add:

```python
parser.add_argument("--frame_save_dir", type=str, default="",
                    help="save camera frames as JPEG every N steps. Empty = disabled")
parser.add_argument("--frame_save_interval", type=int, default=5,
                    help="frame save interval in steps. Default 5")
```

#### 4.4.2 vlm_interval Default Change (line 82)

Change the default from the original 50 to 30:

```python
parser.add_argument("--vlm_interval", type=int, default=30,     # was 50
                    help="VLM query interval (steps, async with _pending auto-throttle)")
```

30 steps at ~10Hz = VLM query every ~3 seconds. This is more responsive for exploration.

#### 4.4.3 Arm Pose Constants in setup_env() (lines 434-456)

Inside setup_env(), after the render product setup and BEFORE the `cams` dict creation, add:

```python
    # After the _set_rp_enabled function definition and big_toggle_works check:

    # Arm pose presets (normalized [-1,1])
    _arm_lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx[:6]]  # (6, 2)
    _arm_lo = _arm_lim[..., 0]
    _arm_hi = _arm_lim[..., 1]
    _arm_center = 0.5 * (_arm_lo + _arm_hi)
    _arm_half = 0.5 * (_arm_hi - _arm_lo)

    def _raw_to_norm(raw_list):
        raw = torch.tensor(raw_list, dtype=torch.float32, device=env.device)
        return ((raw - _arm_center) / _arm_half.clamp(min=1e-6)).clamp(-1.0, 1.0)

    # Navigate tucked pose -- ACTION space values (dataset navigate action mean)
    # These are NOT raw joint states normalized to [-1,1].
    # They are values in the action space that the VLA learned.
    _NAV_TUCKED_ACTION = [-0.001, -1.000, +1.000, +0.658, -0.537, -0.999]
    _tucked_action = np.array(_NAV_TUCKED_ACTION, dtype=np.float32)

    # Approach & Lift demo first-frame ACTION mean (100 episodes)
    # ACTION space -- same space the VLA outputs
    _APPROACH_INIT_ACTION = [-0.00306, -0.89805, 0.94939, 0.87188, -0.5546, -0.71724]
    _approach_init_action = np.array(_APPROACH_INIT_ACTION, dtype=np.float32)

    print(f"  [Tucked]       action: {[f'{v:+.3f}' for v in _tucked_action.tolist()]}")
    print(f"  [ApproachInit] action: {[f'{v:+.3f}' for v in _approach_init_action.tolist()]}")
```

CRITICAL: These are ACTION SPACE values, not raw joint states. The VLA policy was trained to output values in this range. The environment interprets them the same way. Using raw state values here would cause PhysX joint limit violations.

Then include them in the `cams` dict:

```python
    cams = {
        "base_rgb": base_rgb_annot,
        "depth": depth_annot,
        "wrist_rgb": wrist_rgb_annot,
        "_base_vlm_rp": base_vlm_rp,
        "_wrist_rp": wrist_rgb_rp,
        "_depth_rp": depth_rp,
        "_big_toggle_works": big_toggle_works,
        "_set_rp_enabled": _set_rp_enabled,
        "_tucked_action": _tucked_action,          # <-- NEW
        "_approach_init_action": _approach_init_action,  # <-- NEW
    }
```

#### 4.4.4 VIVAOrchestrator Construction with Memory Params (lines 588-606)

In main(), when constructing the VIVAOrchestrator, add the memory parameters:

```python
    if args.mode == "viva":
        memory_interval = args.memory_update_interval if args.memory_update_interval > 0 else args.vlm_interval
        orch = VIVAOrchestrator(
            vlm_server=args.vlm_server,
            vlm_model=args.vlm_model,
            source_object=source,
            dest_object=dest,
            user_request=user_request,
            jpeg_quality=args.jpeg_quality,
            navigate_timeout=args.navigate_timeout,
            approach_lift_timeout=args.approach_lift_timeout,
            carry_timeout=args.carry_timeout,
            approach_place_timeout=args.approach_place_timeout,
            stop_at_carry=args.stop_at_carry,
            s2_max_attempts=args.s2_max_attempts,
            s4_max_attempts=args.s4_max_attempts,
            memory_max_entries=args.memory_max_entries,          # <-- NEW
            memory_update_interval=memory_interval,              # <-- NEW
        )
```

#### 4.4.5 Memory Log File Setup (lines 645-650)

Before the main trial loop, add:

```python
    # Memory log (data collection)
    mem_log_file = None
    if args.mode == "viva" and args.memory_log:
        mem_log_file = open(args.memory_log, "w", buffering=1)
        mem_log_file.write("trial\tstep\tskill\tinstruction\tmem_count\tmem_entries\n")
        print(f"  Memory log: {args.memory_log}")
```

#### 4.4.6 Frame Save Setup (lines 652-656)

```python
    # Frame save
    _frame_save_dir = None
    if args.frame_save_dir:
        _frame_save_dir = os.path.expanduser(args.frame_save_dir)
        print(f"  Frame save: {_frame_save_dir} (every {args.frame_save_interval} steps)")
```

#### 4.4.7 BRAKE Code -- Skill Transition (lines 956-994)

This is the most critical addition. Inside the main loop, after detecting a skill change (`prev_skill != orch.current_skill`), add:

```python
                if args.mode == "viva" and prev_skill != orch.current_skill:
                    vla.reset_buffer()

                    # NEW: Navigate/Carry -> Approach/Place transition: BRAKE
                    if prev_skill in (SkillState.NAVIGATE, SkillState.CARRY) and \
                       orch.current_skill in (SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE):
                        # BRAKE + ARM TRANSITION
                        if orch.current_skill == SkillState.APPROACH_AND_LIFT:
                            _target = np.array(cams["_approach_init_action"], dtype=np.float32)
                            _label = "approach init"
                        else:
                            _target = np.array(cams["_tucked_action"], dtype=np.float32)
                            _label = "tucked"
                        _BRAKE_STEPS = 30
                        _cur_arm = np.array(cams["_tucked_action"], dtype=np.float32)
                        print(f"  [BRAKE] base stop + arm->{_label} ({_BRAKE_STEPS} steps)")
                        print(f"    from: {[f'{v:+.3f}' for v in _cur_arm.tolist()]}")
                        print(f"    to:   {[f'{v:+.3f}' for v in _target.tolist()]}")
                        for _br in range(_BRAKE_STEPS):
                            alpha = (_br + 1) / _BRAKE_STEPS
                            _act = np.zeros(9, dtype=np.float32)
                            _act[:6] = np.clip(
                                _cur_arm * (1.0 - alpha) + _target * alpha,
                                -0.95, 0.95
                            )
                            # base velocity = 0 (indices 6,7,8)
                            action_t = torch.tensor(_act, dtype=torch.float32,
                                                    device=device).unsqueeze(0)
                            obs, rew, term, trunc, info = env.step(action_t)
                            env.sim.render()
                            total_steps += 1
                        print(f"  [BRAKE] complete (arm at {_label})")
                        prev_skill = orch.current_skill
                        continue  # restart loop with fresh image + instruction

                    prev_skill = orch.current_skill
```

HOW IT WORKS:
1. When switching from navigate/carry to approach/place, the robot's arm is in "tucked" position (arms folded).
2. The approach VLA expects the arm to start in "approach init" position (arms extended forward).
3. BRAKE linearly interpolates the arm from tucked to approach-init over 30 steps.
4. During BRAKE, base velocity is zero (indices 6-8 of the 9D action are 0).
5. All arm values are clamped to +/-0.95 to stay within PhysX joint limits.
6. After BRAKE completes, `continue` restarts the main loop so the VLA gets a fresh camera image with the arm already in the correct position.

#### 4.4.8 State Override for S2/S4 (lines 907-912)

After getting the state, before calling VLA:

```python
                # S2/S4: zero out base velocity in VLA input
                # Demo data was collected from stationary starts.
                # Residual velocity from navigate would confuse the VLA.
                if args.mode == "viva" and orch.current_skill in (
                    SkillState.APPROACH_AND_LIFT, SkillState.APPROACH_AND_PLACE
                ):
                    state[6] = 0.0  # vx
                    state[7] = 0.0  # vy
                    state[8] = 0.0  # wz
```

WHY: The approach_and_lift demos were all collected with the robot stationary (vx=vy=wz=0). During live eval, when transitioning from navigate, there is residual velocity. If passed to the VLA, it amplifies the velocity (positive feedback loop) causing the robot to shoot forward uncontrollably.

#### 4.4.9 Arm Action Clamp (line 921)

After getting the VLA action, before safety checks:

```python
                # Clamp arm actions to +/-0.95 (prevent PhysX joint limit violation)
                action[:6] = np.clip(action[:6], -0.95, 0.95)
```

WHY: The VLA occasionally outputs values slightly beyond +/-1.0. When the environment denormalizes these to raw joint angles, values > 1.0 can exceed the PhysX joint limits [-2pi, 2pi], causing simulation instability.

#### 4.4.10 Frame Save in Main Loop (lines 1023-1032)

After env.step(), add:

```python
                # Frame save
                if _frame_save_dir is not None and total_steps % args.frame_save_interval == 0:
                    _trial_dir = os.path.join(_frame_save_dir, f"trial_{trial+1}")
                    os.makedirs(_trial_dir, exist_ok=True)
                    sk = orch.current_skill.value if args.mode == "viva" else "single"
                    _fname = f"step_{total_steps:05d}_{sk}"
                    if base_rgb is not None:
                        Image.fromarray(base_rgb).save(
                            os.path.join(_trial_dir, f"{_fname}_base.jpg"), quality=90)
                    if wrist_rgb is not None:
                        Image.fromarray(wrist_rgb).save(
                            os.path.join(_trial_dir, f"{_fname}_wrist.jpg"), quality=90)
```

Filename pattern: `step_00150_navigate_base.jpg`, `step_00150_navigate_wrist.jpg`.

#### 4.4.11 Memory Log in Main Loop (lines 1034-1042)

After frame save:

```python
                # Memory log
                if mem_log_file is not None:
                    entries = orch.memory_entries
                    entries_str = "|".join(e.replace("|", "/") for e in entries)
                    inst_safe = instruction.replace("\t", " ").replace("\n", " ")
                    mem_log_file.write(
                        f"{trial+1}\t{total_steps}\t{orch.current_skill.value}\t"
                        f"{inst_safe}\t{len(entries)}\t{entries_str}\n"
                    )
```

#### 4.4.12 Memory Log Close (lines 1233-1235)

At the end of main(), after the trial results summary:

```python
    if mem_log_file is not None:
        mem_log_file.close()
        print(f"  [MemoryLog] Saved: {args.memory_log}")
```

---

## 5. Server Startup

### 5.1 VLM Server (Qwen3-VL via vLLM)

```bash
conda activate vllm

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.75 \
    --port 8000
```

Flag explanation:
- `--model Qwen/Qwen3-VL-8B-Instruct` -- HuggingFace model ID (loads from cache)
- `--trust-remote-code` -- Required for Qwen3-VL custom modeling code
- `--dtype bfloat16` -- Halves VRAM usage vs float32
- `--max-model-len 4096` -- Max context length. 4096 is enough for our prompts + image tokens. Higher values waste VRAM.
- `--gpu-memory-utilization 0.75` -- Reserve 75% of GPU for KV cache. Leaves ~10GB for the VLA server.
- `--port 8000` -- OpenAI-compatible API on port 8000.

Expected startup output:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO 04-15 10:00:00 model_runner.py:XXX] Loading model weights took 15.XX GB
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Health check:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
# Expected:
# {
#   "data": [
#     {
#       "id": "Qwen/Qwen3-VL-8B-Instruct",
#       ...
#     }
#   ]
# }
```

VRAM usage: ~22GB.

### 5.2 VLA Server (Pi0.5)

```bash
conda activate lerobotpi0v2

python /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vla_inference_server.py \
    --checkpoint outputs/train/pi05_fixed_stats/checkpoints/060000/pretrained_model \
    --port 8002
```

Flag explanation:
- `--checkpoint` -- Path to the Pi0.5 fine-tuned checkpoint directory. Must contain `config.json`, `model-*.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`.
- `--port 8002` -- FastAPI server port. Must not conflict with VLM (8000).

Expected startup output:

```
2026-04-15 10:05:00 [INFO] Loading PI0.5 from outputs/train/.../pretrained_model ...
2026-04-15 10:05:00 [INFO] Preprocessor steps: ['RenameStep', 'BatchStep', 'QuantileNormStep', ...]
2026-04-15 10:05:15 [INFO] Policy loaded in 15.2s, device=cuda, GPU=9.8GB
2026-04-15 10:05:15 [INFO] Starting VLA server on 0.0.0.0:8002
INFO:     Uvicorn running on http://0.0.0.0:8002
```

Health check:

```bash
curl -s http://localhost:8002/health | python -m json.tool
# Expected:
# {
#   "status": "ok",
#   "model": "pi05-lekiwi-60k",
#   "checkpoint": "outputs/train/.../pretrained_model",
#   "device": "cuda",
#   "gpu_memory_mb": 9800.0
# }
```

VRAM usage: ~10GB.

### 5.3 SSH Tunnel (Local Machine to Server)

If running Isaac Sim locally and servers on a remote A100:

```bash
ssh -f -N \
    -L 8000:localhost:8000 \
    -L 8002:localhost:8002 \
    A100
```

- `-f` -- fork to background
- `-N` -- no remote command (tunnel only)
- `-L 8000:localhost:8000` -- forward local port 8000 to remote port 8000
- `-L 8002:localhost:8002` -- forward local port 8002 to remote port 8002

Verify tunnel:

```bash
curl -s http://localhost:8000/v1/models | head -1
# Should return JSON (not connection refused)
curl -s http://localhost:8002/health | head -1
# Should return JSON
```

### 5.4 GPU Memory Budget

```
VLA (Pi0.5 60k checkpoint):  ~10 GB
VLM (Qwen3-VL 0.75 util):   ~22 GB
Total:                       ~32 GB / 40 GB A100
Free:                        ~8 GB headroom
```

Both servers run on the SAME GPU. The VLM's `--gpu-memory-utilization 0.75` is calibrated to leave enough room for the VLA.

---

## 6. Live Eval Execution

### 6.1 Full Command

```bash
PYTHONUNBUFFERED=1 python vllm/run_full_task.py \
    --mode viva \
    --user_command "find the medicine bottle and place it next to the red cup" \
    --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \
    --dest_object_usd ~/isaac-objects/.../ACE_Coffee_Mug_.../model_clean.usd \
    --scene_idx 1302 --scene_scale 1.0 \
    --num_trials 1 \
    --vlm_interval 30 \
    --memory_update_interval 30 \
    --memory_max_entries 12 \
    --action_log ~/eval_viva_actions.tsv \
    --frame_save_dir ~/eval_frames \
    --frame_save_interval 5 \
    --memory_log ~/eval_memory.tsv \
    2> >(grep -v "omni.physx.tensors.plugin") | tee ~/eval_viva.log
```

- `PYTHONUNBUFFERED=1` -- Force immediate stdout flushing (critical for real-time log monitoring)
- `--mode viva` -- Use the 4-skill VIVA orchestrator (not single_vla)
- `--scene_idx 1302` -- ProcTHOR scene index (0 = no scene, flat ground)
- `2> >(grep -v "omni.physx.tensors.plugin")` -- Suppress noisy PhysX warnings from stderr
- `| tee ~/eval_viva.log` -- Save full output while also displaying live

### 6.2 Expected Output Sequence

```
============================================================
  Full Task: VLM + VLA
  VLM: http://localhost:8000 (Qwen/Qwen3-VL-8B-Instruct)
  VLA: http://localhost:8002
============================================================

  [Health Check]
  VLM: OK (Qwen/Qwen3-VL-8B-Instruct)
  VLA: OK (pi05-lekiwi-60k, 9800MB)

  [Classify] User command: "find the medicine bottle..."
  [Classify] Result: mode=relative_placement, source="medicine bottle", dest="red cup"

  [Tucked]       action: ['-0.001', '-1.000', '+1.000', '+0.658', '-0.537', '-0.999']
  [ApproachInit] action: ['-0.003', '-0.898', '+0.949', '+0.872', '-0.555', '-0.717']

  === Trial 1/1 (easy) ===
  [Spawn] robot=... | source=... | dest=...

    [t=  10  2s 5.0Hz] skill=navigate inst="navigate forward" vlm=450ms(1)
  [MEMORY t=30] +living room. sofa, coffee table, lamp
  [VLM-RAW] call=2 skill=navigate raw="I see a living room..."
    [t=  60 12s 5.0Hz] skill=navigate inst="navigate turn right" vlm=420ms(2)
  ...
  [SKILL] navigate -> approach_and_lift (S2=1/3, S4=0/3)
  [BRAKE] base stop + arm->approach init (30 steps)
  [BRAKE] complete (arm at approach init)
  ...
  [SKILL] approach_and_lift -> carry (S2=1/3, S4=0/3)
  ...

  Trial 1 GT: S1=True S2=True S3=True FULL=True
```

---

## 7. Offline Eval (Dry Test)

For testing the VLA inference pipeline without Isaac Sim:

```bash
# eval_pi05_full.py: Change CKPT path to desired checkpoint, then:
python /tmp/eval_pi05_full.py
```

This script loads dataset samples, runs them through the VLA server's preprocessor/postprocessor pipeline, and compares predicted actions to ground truth. It validates the entire normalization chain end-to-end.

---

## 8. Known Issues and Solutions

### 8.1 State Normalization Explosion (Root Cause)

- **Problem**: Quantile q01/q99 biased by navigate data (63%) -- approach arm4 normalizes to +5.60
- **Solution**: Expand stats.json q01/q99 to full min/max + 5% margin (Section 2)
- **Retraining required**: After modifying stats.json, retrain from base model. Old checkpoints are incompatible.

### 8.2 Base Velocity Amplification

- **Problem**: When transitioning navigate->approach, residual base velocity in VLA state input causes positive feedback. VLA sees vy=0.3, outputs vy=0.5, next step sees vy=0.5, outputs vy=0.8...
- **Solution**: Zero out state[6:9] (vx, vy, wz) for S2/S4 (Section 4.4.8). Demo data always had zero base velocity.

### 8.3 Arm State Mismatch at Skill Transition

- **Problem**: Navigate keeps arm tucked (folded). Approach demos start with arm extended. VLA sees a state it never encountered in training.
- **Solution**: BRAKE transition (Section 4.4.7) interpolates arm from tucked to approach-init over 30 steps before starting S2/S4.

### 8.4 BRAKE Action Space Confusion

- **Problem**: Early attempts used raw joint states as BRAKE targets. Raw states are in radians, but the env.step() expects action-space values (normalized by joint limits). Sending raw states caused PhysX joint limit violations.
- **Solution**: Use action-space values for BRAKE targets (_NAV_TUCKED_ACTION and _APPROACH_INIT_ACTION are dataset action averages). Clamp all arm actions to +/-0.95.

### 8.5 VLM TARGET_FOUND Triggers Too Early

- **Problem**: VLM would trigger TARGET_FOUND when the object was barely visible (far away, small in frame).
- **Solution**: Added strict criteria: "at least 1/4 of image height AND centered". Added "when in doubt, keep approaching". Getting too close is better than triggering too early (approach VLA can handle close proximity).

### 8.6 ExplorationMemory Hallucination

- **Problem**: Scene summary VLM knew the target object name (passed in prompt). It would hallucinate seeing the target in rooms where it did not exist. Memory then contained false "target found here" entries.
- **Solution**: Scene summary prompt (VIVA_SCENE_SUMMARY_SYSTEM_PROMPT) does NOT mention the target name. It only describes room type and furniture. The orchestrator's navigation prompt handles target detection separately.

### 8.7 Memory vs TARGET_FOUND Conflict

- **Problem**: Memory says "already explored this area" but the target is actually visible. Robot turns away instead of approaching.
- **Solution**: PRIORITY rules in both NAVIGATE and CARRY prompts: "Rules 1-3 (target visible) ALWAYS override rules 10-12 (exploration memory)." If the VLM can see the target, it MUST approach regardless of memory.

### 8.8 VLM Response Parsing Failure

- **Problem**: VLM outputs `[command] navigate strafe right`. Bracket extractor gets "command" instead of the actual command.
- **Solution**: Fallback parser in _process_vlm_response() strips bracket prefixes from each line and checks against valid command sets. If a valid command is found in any line, it is used (Section 4.3.9).

---

## 9. Warnings

1. `git pull` will overwrite ALL code modifications -- backup modified files BEFORE pulling.
2. stats.json changes require retraining from the base model. Existing checkpoints are INCOMPATIBLE with different normalization stats.
3. The VLA server loads preprocessor/postprocessor from the checkpoint directory. The checkpoint and stats.json must be from the SAME training run.
4. BRAKE action values are in action space (not raw joint angles, not normalized states). Do not confuse the three coordinate systems.
5. HuggingFace login may be required for google/paligemma-3b-pt-224 (gated model, dependency of Pi0.5 tokenizer). Run `huggingface-cli login` if you see authentication errors during training.
6. The VLM and VLA servers share one GPU. Starting the VLM with `--gpu-memory-utilization` higher than 0.75 will leave insufficient VRAM for the VLA.
7. `--memory_update_interval 0` means "use the same value as vlm_interval". It does NOT mean "disable memory".
8. **Memory feature (ExplorationMemory) is currently DISABLED** via `_MEMORY_ENABLED=False` flag in `vlm_orchestrator.py`. See Section 11 for rationale and future work plan.

---

## 10. Parallel Dataset Conversion (for New Dataset Fine-Tuning)

Use this workflow when collecting a NEW dataset (or adding to an existing one) and need to avoid the ~20-hour single-process conversion.

### !!! CRITICAL — stats.json Fix is MANDATORY !!!

EVERY time a new dataset is converted (regardless of whether single-process or parallel+merge is used), the resulting `meta/stats.json` WILL have the q01/q99 quantile bias problem described in Section 2.1. This is NOT optional, NOT a one-time fix:

- The current dataset `lekiwi_viva_v2` already has the fix applied -- no action needed for current training.
- Any FUTURE dataset (freshly converted OR merged via `aggregate_datasets`) starts with broken stats.
- The Section 2.2 "min/max + 5% margin" script MUST be run before training on any new dataset.
- Skipping this step silently breaks approach_and_lift (arm4 normalizes to +5.60, see Section 2.1).
- This is a routine post-conversion step for every new dataset, not a one-shot historical fix.

See Section 10.3 Step 4 for the exact command to run after merge.

### 10.1 Why the Default Conversion is Slow

Measured baseline on current dataset (`lekiwi_viva_v2`, 209,036 frames @ 640x400, libsvtav1):

- Total: ~20 hours end-to-end (~2.9 fps throughput).
- Not a bug, not broken code. Breakdown:
  - libsvtav1 default preset prioritizes compression over speed.
  - 978 `save_episode()` calls × ffmpeg subprocess spawn overhead.
  - Python main loop + HDF5 read is single-threaded.
  - SVT-AV1 at 640x400 has limited tile-level parallelism.
- Switching codec alone only saves 30-50%. Real speedup requires parallelism.

### 10.2 Why This Workflow is SAFE

Previously I believed merging separate LeRobot datasets was a custom-script risk. **This was wrong.** LeRobot v0.5.0 ships an official merge API:

- Low-level: `lerobot.datasets.aggregate.aggregate_datasets(repo_ids, aggr_repo_id, roots, aggr_root)`
- High-level wrapper: `lerobot.datasets.dataset_tools.merge_datasets(datasets, output_repo_id, output_dir)`

What the official code actually does (verified by reading `aggregate.py` and `video_utils.py`):

| Step | Implementation | Risk |
|---|---|---|
| Video merge | `concatenate_video_files()` — ffmpeg concat demuxer with stream copy (no re-encode) | None |
| Data parquet merge | `update_data_df()` renumbers `episode_index`, `index`, `task_index` | None |
| Tasks merge | `pd.concat(...).index.unique()` — dedup by task string | None |
| Stats merge | `aggregate_stats()` — correct weighted mean/std, min/max union | None |
| Validation | `validate_all_metadata()` — fps/robot_type/features must match (raises if not) | Catches silent bugs |

So the pipeline = (existing converter) + (official LeRobot merge) + (existing stats fix). No custom code.

### 10.3 Full Workflow

#### Step 1: Collect into multiple HDF5 files

Critical prerequisite. The strategy depends on multiple input files. If `collect_demos.py` only produces a single HDF5, either:
- Modify collection to emit one file per 50-100 episodes, OR
- Post-process single HDF5 by splitting into N files before conversion.

Recommended: 8-10 input files (matches server's 32 physical cores well).

#### Step 2: Parallel conversion

```bash
# Example: 8 HDF5 files, 8 parallel processes (each uses 8 threads)
# Server: 64 logical cores, 32 physical. 8 x 8 threads = fits.
ls /home/jovyan/outputs/rl_demos_new/*.hdf5 | \
  xargs -P 8 -I {} bash -c '
    f="$1"
    name=$(basename "$f" .hdf5)
    /home/jovyan/yes/envs/lerobotpi0v2/bin/python \
      /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/convert_hdf5_to_lerobot_v3.py \
      --input "$f" \
      --output_root /home/jovyan/tmp_parts/lekiwi_part_$name \
      --repo_id local/lekiwi_part_$name \
      --vcodec libsvtav1 \
      --overwrite \
      > /home/jovyan/tmp_parts/convert_$name.log 2>&1
  ' _ {}
```

KEY POINTS:
- `--vcodec libsvtav1` — MUST match existing dataset's codec (see `meta/info.json` → `video.codec: "av1"`). Mixing codecs makes `concatenate_video_files` fail.
- Separate `--output_root` per process (cannot share: each writer opens its own parquet/stats).
- Separate `--repo_id` per process (just a label, does not need to be unique in HF — only locally).
- `--overwrite` is safe because each part is in its own tmp dir.
- Redirect each log separately to diagnose any failure.

Expected wall time (8 parallel): **~3 hours** (vs 20h serial).

Monitor: `tail -f /home/jovyan/tmp_parts/convert_*.log` or `nvidia-smi` / `htop` to check saturation.

IMPORTANT: **DO NOT use `/tmp`** for part outputs. `/tmp` is on the container overlay filesystem (`/`, ~112GB free) which can fill up, get cleared on container restart, and has worse I/O than gpfs. Use `/home/jovyan/tmp_parts/` (on gpfs01, 1.1TB+ free, persistent).

#### Step 2.5: Per-Part Validation (MUST DO before merge)

If any parallel conversion crashed mid-way, the part's `finalize()` was NOT called → parquet footer missing → merge will fail or corrupt silently. Validate each part LOADS cleanly before merging:

```python
# validate_parts.py
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PARTS_ROOT = Path("/home/jovyan/tmp_parts")
PART_DIRS = sorted([p for p in PARTS_ROOT.glob("lekiwi_part_*") if p.is_dir()])

all_ok = True
for p in PART_DIRS:
    try:
        ds = LeRobotDataset(repo_id=f"local/{p.name}", root=p)
        # Try reading last frame to force parquet footer read
        _ = ds[ds.meta.total_frames - 1]
        print(f"  OK  {p.name}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
    except Exception as e:
        print(f"  FAIL {p.name}: {type(e).__name__}: {e}")
        all_ok = False

if not all_ok:
    print("\n[ABORT] Some parts failed validation. Re-run those conversions with --overwrite before merging.")
    exit(1)
print("\n[OK] All parts valid. Safe to merge.")
```

If any part fails, re-run ONLY that part's conversion with the same command (with `--overwrite`). Do NOT proceed to merge until all parts validate.

#### Step 3: Merge with official API

```python
# merge_new_parts.py
from pathlib import Path
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PARTS_ROOT = Path("/home/jovyan/tmp_parts")
PART_NAMES = sorted([p.name for p in PARTS_ROOT.glob("lekiwi_part_*") if p.is_dir()])

parts = [
    LeRobotDataset(
        repo_id=f"local/{name}",
        root=PARTS_ROOT / name,
    )
    for name in PART_NAMES
]

print(f"Merging {len(parts)} parts:")
for p in parts:
    print(f"  - {p.root}  episodes={p.meta.total_episodes}  frames={p.meta.total_frames}")

merged = merge_datasets(
    datasets=parts,
    output_repo_id="local/lekiwi_new",
    output_dir="/home/jovyan/lerobot_data/lekiwi_new",
)

print(f"\nMerged dataset:")
print(f"  episodes: {merged.meta.total_episodes}")
print(f"  frames:   {merged.meta.total_frames}")
print(f"  root:     {merged.root}")
```

Expected wall time: **5-10 minutes** (stream-copy video + parquet concat, no re-encode).

What happens internally:
1. `validate_all_metadata()` — if any part has different fps/features/robot_type, raises ValueError BEFORE modifying disk.
2. For each part: `aggregate_videos` → `aggregate_data` → `aggregate_metadata` appends to the merged output.
3. `finalize_aggregation` writes `info.json`, `tasks.parquet`, `stats.json` (via `aggregate_stats`).

#### Step 4: stats.json Fix (STILL REQUIRED)

The merge aggregates stats mathematically correctly, but the RESULT still has the same quantile bias problem as the original pipeline (q01/q99 dominated by navigate/carry frames). To avoid the approach_and_lift normalization explosion (arm4 +5.60), apply the same fix from Section 2.2:

```bash
/home/jovyan/yes/envs/lerobotpi0v2/bin/python -c "
import json, shutil, numpy as np, pyarrow.parquet as pq
DATA_ROOT = '/home/jovyan/lerobot_data/lekiwi_new'
stats = json.load(open(f'{DATA_ROOT}/meta/stats.json'))

# Load merged parquet (may be multiple chunks)
import glob
parquets = sorted(glob.glob(f'{DATA_ROOT}/data/chunk-*/file-*.parquet'))
tables = [pq.read_table(p) for p in parquets]
states  = np.concatenate([np.array(t.column('observation.state').to_pylist()) for t in tables])
actions = np.concatenate([np.array(t.column('action').to_pylist())            for t in tables])

for feat_key, feat_data in [('observation.state', states), ('action', actions)]:
    for i in range(9):
        lo = float(feat_data[:, i].min()); hi = float(feat_data[:, i].max())
        margin = (hi - lo) * 0.05
        stats[feat_key]['q01'][i] = lo - margin
        stats[feat_key]['q99'][i] = hi + margin

shutil.copy(f'{DATA_ROOT}/meta/stats.json', f'{DATA_ROOT}/meta/stats_backup.json')
json.dump(stats, open(f'{DATA_ROOT}/meta/stats.json', 'w'), indent=2)
print('stats.json updated.')
"
```

Then run the Section 2.3 verification script to confirm every normalized dim falls in [-0.3, 1.3].

#### Step 5: Validation (MUST DO before training)

```python
# validate_merged.py
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="local/lekiwi_new", root="/home/jovyan/lerobot_data/lekiwi_new")
print(f"total_episodes: {ds.meta.total_episodes}")
print(f"total_frames:   {ds.meta.total_frames}")
print(f"features:       {list(ds.meta.features.keys())}")

# Sample frames across different episodes (beginning, middle, end)
for idx in [0, ds.meta.total_frames // 2, ds.meta.total_frames - 1]:
    sample = ds[idx]
    print(f"idx={idx}: state={sample['observation.state'][:3].tolist()}  "
          f"action={sample['action'][:3].tolist()}  "
          f"front={sample['observation.images.front'].shape}  "
          f"wrist={sample['observation.images.wrist'].shape}")
```

Pass criteria:
- `total_frames` == sum of parts' total_frames
- `total_episodes` == sum of parts' total_episodes
- Sample decode at arbitrary frame indices succeeds
- Frame tensors have shape (3, 400, 640) or (400, 640, 3) depending on backend

Then a lerobot-train dry run (1 batch) with `--steps=2 --save_freq=1` and check no exceptions before committing to long training.

#### Step 6: Train

Same command as Section 3.1, but point to the new dataset:
```bash
--dataset.repo_id=local/lekiwi_new \
--dataset.root=/home/jovyan/lerobot_data/lekiwi_new \
...
--output_dir=outputs/train/pi05_lekiwi_new \
```

### 10.4 Time Budget (reference)

Assumes ~200K frame new dataset, similar scale to `lekiwi_viva_v2`:

| Phase | Time | Notes |
|---|---|---|
| Collection | (depends) | Outside this workflow |
| Parallel convert (8 parts) | ~3h | Dominated by ffmpeg + Python loop |
| Merge | 5-10 min | Stream-copy dominated |
| stats.json fix + verify | 5 min | Same scripts as Section 2 |
| Validation (sample + dry run) | 15 min | |
| Training (3 epochs) | 50-60h | Same as current |
| **Total (excl. collection)** | **~55-65h** | Fits 3-day deadline |

For comparison, serial conversion alone takes 20 hours.

### 10.5 Rehearsal (Recommended Before Real Run)

Dry-run the split → merge pipeline on the EXISTING `lekiwi_viva_v2` dataset to verify end-to-end correctness with our specific feature set:

1. Split `lekiwi_viva_v2` parquet into two halves (episode 0-488 and 489-977).
2. Convert each half into separate LeRobot datasets using `--resume` from fresh or pass the split HDF5s through the converter.
3. Run `merge_datasets()` on the two parts.
4. Verify: merged dataset should have exactly 978 episodes, 209,036 frames, 18 tasks matching the original.
5. Run `lerobot-train --steps=2` on merged dataset to confirm it loads.

If all pass, the real new-dataset pipeline is low-risk.

NOTE: This rehearsal should be done BEFORE the new data collection finishes so the workflow is already validated when the real conversion needs to start.

### 10.6 Warnings Specific to Parallel Conversion

1. All parts MUST use identical `--vcodec`, `--fps`, and feature set. `validate_all_metadata()` will refuse mismatches — but double-check BEFORE converting to save time.
2. Do NOT share `--output_root` across parallel processes. Each must have its own tmp directory. The LeRobot writer state (parquet writer, episode buffer, stats accumulator) is not thread-safe across processes.
3. After merge, `meta/stats.json` still requires the min/max + 5% margin fix (Section 2.2). The quantile bias problem persists through aggregation because `aggregate_stats` correctly preserves the original distribution.
4. Free disk: 8 parts × ~100GB raw intermediate (parquet + video in tmp) during parallel conversion. Plus ~200GB for the final merged dataset. Total transient: ~1TB peak on tmpfs or /tmp. If `/tmp` is small, use a persistent disk path (e.g. `/home/jovyan/tmp_parts/`).
5. If ANY part fails mid-conversion, that part's output dir is incomplete. Delete and re-run just that part with the same command. Parallel xargs does NOT auto-retry.
6. `concatenate_video_files` uses stream-copy. If one part used a different codec/resolution by mistake, merge will either silently corrupt or fail at ffmpeg level. Always verify codec identity via `ffprobe /path/to/file-000.mp4 | grep codec` on each part before merging.
7. The `lerobot.datasets.dataset_tools.merge_datasets` wrapper signature may differ slightly between LeRobot versions. Verify with `python -c "from lerobot.datasets.dataset_tools import merge_datasets; help(merge_datasets)"` before the real run.

---

## 11. Memory Feature — Current Status & Future Work

### 11.1 Status: DISABLED

ExplorationMemory (scene-summary-based area tracking) is **currently DISABLED** via a single flag in `vlm_orchestrator.py`:

```python
_MEMORY_ENABLED = False  # near top of file
```

All memory-related calls are gated by this flag:
- `ExplorationMemory.should_summarize()` returns `False` → no async VLM summary calls
- `ExplorationMemory.format_explored()` returns `"(memory disabled)"` placeholder
- `VIVAOrchestrator.tick()` skips `_memory.step` increment
- `VIVAOrchestrator.reset_for_new_trial()` skips `_memory.reset()`
- `VIVAOrchestrator.query_async()` skips `_memory.generate_summary_async()`
- `_build_vlm_payload()` does NOT pass `explored_memory` to prompt formatters

Navigate/Carry prompts no longer contain the `{explored_memory}` placeholder or "EXPLORATION MEMORY" section.

### 11.2 Rationale for Disabling

During development, we observed:

**VLM alone is sufficient for scene recognition + skill transitions**:
- TARGET_FOUND detection works reliably with 1/4-height + center-70% criteria.
- OBSTACLE detection for approach/place skills works via depth-triggered VLM check.
- Skill transitions (S1→S2, S2→S3, etc.) are stable without memory.

**Memory implementation had structural issues preventing dramatic improvement**:
- Scene summary VLM often classified distinct rooms with the same generic label ("room"), causing area-dedup logic to overwrite rather than accumulate. Single memory entry at all times.
- Prompt's "turn away from explored area" rule caused left↔right oscillation when the only memory entry matched the current view.
- Memory didn't capture essential navigation state (position change, command history, target visibility) — only scene descriptions.

### 11.3 Future Work: Redesigned Memory Structure

If re-enabling, redesign memory around **navigation state**, not scene snapshots:

```python
MemoryEntry = {
    "step": int,
    "cmd": str,              # "turn_right" etc.
    "target_status": str,    # "NONE" / "SEEN_LEFT" / "SEEN_RIGHT" / "SEEN_CENTER"
    "pos_change": float,     # distance moved since last entry
    "scene": str,            # brief scene description
}
```

Key principles:
- Preserve temporal order (no area-based dedup)
- Include `pos_change` so VLM can detect "rotating in place" vs "actually moving"
- Include command history (last N) so VLM can detect oscillation patterns
- Keep scene snapshot brief for location context

Re-enablement procedure:
1. Implement new MemoryEntry structure in `ExplorationMemory`
2. Update prompt to include command history + position change fields
3. Set `_MEMORY_ENABLED = True`
4. Re-add `{explored_memory}` to prompts
5. Re-enable `_build_vlm_payload` memory injection

### 11.4 Paper / Report Narrative

For the research report:
> "The VLM-driven orchestration achieves stable situation recognition and skill
> transitions across the 4-skill VIVA pipeline. However, the current
> exploration-memory component showed limited gains due to structural
> dedup issues and lack of navigation-state context. We leave a redesigned
> memory module (tracking pose change, command history, and target visibility)
> as future work."

---

## 12. Major Updates — 2026-04-17 (chunk=50, new safety, dataset v3)

This section documents the coordinated set of changes made on 2026-04-17 that redirect training and inference away from the original chunk=10 approach toward the Pi0.5-standard chunk=50 pipeline, combined with a VLM-based safety layer that handles arm self-occlusion.

### 12.1 Chunk Size: 10 → 50 (retraining from pi05_base)

**Problem observed**: With chunk_size=10 (our original fine-tune), grasp trajectories fragmented across 10-50 separate 10-step chunks. The robot extended toward the bottle but failed to execute the "reach → align → close gripper → lift" sequence coherently — often ending in "pushing the ground" near the object without grasping.

**Root cause**: Pi0.5 base model is designed with `chunk_size=50` (default, ≈ 2 seconds of planning at 25 Hz). Our choice of 10 (≈ 0.4 s) truncated the architecture's planning horizon to below the duration of a single grasp phase.

**Fix**: Retrain from `pi05_base` with chunk=50 and n_action_steps=50. NOT compatible with resume from the old 280K checkpoint — different architecture. The old `pi05_fixed_stats` tree was deleted after confirming the new training run started cleanly.

Training command (chunk=50):

```bash
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm
nohup /home/jovyan/yes/envs/lerobotpi0v2/bin/lerobot-train \
    --dataset.repo_id=local/lekiwi_fetch_v7 \
    --dataset.root=/home/jovyan/lerobot_data/lekiwi_viva_v3 \
    --policy.path=/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base \
    --policy.repo_id=local/pi05_lekiwi_chunk50 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.scheduler_decay_steps=200000 \
    --batch_size=2 --steps=10000000 \
    --save_freq=20000 --log_freq=100 --eval_freq=0 --num_workers=4 \
    --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --output_dir=outputs/train/pi05_chunk50 \
    > /home/jovyan/pi05_chunk50.log 2>&1 &
```

### 12.2 Dataset v3: existing `lekiwi_viva_v2` + new approach demos

58 new expert demos for `approach_and_lift` (29,647 frames) were merged with the existing 978-episode dataset, producing `lekiwi_viva_v3`:

| | Episodes | Frames | Approach frames |
|---|---|---|---|
| lekiwi_viva_v2 (existing) | 978 | 209,036 | 77,372 (37%) |
| New approach demos | 58 | 29,647 | 29,647 (100%) |
| **lekiwi_viva_v3 (merged)** | **1,036** | **238,683** | **107,019 (45%)** |

Pipeline executed: HDF5 split into 4 parts → parallel conversion on server (taskset core pinning) → `lerobot.datasets.aggregate.aggregate_datasets` → stats.json fix (min/max + 5% margin). All scripts at `/home/jovyan/tmp_parts/`. See Section 10 for general procedure; this run verified via `merge_and_fix.py` with 6-step validation (feature compare, task-string align, counts match, stats range).

### 12.3 n_action_steps — Skill-adaptive Inference

Even with chunk=50, the client can execute only the first N of the 50 predicted actions before re-querying VLA (Receding Horizon Control). This gives per-skill reactivity tuning:

| Skill | n_action_steps | Reason |
|---|---|---|
| Navigate / Carry | **10** | High reactivity needed for obstacle response and direction changes |
| Approach_and_lift / Approach_and_place | **50** | Long-horizon grasp planning — full chunk used for coherent grasp |

Implementation: `VLAClient.get_action_9d()` takes an optional `n_use` argument. Main loop sets it by skill:

```python
# run_full_task.py (main loop)
n_use = 10 if orch.current_skill in (NAVIGATE, CARRY) else 50
action = vla.get_action_9d(..., n_use=n_use)
```

Client forces buffer re-query when `buffer_idx >= n_use`, even if more actions remain in the 50-action chunk.

### 12.4 Base Velocity Scaling: 0.7 → 0.5 (Navigate / Carry)

VLA's base velocity commands (indices 6-8 of the 9D action) are scaled by **0.5** during navigate/carry skills. Not applied to approach/place.

Rationale:
- Navigate in demo has `vy=+0.5` ≈ 0.5 m/s. With 0.5x scaling → 0.25 m/s commanded ≈ 0.15 m/s actual.
- Gives VLM obstacle check (~1.5s latency) enough margin before collision:
  - 1.5 s × 0.15 m/s ≈ 0.22 m travel, within the 0.3 m safety trigger.
- Also reduces overshoot past the target during navigate.

```python
# run_full_task.py after vla.get_action_9d
if orch.current_skill in (NAVIGATE, CARRY):
    action[6:9] *= 0.5
```

### 12.5 Clamp Removed (±0.95)

The previously-added `action[:6] = np.clip(action[:6], -0.95, 0.95)` was removed. Reason: it cut off the real training action range (~±1.35 especially for gripper), which prevented gripper full-close during grasp. The env's internal `torch.clamp(arm_targets, arm_lo, arm_hi)` handles final safety via baked joint limits (all within ±2π).

### 12.6 BRAKE Shortened: 30 → 15 Steps

The arm-interpolation BRAKE between navigate and approach was cut from 30 to 15 steps to reduce robot drift during the transition. 15 steps (≈ 0.6 s sim) is sufficient for arm pose change and inertial deceleration, while halving the distance drifted past the target.

### 12.7 VLA Server: Return Full Chunk (not single action)

`vla_inference_server.py` previously cleared `_action_queue` each call and returned a single action. Fixed to use `predict_action_chunk()` directly and return all `n_action_steps` actions. Client buffers them, reducing VLA calls by ~50x and raising sim Hz from ~2 Hz to ~13-18 Hz.

### 12.8 VLM-based Safety Layer for Navigate / Carry (major change)

Previously, S1/S3 (navigate/carry) safety blocked forward motion purely on depth < 0.3 m. Two problems:
1. Robot's arm is always in camera view → depth reading often ≤ 0.3 m → safety would block forward constantly.
2. Depth does not distinguish "arm in view with open space behind" from "wall just behind the arm".

New design (VLM obstacle check for S1/S3, mirroring S2/S4):

**Prompts added** (`vlm_prompts.py`):
- `VIVA_NAVIGATE_OBSTACLE_CHECK_SYSTEM_PROMPT`
- `VIVA_CARRY_OBSTACLE_CHECK_SYSTEM_PROMPT`

Both use a TWO-STEP judgment:
1. Is the close foreground a real obstacle (wall/furniture/door) or the robot's own arm?
2. If arm, look BEYOND/AROUND it: is there a wall within ~30 cm behind the arm? (→ OBSTACLE) Or is the space beyond open? (→ CONTINUE)

**Flow** (per-step in S1/S3):
1. depth < 0.3 m detected.
2. If `obstacle_cleared=False`: fire `orch.query_obstacle_check_async(base_rgb)`, default-block forward motion.
3. VLM returns "CONTINUE" → `obstacle_cleared=True` → release forward block.
4. VLM returns "OBSTACLE" → `obstacle_cleared=False` stays, forward block stays. Regular navigate query (different schedule) generates a new command ("backward", "turn").

**Re-validation**: `obstacle_cleared` automatically resets to False every 150 sim steps (≈ 6 s) during S1/S3, forcing VLM to re-check. Also resets on skill transition.

**Backward/strafe/rotation** are never blocked by this safety — only forward vy is gated.

### 12.9 TARGET_FOUND Criteria

Finalized to **1/4 image height + center 70% width**:

```
TARGET_FOUND fires if:
  (A) target occupies ≥ 1/4 of image height (≈ 0.55 m distance, matching demo start)
  (B) target's horizontal position is within center 70% of image width
```

Rationale: demos collected with object 0.6–0.9 m away. Looser criteria (1/8 height, no center requirement) led to TARGET_FOUND triggering from ~1.5 m, outside the approach VLA's training distribution.

Rule 4 (small target off-center) rewritten to explicitly say "strafe TOWARD target's side (LEFT if target is left, RIGHT if target is right)". Earlier example only showed "strafe right", which the VLM sometimes mis-applied.

### 12.10 VLM Interval Stays at 30

`--vlm_interval` default kept at 30 steps. Reasoning:
- With n_action_steps=10 for navigate, VLA regenerates every 10 steps. vlm_interval=30 means 3 VLA chunks per VLM direction update (most are stale, but safety is handled by obstacle check, not direction VLM).
- Obstacle check fires on-demand per-step (when `depth < 0.3 m` and `obstacle_cleared=False`), independent of vlm_interval.
- VLM latency (~1.5 s ≈ 20 step wall) comfortably fits within the 30-step interval.

### 12.11 Summary of Current Eval Config

| Component | Setting |
|---|---|
| `--policy.chunk_size` | 50 (pi0.5 default) |
| `--policy.n_action_steps` (training) | 50 |
| Client `n_use` at inference | 10 (navigate/carry), 50 (approach/place) |
| Base velocity scaling | 0.5 (navigate/carry), 1.0 (approach/place) |
| Action clamp | None (removed) |
| BRAKE steps | 15 |
| `--vlm_interval` | 30 |
| `--safety_dist` | 0.3 m |
| Safety layer (S1/S3) | VLM obstacle check + default forward-block |
| Safety layer (S2/S4) | VLM obstacle check + soft slowdown while pending |
| Memory feature | **DISABLED** (see Section 11) |

### 12.12 Files Touched (2026-04-17)

- `vlm_orchestrator.py`: `_MEMORY_ENABLED` flag, obstacle check extension to S1/S3, `_in_obstacle_check` flag, prompt routing in `_build_vlm_payload`, obstacle-response handling in `_process_vlm_response`, periodic re-check in `tick()`, `obstacle_cleared` reset on S1/S3 entry.
- `vlm_prompts.py`: Removed EXPLORATION MEMORY sections; added `VIVA_NAVIGATE_OBSTACLE_CHECK_*` and `VIVA_CARRY_OBSTACLE_CHECK_*`; TARGET_FOUND criteria tightened to 1/4 + 70%; Rule 4 clarified.
- `run_full_task.py`: Clamp removed, 0.5x base scaling for S1/S3, `n_use` argument in `VLAClient.get_action_9d()`, skill-adaptive `n_use` in main loop, BRAKE 30→15, safety layer rewritten to use `obstacle_cleared` gate, main loop query switched to obstacle-check-priority in S1/S3.
- `vla_inference_server.py`: `predict_action_chunk()` direct return instead of single action.
- Deleted: entire `outputs/train/pi05_fixed_stats/` (old chunk=10 training, 320 GB).

### 12.13 Paper / Report Narrative (Updated)

> "We train Pi0.5 on a merged dataset (lekiwi_viva_v3: 1,036 episodes, 239 K frames,
>  45 % approach-and-lift frames) with the pi0.5-standard chunk size of 50. At inference,
>  a skill-adaptive receding-horizon controller executes the first 10 of 50 predicted
>  actions during navigation/carry phases for obstacle reactivity, and the full 50 during
>  approach/place for coherent grasp planning. A VLM-based safety layer handles arm
>  self-occlusion by reasoning not only about whether a close depth reading is the arm
>  itself but also whether a wall lies immediately behind the arm. Base velocity during
>  navigation is reduced to 50 % of the demonstration scale to give the VLM obstacle
>  check (~1.5 s latency) sufficient response margin."

---

## 13. End-to-End Training on H100 (2026-04-18)

Parallel to the A100 skill-policy training on `lekiwi_viva_v4` (1036 ep, chunk=50, lr=2.5e-5),
we launched an **end-to-end policy** on a rented H100 via elice cloud. This section documents
the new dataset, adaptive noise strategy, Option-B video sharing, remote conversion pipeline,
and LeRobot pi0.5 training pipeline audit findings.

### 13.1 Why a separate end-to-end run?

The `viva_v4` data is short, per-skill teleop (avg ~230 frames/ep) producing a general
controller that the VLM orchestrator sequences. The new 24 demos are **long-horizon
end-to-end** teleops (avg ~4000 frames/ep) covering the full task
"find the medicine bottle and place it next to the red cup". Training a second policy
on these allows direct comparison: VLM-sequenced skills vs. single policy end-to-end.

### 13.2 Source demos (24 teleops, 2 difficulty levels)

| File | difficulty | episodes | avg frames | total frames |
|---|---|---|---|---|
| `scene_teleop_full_easy_aug4x.hdf5` | easy | 12 anchors + 36 HDF5-aug | ~4700 | 56 K + aug |
| `scene_teleop_full_middle_20260418_152846.hdf5` | middle | 12 (no aug) | ~3300 | 39 K |

Root attrs share the same **instruction** per HDF5: `"find the medicine bottle and place it next to the red cup"`.

**HDF5 hardlink structure (easy only)**: images (base_rgb + wrist_rgb) are internally
hardlinked across 4-episode clusters (ep_k, ep_{k+12}, ep_{k+24}, ep_{k+36}) to save disk
since the user created 4x aug copies of 12 originals. State/action differ per copy;
images are shared.

### 13.3 Noise audit — the original aug4x noise was inappropriate

Measured the HDF5's existing aug noise (anchor vs aug_copy diff):
- **state diff: 0** (state unchanged) ✓
- **action diff: uniform σ ≈ 0.020, gripper σ ≈ 0.010**

Compared to actual action dynamics in the **same teleop data** (step-to-step delta_std over 24 anchors):

| dim | action delta_std (new demos) | uniform σ=0.02 fraction | verdict |
|---|---|---|---|
| arm_pan | 0.00108 | 1852 % | **extreme noise** |
| arm_lift | 0.00521 | 384 % | way too large |
| arm_elbow | 0.00470 | 426 % | way too large |
| arm_wristf | 0.00611 | 327 % | way too large |
| arm_wristr | 0.00255 | 784 % | noise > natural variation |
| gripper | 0.00797 | 125 % | larger than natural |
| x.vel | 0.00228 | 877 % | way too large |
| y.vel | 0.05309 | 38 % | small relative to natural |
| theta.vel | 0.10074 | 20 % | small |

**Core problem with uniform σ=0.02:** arm teleop is extremely smooth (delta_std ≈ 0.001–0.008),
so adding 0.02 noise is 2–18× the natural variation — completely destroys the
trajectory smoothness the policy should learn. Meanwhile base-rotation dims
(y.vel, theta.vel) have large natural variation, so 0.02 there is an effective
regularization scale.

### 13.4 Adaptive noise: σ_i = 0.5 × delta_std_i (measured on new demos)

New per-dim noise:

```python
NOISE_STD = np.array([
    0.00054,  # arm_pan
    0.00260,  # arm_lift
    0.00235,  # arm_elbow
    0.00305,  # arm_wristf
    0.00128,  # arm_wristr
    0.00398,  # gripper
    0.00114,  # x.vel
    0.02654,  # y.vel
    0.05037,  # theta.vel
], dtype=np.float32)
```

Rationale:
- Preserves per-dim teleop smoothness ratio (≤ ½ of natural step-to-step change).
- Avoids destroying the precise arm trajectory while still regularizing the
  large-magnitude base rotation dims.
- **Measured on the _new_ demos** (not on `v4`), because v4 skill demos have
  completely different dynamics (delta_std 5–20× larger on arm dims).

**state noise = 0** (identity copy) — consistent with original aug4x philosophy.
State is "ground truth observation" and shouldn't be perturbed in sim data.

### 13.5 Option B — single-anchor conversion + metadata-only aug expansion

**Problem:** LeRobot's HDF5→lerobot converter stages each frame as a PNG to disk
(≈ 14 frames/min per process), then encodes to mp4 via ffmpeg. Converting 48 eps
(12 anchors + 36 aug duplicates of the same images) takes ~15 h even on GPFS.

**Option B insight:** The 4× aug copies in aug4x share identical images (HDF5 hardlink).
Converting all 48 re-encodes the same pixels 4 times. Instead:

1. **Convert 12 anchors** (one per original episode) → 12 single-anchor lerobot datasets
   with videos + data parquet + episodes.parquet.
2. **Aggregate** 12 single-anchor datasets into one 12-ep merged dataset via
   `lerobot.datasets.aggregate.aggregate_datasets`.
3. **Expand to 48** by appending 36 aug rows with:
   - `observation.state` = anchor state (copied)
   - `action` = anchor action + Gaussian noise (new NOISE_STD)
   - `episodes.parquet` entry: video pointers identical to the anchor's
     (chunk_index, file_index, from/to_timestamp).

LeRobot's DataLoader treats each episode independently; two episodes pointing at the
same mp4 range simply decode the same frames and pair them with different state/action.

**Result:** 4× faster conversion (≈ 2 h instead of 15 h), mathematically equivalent
training signal.

Pipeline scripts (on A1002, see §13.7):
- `split_12_anchors.py`: splits `anchors_part_{0..3}.hdf5` into 12 single-anchor hdf5s.
- `run_convert_12parallel.sh`: 12 parallel processes, each 5 cores, output to `/dev/shm`.
- `phase4_expand_easy_v2.py` / `middle_pipeline.py`: Option B expand + stats refresh.
- `redo_aug4x_noise.py`: in-place aug re-noise if wrong NOISE_STD was applied.

### 13.6 Middle demos (no pre-existing aug)

`scene_teleop_full_middle_20260418_152846.hdf5` is a fresh teleop with 12 eps, no aug.
Processed identically to easy but without the HDF5-hardlink shortcut: the 36 aug
entries are generated purely in lerobot-format (state copied, action noise from
NOISE_STD).

### 13.7 Remote conversion server (A1002)

Why: the A100 machine is running the `viva_v4` training. Running conversion in parallel
degrades training throughput by ~26 % (GPFS I/O contention on PNG staging).

Setup:
- Target: `218.148.55.186:30380` (ssh A1002). Internal IP `10.47.128.6` (reachable
  from A100 via the shared cluster subnet after adding A100's pubkey to `~/.ssh/authorized_keys` on A1002).
- Env: `conda create -n aug4x python=3.12 -y`, `pip install lerobot h5py pandas pyarrow`
  (lerobot 0.5.0 requires Python ≥ 3.12; lerobot 0.5.1 was actually installed, see §13.11).
- ffmpeg: system ffmpeg has `libx264` and `libx265` but **no `libsvtav1`** — use `--vcodec h264`.
  Option: `h264_nvenc` is available (GPU-accelerated) but the converter doesn't expose it as a choice.
- tmpfs: `/dev/shm` = 202 GB, used for PNG staging (dramatically faster than GPFS).
- Work dir: `/home/jovyan/viva/`.
- Backups: always copy `/dev/shm/viva_output/*` → `/home/jovyan/viva/backups/` before deletion.

SSH config entry on A100:
```
Host h100_remote
    HostName 10.47.128.6
    User jovyan
    IdentityFile /home/jovyan/.ssh/new_server_key
    StrictHostKeyChecking accept-new
```
(historically named A1002 since it shares the public IP of A100 on port 30380).

### 13.8 Combined dataset structure

`lekiwi_full_teleop_combined` = `aug4x_lerobot_final` (48 eps) ⊕ `middle_lerobot_final` (48 eps):

| | easy (aug4x) | middle | combined |
|---|---|---|---|
| anchors | 12 | 12 | 24 |
| aug copies | 36 | 36 | 72 |
| total eps | 48 | 48 | **96** |
| anchor frames | 56,457 | 39,439 | 95,896 |
| total frames | 225,828 | 157,756 | **383,584** |
| fps | 25 | 25 | 25 |
| task_index | 4 (single unified task) |
| instruction | "find the medicine bottle and place it next to the red cup" |

Dataset root on H100: `/home/elicer/h100_deploy/dataset/`.

### 13.9 H100 training command

```
lerobot-train \
  --dataset.repo_id=local/lekiwi_full_teleop_combined \
  --dataset.root=./dataset \
  --policy.path=./base_model \
  --policy.repo_id=local/pi05_h100_endtoend \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_state_dim=32 \
  --policy.max_action_dim=32 \
  --policy.scheduler_decay_steps=60000 \
  --policy.optimizer_lr=7e-5 \
  --batch_size=16 \
  --steps=80000 \
  --save_freq=8000 \
  --log_freq=100 \
  --eval_freq=0 \
  --num_workers=8 \
  --tolerance_s=0.1 \
  --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
  --output_dir=./outputs/h100_endtoend
```

Comparison with A100 skill training:

| | A100 (viva_v4) | H100 (end-to-end combined) |
|---|---|---|
| GPU | A100 40GB | H100 80GB |
| batch | 2 | **16** |
| lr | 2.5e-5 | **7e-5** (scaled for 8× batch) |
| warmup | 1000 | 1000 |
| decay | 200 000 | 60 000 |
| total steps | 358 K (3 epochs) | 80 K (≈ 3.3 epochs) |
| save_freq | 20 000 | 8 000 |
| ckpt per epoch | 6 | 3 |
| throughput | 1.5 step/s | 0.76 step/s |

### 13.10 LeRobot pi0.5 pipeline audit — findings

Read lerobot sources on H100 (`pi05/modeling_pi05.py`, `pi05/configuration_pi05.py`,
`pi05/processor_pi05.py`, `datasets/lerobot_dataset.py`, `datasets/sampler.py`,
`scripts/lerobot_train.py`).

**Correct (verified in source):**

1. **Flow matching loss**: `u_t = noise − actions`, `x_t = t·noise + (1−t)·actions`,
   MSE over all `chunk_size=50` timesteps (`modeling_pi05.py:738-740, 775-783`).
2. **Time sampling**: Beta(α=1.5, β=1.0) scaled to [0.001, 0.999] (`modeling_pi05.py:633-638`).
3. **QUANTILES normalization**: `[q01, q99] → [-1, 1]` (`normalize_processor.py:358-368`).
4. **Stats source**: `dataset.meta.stats` from `dataset/meta/stats.json` overrides the
   base_model's stats (`lerobot_train.py:270`). Our normalization IS applied.
5. **Image preprocessing**: manual `img * 2 − 1` in the policy forward (siglip convention),
   VISUAL=IDENTITY bypasses the generic normalizer (`modeling_pi05.py:1186-1187`).
6. **Missing 3rd camera** (`right_wrist_0_rgb` absent in our dataset): filled with `-1`
   and mask=0, so the model doesn't attend (`modeling_pi05.py:1199-1204`). Graceful.
7. **Resume**: RNG state, optimizer state, scheduler state, training step all saved/loaded
   (`train_utils.py:150-167`). Warmup continues from saved scheduler state (not re-run).

**Issue found — chunks cross episode boundaries (`lerobot_train.py:281`):**

```python
if hasattr(cfg.policy, "drop_n_last_frames"):
    sampler = EpisodeAwareSampler(...)  # drops last frames of each ep
else:
    sampler = None  # pi0.5 falls here
```

- `diffusion` policy defines `drop_n_last_frames=7`.
- `sarm` policy defines `drop_n_last_frames=1`.
- **pi0.5 has no `drop_n_last_frames`** → sampler is `None` → random sampling, no dropping.

When sampled frame index `t` + `delta ∈ [0, 49]` exceeds episode end, `_get_query_indices`
clamps to `min(ep_end-1, t+delta)` (`lerobot_dataset.py:990`). This means out-of-episode
action chunk positions get **the last action of the episode repeated**. The model
still computes loss on those padded positions (no masking in the PI05Policy.forward),
effectively training "predict last action at episode boundary".

Impact:
- **H100 (avg 4000 frame/ep, chunk 50)**: 50/4000 = **1.25 % of samples** have padded
  positions — negligible.
- **A100 (avg 230 frame/ep in v4 skill data, chunk 50)**: 50/230 ≈ **22 % of samples**
  — moderate. The policy wastes capacity learning a "hold last action" pattern at
  skill endings. Not catastrophic but suboptimal.

### 13.11 Other minor observations

- `stats.json` `count = 95896` = anchor-only sum (24 × avg ~4000 frames). Stats were
  computed before aug expansion. **Impact on training: minimal** — state stats unchanged
  (aug copies anchor state), action q01/q99 barely shift (noise σ ≤ 0.05 vs ranges 0.1–2.0).
- `stats.json` `episode_index` reports `min=0, max=0`. Likely aggregation bug in lerobot
  for constant-per-episode columns. Not used by the policy. Harmless.
- `tasks.parquet` carries 5 tasks (indices 0–4): the first 4 are the converter's
  `DEFAULT_SUBTASK_ID_TO_TEXT` legacy; only task_index=4 is actually assigned to frames.
  Harmless.
- A1002 installed `lerobot==0.5.1` (pip auto-pin from 0.5.0 when numpy <2 constraint
  dropped). Format is identical to 0.5.0 (both `codebase_version: v3.0`), confirmed
  by successful aggregate across versions.

### 13.12 Conclusion for end-to-end training

- `lekiwi_full_teleop_combined` (96 ep, 383 K frame) is correctly built with
  **adaptive noise matched to the new demos' own dynamics**.
- H100 training is exercising the correct flow-matching loss over the full 50-step
  chunk with proper QUANTILES normalization from our stats.
- Training is progressing normally: at step 30 K (38 % of 80 K) loss has dropped to
  **0.010**, ≈ 2× lower than the A100 skill training's 0.020 at similar epoch,
  reflecting the simpler unified task.

### 13.13 Files (H100 training dataset pipeline)

| File | Purpose | Location |
|---|---|---|
| `aug4x_lerobot_final/` | 48-ep easy dataset (Option B) | A1002 `/home/jovyan/viva/` |
| `middle_lerobot_final/` | 48-ep middle dataset | A1002 `/home/jovyan/viva/` |
| `combined_aug4x_middle/` (or delivered to H100 `./dataset/`) | 96-ep merged | H100 `/home/elicer/h100_deploy/dataset/` |
| `split_aug4x.py`, `split_12_anchors.py` | HDF5 splitting (hardlink-preserving for easy) | A1002 `/home/jovyan/viva/` |
| `run_convert_12parallel.sh`, `run_convert_middle.sh` | Parallel converter launchers | A1002 |
| `phase4_expand_easy_v2.py`, `middle_pipeline.py` | Option-B expand + stats refresh | A1002 |
| `redo_aug4x_noise.py` | In-place aug re-noise | A1002 |
| CONVERSION_GUIDE_REMOTE.md | Full remote-machine conversion guide | A100 `/home/jovyan/tmp_parts/` |

### 13.14 Paper narrative (H100 supplement)

> "In parallel to the skill-decomposition model trained on the short-horizon `viva_v4`
>  dataset (1,036 ep, ~230 frames/ep), we trained an **end-to-end Pi0.5 policy** on 24
>  long-horizon teleoperations (avg ~4,000 frames/ep) covering the full pick-and-place
>  task. To regularize without distorting the precise teleoperated arm trajectories,
>  action-only augmentation with **per-dimension Gaussian noise σ_i = 0.5 × Δ_i** was
>  applied, where Δ_i is the empirical step-to-step action-delta std of each dim
>  measured on the same 24 demos. This preserves the teleop smoothness prior while
>  regularizing the large-amplitude base-rotation dims. 36 aug copies were generated
>  per-dim and pointed to the anchor episode's video via the LeRobot `episodes.parquet`
>  video-reference mechanism (Option B), avoiding a 4× video re-encode cost. The
>  resulting dataset (96 ep, 383 K frames) was trained on an H100 with batch size 16
>  and learning rate 7·10⁻⁵, converging to a flow-matching loss of ≈0.01 within 1.25
>  epochs — roughly 2× lower than the skill model at equivalent dataset coverage,
>  consistent with the simpler label distribution of a single unified task."

---

## 14. Multi-server Eval Setup (2026-04-19)

This section captures the state after H100 disconnect + A100 → A1002 training relocation.

### 14.1 Server topology (historical snapshot — see §16 for current)

> **UPDATE 2026-04-19 17:xx**: 아래 표는 초기 (H100 end-to-end 끝나고 A1002가 skill 재개했을 때) snapshot.
> 현재 topology는 **§16**에서 다시 정리됨. H100은 **재대여해서 viva_v4 skill fine-tune 중**,
> A1002는 중단. A100은 eval host.

| Server | Hostname / alias | GPU | Role (당시) |
|---|---|---|---|
| A100 (this machine) | `cheetah-...`, local | A100 40GB | Eval host (VLM + VLA + Isaac Sim) |
| A1002 | `ssh A1002` (elice config), internal IP `10.47.128.6` | A100 40GB | Training host (skill policy resume) — **중단됨** |
| H100 (round 1, end-to-end) | `ssh h100` (elice tunnel 25104) | H100 80GB | **완료** (64K ckpt backed up, disconnected) |

### 14.2 Checkpoint inventory (on A100)

| Name | Path on A100 | Size | Epoch | Loss |
|---|---|---|---|---|
| **Skill policy** (`pi05_chunk50_v4`) | `/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/outputs/train/pi05_chunk50_v4/checkpoints/` | 9 ckpts × 23 GB | 1.68 (200K) | 0.020 |
| **End-to-end** (H100 backup) | `/home/jovyan/h100_endtoend_backup/outputs/h100_endtoend/checkpoints/{048000,056000,064000}/` | 3 ckpts × 23 GB | 2.00 / 2.33 / 2.67 | 0.009 / 0.007 / **0.006** |

### 14.3 A1002 resume state (skill training continues there)

Paths on A1002:
```
/home/jovyan/viva/
├── envs/lerobotpi0v2/                        ← conda-pack'd from A100, 8.6 GB
├── lekiwi_viva_v4/                           ← dataset (1036 ep)
├── outputs/train/pi05_chunk50_v4/
│   └── checkpoints/
│       ├── 00200000/                          ← resumed from this
│       └── last → 00200000
├── pi05_chunk50_v4_resume.log                ← training log
├── start_a1002_resume.sh                     ← launcher (accepts <ckpt_step>)
└── (conversion artifacts from earlier: aug4x_*, middle_*, combined_aug4x_middle/)
```

Launch command (if you need to re-run):
```bash
# On A100 (this machine):
ssh A1002 "bash /home/jovyan/viva/start_a1002_resume.sh 200000"
```

The launcher:
1. Updates `checkpoints/last` symlink
2. Patches `train_config.json:dataset.root` → `/home/jovyan/viva/lekiwi_viva_v4`
3. nohup launches `lerobot-train --config_path=.../last/pretrained_model/train_config.json --resume=true`

### 14.4 Eval server launch (A100)

Both servers fit in 40 GB GPU:
- VLM (Qwen3-VL-8B, bf16, util 0.75) = ~30 GB
- VLA (Pi0.5, bf16) = ~9.4 GB
- Total ~39 GB → Isaac Sim has ~1 GB headroom (may OOM; reduce VLM util if needed)

**Correct launch (as of 2026-04-19):**

```bash
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env

# End-to-end (H100 64K):
VLA_CKPT=/home/jovyan/h100_endtoend_backup/outputs/h100_endtoend/checkpoints/064000/pretrained_model

# Skill (latest A100 save, 200K):
# VLA_CKPT=/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/outputs/train/pi05_chunk50_v4/checkpoints/last/pretrained_model

bash launch_servers.sh all --checkpoint "$VLA_CKPT"
```

The launcher (fixed 2026-04-19):
- VLM: Qwen3-VL-8B at port 8000, gpu-mem 0.75
- VLA: **Pi0.5 via `vllm/vla_inference_server.py`**, port 8002
- VLA arg is `--checkpoint` (not `--model` — that was the old PI0-FAST server signature)
- `rm -f logs/vla_server.log` before start (stale-log mitigation)

### 14.5 Single canonical VLA server file

As of 2026-04-19, the **only** VLA server script is:

```
/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/vla_inference_server.py  (Pi0.5, PI05Policy)
```

The parent-dir duplicate (`lekiwi_nav_env/vla_inference_server.py`) was **deleted** to avoid import-path confusion. Previously there were two copies with conflicting implementations (old PI0-FAST vs new PI0.5), which caused repeated launch failures.

### 14.6 Troubleshooting lessons (from 2026-04-19)

| Symptom | Root cause | Fix |
|---|---|---|
| `AttributeError: 'PI05Config' object has no attribute 'action_tokenizer_name'` | Running old PI0-FAST server against PI05 checkpoint | Use the canonical `vllm/vla_inference_server.py` (uses PI05Policy), not any backup copy |
| `--model` arg unrecognized | New Pi0.5 server takes `--checkpoint` | Pass `--checkpoint <path>` |
| Re-launch shows stale error log | Log file not truncated; new process failed silently before writing | `rm -f logs/vla_server.log` before each restart |
| `Failed to load ... gated repo 401` | HF token missing on fresh A1002 | Copy `~/.cache/huggingface/token` + `stored_tokens` from A100 |
| `launch_servers.sh` loads wrong tokenizer | Historical: vllm/ dir had PI0-FAST version | Updated 2026-04-19 to use PI05 server with correct arg |

### 14.7 Quick sanity check script

After launching servers, verify both:

```bash
curl -s http://localhost:8000/v1/models | head -c 300   # VLM
curl -s http://localhost:8002/health                     # VLA
```

Expected:
- VLM: JSON with `Qwen/Qwen3-VL-8B-Instruct`
- VLA: `{"status":"ok","model":"pi05-...","device":"cuda","gpu_memory_mb":...}`

### 14.8 A100 stopped at 200K — progress counter note

A100 viva_v4 training was stopped at step **210,761** (last ckpt at 200,000) on 2026-04-19 to free A100 GPU for eval. Training continues on A1002 from 200K. Original training plan was 3 epochs = 358K; A1002 will complete the remaining 158K steps.

Note: lerobot's `epch` counter displays `smpl / current_dataset_frames`, so resume from checkpoint inherits the absolute smpl counter. Useful but can be misleading if the ckpt was trained on a *different* dataset earlier (e.g., viva_v3 before the v4 fix). See §12 for v3→v4 context.

---

## 15. End-to-End Evaluation (VLA only, no VLM) — 2026-04-19

### 15.1 Why a separate runner?

`run_full_task.py` assumes the **skill-aware** policy (`lekiwi_viva_v4` training) where:
- VLM orchestrator produces skill-specific instructions ("navigate forward", "approach and lift", ...)
- VLA is conditioned on these phased instructions

The H100 end-to-end checkpoint (`lekiwi_full_teleop_combined`) was trained on:
- **One single instruction** (`"find the medicine bottle and place it next to the red cup"`) applied for the entire episode
- No phase decomposition
- Single unified trajectory from start to grasp to place

Running `run_full_task.py` against this checkpoint would feed it **out-of-distribution instructions**
(e.g., "navigate turn left") that it never saw during training.

### 15.2 The runner: `vllm/run_only_vla.py`

Standalone Pi0.5 evaluation — no VLM at all.

Architecture:
```
Isaac Sim
  └─ base_cam + wrist_cam + depth

VLA server (port 8002)
  └─ (base_rgb, wrist_rgb, state_9d, FIXED_INSTRUCTION) → 50-step action chunk

Main loop:
  capture → get_state → vla.query (or consume buffer) → safety check → env.step
```

**No VLM**, no skill state machine, no phase transition, no obstacle-check VLM call.

### 15.3 Launch procedure

```bash
# 1) VLA 서버만 기동 (end-to-end ckpt)
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env
bash launch_servers.sh vla --checkpoint \
  /home/jovyan/h100_endtoend_backup/outputs/h100_endtoend/checkpoints/064000/pretrained_model

# 2) 실행
cd vllm
python run_only_vla.py --headless
#   또는 instruction 변경:
# python run_only_vla.py --headless --instruction "find the bottle and place it on the table"
```

### 15.4 Key CLI options (differences from run_full_task.py)

| Option | Default | Note |
|---|---|---|
| `--instruction` | `"find the medicine bottle and place it next to the red cup"` | **학습 데이터와 동일해야 함**. 변경 시 OOD 위험 |
| `--n_use` | `50` | Chunk 전체 실행 후 재쿼리 (end-to-end는 긴 horizon 안정적). 작게 하면 반응성↑ |
| `--safety_dist` | `0.3` | depth 기반 **단순** safety (VLM 없음). 0이면 비활성 |
| `--render_wide` | `false` | base_cam을 640x400로 직접 렌더 (VLM 없으니 1280x800 불필요) |

제외된 run_full_task.py 인자들:
- `--vlm_server`, `--vlm_model`, `--user_command`, `--target_object`, `--dest_object`
- `--vlm_interval`, `--memory_*`
- `--navigate_timeout`, `--approach_lift_timeout`, `--carry_timeout`, `--approach_place_timeout`
- `--s2_max_attempts`, `--s4_max_attempts`
- `--mode`, `--stop_at_carry`

### 15.5 GPU memory budget (VLA-only)

A100 40GB에서:
- VLA (Pi0.5 bf16) = ~9.4 GB
- Isaac Sim rendering = ~5 GB
- 남는 여유 = ~25 GB

VLM 없으니 OOM 걱정 없음. 평가 안정성↑.

### 15.6 Use cases

| 목적 | 사용 runner |
|---|---|
| VIVA 시스템 (VLM + VLA + skill state machine) eval | `run_full_task.py` with A100/A1002 skill ckpt |
| End-to-end Pi0.5 (단일 instruction) eval | `run_only_vla.py` with H100 end-to-end ckpt |
| Ablation: VIVA 없는 순수 VLA 성능 비교 | `run_only_vla.py` (공정 비교용) |

### 15.7 Expected behavior (end-to-end policy)

Single instruction이 전체 episode 내내 고정되어 있으므로, VLA는 학습 분포에 따라 **스스로 phase를 전환** (navigate → approach → lift → carry → place). Phase 전환이 부정확하면 OOD로 빠져 recovery 안 됨 — 이게 VIVA (skill decomposition) 대비 end-to-end의 구조적 약점.

Eval 비교 관심 항목:
- `run_full_task.py` (VIVA) success rate
- `run_only_vla.py` (end-to-end) success rate
- phase transition failures (end-to-end에서 특정 phase에서 멈춤)




---

## 16. H100 skill policy fine-tune (viva_v4 resume at bs=16) — 2026-04-19

### 16.1 배경

A100 viva_v4 skill training (bs=2, lr=2.5e-5)이 step 200K 도달 후 loss 0.020에서
수렴 조짐. 추가 fine-tune을 더 큰 batch + 더 높은 lr로 빠르게 하기 위해
**H100 80GB를 재대여**해서 `bs=16, lr=7e-5`로 resume.

목표: step 200K → step 210K (2.35 v4 epoch) 까지 ~3.7h 이내에 완주.

### 16.2 Server topology (현재)

| Server | 접속 | GPU | 역할 |
|---|---|---|---|
| **A100** (this machine) | local | A100 40GB | **Eval host** (VLM + VLA 64K ckpt) |
| **A1002** | `ssh A1002` (port 30380) | A100 40GB | **유휴** (conversion 이후 대기) |
| **H100 (round 2)** | `ssh h100` (elice tunnel **port 54915** — 매 대여마다 바뀜) | H100 80GB | **viva_v4 fine-tune 중** (bs=16, step 200K→210K) |

H100 key: `/home/jovyan/.ssh/elice.pem` (이전 대여와 동일 key 재사용됨).

### 16.3 환경 호환성 (확정된 지식)

이전에 잘못 걱정한 것 + 실제 발견:

| 요소 | A100 상태 | H100 상태 | 호환? |
|---|---|---|---|
| torch 2.10.0+cu128 (CUDA 12.8) | 설치됨 | **driver 535.216.03 (최대 CUDA 12.2)에서도 동작 OK** (sm_90 pre-compiled) | ✅ |
| lerobot 0.5.0 | 설치됨 | conda-pack으로 이식 | ✅ |
| paligemma tokenizer | HF cache 있음 | token + cache 전송하면 OK | ✅ |
| **torchcodec (video decode)** | `libavutil.so.56` (FFmpeg 4) 시스템 의존 — A100 Ubuntu 22.04엔 있음 | ❌ H100 이미지에 ffmpeg 없음, libavutil 없음 | **불호환** |
| **pyav (video decode 대체)** | bundled ffmpeg 포함 | 동일 | ✅ 이걸 쓰면 됨 |
| bf16 matmul on H100 | — | 정상 동작 | ✅ |

**핵심 fix**: `train_config.json` 의 `dataset.video_backend` 를 `torchcodec` → `pyav` 로 변경.

### 16.4 Deploy 패키지 구성

`/home/jovyan/h100_viva_deploy/` (A100 측, 30 GB)

```
├── env.tar.gz              4.8 GB  conda-pack한 lerobotpi0v2
├── ckpt/00200000/          23 GB   A100 viva_v4 step 200K
├── dataset/lekiwi_viva_v4/ 1.8 GB  (1036 ep, 238K frames)
├── hf_cache/               21 MB   token + stored_tokens + paligemma-3b-pt-224
└── scripts/
    ├── deploy.sh           unpack + 파일 배치 + config patch + shebang 수정
    ├── patch_configs.py    bs=16/lr=7e-5 패치 + scheduler_state.json base_lrs 덮어쓰기
    └── launch.sh           lerobot-train 시작
```

### 16.5 Deploy 순서 (A100 → H100)

```bash
# 1) rsync (A100 local, ~15분, 내부망 23MB/s)
rsync -avz --progress -e "ssh -i /home/jovyan/.ssh/elice.pem" \
  /home/jovyan/h100_viva_deploy/ \
  h100:/home/elicer/h100_viva_deploy/

# 2) deploy (H100, ~5분: env 풀기 + 배치 + shebang 수정)
ssh h100 "cd /home/elicer/h100_viva_deploy && bash scripts/deploy.sh"

# 2-a) deploy.sh 안의 conda-unpack이 `#!/usr/bin/env python` shebang 때문에 실패할 수 있음
#      → shebang 교정 step이 포함되어 있으나 conda-unpack 자체도 실패 시 수동으로:
# ssh h100 "/home/elicer/h100_viva/envs/lerobotpi0v2/bin/python3.12 \
#           /home/elicer/h100_viva/envs/lerobotpi0v2/bin/conda-unpack"

# 3) video_backend 패치 (deploy.sh가 patch_configs.py 실행 후 추가로):
ssh h100 '/home/elicer/h100_viva/envs/lerobotpi0v2/bin/python3.12 -c "
import json
p = \"/home/elicer/h100_viva/outputs/train/pi05_viva_h100/checkpoints/00200000/pretrained_model/train_config.json\"
cfg = json.load(open(p))
cfg[\"dataset\"][\"video_backend\"] = \"pyav\"
cfg[\"tolerance_s\"] = 0.1
cfg[\"steps\"] = 10000000          # ★ 중요: auto-scale 방지 (16.7 참고)
json.dump(cfg, open(p, \"w\"), indent=4)
"'

# 4) 학습 시작
ssh h100 "bash /home/elicer/h100_viva_deploy/scripts/launch.sh"
```

### 16.6 핵심 config (viva_v4 resume용 bs=16)

| 필드 | 값 | 주의 |
|---|---|---|
| batch_size | **16** | H100 80GB에 여유 (VRAM 42/80 GB 사용) |
| optimizer_lr (peak) | **7e-5** | sqrt rule: `2.5e-5 × √8 = 7.07e-5` |
| scheduler_warmup_steps | 500 | 이미 200K 학습됨, 짧게 |
| scheduler_decay_steps | **1,000,000** | step 200K에서 lr이 peak의 ~91% (= 6.36e-5) 유지 |
| **steps (num_training_steps)** | **10,000,000** | **반드시 num_decay_steps 보다 커야 함** (§16.7 참고) |
| save_freq | 5000 | 205K, 210K, 215K... |
| dataset.video_backend | `pyav` | torchcodec은 FFmpeg 시스템 의존 → H100엔 없음 |
| tolerance_s | 0.1 | 이전 H100 end-to-end 세팅 그대로 |
| dataset.root | `/home/elicer/h100_viva/dataset/lekiwi_viva_v4` | H100 절대경로 |

### 16.7 LR 계산 완전 정복 (이번 세션 큰 gotcha)

#### 공식 (lerobot/optim/schedulers.py의 `CosineDecayWithWarmup`)
```python
cosine_decay = 0.5 * (1 + cos(π * step / decay_steps))
alpha = decay_lr / peak_lr
lambda = (1 - alpha) * cosine_decay + alpha
lr = base_lrs[0] * lambda
```

Step 200K, peak=7e-5, decay=1M, decay_lr=2.5e-6:
```
cosine_decay = 0.5 × (1 + cos(π × 0.2)) = 0.5 × 1.809 = 0.9045
alpha = 2.5e-6 / 7e-5 = 0.0357
lambda = 0.9643 × 0.9045 + 0.0357 = 0.908
lr = 7e-5 × 0.908 = 6.36e-5  ✓
```

#### ⚠ 트랩 1: Auto-scale

lerobot은 `num_training_steps < num_decay_steps`이면 decay를 강제로 `num_training_steps`로 축소.

우리가 처음 `cfg.steps=215000` (train 목표) + `num_decay_steps=1M` 으로 설정 → lerobot이
`actual_decay_steps = 215000` 으로 바꿔버림 → step 200K가 93% 지점 → **lr = 3.3e-6** (target의 1/20).

**Fix**: `cfg.steps = 10,000,000` (auto-scale 회피). 학습 종료는 **수동 kill**로 조절.

#### ⚠ 트랩 2: PyTorch LambdaLR state_dict가 base_lrs 덮어쓰기

`scheduler.load_state_dict(saved_state)` 호출 시 `base_lrs` 가 saved 값으로 overwrite됨.
A100에서 200K ckpt 저장 시 base_lrs=[2.5e-5] (bs=2 때 값). Resume + config만 5e-5/7e-5로
바꿔도 state dict 로드 후 base_lrs가 옛날 값으로 돌아감.

**Fix**: `training_state/scheduler_state.json` 의 `base_lrs` 필드를 **직접 원하는 값으로
수정** 후 resume. (`patch_configs.py` 가 자동으로 이것을 함.)

```json
// 수정 전
{"base_lrs": [2.5e-05], "last_epoch": 200000, ...}
// 수정 후
{"base_lrs": [7e-05], "last_epoch": 200000, ...}
```

### 16.8 목표 도달 시 회수

목표: step 210K (= 2.35 v4 epoch, 200K→210K 10,000 step 추가 ~3h 45m)

```bash
# 1) H100 학습 중단
ssh h100 "pkill -9 -f lerobot-train"

# 2) 210K ckpt A100으로 rsync
mkdir -p /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/outputs/train/pi05_viva_h100/checkpoints
rsync -avz --progress -e "ssh -i /home/jovyan/.ssh/elice.pem" \
  h100:/home/elicer/h100_viva/outputs/train/pi05_viva_h100/checkpoints/00210000/ \
  /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm/outputs/train/pi05_viva_h100/checkpoints/00210000/
```

회수 후 `run_full_task.py` (VIVA) 의 VLA ckpt로 210K 지정해서 eval.

### 16.9 재발 방지 체크리스트 (다음 H100 대여 시)

1. ✅ `torch.cuda.is_available()` 반환 확인 (cu128 + 535 driver 조합 OK, 추가 설치 불필요)
2. ✅ `dataset.video_backend = "pyav"` 항상 패치 (torchcodec 쓰면 ffmpeg 시스템 의존)
3. ✅ `scheduler_state.json` 의 `base_lrs` 를 새 peak_lr로 수정 (PyTorch LambdaLR 덮어쓰기 버그)
4. ✅ `cfg.steps > num_decay_steps` 보장 (auto-scale 회피)
5. ✅ conda-pack 후 `lerobot-train` 의 shebang을 env의 절대경로 python3.12로 교정
6. ✅ HF `token` + `stored_tokens` + paligemma tokenizer cache 동봉 (paligemma는 gated)
7. ✅ 학습 시작 후 첫 INFO step에서 `lr` 값 직접 확인 (수식 역산과 비교)

### 16.10 이번에 **잘못된** 이전 주장 정정

- ❌ "A100 40GB는 batch=2가 한계" — 실측 **bs=8까지 OK** (VRAM 98%). bs=8은 bs=2 대비 1.7x throughput.
- ❌ "torch cu128은 H100 driver 535와 호환 안 됨" — 실측 **동작**. PyTorch sm_90 pre-compiled kernel 포함.
- ❌ "video codec이 libsvtav1이라 H100에서 decode 불가" — `pyav` 가 bundled ffmpeg로 AV1 decode 지원 (`libdav1d`).

(이전 섹션들의 이런 기술적 추정은 사실 확인을 거치지 않은 부분이라 이번에 교정됨.)
