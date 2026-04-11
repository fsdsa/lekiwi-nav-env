# ACT State-Only Fork for LeKiwi

Fork of [tonyzhaozh/act](https://github.com/tonyzhaozh/act) modified for **state-only** operation with configurable dimensions.

## Changes from Original

1. **`detr/models/detr_vae.py`** — Configurable `state_dim`/`action_dim` (was hardcoded 14), proper state-only path with latent z injection
2. **`detr/main.py`** — `state_only` flag, no argparse conflicts
3. **`policy.py`** — State-only mode skips image normalization
4. **`utils.py`** — Reads LeKiwi HDF5 format (`episode_*/obs`, `episode_*/actions`, `episode_*/teleop_active`), preserves official normalization
5. **`imitate_episodes.py`** — Simplified for state-only, AMP (bf16) for A100
6. **`eval_act_bc.py`** — Inference with temporal ensembling + denormalization

## Quick Start

### Training
```bash
python imitate_episodes.py \
    --demo_dir /path/to/demos \
    --skill approach_and_grasp \
    --ckpt_dir checkpoints/act_skill2 \
    --chunk_size 20 \
    --batch_size 32 \
    --num_epochs 3000 \
    --lr 1e-4 \
    --kl_weight 10 \
    --hidden_dim 256 \
    --dim_feedforward 1024
```

### Evaluation
```python
from eval_act_bc import ACTInference

act = ACTInference("checkpoints/act_skill2", temporal_agg=True)
act.reset()
action = act.get_action(obs)  # obs: numpy (30,), action: numpy (9,)
```

## Data Format

HDF5 files with structure:
```
episode_0/
  obs: (T, obs_dim)        # e.g. 30D
  actions: (T, action_dim)  # e.g. 9D
  teleop_active: (T,)       # bool, optional
episode_1/
  ...
```
