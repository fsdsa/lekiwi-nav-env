#!/bin/bash
# Pi0.5 fine-tuning script for lekiwi mobile manipulator
# v2 (2026-04-09): added LR schedule fix + carry turn label sanity check
#
# Fixes from v1 (60K analysis):
#   1. scheduler_decay_steps=100000 (was 30000 default)
#      → previous run reached LR floor at step 30K, leaving 170K steps at minimum LR
#      → fix: extend cosine decay to 100K so model keeps learning meaningfully
#   2. steps=150000 (was 200000)
#      → 100K decay + 50K floor for fine-tuning, more efficient than 200K
#   3. Pre-flight carry turn label sanity check
#      → catch the carry turn label bug (task 16/17 swap) before it propagates
#
# Key differences from pi0fast:
#   - Continuous flow matching (not FAST tokens)
#   - QUANTILES normalization (default, our stats has q01/q99)
#   - chunk_size 50 default (we override to 10)
#   - More memory than pi0fast (~9GB inference, ~30GB training)
#
# IMPORTANT - Image key mapping:
#   pi05_base expects 3 images (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)
#   our dataset has 2 images (front, wrist)
#   → rename_map maps our keys to base config keys
#   → right_wrist_0_rgb is missing → pi05 _preprocess_images auto-pads with -1
#
# Memory budget (A100 40GB):
#   - batch_size=2, gradient_checkpointing=true, bfloat16 → ~39 GB used (1 GB free)

set -e

cd "$(dirname "$0")"

OUTPUT_DIR="outputs/train/pi05_lekiwi_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/home/jovyan/pi05_train_$(date +%Y%m%d_%H%M%S).log"
LEROBOT_BIN="/home/jovyan/yes/envs/lerobotpi0v2/bin/lerobot-train"
PYTHON_BIN="/home/jovyan/yes/envs/lerobotpi0v2/bin/python"
PI05_BASE="/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/pi05_base"

echo "=========================================="
echo "  Pi0.5 fine-tuning for lekiwi"
echo "  output: $OUTPUT_DIR"
echo "  log: $LOG_FILE"
echo "  pi05_base: $PI05_BASE"
echo "=========================================="

# Pre-flight checks
if [ ! -d "$PI05_BASE" ]; then
    echo "ERROR: pi05_base model not found at $PI05_BASE"
    exit 1
fi

if [ ! -d "/home/jovyan/lerobot_data/lekiwi_viva_v2" ]; then
    echo "ERROR: dataset not found"
    exit 1
fi

# Verify q01/q99 exists in stats (required for QUANTILES)
$PYTHON_BIN -c "
import json
with open('/home/jovyan/lerobot_data/lekiwi_viva_v2/meta/stats.json') as f:
    s = json.load(f)
assert 'q01' in s['action'], 'q01 missing from action stats'
assert 'q99' in s['action'], 'q99 missing from action stats'
assert 'q01' in s['observation.state'], 'q01 missing from state stats'
assert 'q99' in s['observation.state'], 'q99 missing from state stats'
print('[1/2] Stats q01/q99 OK for QUANTILES normalization')
"

# Verify carry turn label fix (task 16=carry turn left, task 17=carry turn right)
# Critical: ensures navigate vs carry turn convention is consistent
$PYTHON_BIN -c "
import json
import pandas as pd
import numpy as np

with open('/home/jovyan/lerobot_data/lekiwi_viva_v2/meta/tasks.jsonl') as f:
    task_lookup = {json.loads(l)['task_index']: json.loads(l)['task'] for l in f}

# Expected: env wz convention is action+ = right (CW), action- = left (CCW)
expected = {
    'navigate turn left':  ('-', 8),
    'navigate turn right': ('+', 9),
    'carry turn left':     ('-', 16),
    'carry turn right':    ('+', 17),
}
data = pd.read_parquet('/home/jovyan/lerobot_data/lekiwi_viva_v2/data/chunk-000/file-000.parquet')

errors = []
for name, (sign, ti) in expected.items():
    actual_name = task_lookup.get(ti)
    if actual_name != name:
        errors.append(f'task_index {ti} expected \"{name}\" but got \"{actual_name}\"')
        continue
    actions = np.stack(data[data['task_index']==ti]['action'].values)
    wz_mean = actions[:,8].mean()
    actual_sign = '+' if wz_mean > 0 else '-'
    if actual_sign != sign:
        errors.append(f'{name} expected wz {sign} but got {actual_sign} ({wz_mean:+.4f})')

if errors:
    print('[2/2] CARRY TURN LABEL CHECK FAILED:')
    for e in errors:
        print(f'  - {e}')
    exit(1)
print('[2/2] Carry turn labels consistent (navigate ↔ carry signs match)')
"

# Kill any existing training
ps aux | grep lerobot-train | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 2

# Free GPU memory check
$PYTHON_BIN -c "
import torch
free_gb = (torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0) / 1e9
print(f'Free GPU memory: {free_gb:.1f} GB')
if free_gb < 30:
    print('WARNING: Less than 30GB free, may OOM. Lower batch_size or kill other processes.')
"

# Image key rename map (dataset → pi05 base config)
# - observation.images.front → observation.images.base_0_rgb
# - observation.images.wrist → observation.images.left_wrist_0_rgb
# - right_wrist_0_rgb: missing, auto-padded by pi05
RENAME_MAP='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

# Start training
# v2 changes (vs 60K run):
#   - scheduler_decay_steps=100000 (was 30000) → meaningful LR until 100K
#   - steps=150000 (was 200000) → 100K decay + 50K floor fine-tune
nohup $LEROBOT_BIN \
    --dataset.repo_id=local/lekiwi_fetch_v6 \
    --dataset.root=/home/jovyan/lerobot_data/lekiwi_viva_v2 \
    --policy.path=$PI05_BASE \
    --policy.repo_id=local/pi05_lekiwi \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.scheduler_decay_steps=100000 \
    --batch_size=2 \
    --steps=150000 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=0 \
    --num_workers=4 \
    --rename_map="$RENAME_MAP" \
    --output_dir="$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
sleep 5
if ps -p $PID > /dev/null; then
    echo "Started PID: $PID"
    echo "Output: $OUTPUT_DIR"
    echo "Log: $LOG_FILE"
    echo ""
    echo "Monitor: tail -f $LOG_FILE"
    echo "Stop:    kill -9 $PID"
else
    echo "ERROR: Process died immediately. Check log: $LOG_FILE"
    tail -30 "$LOG_FILE"
    exit 1
fi
