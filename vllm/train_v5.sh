#!/bin/bash
# Pi0.5 fine-tuning on lekiwi_viva_v5 (2026-04-20)
#
# v5 dataset:
#   - v4 nav(446) + carry(432) 유지
#   - v4 approach 158개 전부 drop, 새로 수집한 100개로 교체
#     (approach 에피소드는 fr=0 artifact 없이 tucked pose부터 녹화됨)
#   - 총 978 episodes, stats.json 실 데이터로 정확히 재계산됨
#
# 학습 config (A100 재현, 배포 ckpt=250K):
#   - batch_size=8 (A100 40GB)
#   - chunk_size=50 (approach lift trajectory에 필요)
#   - n_action_steps=50
#   - steps=300000 (배포된 ckpt = 250000 step)
#   - lr=5e-5, warmup=500, decay=1000000 (cosine)
#   - gradient_checkpointing=true, bfloat16
#
# 추론 시 n_use 분기 (run_full_task.py 내):
#   - nav/carry: n_use=10 (receding horizon, drift 방지)
#   - approach/place: n_use=50 (full chunk 필요)

set -e

cd "$(dirname "$0")"

OUTPUT_DIR="outputs/train/pi05_viva_v5_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/home/jovyan/pi05_v5_train_$(date +%Y%m%d_%H%M%S).log"
LEROBOT_BIN="/home/jovyan/yes/envs/lerobotpi0v2/bin/lerobot-train"
PYTHON_BIN="/home/jovyan/yes/envs/lerobotpi0v2/bin/python"
# Pi0.5 base (H100 deploy에서 다운로드된 14GB, type=pi05, chunk_size=50)
PI05_BASE="/home/jovyan/h100_deploy/base_model"
DATASET_ROOT="/home/jovyan/lerobot_data/lekiwi_viva_v5"

echo "=========================================="
echo "  Pi0.5 fine-tuning on lekiwi_viva_v5"
echo "  output: $OUTPUT_DIR"
echo "  log:    $LOG_FILE"
echo "=========================================="

# ── Pre-flight ──
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: v5 dataset not found at $DATASET_ROOT"
    echo "       run build_v5.py first"
    exit 1
fi

if [ ! -f "$PI05_BASE/model.safetensors" ]; then
    echo "ERROR: Pi0.5 base 없음 at $PI05_BASE"
    echo "       기대 위치: model.safetensors + config.json (type=pi05)"
    exit 1
fi
echo "  Pi0.5 base: $PI05_BASE"

# Stats sanity: q01/q99 must match actual data (build_v5.py recomputes exactly)
$PYTHON_BIN -c "
import json, pandas as pd, numpy as np
s = json.load(open('$DATASET_ROOT/meta/stats.json'))
df = pd.read_parquet('$DATASET_ROOT/data/chunk-000/file-000.parquet')
act = np.stack([np.array(r) for r in df['action']])
true_q99 = np.percentile(act, 99, axis=0)
stats_q99 = np.array(s['action']['q99'])
diff = np.abs(true_q99 - stats_q99).max()
assert diff < 1e-3, f'stats.json q99 mismatch: {diff:.6f}'
print(f'[1/2] stats.json verified (action q99 max diff {diff:.2e})')
print(f'[1/2]   action q99[arm[1]] = {stats_q99[1]:.4f}  (tight, edge anchor for sharp learning)')
print(f'[1/2]   total frames = {len(df)}, episodes = {df[\"episode_index\"].nunique()}')
"

# Task count sanity
$PYTHON_BIN -c "
import pandas as pd
tasks = pd.read_parquet('$DATASET_ROOT/meta/tasks.parquet')
print(f'[2/2] tasks.parquet has {len(tasks)} labels')
print(tasks.to_string())
"

# ── Free GPU ──
ps aux | grep -E 'lerobot-train|vla_inference_server|vllm' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 2
$PYTHON_BIN -c "
import torch
free_gb = (torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0) / 1e9
print(f'GPU free: {free_gb:.1f} GB')
assert free_gb > 30, f'Need >30GB free, got {free_gb:.1f}'
"

# Image key rename (dataset front/wrist → pi05 base_0/left_wrist_0)
RENAME_MAP='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'

# ── Launch ──
nohup $LEROBOT_BIN \
    --dataset.repo_id=local/lekiwi_viva_v5 \
    --dataset.root=$DATASET_ROOT \
    --dataset.video_backend=pyav \
    --policy.path=$PI05_BASE \
    --policy.repo_id=local/pi05_lekiwi_v5 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --policy.optimizer_lr=5e-5 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=1000000 \
    --batch_size=8 \
    --steps=300000 \
    --save_freq=10000 \
    --log_freq=100 \
    --eval_freq=0 \
    --num_workers=16 \
    --rename_map="$RENAME_MAP" \
    --output_dir="$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
sleep 5
if ps -p $PID > /dev/null; then
    echo "학습 시작됨 (PID $PID)"
    echo "Output: $OUTPUT_DIR"
    echo "Log:    $LOG_FILE"
    echo ""
    echo "모니터링:"
    echo "  tail -f $LOG_FILE"
    echo "  strings $LOG_FILE | grep -oP '\d+/300000' | tail -1"
    echo ""
    echo "예상 시간: A100 bs=8 ~0.7 s/step → 300K steps ≈ 58h (배포 ckpt=250K)"
else
    echo "ERROR: 학습 프로세스 죽음"
    tail -30 "$LOG_FILE"
    exit 1
fi
