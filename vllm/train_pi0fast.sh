#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Pi0-FAST Fine-tuning Pipeline for LeKiwi
# Run on A100 server (lerobotpi0v2 conda env)
#
# Full pipeline:
#   1. Convert HDF5 → LeRobot v3 dataset
#   2. Fine-tune Pi0-FAST on the dataset
#   3. Serve the fine-tuned model
#
# Usage examples:
#   # Step 1: Convert HDF5 data
#   bash train_pi0fast.sh convert \
#       --input "/path/to/teleop_scene_data/*.hdf5" \
#       --output_root ~/lerobot_data/lekiwi_scene
#
#   # Step 2: Train (full fine-tuning, default)
#   bash train_pi0fast.sh train \
#       --data_root ~/lerobot_data/lekiwi_scene \
#       --steps 10000
#
#   # Step 2 alt: Train with LoRA (requires peft, auto-installed)
#   bash train_pi0fast.sh train \
#       --data_root ~/lerobot_data/lekiwi_scene \
#       --mode lora \
#       --steps 10000
#
#   # Step 3: Serve fine-tuned model
#   bash train_pi0fast.sh serve \
#       --checkpoint outputs/train/pi0fast_lekiwi/checkpoints/last
#
#   # One-shot: convert + train
#   bash train_pi0fast.sh all \
#       --input "/path/to/data/*.hdf5" \
#       --output_root ~/lerobot_data/lekiwi_scene \
#       --steps 10000
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──
CONDA_ENV="lerobotpi0v2"
BASE_MODEL="lerobot/pi0fast-base"
REPO_ID="local/lekiwi_scene"
FPS=25
VCODEC="h264"              # h264 is universally supported
TASK_SOURCE="full_task"

MODE="full"                # "full" or "lora"
BATCH_SIZE=4
STEPS=30000
SAVE_FREQ=5000
LOG_FREQ=100
EVAL_FREQ=0                # 0 = no eval (no sim env on server)
LORA_R=16
NUM_WORKERS=4
OUTPUT_DIR=""
GRADIENT_CHECKPOINTING="true"

# VLA server defaults
VLA_PORT=8002
VLA_HOST="0.0.0.0"

# ── Parse command ──
COMMAND="${1:-help}"
shift || true

# ── Parse args ──
INPUT_PATTERN=""
OUTPUT_ROOT=""
DATA_ROOT=""
CHECKPOINT=""
EXTRA_TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)         INPUT_PATTERN="$2"; shift 2 ;;
        --output_root)   OUTPUT_ROOT="$2"; shift 2 ;;
        --data_root)     DATA_ROOT="$2"; shift 2 ;;
        --checkpoint)    CHECKPOINT="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --batch_size)    BATCH_SIZE="$2"; shift 2 ;;
        --steps)         STEPS="$2"; shift 2 ;;
        --save_freq)     SAVE_FREQ="$2"; shift 2 ;;
        --lora_r)        LORA_R="$2"; shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --base_model)    BASE_MODEL="$2"; shift 2 ;;
        --repo_id)       REPO_ID="$2"; shift 2 ;;
        --port)          VLA_PORT="$2"; shift 2 ;;
        --num_workers)   NUM_WORKERS="$2"; shift 2 ;;
        --no_gradient_checkpointing) GRADIENT_CHECKPOINTING="false"; shift ;;
        --extra)         EXTRA_TRAIN_ARGS="$2"; shift 2 ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Activate conda ──
activate_env() {
    # Server-specific conda paths
    export PATH="/home/jovyan/yes/bin:/home/jovyan/miniconda3/bin:$PATH"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "[OK] conda env: $CONDA_ENV ($(python --version))"
}

# ── Step 1: Convert HDF5 → LeRobot v3 ──
do_convert() {
    if [[ -z "$INPUT_PATTERN" ]]; then
        echo "ERROR: --input required for convert"
        exit 1
    fi
    if [[ -z "$OUTPUT_ROOT" ]]; then
        echo "ERROR: --output_root required for convert"
        exit 1
    fi

    activate_env

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CONVERTER="${SCRIPT_DIR}/../convert_hdf5_to_lerobot_v3.py"

    if [[ ! -f "$CONVERTER" ]]; then
        echo "ERROR: converter not found at $CONVERTER"
        echo "Make sure the repo is synced (git pull)"
        exit 1
    fi

    echo "══════════════════════════════════════════════════════════════"
    echo "Converting HDF5 → LeRobot v3"
    echo "  input:       $INPUT_PATTERN"
    echo "  output_root: $OUTPUT_ROOT"
    echo "  repo_id:     $REPO_ID"
    echo "  vcodec:      $VCODEC"
    echo "  task_source: $TASK_SOURCE"
    echo "══════════════════════════════════════════════════════════════"

    python "$CONVERTER" \
        --input $INPUT_PATTERN \
        --output_root "$OUTPUT_ROOT" \
        --repo_id "$REPO_ID" \
        --fps "$FPS" \
        --vcodec "$VCODEC" \
        --task_source "$TASK_SOURCE" \
        --overwrite \
        --skip_episodes_without_images

    echo "[OK] Conversion complete: $OUTPUT_ROOT"
}

# ── Step 2: Train Pi0-FAST ──
do_train() {
    if [[ -z "$DATA_ROOT" ]]; then
        if [[ -n "$OUTPUT_ROOT" ]]; then
            DATA_ROOT="$OUTPUT_ROOT"
        else
            echo "ERROR: --data_root required for train"
            exit 1
        fi
    fi

    activate_env

    # Install peft if using LoRA
    if [[ "$MODE" == "lora" ]]; then
        if ! python -c "import peft" 2>/dev/null; then
            echo "[INFO] Installing peft for LoRA fine-tuning..."
            pip install peft -q
        fi
    fi

    # Generate output dir with timestamp
    if [[ -z "$OUTPUT_DIR" ]]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="outputs/train/pi0fast_lekiwi_${MODE}_${TIMESTAMP}"
    fi

    # ── Patch base model config for our robot ──
    # The pretrained pi0fast-base has hardcoded input_features for 3 cameras
    # (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb) and 32D state.
    # We clear input/output_features so make_policy() infers them from our dataset
    # (2 cameras: front+wrist, 9D state, 9D action).
    LOCAL_MODEL="${BASE_MODEL}"
    if [[ "$BASE_MODEL" == lerobot/* ]] || [[ ! -d "$BASE_MODEL" ]]; then
        LOCAL_MODEL="./pi0fast_base_lekiwi"
        if [[ ! -f "${LOCAL_MODEL}/model.safetensors" ]]; then
            echo "[INFO] Downloading and patching base model for LeKiwi..."
            python -c "
from huggingface_hub import snapshot_download
import json, os

local_dir = snapshot_download('${BASE_MODEL}', local_dir='${LOCAL_MODEL}')
config_path = os.path.join(local_dir, 'config.json')
with open(config_path) as f:
    cfg = json.load(f)

# Clear features → make_policy() will infer from dataset
cfg['input_features'] = {}
cfg['output_features'] = {}

with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=4)

print(f'[OK] Patched config: input_features={{}}, output_features={{}}')
print(f'     Weights preserved at {local_dir}')
"
        else
            echo "[INFO] Using existing patched model at ${LOCAL_MODEL}"
        fi

        # Fix tokenizer: physical-intelligence/fast → lerobot/fast-action-tokenizer
        if [[ -f "${LOCAL_MODEL}/policy_preprocessor.json" ]]; then
            python -c "
import json
for fname in ['policy_preprocessor.json', 'policy_postprocessor.json']:
    fpath = '${LOCAL_MODEL}/' + fname
    try:
        with open(fpath) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        continue
    changed = False
    for step in cfg.get('steps', []):
        c = step.get('config', {})
        if c.get('action_tokenizer_name') == 'physical-intelligence/fast':
            c['action_tokenizer_name'] = 'lerobot/fast-action-tokenizer'
            changed = True
    if changed:
        with open(fpath, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f'[OK] Fixed tokenizer in {fname}')
"
        fi
    fi

    echo "══════════════════════════════════════════════════════════════"
    echo "Training Pi0-FAST"
    echo "  mode:        $MODE"
    echo "  base_model:  $LOCAL_MODEL (patched for LeKiwi)"
    echo "  data_root:   $DATA_ROOT"
    echo "  repo_id:     $REPO_ID"
    echo "  batch_size:  $BATCH_SIZE"
    echo "  steps:       $STEPS"
    echo "  output_dir:  $OUTPUT_DIR"
    if [[ "$MODE" == "lora" ]]; then
        echo "  lora_r:      $LORA_R"
    fi
    echo "══════════════════════════════════════════════════════════════"

    # Build training command
    TRAIN_CMD=(
        lerobot-train
        --dataset.repo_id="$REPO_ID"
        --dataset.root="$DATA_ROOT"
        --policy.path="$LOCAL_MODEL"
        --policy.repo_id="local/pi0fast_lekiwi_${MODE}"
        --batch_size="$BATCH_SIZE"
        --steps="$STEPS"
        --save_freq="$SAVE_FREQ"
        --log_freq="$LOG_FREQ"
        --eval_freq="$EVAL_FREQ"
        --num_workers="$NUM_WORKERS"
        --output_dir="$OUTPUT_DIR"
    )

    # Pi0-FAST specific: smaller action chunks (LIBERO reference uses 10, not default 50)
    TRAIN_CMD+=(
        --policy.chunk_size=10
        --policy.n_action_steps=10
        --policy.max_action_tokens=256
    )

    # LoRA-specific args
    if [[ "$MODE" == "lora" ]]; then
        TRAIN_CMD+=(
            --peft.method_type=LORA
            --peft.r="$LORA_R"
            --peft.target_modules=all-linear
        )
    fi

    # Gradient checkpointing for memory efficiency
    if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
        TRAIN_CMD+=(--policy.gradient_checkpointing=true)
    fi

    # dtype = bfloat16 for A100
    TRAIN_CMD+=(--policy.dtype=bfloat16)

    # Disable action token prefix validation (base model can produce garbage initially)
    TRAIN_CMD+=(--policy.validate_action_token_prefix=false)

    # Extra args
    if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
        TRAIN_CMD+=($EXTRA_TRAIN_ARGS)
    fi

    echo "[CMD] ${TRAIN_CMD[*]}"
    echo ""

    PYTHONUNBUFFERED=1 "${TRAIN_CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}.log"

    echo ""
    echo "[OK] Training complete"
    echo "  output:     $OUTPUT_DIR"
    echo "  log:        ${OUTPUT_DIR}.log"
    echo "  checkpoint: $OUTPUT_DIR/checkpoints/last/"
}

# ── Step 3: Serve fine-tuned model ──
do_serve() {
    if [[ -z "$CHECKPOINT" ]]; then
        echo "ERROR: --checkpoint required for serve"
        echo "Example: --checkpoint outputs/train/pi0fast_lekiwi_full_xxx/checkpoints/last/pretrained_model"
        exit 1
    fi

    activate_env

    # Resolve checkpoint path: if pointing to a step dir (containing pretrained_model/),
    # auto-append pretrained_model/
    MODEL_PATH="$CHECKPOINT"
    if [[ -d "${CHECKPOINT}/pretrained_model" ]]; then
        MODEL_PATH="${CHECKPOINT}/pretrained_model"
        echo "[INFO] Auto-resolved to ${MODEL_PATH}"
    fi

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SERVER="${SCRIPT_DIR}/vla_inference_server.py"

    echo "══════════════════════════════════════════════════════════════"
    echo "Serving fine-tuned Pi0-FAST"
    echo "  model_path: $MODEL_PATH"
    echo "  port:       $VLA_PORT"
    echo "══════════════════════════════════════════════════════════════"

    python "$SERVER" \
        --model "$MODEL_PATH" \
        --port "$VLA_PORT" \
        --host "$VLA_HOST"
}

# ── All: convert + train ──
do_all() {
    do_convert
    DATA_ROOT="$OUTPUT_ROOT"
    do_train
}

# ── Help ──
do_help() {
    cat <<'HELP'
Pi0-FAST Fine-tuning Pipeline for LeKiwi
=========================================

Commands:
  convert   Convert HDF5 → LeRobot v3 dataset
  train     Fine-tune Pi0-FAST on LeRobot v3 dataset
  serve     Serve fine-tuned model via VLA inference server
  all       Convert + Train in one step
  help      Show this message

Typical workflow:
  1. Collect data on 3090:
     python vllm/record_teleop_scene.py \
       --skill approach_and_grasp \
       --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
       --resip_checkpoint checkpoints/resip/resip_best.pt \
       --instruction "approach and grasp the medicine bottle"

  2. Transfer HDF5 to server:
     scp -P 30179 -i ~/.ssh/private.pem \
       teleop_scene_data/*.hdf5 \
       jovyan@218.148.55.186:~/data/lekiwi_hdf5/

  3. Convert + Train on server:
     bash train_pi0fast.sh all \
       --input "~/data/lekiwi_hdf5/*.hdf5" \
       --output_root ~/lerobot_data/lekiwi_scene \
       --steps 10000

  4. Serve fine-tuned model:
     bash train_pi0fast.sh serve \
       --checkpoint outputs/train/pi0fast_lekiwi_full_xxx/checkpoints/last

  5. Run evaluation on 3090:
     python vllm/run_full_task.py \
       --user_command "find the medicine bottle and place it next to the red cup" \
       --object_usd /path/to/model.usd \
       --dest_object_usd /path/to/cup.usd

Options:
  --input <glob>         HDF5 input file(s) for convert
  --output_root <dir>    Output directory for LeRobot v3 dataset
  --data_root <dir>      LeRobot v3 dataset path for training
  --checkpoint <dir>     Fine-tuned model checkpoint for serving
  --mode full|lora       Training mode (default: full)
  --batch_size <N>       Batch size (default: 4 for full, 8 for LoRA)
  --steps <N>            Training steps (default: 30000)
  --save_freq <N>        Checkpoint save frequency (default: 5000)
  --lora_r <N>           LoRA rank (default: 16)
  --base_model <path>    Pretrained model (default: lerobot/pi0fast-base)
  --repo_id <id>         Dataset repo ID (default: local/lekiwi_scene)
  --output_dir <dir>     Training output directory (auto-generated if omitted)
  --port <N>             VLA server port (default: 8002)
  --num_workers <N>      Dataloader workers (default: 4)
  --no_gradient_checkpointing  Disable gradient checkpointing
  --extra "<args>"       Extra lerobot CLI args (quoted string)

Memory estimates (A100 40GB):
  LoRA (default):  ~12-15 GB → batch_size=8 OK
  Full fine-tune:  ~25-35 GB → batch_size=2-4, gradient_checkpointing required
HELP
}

# ── Dispatch ──
case "$COMMAND" in
    convert) do_convert ;;
    train)   do_train ;;
    serve)   do_serve ;;
    all)     do_all ;;
    help|*)  do_help ;;
esac
