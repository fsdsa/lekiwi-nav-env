#!/bin/bash
# ==============================================================
# A100 Server Environment Setup
# ==============================================================
# 1. VLM: SmolVLM-2 제거 → Qwen2.5-VL-7B-Instruct 설치
# 2. BC/RL 학습 환경: PyTorch + Isaac Sim (headless) + Isaac Lab + skrl
# ==============================================================
#
# 사용법:
#   chmod +x setup_server_env.sh
#   bash setup_server_env.sh          # 전체 설치
#   bash setup_server_env.sh vlm      # VLM 환경만
#   bash setup_server_env.sh rl       # BC/RL 환경만
#
# 사전 조건:
#   - conda 설치됨
#   - NVIDIA 드라이버 525+ (A100 호환)
#   - huggingface-cli login 완료
#
# ==============================================================

set -e

# ======================== Configuration ========================
QWEN_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
SMOLVLM_CACHE_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-2-2.2B-Instruct"

PYTHON_VERSION="3.11"
# 서버 CUDA 버전 확인: nvcc --version 또는 nvidia-smi
# cu118, cu121, cu124, cu128 중 서버에 맞는 것 선택
CUDA_INDEX="https://download.pytorch.org/whl/cu124"
SKRL_VERSION="1.4.3"

# Isaac Sim pip 버전 (데스크탑과 맞춰야 함)
# 데스크탑 확인: python -c "import isaacsim; print(isaacsim.__version__)"
ISAACSIM_PIP_VERSION="4.5.0"

# 프로젝트 경로
PROJECT_DIR="$HOME/IsaacLab/scripts/lekiwi_nav_env"

eval "$(conda shell.bash hook)"
MODE=${1:-all}

# ======================== Part 1: VLM ========================
setup_vlm() {
    echo ""
    echo "=========================================="
    echo " Part 1: VLM (SmolVLM-2 → Qwen2.5-VL-7B)"
    echo "=========================================="

    # --- 1-1. SmolVLM-2 캐시 삭제 ---
    if [ -d "$SMOLVLM_CACHE_DIR" ]; then
        echo "[1/4] SmolVLM-2 모델 캐시 삭제..."
        rm -rf "$SMOLVLM_CACHE_DIR"
        echo "  삭제 완료"
    else
        echo "[1/4] SmolVLM-2 캐시 없음 (skip)"
    fi

    # --- 1-2. 기존 inference 환경 제거 ---
    echo "[2/4] 기존 inference 환경 제거..."
    conda deactivate 2>/dev/null || true
    conda env remove -n inference -y 2>/dev/null || true

    # --- 1-3. 새 inference 환경 생성 ---
    echo "[3/4] Qwen2.5-VL inference 환경 생성..."
    conda create -n inference python=$PYTHON_VERSION -y
    conda activate inference

    pip install --upgrade pip
    pip install torch torchvision --index-url $CUDA_INDEX
    pip install "transformers>=4.49.0" accelerate qwen-vl-utils
    pip install huggingface_hub

    # flash-attn: A100에서 추론 속도 ~2x 향상. 빌드 실패 시 없어도 동작함
    echo "  flash-attn 설치 시도 (optional)..."
    pip install flash-attn --no-build-isolation 2>/dev/null || \
        echo "  [WARN] flash-attn 빌드 실패. CUDA toolkit/gcc 확인. 없어도 정상 동작."

    # --- 1-4. 모델 다운로드 ---
    echo "[4/4] $QWEN_MODEL_ID 다운로드..."
    huggingface-cli download $QWEN_MODEL_ID

    # --- 검증 ---
    echo ""
    echo "  설치 검증..."
    python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print('Qwen2.5-VL import OK')
print('Model ID: $QWEN_MODEL_ID')
print('VRAM (bf16): ~15GB')
"

    conda deactivate
    echo ""
    echo "VLM 설치 완료"
    echo "  활성화: conda activate inference"
    echo "  테스트: python -c \"from transformers import Qwen2_5_VLForConditionalGeneration; print('OK')\""
}

# ======================== Part 2: BC/RL ========================
setup_rl() {
    echo ""
    echo "=========================================="
    echo " Part 2: BC/RL Training Environment"
    echo "=========================================="
    echo ""
    echo "  BC 학습: PyTorch만 필요 (Isaac Sim 불필요)"
    echo "  RL 학습: Isaac Sim headless + Isaac Lab + skrl 필요"
    echo ""

    # --- 2-1. conda 환경 생성 ---
    echo "[1/5] rl_train conda 환경 생성..."
    conda deactivate 2>/dev/null || true
    conda env remove -n rl_train -y 2>/dev/null || true
    conda create -n rl_train python=$PYTHON_VERSION -y
    conda activate rl_train

    pip install --upgrade pip

    # --- 2-2. PyTorch ---
    echo "[2/5] PyTorch 설치..."
    pip install torch torchvision --index-url $CUDA_INDEX

    # GPU 확인
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

    # --- 2-3. Isaac Sim (headless, pip) ---
    echo "[3/5] Isaac Sim headless 설치..."
    echo "  버전: $ISAACSIM_PIP_VERSION"
    echo "  (데스크탑과 동일 버전이어야 함)"
    echo ""
    # A100은 RT Core 없음 → 렌더링 불가, physics headless만 사용
    # RL 학습은 state-only (렌더링 없음)이므로 문제없음
    pip install "isaacsim==${ISAACSIM_PIP_VERSION}" \
        isaacsim-extscache-physics \
        isaacsim-extscache-kit-sdk \
        --extra-index-url https://pypi.nvidia.com

    # --- 2-4. Isaac Lab ---
    echo "[4/5] Isaac Lab 설치..."
    ISAACLAB_DIR="$HOME/IsaacLab"
    if [ ! -d "$ISAACLAB_DIR" ]; then
        echo "  Isaac Lab 클론..."
        git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
    fi
    cd "$ISAACLAB_DIR"
    # Isaac Lab 버전 태그가 있다면 checkout (예: v0.44.9)
    # git checkout v0.44.9 2>/dev/null || true
    ./isaaclab.sh --install

    # --- 2-5. 추가 패키지 ---
    echo "[5/5] 추가 패키지 설치 (skrl, h5py, etc.)..."
    pip install "skrl==${SKRL_VERSION}" h5py tensorboard scipy matplotlib

    # --- 프로젝트 파일 확인 ---
    if [ -d "$PROJECT_DIR" ]; then
        echo ""
        echo "  프로젝트 디렉토리 확인: $PROJECT_DIR"
    else
        echo ""
        echo "  [ACTION] 프로젝트 파일을 서버로 복사해야 합니다:"
        echo "    Desktop에서:"
        echo "    scp -r ~/IsaacLab/scripts/lekiwi_nav_env/ server:~/IsaacLab/scripts/"
    fi

    conda deactivate
    echo ""
    echo "RL/BC 환경 설치 완료"
    echo "  활성화: conda activate rl_train"
}

# ======================== Main ========================
echo "=============================================="
echo " A100 Server Environment Setup"
echo " Mode: $MODE"
echo "=============================================="

case $MODE in
    vlm)  setup_vlm ;;
    rl)   setup_rl ;;
    all)  setup_vlm; setup_rl ;;
    *)
        echo "Usage: $0 [all|vlm|rl]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo " Setup Complete!"
echo "=============================================="
echo ""
echo "서버 conda 환경:"
echo "  inference  : Qwen2.5-VL-7B-Instruct (VLM, ~15GB VRAM)"
echo "  rl_train   : Isaac Sim + Isaac Lab + skrl (BC/RL 학습)"
echo "  lerobotpi0 : pi0-FAST + LeRobot (VLA 파인튜닝, 기존 유지)"
echo "  groot      : GR00T N1.6 (VLA 파인튜닝, 기존 유지)"
echo ""
echo "워크플로우:"
echo "  1. Desktop: 텔레옵 → scp demos/ → 서버"
echo "  2. 서버:    conda activate rl_train"
echo "             python train_bc.py --demo_dir demos/ --expected_obs_dim 30"
echo "             python train_lekiwi.py --skill approach_and_grasp --num_envs 2048 --headless"
echo "  3. 서버 → Desktop: scp checkpoint → collect_demos.py (렌더링)"
echo "  4. Desktop → 서버: scp HDF5 → 변환 → VLA 파인튜닝"
