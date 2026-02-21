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
# 서버 CUDA 버전 확인: nvidia-smi → "CUDA Version" 확인
# Desktop은 CUDA 13.1 / PyTorch cu128
# 서버도 CUDA 12.8+ 이면 cu128, 아니면 cu124 사용
CUDA_INDEX="https://download.pytorch.org/whl/cu128"
SKRL_VERSION="1.4.3"

# Isaac Sim pip 버전 (데스크탑과 맞춰야 함)
# 데스크탑: pip show isaacsim → 5.0.0.0
ISAACSIM_PIP_VERSION="5.0.0.0"

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
    # 데스크탑: git clone + ./isaaclab.sh --install (editable, v2.2.0 태그, 내부 버전 0.44.9)
    ISAACLAB_DIR="$HOME/IsaacLab"
    if [ ! -d "$ISAACLAB_DIR/.git" ]; then
        echo "  Isaac Lab 클론..."
        git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
    fi
    cd "$ISAACLAB_DIR"
    git checkout v2.2.0
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
    echo "  검증: bash feedback/setup_server_env.sh verify"
}

# ======================== Verify ========================
verify_all() {
    echo ""
    echo "=========================================="
    echo " 설치 검증"
    echo "=========================================="

    FAIL=0

    # --- VLM 환경 ---
    echo ""
    echo "[1/6] VLM 환경 (inference)..."
    if conda activate inference 2>/dev/null; then
        python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print('  Qwen2.5-VL import: OK')
" 2>/dev/null || { echo "  Qwen2.5-VL import: FAIL"; FAIL=1; }

        # 모델 캐시 확인
        if [ -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct" ]; then
            echo "  모델 캐시: OK"
        else
            echo "  모델 캐시: FAIL (huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct)"
            FAIL=1
        fi
        conda deactivate
    else
        echo "  conda inference: FAIL (환경 없음)"
        FAIL=1
    fi

    # --- RL 환경 ---
    echo ""
    echo "[2/6] RL/BC 환경 (rl_train)..."
    if conda activate rl_train 2>/dev/null; then
        # PyTorch + CUDA
        python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  PyTorch {torch.__version__}: OK (GPU: {torch.cuda.get_device_name(0)})')
" 2>/dev/null || { echo "  PyTorch/CUDA: FAIL"; FAIL=1; }

        # Isaac Sim
        python -c "
import isaacsim
print(f'  Isaac Sim {isaacsim.__version__}: OK')
" 2>/dev/null || { echo "  Isaac Sim: FAIL"; FAIL=1; }

        # skrl
        python -c "
import skrl
print(f'  skrl {skrl.__version__}: OK')
" 2>/dev/null || { echo "  skrl: FAIL"; FAIL=1; }

        # h5py
        python -c "
import h5py
print(f'  h5py {h5py.version.version}: OK')
" 2>/dev/null || { echo "  h5py: FAIL"; FAIL=1; }

        conda deactivate
    else
        echo "  conda rl_train: FAIL (환경 없음)"
        FAIL=1
    fi

    # --- 전송된 파일 ---
    echo ""
    echo "[3/6] 로봇 USD..."
    if [ -f "$HOME/Downloads/lekiwi_robot.usd" ]; then
        echo "  ~/Downloads/lekiwi_robot.usd: OK ($(du -h ~/Downloads/lekiwi_robot.usd | cut -f1))"
    else
        echo "  ~/Downloads/lekiwi_robot.usd: FAIL"
        FAIL=1
    fi

    echo ""
    echo "[4/6] 물체 USD..."
    OBJ_COUNT=$(find ~/isaac-objects -name "*.usd" 2>/dev/null | wc -l)
    if [ "$OBJ_COUNT" -gt 100 ]; then
        echo "  ~/isaac-objects/: OK ($OBJ_COUNT USD files, $(du -sh ~/isaac-objects/ 2>/dev/null | cut -f1))"
    else
        echo "  ~/isaac-objects/: FAIL ($OBJ_COUNT files, 1000+ expected)"
        FAIL=1
    fi

    # 물체 경로 심링크 확인
    if [ -d "/home/yubin/isaac-objects" ] || [ -L "/home/yubin/isaac-objects" ]; then
        echo "  /home/yubin/isaac-objects 심링크: OK"
    else
        echo "  /home/yubin/isaac-objects 심링크: MISSING"
        echo "    → sudo mkdir -p /home/yubin && sudo ln -s /home/jovyan/isaac-objects /home/yubin/isaac-objects"
        echo "    → 또는: sed -i 's|/home/yubin/|/home/jovyan/|g' object_catalog.json object_catalog_all.json"
        FAIL=1
    fi

    echo ""
    echo "[5/6] 캘리브레이션 파일..."
    for f in ~/IsaacLab/calibration/tuned_dynamics.json \
             ~/IsaacLab/calibration/arm_limits_real2sim.json; do
        if [ -f "$f" ]; then
            echo "  $(basename $f): OK"
        else
            echo "  $(basename $f): FAIL"
            FAIL=1
        fi
    done

    echo ""
    echo "[6/6] 프로젝트 코드..."
    PY_COUNT=$(ls "$PROJECT_DIR"/*.py 2>/dev/null | wc -l)
    DEMO_COUNT=$(ls "$PROJECT_DIR"/demos/*.hdf5 2>/dev/null | wc -l)
    echo "  Python 파일: $PY_COUNT개"
    echo "  텔레옵 HDF5: $DEMO_COUNT개"

    for f in train_lekiwi.py train_bc.py models.py lekiwi_skill1_env.py \
             lekiwi_skill2_env.py lekiwi_robot_cfg.py aac_wrapper.py aac_ppo.py; do
        if [ ! -f "$PROJECT_DIR/$f" ]; then
            echo "  $f: MISSING"
            FAIL=1
        fi
    done

    if [ "$PY_COUNT" -gt 30 ] && [ "$DEMO_COUNT" -gt 0 ]; then
        echo "  프로젝트 파일: OK"
    else
        echo "  프로젝트 파일: FAIL (py=$PY_COUNT, demos=$DEMO_COUNT)"
        FAIL=1
    fi

    # --- 결과 ---
    echo ""
    echo "=========================================="
    if [ "$FAIL" -eq 0 ]; then
        echo " ALL CHECKS PASSED"
    else
        echo " SOME CHECKS FAILED — 위 FAIL 항목 확인"
    fi
    echo "=========================================="
    return $FAIL
}

# ======================== Main ========================
echo "=============================================="
echo " A100 Server Environment Setup"
echo " Mode: $MODE"
echo "=============================================="

case $MODE in
    vlm)    setup_vlm ;;
    rl)     setup_rl ;;
    all)    setup_vlm; setup_rl ;;
    verify) verify_all ;;
    *)
        echo "Usage: $0 [all|vlm|rl|verify]"
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
