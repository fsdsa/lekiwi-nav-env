#!/usr/bin/env bash
###############################################################################
# setup_env.sh - 새 서버에 Isaac Sim + Isaac Lab + rl_train conda 환경 설치
#
# 사용법:
#   bash setup_env.sh
#
# 전제 조건:
#   - Ubuntu 22.04 LTS
#   - NVIDIA Driver >= 535 (CUDA 12.x)
#   - nvidia-smi 정상 작동
#   - 인터넷 연결
#
# 설치되는 것:
#   1. Miniconda (없으면)
#   2. Isaac Sim 5.0.0 (pip)
#   3. Isaac Lab 2.2.0 (git clone + editable install)
#   4. rl_train conda 환경 (Python 3.11, PyTorch 2.7.0+cu128)
#   5. 추가 pip 의존성 (skrl, diffusers, etc.)
#
# 현재 환경 기준 버전:
#   Isaac Sim: 5.0.0-rc.45
#   Isaac Lab: 2.2.0
#   Python:    3.11.14
#   PyTorch:   2.7.0+cu128
#   skrl:      1.4.3
#   diffusers: 0.36.0
###############################################################################
set -euo pipefail

echo "=========================================="
echo "  LeKiwi ResiP Training - Server Setup"
echo "=========================================="

# ------------------------------------------------------------------
# 0. 기본 시스템 패키지 확인
# ------------------------------------------------------------------
echo ""
echo "[0/6] 시스템 확인..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi 없음. NVIDIA Driver를 먼저 설치하세요."
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)
echo "  GPU: $GPU_INFO"

# 필수 시스템 패키지
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs curl wget build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libegl1 libx11-6 libxt6 2>/dev/null || true
echo "  시스템 패키지 OK"

# ------------------------------------------------------------------
# 1. Miniconda 설치 (없으면)
# ------------------------------------------------------------------
echo ""
echo "[1/6] Miniconda 확인..."

CONDA_DIR="$HOME/miniconda3"
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "  Miniconda 설치 중..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "  Miniconda 설치 완료: $CONDA_DIR"
else
    echo "  Miniconda 이미 존재: $CONDA_DIR"
fi

# conda 초기화
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda config --set auto_activate_base false 2>/dev/null || true

# ------------------------------------------------------------------
# 2. Conda 환경 생성: rl_train (Python 3.11)
# ------------------------------------------------------------------
echo ""
echo "[2/6] Conda 환경 생성 (rl_train, Python 3.11)..."

if conda env list | grep -q "rl_train"; then
    echo "  rl_train 환경 이미 존재. 스킵."
else
    conda create -n rl_train python=3.11 -y
    echo "  rl_train 환경 생성 완료."
fi

conda activate rl_train

# ------------------------------------------------------------------
# 3. Isaac Sim 5.0 설치 (pip)
# ------------------------------------------------------------------
echo ""
echo "[3/6] Isaac Sim 5.0 설치..."

ISAACSIM_DIR="$HOME/isaacsim"

# Isaac Sim pip 설치 확인
if python -c "import isaacsim" 2>/dev/null; then
    ISAACSIM_VER=$(python -c "import isaacsim; print(isaacsim.__version__)" 2>/dev/null || echo "unknown")
    echo "  Isaac Sim 이미 설치됨: $ISAACSIM_VER"
else
    echo "  Isaac Sim pip 패키지 설치 중..."
    echo "  (이 과정은 10-20분 소요될 수 있습니다)"

    # PyTorch 먼저 설치 (Isaac Sim이 의존)
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

    # Isaac Sim pip 설치
    # 참고: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_pip.html
    pip install isaacsim==5.0.0.0 \
        --extra-index-url https://pypi.nvidia.com

    echo "  Isaac Sim 설치 완료."
fi

# isaacsim 설치 경로 확인 및 심볼릭 링크
ISAACSIM_SITE=$(python -c "import isaacsim, os; print(os.path.dirname(isaacsim.__file__))" 2>/dev/null || echo "")
if [ -n "$ISAACSIM_SITE" ] && [ ! -d "$ISAACSIM_DIR" ]; then
    ln -sf "$ISAACSIM_SITE" "$ISAACSIM_DIR"
    echo "  심볼릭 링크: $ISAACSIM_DIR -> $ISAACSIM_SITE"
fi

# ------------------------------------------------------------------
# 4. Isaac Lab 2.2.0 설치
# ------------------------------------------------------------------
echo ""
echo "[4/6] Isaac Lab 2.2.0 설치..."

ISAACLAB_DIR="$HOME/IsaacLab"

if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "  Isaac Lab 클론 중..."
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
    cd "$ISAACLAB_DIR"
    # 2.2.0 태그로 체크아웃
    git checkout v2.2.0 2>/dev/null || git checkout 46dff13
    echo "  Isaac Lab 클론 완료."
else
    echo "  Isaac Lab 이미 존재: $ISAACLAB_DIR"
    cd "$ISAACLAB_DIR"
fi

# _isaac_sim 심볼릭 링크 (pip 설치 시 자동 감지하지만 명시적 생성)
if [ -n "$ISAACSIM_SITE" ] && [ ! -L "$ISAACLAB_DIR/_isaac_sim" ]; then
    ln -sf "$ISAACSIM_SITE" "$ISAACLAB_DIR/_isaac_sim"
    echo "  _isaac_sim 심볼릭 링크 생성."
fi

# Isaac Lab 확장 설치 (editable mode)
echo "  Isaac Lab 확장 설치 중..."
if [ -f "$ISAACLAB_DIR/isaaclab.sh" ]; then
    cd "$ISAACLAB_DIR"
    # isaaclab.sh -i: 모든 소스 확장 editable 설치
    bash isaaclab.sh -i 2>&1 | tail -5
    echo "  Isaac Lab 확장 설치 완료."
else
    echo "  WARNING: isaaclab.sh 없음. 수동으로 확장을 설치하세요:"
    echo "    cd $ISAACLAB_DIR && pip install -e source/isaaclab"
    echo "    pip install -e source/isaaclab_assets"
    echo "    pip install -e source/isaaclab_rl"
    echo "    pip install -e source/isaaclab_tasks"
fi

# ------------------------------------------------------------------
# 5. 추가 pip 의존성 설치
# ------------------------------------------------------------------
echo ""
echo "[5/6] 추가 pip 의존성 설치..."

# 핵심 RL/BC 의존성
pip install \
    "skrl==1.4.3" \
    "diffusers>=0.36.0" \
    "h5py>=3.10" \
    "tensorboard>=2.15" \
    "gymnasium>=1.0" \
    2>&1 | tail -3

# 유틸 의존성
pip install \
    "pillow>=11.0" \
    "scipy>=1.11" \
    "pydantic>=2.0" \
    2>&1 | tail -3

echo "  pip 의존성 설치 완료."

# ------------------------------------------------------------------
# 6. 프로젝트 디렉토리 구조 생성
# ------------------------------------------------------------------
echo ""
echo "[6/6] 프로젝트 디렉토리 구조 생성..."

PROJECT_DIR="$ISAACLAB_DIR/scripts/lekiwi_nav_env"
mkdir -p "$PROJECT_DIR/calibration"
mkdir -p "$PROJECT_DIR/checkpoints/dp_bc_small"
mkdir -p "$PROJECT_DIR/checkpoints/dp_bc_skill3_55d_fixed_1e-4"
mkdir -p "$PROJECT_DIR/checkpoints/resip_s3_v19"
mkdir -p "$PROJECT_DIR/backup/appoachandlift"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$HOME/Downloads"
mkdir -p "$HOME/isaac-objects/mujoco_scanned_objects/models"

echo "  디렉토리 구조 생성 완료."

# ------------------------------------------------------------------
# 검증
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  설치 검증"
echo "=========================================="

PASS=0
FAIL=0

check() {
    if eval "$2" &>/dev/null; then
        echo "  [OK] $1"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $1"
        FAIL=$((FAIL + 1))
    fi
}

check "Python 3.11"      "python --version 2>&1 | grep -q '3.11'"
check "PyTorch"           "python -c 'import torch; assert torch.cuda.is_available()'"
check "PyTorch CUDA"      "python -c 'import torch; print(torch.version.cuda)'"
check "Isaac Sim"         "python -c 'import isaacsim'"
check "Isaac Lab"         "python -c 'import isaaclab'"
check "skrl"              "python -c 'import skrl; assert skrl.__version__ >= \"1.4\"'"
check "diffusers"         "python -c 'import diffusers'"
check "h5py"              "python -c 'import h5py'"
check "gymnasium"         "python -c 'import gymnasium'"

echo ""
echo "  결과: $PASS 통과, $FAIL 실패"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  WARNING: 일부 검증 실패. 위 [FAIL] 항목을 확인하세요."
    echo "  Isaac Sim pip 설치가 실패한 경우:"
    echo "    - NVIDIA Driver >= 535 확인"
    echo "    - pip install isaacsim==5.0.0.0 --extra-index-url https://pypi.nvidia.com"
fi

echo ""
echo "=========================================="
echo "  다음 단계"
echo "=========================================="
echo ""
echo "  1. 파일 전송 (로컬에서 실행):"
echo "     bash secondserver/transfer_files.sh <user> <host> <port>"
echo ""
echo "  2. 환경변수 설정:"
echo "     export LEKIWI_USD_PATH=\$HOME/Downloads/lekiwi_robot.usd"
echo ""
echo "  3. 학습 시작:"
echo "     bash secondserver/run_training.sh"
echo ""
