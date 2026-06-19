#!/usr/bin/env bash
# =============================================================================
# 데이터 생성 파이프라인 — 환경 설치 (이 파일 하나로 순서대로 전부)
#
#   bash setup.sh          # 데스크탑: 텔레옵·BC·RL·eval·record_scene·변환
#   bash setup.sh server   # A100 서버: BC/RL 학습 env(rl_train) — 선택
#
# 검증 환경(2026-06-18): Ubuntu 22.04.5 · Isaac Sim 5.0.0 · PyTorch 2.7.0+cu128
#   데스크탑 env_isaaclab(Isaac Lab 0.44.9) / lerobot050(LeRobot 0.5.0) · 서버 rl_train(Isaac Lab 2.2.0)
#   molmospaces(ProcTHOR scene) · ROS2 Humble
# 전제: RTX GPU(데스크탑) 또는 A100(서버) + NVIDIA 드라이버.
#       로봇·물체 USD는 「Lekiwi 환경 구축 및 원격조종 물체 스폰」 매뉴얼에서 준비.
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
MODE="${1:-desktop}"

# Miniconda 설치 (없으면 — 데스크탑·서버 공통)
if ! command -v conda >/dev/null 2>&1 && [ ! -x "$CONDA_BASE/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_BASE"
fi
# conda 활성화
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ════════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "server" ]; then
echo "==== A100 서버: rl_train (BC/RL 학습) ===="

  # Isaac Sim 런타임 시스템 라이브러리
  sudo apt-get update -qq && sudo apt-get install -y -qq git curl wget build-essential libgl1-mesa-glx libglib2.0-0 libegl1 libsm6 libxext6 libxrender-dev || true

  # rl_train conda env 생성 (Python 3.11)
  conda create -n rl_train python=3.11 -y
  conda activate rl_train

  # PyTorch 2.7.0 + CUDA 12.8 (Isaac Sim 의존, 먼저 설치)
  pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

  # Isaac Sim 5.0.0 (pip, NVIDIA 인덱스)
  pip install isaacsim==5.0.0.0 --extra-index-url https://pypi.nvidia.com

  # Isaac Lab 2.2.0 클론 (없으면)
  [ -d "$ISAACLAB_DIR/.git" ] || git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  # Isaac Lab 2.2.0 체크아웃 + 확장 editable 설치
  ( cd "$ISAACLAB_DIR" && { git checkout v2.2.0 2>/dev/null || git checkout 46dff13; } && bash isaaclab.sh -i )

  # RL/BC 의존성 (skrl, diffusion policy, 데이터 IO)
  pip install skrl==1.4.3 "diffusers>=0.36.0" "h5py>=3.10" tensorboard "gymnasium>=1.0" pillow scipy pydantic

  # 정확 버전 고정 (lockfile — 검증 시점과 동일)
  pip install -r "$SCRIPT_DIR/secondserver/env_rl_train.lock.txt" --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128 || true

  echo "[rl_train] 완료. 학습 예: PYTHONUNBUFFERED=1 python train_resip.py --skill approach_and_grasp ... --headless"
  exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
echo "==== 데스크탑: env_isaaclab + lerobot050 + molmospaces + ROS2 ===="

# Isaac Lab 레포 클론 (없으면, 고정 커밋 = Isaac Lab 0.44.9)
[ -f "$ISAACLAB_DIR/isaaclab.sh" ] || { git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"; ( cd "$ISAACLAB_DIR" && git checkout 46dff135f44683f031edf346e544fcfd8456b2bb ); }

# env_isaaclab conda env 생성 (Python 3.11)
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

# Isaac Sim 5.0.0 설치 (pip, NVIDIA 인덱스)
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# Isaac Lab 확장 + RL 라이브러리 editable 설치
( cd "$ISAACLAB_DIR" && ./isaaclab.sh -i )

# 데이터젠 의존성 (diffusion policy=diffusers, VLA client=requests, 키보드 텔레옵=pynput)
pip install "diffusers>=0.36" requests numpy pynput

# 정확 버전 고정 (lockfile — isaacsim/torch는 인덱스 필요)
pip install -r "$SCRIPT_DIR/export/env_isaaclab.lock.txt" --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128 || true

# molmospaces(ProcTHOR scene, 10주차) 클론 (없으면, 고정 커밋)
[ -d "$HOME/molmospaces/.git" ] || { git clone https://github.com/allenai/molmospaces.git "$HOME/molmospaces"; ( cd "$HOME/molmospaces" && git checkout 0939e18b14a3650ae8094ff516a4e244fca82198 ); }
# molmospaces Isaac 확장 editable 설치
( cd "$HOME/molmospaces/molmo_spaces_isaac" && pip install -e ".[dev]" )
# ProcTHOR scene 다운로드 (scene 8·3614·100·9999·1302 → assets/usd)
( cd "$HOME/molmospaces" && python download_few_scenes.py )

# lerobot050 conda env 생성 (Python 3.12) — HDF5 → LeRobot v3 변환용
conda create -n lerobot050 python=3.12 -y
conda activate lerobot050
# LeRobot 0.5.0 + 변환 의존성 (lockfile로 정확 버전)
pip install -r "$SCRIPT_DIR/export/env_lerobot050.lock.txt"

# ROS2 Humble (텔레옵 브리지, 없으면 설치)
if [ ! -d /opt/ros/humble ]; then
  sudo apt update && sudo apt install -y software-properties-common curl
  sudo add-apt-repository -y universe
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt update && sudo apt install -y ros-humble-desktop
fi

# 검증 (설치된 버전 출력)
conda run -n env_isaaclab python -c "import isaacsim,torch; print('  env_isaaclab : torch', torch.__version__, '| isaacsim', isaacsim.__version__)" 2>/dev/null || echo "  env_isaaclab 확인 실패 (conda activate 후 재시도)"
conda run -n lerobot050  python -c "import lerobot; print('  lerobot050   : lerobot', lerobot.__version__)" 2>/dev/null || echo "  lerobot050 확인 실패"
echo "데스크탑 환경 설치 완료. 셸마다: export OMNI_KIT_ACCEPT_EULA=YES LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd"
echo "(기존 데스크탑 env를 그대로 복제하려면 export/setup_env.sh 의 MODE=mirror 참고)"
