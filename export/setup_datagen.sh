#!/usr/bin/env bash
# =============================================================================
# 데이터 생성 파이프라인 — 데스크탑 환경 전체 설치 (한 번에)
#
# 이 스크립트만 실행하면 현재 데스크탑과 동일한 환경이 재현된다.
#   [1] env_isaaclab  — Isaac Sim 5.0 + Isaac Lab 0.44.9 + BC/RL/eval/record_scene
#                       (Python 3.11, PyTorch 2.7.0+cu128)   ← export/setup_env.sh
#   [2] lerobot050    — HDF5 → LeRobot v3 변환 (Python 3.12, LeRobot 0.5.0)
#                       ← export/setup_lerobot050_env.sh
#   [3] ROS2 Humble   — 텔레옵 브리지 (이미 있으면 건너뜀)
#
# 검증 환경(2026-06-18 실측): Ubuntu 22.04.5 · Isaac Sim 5.0.0 · Isaac Lab 0.44.9
#                            · PyTorch 2.7.0+cu128 · LeRobot 0.5.0 · ROS2 Humble
#
# BC/RL 학습을 A100 서버에서 돌릴 경우: 서버에서 secondserver/setup_env.sh 로 rl_train 설치.
#
# 사용:
#   bash export/setup_datagen.sh
#   # 기존 env를 그대로 복제(가장 확실): env_isaaclab은 mirror 모드 사용
#   MODE=mirror OLD_HOST=user@old-desktop bash export/setup_datagen.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== [1/3] env_isaaclab (Isaac Sim 5.0 + BC/RL/eval/record_scene) ===="
bash "$SCRIPT_DIR/setup_env.sh"

echo "==== [2/3] lerobot050 (HDF5 -> LeRobot v3 변환) ===="
bash "$SCRIPT_DIR/setup_lerobot050_env.sh"

echo "==== [3/3] ROS2 Humble (텔레옵 브리지) ===="
if [ -d /opt/ros/humble ]; then
  echo "ROS2 Humble 이미 설치됨 (/opt/ros/humble) — 건너뜀"
else
  echo "ROS2 Humble 설치 (Ubuntu 22.04)..."
  sudo apt update && sudo apt install -y software-properties-common curl
  sudo add-apt-repository -y universe
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
       | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt update && sudo apt install -y ros-humble-desktop
  echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
fi

echo ""
echo "==== 검증 (설치된 버전) ===="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda run -n env_isaaclab python -c "import torch,isaacsim; print('  env_isaaclab : python', __import__('sys').version.split()[0], '| torch', torch.__version__, '| isaacsim', isaacsim.__version__)" 2>/dev/null || echo "  env_isaaclab 확인 실패"
conda run -n lerobot050  python -c "import lerobot,torch; print('  lerobot050   : python', __import__('sys').version.split()[0], '| lerobot', lerobot.__version__, '| torch', torch.__version__)" 2>/dev/null || echo "  lerobot050 확인 실패"
[ -d /opt/ros/humble ] && echo "  ROS2         : humble (/opt/ros/humble)"
echo "==== 데이터 생성 데스크탑 환경 설치 완료 ===="
