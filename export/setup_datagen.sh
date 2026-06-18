#!/usr/bin/env bash
# =============================================================================
# 데이터 생성 파이프라인 — 데스크탑 환경 "전체" 설치 (아무것도 없는 PC → 현재와 동일)
#
# 이 스크립트 하나만 실행하면 아래가 전부 설치/검증된다:
#   [0] Miniconda            (없으면 설치)
#   [1] Isaac Lab repo       (없으면 isaac-sim/IsaacLab clone + 고정 커밋 checkout)
#   [2] env_isaaclab         Python 3.11 + Isaac Sim 5.0.0 + Isaac Lab 0.44.9
#                            + 데이터젠 의존(diffusers/skrl/...) — lockfile로 정확 버전 고정
#   [3] molmospaces          ProcTHOR scene (10주차) — clone + pip -e + scene 다운로드(1302 포함)
#   [4] lerobot050           Python 3.12 + LeRobot 0.5.0 (HDF5 → LeRobot v3 변환)
#   [5] ROS2 Humble          텔레옵 브리지 (없으면 설치)
#   [6] 검증                 설치된 버전 출력
#
# 검증 환경(2026-06-18 실측): Ubuntu 22.04.5 · Isaac Sim 5.0.0 · Isaac Lab 0.44.9
#   · PyTorch 2.7.0+cu128 · LeRobot 0.5.0 · ROS2 Humble · NVIDIA Driver 590.48
#
# 전제: RTX GPU + NVIDIA 드라이버. 로봇/물체 USD(~/Downloads/lekiwi_robot.usd, ~/lekiwi,
#   ~/isaac-objects)는 "Lekiwi 환경 구축 및 원격조종 매뉴얼"에서 준비된다(비공개 자산).
#   이 스크립트는 소프트웨어 환경 + ProcTHOR scene 을 설치한다.
#
# 사용:  bash export/setup_datagen.sh
#   기존 데스크탑 env를 그대로 복제(가장 확실): MODE=mirror OLD_HOST=user@old bash export/setup_env.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-46dff135f44683f031edf346e544fcfd8456b2bb}"   # isaaclab 0.44.9
MOLMO_DIR="${MOLMO_DIR:-$HOME/molmospaces}"
MOLMO_COMMIT="${MOLMO_COMMIT:-0939e18b14a3650ae8094ff516a4e244fca82198}"

# ── [0] Miniconda ────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1 && [ ! -x "$CONDA_BASE/bin/conda" ]; then
  echo "==== [0] Miniconda 설치 ===="
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_BASE"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "[WARN] nvidia-smi 없음 — Isaac Sim 렌더링엔 RTX GPU + NVIDIA 드라이버 필수"
fi

# ── [1] Isaac Lab repo (isaac-sim/IsaacLab, 고정 커밋) ─────────────────────────
if [ ! -f "$ISAACLAB_DIR/isaaclab.sh" ]; then
  echo "==== [1] IsaacLab clone → $ISAACLAB_DIR ===="
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  ( cd "$ISAACLAB_DIR" && git checkout "$ISAACLAB_COMMIT" )
fi

# ── [2] env_isaaclab (Isaac Sim 5.0 + Isaac Lab + 데이터젠 의존, 정확버전) ──────
if conda env list | grep -qE "/env_isaaclab$"; then
  echo "==== [2] env_isaaclab 이미 존재 — 건너뜀 ===="
else
  echo "==== [2] env_isaaclab 생성 + Isaac Sim 5.0.0 + Isaac Lab 설치 ===="
  conda create -y -n env_isaaclab python=3.11
  conda activate env_isaaclab
  pip install --upgrade pip
  pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
  ( cd "$ISAACLAB_DIR" && ./isaaclab.sh -i )
  pip install "diffusers>=0.36" requests numpy pynput
  echo "==== [2b] 정확 버전 고정 (lockfile) ===="
  pip install -r "$SCRIPT_DIR/env_isaaclab.lock.txt" \
      --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
    || echo "[WARN] lockfile 정확고정 일부 실패 — 위 major 버전으로 동작 (editable isaaclab/molmo는 [1][3]에서 설치)"
fi

# ── [3] molmospaces (ProcTHOR scene, 10주차) ──────────────────────────────────
echo "==== [3] molmospaces (ProcTHOR scene) ===="
conda activate env_isaaclab
if [ ! -d "$MOLMO_DIR/.git" ]; then
  git clone https://github.com/allenai/molmospaces.git "$MOLMO_DIR"
  ( cd "$MOLMO_DIR" && git checkout "$MOLMO_COMMIT" )
fi
( cd "$MOLMO_DIR/molmo_spaces_isaac" && pip install -e ".[dev]" )
( cd "$MOLMO_DIR" && python download_few_scenes.py )   # scenes 8,3614,100,9999,1302 → assets/usd

# ── [4] lerobot050 (HDF5 → LeRobot v3 변환) ──────────────────────────────────
echo "==== [4] lerobot050 (LeRobot 0.5.0 변환 env) ===="
bash "$SCRIPT_DIR/setup_lerobot050_env.sh"

# ── [5] ROS2 Humble (텔레옵 브리지) ──────────────────────────────────────────
echo "==== [5] ROS2 Humble ===="
if [ -d /opt/ros/humble ]; then
  echo "ROS2 Humble 이미 설치됨 (/opt/ros/humble) — 건너뜀"
else
  sudo apt update && sudo apt install -y software-properties-common curl
  sudo add-apt-repository -y universe
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
       | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt update && sudo apt install -y ros-humble-desktop
  echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
fi

# ── [6] 검증 ─────────────────────────────────────────────────────────────────
echo ""
echo "==== [6] 검증 (설치 버전) ===="
conda run -n env_isaaclab python - <<'PY' 2>/dev/null || echo "  env_isaaclab 확인 실패 (conda activate env_isaaclab 후 재시도)"
import sys, importlib
print("  env_isaaclab : python", sys.version.split()[0])
for m in ["isaacsim","isaaclab","torch","diffusers","skrl","molmo_spaces_isaac"]:
    try:
        mod = importlib.import_module(m); print(f"    OK  {m:18s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"    FAIL {m:18s} {type(e).__name__}")
PY
conda run -n lerobot050 python -c "import sys,lerobot,torch; print('  lerobot050   : python', sys.version.split()[0], '| lerobot', lerobot.__version__, '| torch', torch.__version__)" 2>/dev/null || echo "  lerobot050 확인 실패"
[ -d /opt/ros/humble ] && echo "  ROS2         : humble (/opt/ros/humble)"
echo ""
echo "==== 설치 완료. 셸마다 환경변수 설정 ===="
echo '  export OMNI_KIT_ACCEPT_EULA=YES'
echo '  export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd'
echo "  (로봇/물체 USD 자산은 Lekiwi 환경 구축 매뉴얼에서 준비)"
