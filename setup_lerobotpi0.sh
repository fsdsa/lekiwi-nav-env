#!/bin/bash
# lerobotpi0 conda 환경 생성 — Pi0-FAST VLA 추론용
#
# 사용법 (A100 서버에서):
#   bash setup_lerobotpi0.sh
#
# 생성되는 환경:
#   conda activate lerobotpi0
#   python vla_inference_server.py --checkpoint <path> --port 8002

set -euo pipefail

ENV_NAME="lerobotpi0"
CONDA_DIR="${HOME}/miniconda3"

echo "============================================================"
echo "  Setting up ${ENV_NAME} conda environment"
echo "============================================================"

# conda 초기화
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# 기존 환경 확인
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  [SKIP] ${ENV_NAME} already exists"
    echo "  삭제하려면: conda env remove -n ${ENV_NAME}"
    conda activate ${ENV_NAME}
else
    echo "  [1/4] Creating conda env: ${ENV_NAME} (python 3.11)"
    conda create -n ${ENV_NAME} python=3.11 -y

    conda activate ${ENV_NAME}

    echo "  [2/4] Installing PyTorch (CUDA 12.4)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    echo "  [3/4] Installing LeRobot + Pi0-FAST"
    pip install "lerobot[pi0]"

    echo "  [4/4] Installing server dependencies"
    pip install fastapi uvicorn pydantic pillow
fi

echo ""
echo "  Verifying installation..."
python -c "
import torch
print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
print('  PI0FASTPolicy: OK')
import fastapi, uvicorn
print(f'  fastapi={fastapi.__version__}, uvicorn OK')
print('  All checks passed!')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Usage:"
echo "    conda activate ${ENV_NAME}"
echo "    python vla_inference_server.py --checkpoint <path> --port 8002"
echo "============================================================"
