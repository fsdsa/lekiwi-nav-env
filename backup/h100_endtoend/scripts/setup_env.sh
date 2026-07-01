#!/bin/bash
# H100 클라우드 초기 환경 세팅 (sudo 없어도 됨, conda 기반)

set -e

echo "=== 1. Miniconda 설치 ==="
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
    rm Miniconda3-latest-Linux-x86_64.sh
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo ""
echo "=== 2. pi05_h100 env 생성 (Python 3.12) ==="
if ! conda env list | grep -q '^pi05_h100'; then
    conda create -n pi05_h100 python=3.12 -y
fi
conda activate pi05_h100

echo ""
echo "=== 3. PyTorch 2.4.1 + CUDA 12.1 ==="
pip install --quiet --upgrade pip
pip install --quiet torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== 4. lerobot 0.5.0 + deps ==="
pip install --quiet 'lerobot==0.5.0' h5py pandas pyarrow

echo ""
echo "=== 5. ffmpeg (sudo 없이 conda-forge에서) ==="
if ! command -v ffmpeg &> /dev/null; then
    conda install -c conda-forge ffmpeg -y -q
fi

echo ""
echo "=== 6. 환경 검증 ==="
python - <<'PYEOF'
import torch, lerobot, sys
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"lerobot: {lerobot.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu}")
    print(f"VRAM: {vram:.1f} GB")
    if 'H100' not in gpu:
        print(f"  WARNING: GPU가 H100이 아님 ({gpu}). config 재조정 필요할 수 있음.")
    if vram < 75:
        print(f"  WARNING: VRAM {vram:.1f}GB < 75GB. batch=16 위험. BATCH=8 시도하거나 GRAD_CKPT=true.")
else:
    print("FATAL: CUDA 없음")
    sys.exit(1)
PYEOF

which ffmpeg && ffmpeg -codecs 2>&1 | grep -E 'libx264' | head -1

echo ""
echo "✓ 환경 준비 완료. 'conda activate pi05_h100' 후 './preflight.sh' 또는 './train_h100.sh'"
