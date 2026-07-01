#!/bin/bash
# H100 서버에서 학습 시작 전에 모든 컴포넌트 검증
# setup_env.sh 완료 후, train_h100.sh 전에 실행

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pi05_h100

cd "$(dirname "$0")"

python - <<'PYEOF'
import json, pandas as pd, numpy as np, torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path('./dataset')
BASE = Path('./base_model')

# 1. GPU 체크
print("[1] GPU")
assert torch.cuda.is_available(), "CUDA unavailable"
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"   GPU: {gpu}, VRAM: {vram:.1f} GB")

# 2. Dataset 로드
print("[2] Dataset load")
ds = LeRobotDataset(repo_id='local/lekiwi_full_teleop_combined', root=ROOT)
print(f"   eps={ds.meta.total_episodes}, frames={ds.meta.total_frames}, fps={ds.meta.fps}")
assert ds.meta.total_episodes == 96

# 3. Frame decode
print("[3] Frame decode (ep 0, 50, 95)")
for ep in [0, 50, 95]:
    start = ds.meta.episodes[ep]['dataset_from_index']
    s = ds[start + 10]
    assert s['observation.images.front'].shape == (3, 400, 640)
    assert s['observation.state'].shape[0] == 9
print("   ✓ 비디오 + state + action 모두 정상")

# 4. stats.json 정합성
print("[4] stats.json 5% margin")
with open(ROOT / 'meta/stats.json') as f:
    stats = json.load(f)
df = pd.read_parquet(ROOT / 'data/chunk-000/file-000.parquet')
states = np.stack([np.array(s) for s in df['observation.state']]).astype(np.float64)
actions = np.stack([np.array(a) for a in df['action']]).astype(np.float64)
for key, arr in [('observation.state', states), ('action', actions)]:
    mn, mx = arr.min(axis=0), arr.max(axis=0); rng = mx - mn
    assert np.allclose(np.array(stats[key]['q01']), mn - 0.05 * rng), f"{key} q01"
    assert np.allclose(np.array(stats[key]['q99']), mx + 0.05 * rng), f"{key} q99"
print("   ✓ min/max ± 5% margin (v4 convention)")

# 5. Base model 존재
print("[5] Base model")
assert BASE.exists()
config = json.load(open(BASE / 'config.json'))
print(f"   type={config.get('type')}, chunk_size={config.get('chunk_size')}")

# 6. 비디오 파일 존재
print("[6] Video files")
v = list((ROOT / 'videos').rglob('*.mp4'))
print(f"   {len(v)} mp4 files present")
assert len(v) == 4

print("\n✅ PREFLIGHT 통과 — train_h100.sh 실행 가능")
PYEOF
