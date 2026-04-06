# Pi0-FAST VLA 파인튜닝 가이드

Pi0-FAST (2.9B) 모델을 LeKiwi 데이터로 파인튜닝하는 전체 과정을 기록한다.

---

## 1. 전체 흐름

```
[1] Sim 데이터 수집 (HDF5)
    ↓
[2] HDF5 merge (approach_lift + navigate 등)
    ↓
[3] HDF5 → LeRobot v3 변환
    ↓
[4] 데이터 검증 (NaN, stats.json)
    ↓
[5] Pi0-FAST 파인튜닝
    ↓
[6] 파인튜닝 모델 서빙
```

---

## 2. 서버 환경

| 항목 | 값 |
|------|-----|
| 서버 | `jovyan@218.148.55.186:30179` |
| SSH | `ssh -i ~/.ssh/private.pem -p 30179 jovyan@218.148.55.186` |
| conda env | `lerobotpi0v2` |
| conda path | `/home/jovyan/yes/envs/lerobotpi0v2` |
| Python | 3.12 |
| lerobot | 0.5.0 (git source v0.5.0 tag + `[pi]` extra) |
| transformers | 5.3.0 |
| torch | 2.10.0 |
| GPU | A100 40GB |

### 환경 활성화

```bash
export PATH="/home/jovyan/yes/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate lerobotpi0v2
```

### 필수 패키지

```bash
pip install sentencepiece tiktoken  # FAST tokenizer 의존성
huggingface-cli login               # google/paligemma-3b-pt-224 gated repo 접근
```

---

## 3. 데이터 수집 (Step 1)

### 3-1. Expert 데이터 수집 (record_teleop_scene.py)

```bash
# Approach & Lift (RL expert, headless, 100 episodes)
PYTHONUNBUFFERED=1 python vllm/record_teleop_scene.py \
  --skill approach_and_grasp \
  --instruction "pick up the medicine bottle" \
  --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
  --resip_checkpoint checkpoints/resip/resip_best.pt \
  --object_usd .../5_HTP/model_clean.usd \
  --dest_object_usd .../ACE_Coffee_Mug_.../model_clean.usd \
  --scene_idx 1302 --scene_scale 1.0 \
  --num_demos 100 --only_success --headless

# Navigate (lookup table, headless, 480 episodes)
PYTHONUNBUFFERED=1 python vllm/record_teleop_scene.py \
  --skill navigate \
  --instruction "navigate forward" \
  --scene_idx 1302 --scene_scale 1.0 \
  --num_demos 480 --headless
```

### 3-2. NaN 방어 (2026-03-27 수정)

`record_teleop_scene.py`에 다음 안전장치가 추가됨:

1. **`get_state_9d()`**: physics settle 부족 시 velocity NaN → 0 대체 + 경고 출력
2. **`save_episode()`**: state/action에 NaN 있으면 저장 거부 (`[REJECT]` 출력)
3. **`--scene_settle_steps`**: 기본값 30 → 60으로 증가 (physics 안정화 시간 확보)

**NaN 발생 원인**: `apply_scene_task_layout()` 후 physics settle 부족으로 `root_lin_vel_b`/`root_ang_vel_b`가 NaN. 특히 navigate 모드에서 직접 layout 생성 시 발생 빈도 높음.

---

## 4. 데이터 Merge (Step 2)

서버에서 HDF5 합치기:

```bash
cd /home/jovyan/data/lekiwi_hdf5

# approach_lift + navigate 합치기
python3 << 'EOF'
import h5py, os, numpy as np

output = "viva_merged.hdf5"
inputs = [
    ("100success_approach_lift.hdf5", None),  # 100 episodes
    ("navigate_scene_480ep.hdf5", None),       # 480 episodes
]

fo = h5py.File(output, "w")
ep_idx = 0
for fname, _ in inputs:
    fi = h5py.File(fname, "r")
    for key in sorted(fi.keys(), key=lambda x: int(x.split("_")[1])):
        fi.copy(key, fo, name=f"episode_{ep_idx}")
        ep_idx += 1
    fi.close()
fo.attrs["total_episodes"] = ep_idx
fo.close()
print(f"Merged: {ep_idx} episodes → {output}")
EOF
```

---

## 5. LeRobot v3 변환 (Step 3)

### 5-1. 변환 명령어

```bash
# 서버에서 실행 (lerobotpi0v2 env)
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env

python convert_hdf5_to_lerobot_v3.py \
  --input "/home/jovyan/data/lekiwi_hdf5/viva_merged.hdf5" \
  --output_root /home/jovyan/lerobot_data/lekiwi_viva \
  --repo_id local/lekiwi_viva \
  --fps 25 \
  --vcodec h264 \
  --task_source full_task \
  --overwrite \
  --skip_episodes_without_images
```

### 5-2. NaN 자동 스킵 (2026-03-27 수정)

`convert_hdf5_to_lerobot_v3.py`에 다음 안전장치가 추가됨:

1. **에피소드 NaN 검증**: `robot_state` 또는 `actions`에 NaN 있으면 `[SKIP]` 출력 후 해당 에피소드 건너뜀
2. **stats.json 최종 검증**: 변환 완료 후 stats.json의 모든 값에 NaN/None/inf 검사 → 문제 시 `[ERROR]` 출력

```
# 변환 시 NaN 에피소드 자동 스킵 예시
[SKIP] viva_merged.hdf5:episode_218 - robot_state NaN (150/150 rows)
[SKIP] viva_merged.hdf5:episode_219 - robot_state NaN (150/150 rows)
...
[OK] stats.json 검증 통과 (NaN/None/inf 없음)
```

### 5-3. 변환 결과 구조

```
lekiwi_viva/
├── meta/
│   ├── info.json              # total_episodes, total_frames, fps
│   ├── stats.json             # 전체 데이터셋 통계
│   ├── tasks.jsonl            # task index → text 매핑
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  # 에피소드별 메타 (length, from/to_index, per-ep stats)
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # 전체 프레임 데이터 (state, action, indices)
└── videos/
    ├── observation.images.front/
    │   └── chunk-000/
    │       └── file-000.mp4   # base camera 비디오
    └── observation.images.wrist/
        └── chunk-000/
            └── file-000.mp4   # wrist camera 비디오
```

### 5-4. Resume 기능

`--resume` 플래그로 중단된 변환 이어서 가능:

```bash
python convert_hdf5_to_lerobot_v3.py \
  --input "..." --output_root ... --repo_id ... \
  --resume
```

기존 `info.json`의 `total_episodes`를 읽어 이미 변환된 에피소드를 스킵하고 이어서 추가.

---

## 6. 데이터 검증 (Step 4)

변환 완료 후 반드시 검증:

```python
# lerobotpi0v2 env에서 실행
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

ds = LeRobotDataset("local/lekiwi_viva", root="/home/jovyan/lerobot_data/lekiwi_viva")
print(f"Dataset: {len(ds)} frames, {ds.meta.total_episodes} episodes")

# 샘플 NaN 검사
for i in [0, 1000, len(ds)//2, len(ds)-1]:
    sample = ds[i]
    state = sample["observation.state"]
    action = sample["action"]
    assert not torch.isnan(state).any(), f"NaN in state at idx={i}"
    assert not torch.isnan(action).any(), f"NaN in action at idx={i}"
print("PASSED: no NaN")
```

### stats.json 검증

```python
import json
with open("/home/jovyan/lerobot_data/lekiwi_viva/meta/stats.json") as f:
    stats = json.load(f)

for key in stats:
    for sname, vals in stats[key].items():
        if isinstance(vals, list):
            for i, v in enumerate(vals):
                assert v is not None, f"{key}.{sname}[{i}] is None"
                if isinstance(v, float):
                    assert v == v, f"{key}.{sname}[{i}] is NaN"
                    assert abs(v) != float("inf"), f"{key}.{sname}[{i}] is inf"
print("stats.json: ALL CLEAN")
```

**주의**: stats.json의 quantile 값(q01~q99)이 전부 0이면 NaN→0 패치 흔적. 반드시 clean 데이터로 재계산 필요.

---

## 7. Pi0-FAST 파인튜닝 (Step 5)

### 7-1. 학습 스크립트: train_pi0fast.sh

위치: `vllm/train_pi0fast.sh`

```bash
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm

# Full fine-tuning (기본, 권장)
bash train_pi0fast.sh train \
  --data_root /home/jovyan/lerobot_data/lekiwi_viva \
  --repo_id local/lekiwi_viva \
  --steps 30000 \
  --batch_size 4 \
  --save_freq 5000 \
  --num_workers 4

# LoRA fine-tuning (메모리 절약)
bash train_pi0fast.sh train \
  --data_root /home/jovyan/lerobot_data/lekiwi_viva \
  --repo_id local/lekiwi_viva \
  --mode lora \
  --steps 30000 \
  --batch_size 8 \
  --lora_r 16

# 백그라운드 실행 (nohup)
nohup bash -c "PYTHONUNBUFFERED=1 bash train_pi0fast.sh train \
  --data_root /home/jovyan/lerobot_data/lekiwi_viva \
  --repo_id local/lekiwi_viva \
  --steps 30000 --batch_size 4 --num_workers 4" \
  > /home/jovyan/pi0fast_viva_train.log 2>&1 &
```

### 7-2. 스크립트가 자동으로 하는 것

1. **Base model 다운로드**: `lerobot/pi0fast-base` → `./pi0fast_base_lekiwi/`
2. **Config 패치**: `input_features={}`, `output_features={}` (LeKiwi 로봇 사양에 맞게 추론)
3. **Tokenizer 패치**: `physical-intelligence/fast` → `lerobot/fast-action-tokenizer` (sentencepiece 호환)
4. **학습 파라미터 설정**:
   - `chunk_size=10`, `n_action_steps=10`, `max_action_tokens=256`
   - `gradient_checkpointing=true` (A100 메모리 절약)
   - `dtype=bfloat16`
   - `validate_action_token_prefix=false` (초기 garbage output 방지)

### 7-3. 주요 수정사항 기록

#### Tokenizer 에러 해결 (2026-03-27)

**문제**: pretrained model의 `policy_preprocessor.json`에 `action_tokenizer_name: "physical-intelligence/fast"`가 하드코딩. sentencepiece/tiktoken이 설치되어도 HuggingFace `transformers`가 이 tokenizer를 인스턴스화하지 못함.

```
ValueError: Failed to instantiate processor step 'action_tokenizer_processor'
  with config: {'action_tokenizer_name': 'physical-intelligence/fast'}
  Error: Couldn't instantiate the backend tokenizer
```

**해결**: `policy_preprocessor.json`의 `action_tokenizer_name`을 `lerobot/fast-action-tokenizer`로 변경.

```json
// BEFORE (에러)
"action_tokenizer_name": "physical-intelligence/fast"

// AFTER (정상)
"action_tokenizer_name": "lerobot/fast-action-tokenizer"
```

`train_pi0fast.sh`에 자동 패치 로직 추가됨 (2026-03-27). 수동 수정 불필요.

#### NaN → OverflowError 해결 (2026-03-27)

**문제**: HDF5 원본 데이터에 25개 에피소드의 `robot_state`가 전부 NaN. 변환 시 NaN이 stats.json에 전파 → normalizer가 inf 생성 → tokenizer OverflowError.

```
OverflowError: cannot convert float infinity to integer
```

**해결 (3단계)**:

1. **parquet에서 NaN 에피소드 제거** + episode_index 재번호
2. **stats.json 재계산**: clean 데이터에서 min/max/mean/std/quantile 전부 재계산
3. **episodes parquet 재빌드**: `dataset_from_index`/`dataset_to_index` + per-episode stats 재계산

**근본 방지**: `record_teleop_scene.py`와 `convert_hdf5_to_lerobot_v3.py`에 NaN 검증 추가 (§3-2, §5-2 참조).

#### IndexError 해결 (2026-03-27)

**문제**: parquet에서 NaN 에피소드 제거 후, episodes parquet의 `dataset_to_index`가 여전히 old 범위(149,198)를 가리킴 → `Invalid key: 148448 is out of bounds for size 145448`.

**해결**: episodes parquet의 `dataset_from_index`, `dataset_to_index`, `length`, per-episode stats를 clean 데이터 기준으로 재빌드.

#### push_model_to_hub 403 (무시 가능)

학습 완료 후 HuggingFace에 모델 업로드 시 `local/` namespace 권한 부족으로 403 에러. **로컬 체크포인트는 정상 저장**되므로 무시.

### 7-4. 학습 파라미터

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| base_model | `lerobot/pi0fast-base` | 2.9B params |
| mode | full | LoRA도 가능 (`--mode lora`) |
| batch_size | 4 | full fine-tune 시 A100 40GB |
| steps | 30000 | |
| save_freq | 5000 | |
| log_freq | 100 | |
| chunk_size | 10 | LIBERO reference |
| n_action_steps | 10 | |
| max_action_tokens | 256 | |
| gradient_checkpointing | true | |
| dtype | bfloat16 | |
| num_workers | 4 | dataloader |

### 7-5. 예상 학습 시간

| 모드 | batch_size | 속도 | 30K steps |
|------|-----------|------|-----------|
| Full | 4 | ~1.25 step/s | ~6.7시간 |
| Full | 2 | ~1.8 step/s | ~4.6시간 |
| LoRA | 8 | ~2+ step/s | ~4시간 |

### 7-6. 메모리 사용량

| 모드 | VRAM |
|------|------|
| Full (bs=4, grad_ckpt) | ~30-35 GB |
| Full (bs=2, grad_ckpt) | ~25-28 GB |
| LoRA (bs=8) | ~12-15 GB |

### 7-7. 학습 모니터링

```bash
# 진행 상황
tail -f /home/jovyan/pi0fast_viva_train.log

# 최신 step 확인 (tqdm은 \r로 덮어쓰므로 strings 사용)
strings /home/jovyan/pi0fast_viva_train.log | grep -oP "\d+/30000" | tail -1

# 프로세스 확인
pgrep -af lerobot-train

# 체크포인트 확인
ls -la outputs/train/pi0fast_lekiwi_full_*/checkpoints/
```

### 7-8. 출력 디렉토리 구조

```
outputs/train/pi0fast_lekiwi_full_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── 005000/
│   │   └── pretrained_model/    # 5K step 체크포인트
│   ├── 010000/
│   │   └── pretrained_model/
│   ├── ...
│   └── last/
│       └── pretrained_model/    # 최종 체크포인트
├── config.json
└── train_info.json
```

---

## 8. 모델 서빙 (Step 6)

### 8-1. 서빙 명령어

```bash
# 파인튜닝 모델 서빙
bash train_pi0fast.sh serve \
  --checkpoint outputs/train/pi0fast_lekiwi_full_YYYYMMDD_HHMMSS/checkpoints/last

# 포트 변경
bash train_pi0fast.sh serve \
  --checkpoint ... --port 8002
```

`pretrained_model/` 서브디렉토리가 있으면 자동 resolve.

### 8-2. VLA 추론 서버 (vla_inference_server.py)

```bash
python vllm/vla_inference_server.py \
  --model outputs/train/.../checkpoints/last/pretrained_model \
  --port 8002 \
  --host 0.0.0.0
```

### 8-3. VLM + VLA 동시 서빙

```bash
# VLM (별도 터미널, conda vllm)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code

# VLA (별도 터미널, conda lerobotpi0v2)
python vllm/vla_inference_server.py \
  --model outputs/train/.../checkpoints/last/pretrained_model \
  --port 8002 \
  --host 0.0.0.0
```

**VLM 설정 (Qwen3-VL-8B-Instruct)**:
- `--gpu-memory-utilization 0.75`: VLA 동시 로드 시 필수 (0.50은 KV cache OOM)
- `--max-model-len 4096`: KV cache 절약 (기본 262144는 OOM)
- `--dtype bfloat16`: A100 최적
- VRAM: VLM ~29.8GB + VLA ~8.1GB = ~37.9GB / 40GB
- **SSH 터널 필수**: `ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 -i ~/.ssh/private.pem -p 30179 jovyan@218.148.55.186`

**VRAM**: VLM ~19.8GB + VLA ~7.7GB = ~27.5GB / 40GB

---

## 9. One-shot 파이프라인 (변환 + 학습)

```bash
bash train_pi0fast.sh all \
  --input "/home/jovyan/data/lekiwi_hdf5/viva_merged.hdf5" \
  --output_root /home/jovyan/lerobot_data/lekiwi_viva \
  --repo_id local/lekiwi_viva \
  --steps 30000
```

---

## 10. 데이터 패치 방법 (NaN 에피소드 제거)

만약 변환 후 NaN 에피소드가 발견되었을 때 재변환 없이 패치하는 방법:

### 10-1. NaN 에피소드 확인

```python
import pandas as pd, numpy as np

df = pd.read_parquet("lekiwi_viva/data/chunk-000/file-000.parquet")
state_arr = np.stack(df["observation.state"].values)
action_arr = np.stack(df["action"].values)
nan_mask = np.any(np.isnan(state_arr), axis=1) | np.any(np.isnan(action_arr), axis=1)
nan_eps = sorted(df.loc[nan_mask, "episode_index"].unique().tolist())
print(f"NaN episodes: {len(nan_eps)}, indices: {nan_eps}")
```

### 10-2. 패치 절차

1. **백업**: `meta/` + `data/` 복사
2. **parquet**: NaN 에피소드 제거 → episode_index 재번호 → index/frame_index 재계산
3. **stats.json**: clean 데이터에서 전체 재계산 (min/max/mean/std/q01~q99)
4. **info.json**: total_episodes, total_frames 업데이트
5. **episodes parquet**: dataset_from/to_index + per-episode stats 재빌드

**비디오 파일은 건드리지 않음** — chunk 단위 MP4이므로 episode 삭제와 무관.

### 10-3. 검증

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/lekiwi_viva", root="...")
# 모든 샘플이 NaN 없는지 확인
for i in range(0, len(ds), 1000):
    sample = ds[i]
    assert not torch.isnan(sample["observation.state"]).any()
    assert not torch.isnan(sample["action"]).any()
```

---

## 11. 파인튜닝 기록

### v1 (폐기)
- navigate arm action=0 (잘못), NaN 25ep → 중단

### v2 (폐기)
- navigate arm action=raw joint position (action space 불일치) → 중단

### v3 (lekiwi_viva)
| 항목 | 값 |
|------|-----|
| 데이터 | approach_lift 100ep + navigate 455ep = **555 clean episodes, 145,448 frames** |
| navigate action | **[-1,1] normalized** (arm_action_to_limits 역변환, approach_lift와 동일 space) |
| task 수 | 12 (task_index 0~11) |
| 학습 시작 | 2026-03-27 |
| 데이터셋 경로 | `/home/jovyan/lerobot_data/lekiwi_viva` |

### v4 (lekiwi_viva_v2, 현재 진행 중)
| 항목 | 값 |
|------|-----|
| 데이터 | approach_lift 100ep + navigate 446ep + **carry 432ep** = **978 episodes, 209,036 frames** |
| carry 수집 | `record_teleop_scene.py --skill combined_s2_s3` (S3 phase만 기록, 150 steps/ep) |
| carry instruction | 6방향 × 72ep: carry forward/backward/left/right/turn left/turn right |
| task 수 | 18 (기존 12 + carry 6방향) |
| 전처리 | last frame state=0 수정 (101ep), anomalous ep 삭제 (9ep), stats std floor=0.3 |
| 학습 | 2026-04-05, full fine-tune, bs=4, 300K steps, ~1.25 step/s |
| 데이터셋 경로 | `/home/jovyan/lerobot_data/lekiwi_viva_v2` |
| HDF5 원본 | `/home/jovyan/data/lekiwi_hdf5/viva_merged_with_carry.hdf5` |
| 출력 | `outputs/train/pi0fast_lekiwi_full_20260405_140358/` |
| 로그 | `/home/jovyan/train_pi0fast_viva_v2.log` |

#### v4 carry 데이터 수집

```bash
# 로컬 3090에서 실행 (S2 expert → S3 carry BC+ResiP, 6위치×6방향×4라운드=144ep)
PYTHONUNBUFFERED=1 python vllm/record_teleop_scene.py \
    --skill combined_s2_s3 \
    --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    --resip_checkpoint backup/appoachandlift/resip64%.pt \
    --bc_checkpoint_s3 checkpoints/dp_bc_carry_v4/dp_bc_epoch300.pt \
    --resip_checkpoint_s3 checkpoints/resip_carry_v6/resip_carry_iter240.pt \
    --object_usd ~/isaac-objects/.../5_HTP/model_clean.usd \
    --scene_idx 1302 --scene_scale 1.0 \
    --num_demos 144 --only_success --headless
# 결과: 144ep (SR=56%) → 3배 복사 → 432ep
# instruction: base action 패턴으로 자동 분류 후 HDF5 in-place 수정
```

#### v4 데이터 전처리

1. **Last frame state=0**: approach_lift 100ep + navigate 1ep의 마지막 프레임 robot_state 전부 0 → state[-2] 값으로 복사
2. **Anomalous ep 삭제 (9개)**: episode_216 (벽 충돌 base_wz=±8) + navigate forward 8ep (arm0 ±2.7 드리프트)
3. **Stats std floor=0.3**: 변환 후 stats.json에서 std < 0.3인 dim에 최소값 적용 (arm0/arm4/base_vx/vy/wz). tokenizer overflow 방지
4. **tasks.jsonl KeyError 수정**: `convert_hdf5_to_lerobot_v3.py`에서 LeRobot v3 tasks parquet 구조 변경 대응 (`row["task"]` → `task_text`)

---

## 12. 코드 수정 이력

### 2026-03-27 수정 (NaN 방어 + tokenizer 자동 패치)

**`vllm/record_teleop_scene.py`**:
- `get_state_9d()`: NaN 감지 → 0 대체 + `[WARN]` 출력
- `save_episode()`: state/action NaN 검증 → NaN 시 저장 거부 (`return False`)
- 호출부 2곳: `save_episode` 반환값 체크 → 실패 시 saved_count 미증가
- `--scene_settle_steps` 기본값: 30 → 60

**`convert_hdf5_to_lerobot_v3.py`**:
- robot_state/actions NaN → 에피소드 `[SKIP]`
- 변환 완료 후 stats.json NaN/None/inf 최종 검증

**`vllm/train_pi0fast.sh`**:
- base model 다운로드 후 tokenizer 이름 자동 패치: `physical-intelligence/fast` → `lerobot/fast-action-tokenizer`

### 2026-03-27 수정 (Navigate action space 통일)

**핵심 문제**: navigate 데이터의 arm action이 approach_lift와 다른 action space였음.
- approach_lift: [-1,1] normalized (arm_action_to_limits 매핑)
- navigate (이전): raw joint position 또는 0 → VLA가 일관된 action space를 학습 못 함

**수정 내용**:

1. **`collect_nav_skill2.py`**: arm action에 TUCKED_POSE의 normalized [-1,1] 값 사용
   ```
   TUCKED_ARM_NORM = [-0.001, -1.0, 1.0, 0.659, -0.537]  (j3=-0.4의 역변환)
   TUCKED_GRIP_NORM = -0.999
   ```
2. **`eval_navigate.py`**: monkey-patch `_nav_apply` 제거 → env 기본 `_apply_action` 사용 (arm_action_to_limits 매핑 자동 적용)
3. **`train_resip.py` `main_navigate()`**:
   - monkey-patch 제거 → env 기본 `_apply_action` 사용
   - `ResidualPolicy`: `action_scale=0.1`, `learn_std=True` 추가 (다른 skill과 동일)
   - obs_buf에 pre-computed `res_input` 저장 (PPO에서 dp_agent 재호출 제거 — deque 셔플 NaN 방지)
   - tucked reward σ 확대: arm 0.1→0.3, grip 0.1→0.2, 가중치 0.5→2.0 / 0.2→0.5
   - arm smooth penalty 추가: `-0.01 * delta_arm²`
   - `grasp_success_height=100.0` (navigate에서 task_success 비활성화)
   - reward NaN 방어: `torch.nan_to_num(reward, nan=0.0)`
4. **서버 VLA 데이터 패치**: parquet에서 navigate ep의 action[0:6]을 normalized 값으로 교체 + stats/episodes parquet 재계산
5. **`train_diffusion_bc.py`**: `--down_dims` 기본값 [256,512,1024] → [64,128,256]

**Navigate BC 데모**: 6방향 × 20회 = 120 에피소드 (이전 24개 → 120개로 증량, arm 진동 해소)

**`arm_action_to_limits` 매핑 정리**:
```
action[-1,1] → joint_position = center + action * half
center = (lo + hi) / 2, half = (hi - lo) / 2
역변환: action = (target - center) / half
```

---

## 13. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Couldn't instantiate the backend tokenizer` | `policy_preprocessor.json`의 tokenizer 이름 | `lerobot/fast-action-tokenizer`로 변경 (자동 패치됨) |
| `OverflowError: cannot convert float infinity` | stats.json에 NaN → normalizer가 inf 생성 | NaN 에피소드 제거 + stats 재계산 |
| `IndexError: out of bounds for size` | episodes parquet의 index가 old 범위 | episodes parquet 재빌드 |
| `403 Forbidden: push_model_to_hub` | HF `local/` namespace 권한 | 무시 (로컬 체크포인트 정상) |
| `ModuleNotFoundError: lerobot.common` | lerobot 0.5.0은 `lerobot.datasets` | `from lerobot.datasets.lerobot_dataset import LeRobotDataset` |
| `ImportError: h5py` | rl_train env에는 h5py만, pyarrow 없음 | `lerobotpi0v2` env 사용 |
| 학습 시 GPU OOM | batch_size 너무 큼 | full: bs=2~4 + grad_ckpt, LoRA: bs=8 |
| `attention_mask` 에러 | lerobot Pi0 attention mask dtype | `vla_inference_server.py`에서 `.bool()` 전달 |
| Navigate arm action space 불일치 | action이 raw joint pos (approach_lift는 [-1,1]) | `arm_action_to_limits` 역변환으로 normalized 값 사용 |
| Navigate BC arm 진동 | 데모 24개 부족 | 120 에피소드로 증량 (6방향 × 20회) |
| Navigate RL entropy 고정 3.7704 | `learn_std=False` (누락) + `action_scale=1.0` | `learn_std=True` + `action_scale=0.1` |
| Navigate RL entropy 폭발 12.77 | `ent_coef=0.001` 너무 높음 | `ent_coef=0.005` (+ tucked σ 확대) |
| Navigate RL v_loss=NaN | PPO에서 dp_agent.base_action_normalized 재호출 (deque 셔플) | obs_buf에 pre-computed res_input 저장, PPO에서 직접 사용 |
| tasks.jsonl 비어있음 | 패치 과정에서 누락 | tasks.parquet에서 복원 |
| splits 0:580 불일치 | NaN 제거 전 값 | 0:555로 수정 |

---

## 14. 참조 파일

| 파일 | 역할 |
|------|------|
| `vllm/train_pi0fast.sh` | 변환/학습/서빙 통합 스크립트 |
| `convert_hdf5_to_lerobot_v3.py` | HDF5 → LeRobot v3 변환 |
| `vllm/record_teleop_scene.py` | ProcTHOR scene 데이터 수집 |
| `vllm/vla_inference_server.py` | Pi0-FAST FastAPI 서빙 |
| `vllm/run_full_task.py` | VLM+VLA 전체 파이프라인 실행 |
| `vllm/procthor_scene.py` | ProcTHOR scene 로딩/스폰 |
