# REPRODUCE — 새 서버/데스크탑에서 처음부터 재현하기

이 문서는 **아무것도 없는 새 머신**(다른 username·다른 home 경로 포함)에서 이 레포를
처음부터 재현하기 위한 단일 진입점이다. 상세 배경은 아래 기존 문서를 참조한다.

- 데스크탑 마이그레이션 상세: [`export/MIGRATION.md`](export/MIGRATION.md), [`export/LAPTOP_SETUP.md`](export/LAPTOP_SETUP.md)
- 서버(학습) 상세: [`secondserver/SETUP_GUIDE.md`](secondserver/SETUP_GUIDE.md)
- 추론 서버(VLM/VLA) 상세: [`vllm/SETUP_MANUAL.md`](vllm/SETUP_MANUAL.md), [`vllm/README.md`](vllm/README.md)

> **핵심**: 코드는 git에 있지만 **① 로봇/물체 USD, ② BC/RL 체크포인트, ③ lerobot 데이터셋,
> ④ VLA 파인튜닝 ckpt** 는 git에 없다(gitignore·폴더 밖). 이 4가지는 백업에서 별도로 가져와야 한다.
> 이 레포의 경로는 전부 `$HOME` 기준 + 환경변수 override 로 바뀌었으므로 **username이 달라도 동작**한다
> (기존엔 `/home/yubin11`·`/home/jovyan` 하드코딩이었음 — 이제 제거됨).

---

## 역할별 머신

| 머신 | 역할 | 설치 스크립트 |
|------|------|---------------|
| **데스크탑** (RTX GPU) | 텔레옵·데이터수집·sim eval·로컬 BC/RL 추론 | `bash setup.sh` (= `export/setup_datagen.sh`) |
| **A100 서버 — 학습** | BC/RL(ResiP) 학습 | `bash setup.sh server` |
| **A100 서버 — 추론** | VLM(8000)+VLA(8002) 서빙, VLA 파인튜닝 | `bash secondserver/setup_viva.sh` |

---

## 재현 순서 (checklist)

### 1) 레포 클론
```bash
git clone <this-repo-url> ~/IsaacLab/scripts/lekiwi_nav_env   # 또는 IsaacLab 전체를 rsync
cd ~/IsaacLab/scripts/lekiwi_nav_env
```

### 2) 소프트웨어 환경 설치 (`setup.sh`)
`setup.sh` 하나가 Miniconda→Isaac Sim→Isaac Lab→conda env→(데스크탑은 molmospaces·lerobot050·ROS2)까지 순서대로 설치한다. `$HOME`/`$CONDA_BASE`/`$ISAACLAB_DIR` 기준이라 username 무관.
```bash
# 데스크탑 (env_isaaclab + lerobot050 + ProcTHOR + ROS2)
bash setup.sh

# 학습 서버 (rl_train env: Isaac Sim 5.0 + Isaac Lab 2.2.0 + skrl/diffusers)
bash setup.sh server

# 추론 서버 (vllm + lerobotpi0v2 env + HF 모델 캐싱까지 한 번에)
bash secondserver/setup_viva.sh
```
정확 버전이 필요하면 lockfile 사용: `export/env_*.lock.txt`, `secondserver/env_*.lock.txt`.

### 3) HF 모델 다운로드 (`download_models.sh`)
```bash
huggingface-cli login                     # google/paligemma gated repo 접근 필수
bash secondserver/download_models.sh      # Qwen3-VL-8B (HF 캐시) + lerobot/pi05_base → ./pi05_base
# base 모델 위치 바꾸려면:  PI05_BASE_DIR=~/h100_deploy/base_model bash secondserver/download_models.sh
```
(`setup_viva.sh` 를 돌렸다면 이 단계는 이미 포함됨.)

### 4) USD 자산 확보 (스크립트가 설치하지 않음 — 반드시 별도 확보)
| 자산 | 위치(기본) | 크기 | 확보 방법 |
|------|-----------|------|-----------|
| 로봇 USD `lekiwi_robot.usd` | `~/Downloads/lekiwi_robot.usd` | 7.3M | 백업에서 rsync. **내부적으로 `~/lekiwi/` (4.7G mesh 트리)를 상대참조** → `~/lekiwi/` 전체도 함께 확보 필수 |
| 물체 USD 카탈로그 | `~/isaac-objects/mujoco_scanned_objects/models/` | 3.8G | 백업에서 rsync (최소 `5_HTP`, `ACE_Coffee_Mug_Kristen_16_oz_cup` 두 개는 필수) |
| ProcTHOR scene | `~/molmospaces/` | 24M | `setup.sh`(데스크탑)가 `download_few_scenes.py`로 자동 설치 |

- **원본 출처**: 로봇/물체 USD 제작·스폰 절차는 Notion 매뉴얼 「Lekiwi 환경 구축 및 원격조종 물체 스폰」에 있다(레포 밖).
- **머신 간 복사**: 기존 머신이 살아있으면 `bash secondserver/transfer_files.sh <user> <host> [port]` 가 로봇 USD + 물체 2종 + 체크포인트를 scp 로 밀어준다. 상세 rsync 명령은 `export/MIGRATION.md §2`.

### 5) 데이터셋 · 체크포인트 확보 (백업에서 rsync — git에 없음)
| 자산 | 위치(기본) | 필요 대상 | 확보 방법 |
|------|-----------|-----------|-----------|
| BC/RL ckpt (`dp_bc_epoch150.pt`, `resip64%.pt`, `dp_bc_epoch500.pt` 등) | `checkpoints/`, `backup/` | ResiP 학습·eval | 백업에서 rsync (`transfer_files.sh`가 3개 핵심본 전송) |
| lerobot 데이터셋 (`lekiwi_viva_v5`) | `~/lerobot_data/lekiwi_viva_v5` | VLA 파인튜닝 | 백업 rsync, 또는 `vllm/dataset_tools/build_v5.py`로 재빌드 |
| VLA 파인튜닝 ckpt (`pi05_viva_v5` 250K) | `vllm/outputs/train/...` | VLA 서빙 | 백업 rsync, 또는 `bash vllm/train_v5.sh`로 재학습 |
| calibration (`arm_limits_measured.json`, `tucked_pose.json`) | `calibration/` | 전 파이프라인 | **git tracked — 레포와 함께 옴** ✓ |

### 6) 환경변수 설정 (username/경로가 다르면 필수)
기본값은 전부 `$HOME` 기준이라 표준 레이아웃이면 대개 그대로 동작한다. 자산을 다른 곳에 뒀으면 override:
```bash
export OMNI_KIT_ACCEPT_EULA=YES
export LEKIWI_USD_PATH="$HOME/Downloads/lekiwi_robot.usd"      # 로봇 USD (미설정 시 fallback도 ~/Downloads/...)
export ISAAC_OBJECTS_DIR="$HOME/isaac-objects"                 # 물체 USD 루트 (skill3 dest 기본·run_teleop.sh)
# 학습 launcher(run_train_nav.sh / run_eval_scene.sh)를 conda 미활성 상태로 쓸 때만:
export ISAAC_PATH="$HOME/isaacsim"                             # Isaac Sim 설치 경로
export ENV_ISAACLAB_DIR="$HOME/miniconda3/envs/env_isaaclab"  # env_isaaclab conda env
```
`.bashrc` 에 넣어두면 편하다.

### 7) 실행
```bash
# --- 학습 서버: ResiP 학습 (사전검증 포함, USD/ckpt 경로 자동 확인) ---
bash secondserver/run_training.sh

# --- 추론 서버: VLM+VLA 동시 기동 ---
#   lerobotpi0v2 env가 표준 conda base면:  LEROBOT_CONDA_DIR=$HOME/miniconda3
LEROBOT_CONDA_DIR="$HOME/miniconda3" \
  bash launch_servers.sh all --checkpoint <pi05 ckpt>/pretrained_model

# --- 데스크탑: 전체 태스크(파이프라인) 실행 (SSH 터널 먼저) ---
ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 A100
conda activate env_isaaclab
python vllm/run_full_task.py --headless ...

# --- VLA 파인튜닝 (추론 서버) ---
#   lerobot bin이 표준 conda base면 override:
LEROBOT_BIN=$HOME/miniconda3/envs/lerobotpi0v2/bin/lerobot-train \
PYTHON_BIN=$HOME/miniconda3/envs/lerobotpi0v2/bin/python \
  bash vllm/train_v5.sh
```

---

## 환경변수 레퍼런스 (이번 정리에서 override 가능하게 만든 것)

| 환경변수 | 기본값 | 쓰이는 곳 |
|----------|--------|-----------|
| `LEKIWI_USD_PATH` | `~/Downloads/lekiwi_robot.usd` | `lekiwi_robot_cfg.py` (로봇 USD) |
| `ISAAC_OBJECTS_DIR` | `~/isaac-objects` | `lekiwi_skill3_env.py`(dest object), `run_teleop.sh` |
| `LEROBOT_DATASET_ROOT` | `~/lerobot_data/lekiwi_viva_v4` | `vllm/diagnose_vla.py` |
| `ISAAC_PATH` | `~/isaacsim` | `run_train_nav.sh`, `run_eval_scene.sh` |
| `ENV_ISAACLAB_DIR` | `~/miniconda3/envs/env_isaaclab` | `run_train_nav.sh`, `run_eval_scene.sh` |
| `VLLM_ENV_BIN` | `~/miniconda3/envs/vllm/bin` | `run_vllm_server.sh` |
| `CONDA_DIR` | `~/miniconda3` | `launch_servers.sh` (VLM env) |
| `LEROBOT_CONDA_DIR` | `~/yes` | `launch_servers.sh` (VLA env; 표준설치면 `~/miniconda3`) |
| `PI05_BASE_DIR` / `PI05_BASE` | `pi05_base` / `~/h100_deploy/base_model` | `download_models.sh`, `setup_viva.sh`, `train_v5.sh` |
| `LEROBOT_BIN` / `PYTHON_BIN` | `~/yes/envs/lerobotpi0v2/bin/*` | `train_v5.sh`, `train_pi05.sh` |
| `DATASET_ROOT` | `~/lerobot_data/lekiwi_viva_v{5,2}` | `train_v5.sh`, `train_pi05.sh` |
| `OBJ` | (ISAAC_OBJECTS_DIR 기반) | `run_teleop.sh` (텔레옵 물체 override) |

setup 스크립트 자체의 조정용: `CONDA_BASE`, `ISAACLAB_DIR`, `MOLMO_DIR`, `ENV_NAME`, `MODE`, `OLD_HOST`
(각 스크립트 상단 참조).
