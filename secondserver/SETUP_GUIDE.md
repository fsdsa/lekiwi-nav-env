# Second Server Setup Guide

새 서버에서 ResiP combined_s2_s3 학습을 실행하기 위한 전체 가이드.

## 1. 요구 사항

| 항목 | 최소 사양 |
|------|----------|
| GPU | NVIDIA A100 40GB 이상 (1024 envs 기준 ~5GB VRAM) |
| NVIDIA Driver | >= 535 (CUDA 12.x 호환) |
| RAM | 32GB+ |
| Disk | 50GB+ (Isaac Sim ~15GB, IsaacLab ~2GB, 모델/데이터 ~1GB) |
| OS | Ubuntu 22.04 LTS (권장) |

## 2. 전송해야 할 파일 목록

### 2-1. Python 소스 코드 (6개, ~360KB)

모두 `lekiwi_nav_env/` 디렉토리 내 파일:

| 파일 | 크기 | 역할 |
|------|------|------|
| `train_resip.py` | 196KB | 메인 학습 스크립트 |
| `lekiwi_skill2_eval.py` | 104KB | Skill-2 환경 (combined_s2_s3 모드) |
| `diffusion_policy.py` | 28KB | Diffusion BC + Residual Policy |
| `skill3_bc_obs.py` | 20KB | Skill-3 관측 빌더 |
| `lekiwi_robot_cfg.py` | 8KB | 로봇 USD/관절 설정 |
| `__init__.py` | 4KB | Gym 등록 |

### 2-2. 체크포인트 (3개, ~44MB)

| CLI 인자 | 상대 경로 | 크기 |
|----------|----------|------|
| `--bc_checkpoint` | `checkpoints/dp_bc_small/dp_bc_epoch150.pt` | 21MB |
| `--s2_resip_checkpoint` | `backup/appoachandlift/resip64%.pt` | 1.8MB |
| `--s3_bc_checkpoint` | `checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt` | 21MB |

### 2-3. 보정 파일 (1개, 4KB)

| 파일 | 상대 경로 |
|------|----------|
| 관절 한계 | `calibration/arm_limits_measured.json` |

### 2-4. USD 파일 (3개, ~12MB)

| 용도 | 경로 | 크기 | 참고 |
|------|------|------|------|
| 로봇 | `lekiwi_robot.usd` | 7MB | 단일 파일 |
| Source 물체 (5_HTP) | `5_HTP/` 전체 디렉토리 | 2.9MB | **디렉토리 전체 복사** (colliders/, texture.png 포함) |
| Dest 물체 (ACE Coffee Mug) | `ACE_Coffee_Mug_Kristen_16_oz_cup/` 전체 디렉토리 | 1.9MB | **디렉토리 전체 복사** |

> **중요**: USD 물체는 `model_clean.usd` 단독이 아닌 **디렉토리 전체**를 복사해야 합니다.
> 내부에 `colliders/*.usd`, `texture.png` 등이 참조됩니다.

## 3. 서버 디렉토리 구조 (목표)

```
$HOME/
├── Downloads/
│   └── lekiwi_robot.usd                         # 로봇 USD
├── isaac-objects/
│   └── mujoco_scanned_objects/models/
│       ├── 5_HTP/                                # source 물체 (전체 디렉토리)
│       │   ├── model_clean.usd
│       │   ├── colliders/
│       │   └── texture.png
│       └── ACE_Coffee_Mug_Kristen_16_oz_cup/     # dest 물체 (전체 디렉토리)
│           ├── model_clean.usd
│           ├── colliders/
│           └── texture.png
├── isaacsim/                                      # Isaac Sim 5.0 설치
├── IsaacLab/                                      # Isaac Lab 2.2.0 클론
│   ├── _isaac_sim -> $HOME/isaacsim               # 심볼릭 링크
│   └── scripts/
│       └── lekiwi_nav_env/                        # 프로젝트 코드
│           ├── train_resip.py
│           ├── diffusion_policy.py
│           ├── skill3_bc_obs.py
│           ├── lekiwi_skill2_eval.py
│           ├── lekiwi_robot_cfg.py
│           ├── __init__.py
│           ├── calibration/
│           │   └── arm_limits_measured.json
│           ├── checkpoints/
│           │   ├── dp_bc_small/
│           │   │   └── dp_bc_epoch150.pt
│           │   ├── dp_bc_skill3_55d_fixed_1e-4/
│           │   │   └── dp_bc_epoch500.pt
│           │   └── resip_s3_v19/                  # 학습 결과 저장 (자동 생성)
│           └── backup/
│               └── appoachandlift/
│                   └── resip64%.pt
└── miniconda3/
    └── envs/
        └── rl_train/                              # Conda 환경
```

## 4. 파일 전송 방법

로컬 머신에서 실행 (secondserver 디렉토리의 `transfer_files.sh` 사용):

```bash
# 1. 먼저 서버 정보 설정
export SERVER_USER=<username>
export SERVER_HOST=<ip>
export SERVER_PORT=<ssh_port>  # 기본 22

# 2. 전송 스크립트 실행
bash secondserver/transfer_files.sh $SERVER_USER $SERVER_HOST $SERVER_PORT
```

또는 수동 전송:

```bash
# 소스 코드 + 체크포인트 + 보정 파일 (tar로 묶어서 전송)
cd ~/IsaacLab/scripts/lekiwi_nav_env
tar czf /tmp/lekiwi_train_bundle.tar.gz \
    train_resip.py diffusion_policy.py skill3_bc_obs.py \
    lekiwi_skill2_eval.py lekiwi_robot_cfg.py __init__.py \
    calibration/arm_limits_measured.json \
    checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    backup/appoachandlift/resip64%.pt \
    checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt

scp -P $SERVER_PORT /tmp/lekiwi_train_bundle.tar.gz $SERVER_USER@$SERVER_HOST:~/

# 로봇 USD
scp -P $SERVER_PORT ~/Downloads/lekiwi_robot.usd $SERVER_USER@$SERVER_HOST:~/Downloads/

# 물체 USD (디렉토리 전체)
scp -rP $SERVER_PORT ~/isaac-objects/mujoco_scanned_objects/models/5_HTP \
    $SERVER_USER@$SERVER_HOST:~/isaac-objects/mujoco_scanned_objects/models/
scp -rP $SERVER_PORT ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup \
    $SERVER_USER@$SERVER_HOST:~/isaac-objects/mujoco_scanned_objects/models/
```

## 5. 환경 설치

서버에서 `setup_env.sh` 실행:

```bash
bash secondserver/setup_env.sh
```

상세 단계는 `setup_env.sh` 내부 주석 참조.

## 6. 학습 실행

```bash
bash secondserver/run_training.sh
```

또는 수동 실행:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_train
cd ~/IsaacLab/scripts/lekiwi_nav_env

export LEKIWI_USD_PATH=$HOME/Downloads/lekiwi_robot.usd
export PYTHONUNBUFFERED=1

python train_resip.py \
  --skill combined_s2_s3 \
  --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
  --s2_resip_checkpoint 'backup/appoachandlift/resip64%.pt' \
  --s3_bc_checkpoint checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt \
  --object_usd $HOME/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd $HOME/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --num_envs 1024 \
  --num_env_steps 3000 \
  --total_timesteps 200000000 \
  --s2_lift_hold_steps 200 \
  --s3_curriculum_stage v15_dense \
  --normalize_reward True \
  --init_logstd -2.0 \
  --lr_actor 1e-3 \
  --lr_critic 5e-3 \
  --target_kl 1.5 \
  --ent_coef 0.001 \
  --save_dir checkpoints/resip_s3_v19 \
  --seed 82 \
  --headless
```

## 7. 검증 체크리스트

- [ ] `nvidia-smi` 에서 GPU 인식 확인
- [ ] `conda activate rl_train` 정상 작동
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `python -c "import isaacsim"` 에러 없음
- [ ] `python -c "import isaaclab"` 에러 없음
- [ ] `python -c "import skrl; print(skrl.__version__)"` → 1.4.3
- [ ] `python -c "import diffusers"` 에러 없음
- [ ] `ls $LEKIWI_USD_PATH` → 파일 존재
- [ ] 3개 체크포인트 파일 존재 확인
- [ ] 2개 물체 USD 디렉토리 + colliders/ 존재 확인
- [ ] `calibration/arm_limits_measured.json` 존재 확인

## 8. 버전 정보 (현재 로컬 기준, 2026-04-13)

| 컴포넌트 | 버전 |
|----------|------|
| Isaac Sim | 5.0.0-rc.45 |
| Isaac Lab | 2.2.0 (git tag: 46dff13) |
| Python | 3.11.14 |
| PyTorch | 2.7.0+cu128 |
| CUDA (PyTorch) | 12.8 |
| cuDNN | 9.07.01 |
| skrl | 1.4.3 |
| diffusers | 0.36.0 |
| numpy | 1.26.0 |
| gymnasium | 1.2.0 |
| h5py | 3.15.1 |

## 9. 주의 사항

1. **`LEKIWI_USD_PATH` 환경변수 필수** - 설정 안 하면 `/home/yubin11/Downloads/lekiwi_robot.usd` 기본값이 적용되어 서버에서 파일 없음 에러 발생
2. **`--headless` 필수** - 서버에는 디스플레이 없음
3. **`PYTHONUNBUFFERED=1`** - nohup 실행 시 로그 실시간 flush
4. **USD 물체 디렉토리 전체 복사** - `model_clean.usd`만 복사하면 collider/texture 참조 실패
5. **A100 40GB에서 1024 envs ~5GB VRAM** - 2048 envs도 가능하지만 `gpu_max_rigid_patch_count` 설정 필요할 수 있음 (8192+ envs일 때 `2**18`)
6. **서버 학습 로그**: 매번 새 파일명 사용 (덮어쓰기 금지). 형식: `logs/resip_s3_v19_YYYYMMDD_HHMMSS.log`
