# A100 Server Setup & Transfer Guide

## 1. SSH 접속 정보

```
Host: 218.148.55.186
Port: 30179
User: jovyan
Key: ~/.ssh/private.pem

ssh -p 30179 -i ~/.ssh/private.pem jovyan@218.148.55.186
```

---

## 2. 전송 완료 파일

Desktop에서 서버로 전송 완료된 파일 목록:

| 파일 | 서버 경로 | 크기 | 용도 |
|------|----------|------|------|
| 로봇 USD | `~/Downloads/lekiwi_robot.usd` | 7MB | Isaac Sim 로봇 로드 |
| **로봇 URDF/메쉬** | **`~/lekiwi/`** | **4.7GB** | **USD 내부 참조 (mesh, URDF)** |
| `tuned_dynamics.json` | `~/IsaacLab/calibration/` | 141KB | dynamics DR 파라미터 |
| `arm_limits_measured.json` | `~/IsaacLab/calibration/` | 2.1KB | arm joint limits (**신버전, 반드시 이 파일 사용**) |
| `calibration_latest.json` | `~/IsaacLab/calibration/` | 974KB | 실로봇 캘리브레이션 참조 |
| 프로젝트 코드 (41 py) | `~/IsaacLab/scripts/lekiwi_nav_env/` | - | 전체 학습/수집 코드 |
| 텔레옵 HDF5 (10개) | `~/IsaacLab/scripts/lekiwi_nav_env/demos/` | 92KB | BC 학습 데이터 |
| `object_catalog.json` | `~/IsaacLab/scripts/lekiwi_nav_env/` | 5KB | 대표 12종 물체 |
| `object_catalog_all.json` | `~/IsaacLab/scripts/lekiwi_nav_env/` | 445KB | 전체 1030종 물체 |
| 물체 USD (1030종) | `~/isaac-objects/` | 3.8GB | 다중 물체 RL 학습 |

### 2-1. 로봇 URDF/메쉬 전송 (필수)

`lekiwi_robot.usd`는 내부적으로 `/home/jovyan/lekiwi/urdf/lekiwi/lekiwi.usd`를 참조한다. 이 USD가 다시 mesh 파일들을 참조하므로 **`~/lekiwi/` 디렉토리 전체가 서버에 있어야 한다.**

```bash
# Desktop → 서버 전체 전송 (최초 1회)
rsync -avz --progress -e "ssh -i ~/.ssh/private.pem -p 30179" \
    ~/lekiwi/ jovyan@218.148.55.186:~/lekiwi/
```

**메쉬가 없으면**: Isaac Sim이 로봇을 로드하지만 PhysX articulation만 생성되고 시각적 메쉬가 없는 "빈 깡통" 상태가 된다. RL 학습(state-only)은 물리 엔진만 사용하므로 **메쉬 없이도 학습은 정상 동작**하지만, 데이터 수집(카메라 렌더링)에서는 메쉬가 필수.

---

## 3. 서버 환경 설치

### 3-1. Conda 설치 위치 (주의)

서버에 **두 개의 conda 설치**가 있을 수 있다:
- `~/yes/` — 기존 설치 (일부 Isaac Sim 확장 누락 가능)
- `~/miniconda3/` — 새 설치 (**학습에 사용하는 환경**)

**학습에 사용하는 conda**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl_train
```

> **참고**: `~/.bashrc`에 conda init이 추가되어 있으면 interactive shell에서는 자동 활성화되지만, `nohup`이나 `ssh remote command`는 non-interactive이므로 반드시 `source ~/miniconda3/etc/profile.d/conda.sh`를 명시해야 한다.

### 3-2. Isaac Sim EULA 수락 (non-interactive 필수)

`nohup`으로 Isaac Sim을 처음 실행하면 EULA 동의 프롬프트에서 `EOFError`가 발생한다. 사전에 수동으로 EULA 파일을 생성해야 한다:

```bash
echo 'yes' > ~/miniconda3/envs/rl_train/lib/python3.11/site-packages/omni/EULA_ACCEPTED
```

### 3-3. 환경 설치 스크립트

```bash
cd ~/IsaacLab/scripts/lekiwi_nav_env
bash feedback/setup_server_env.sh          # 전체 (VLM + RL)
bash feedback/setup_server_env.sh vlm      # VLM만
bash feedback/setup_server_env.sh rl       # BC/RL만
```

스크립트가 생성하는 conda 환경:
- `inference`: Qwen2.5-VL-7B-Instruct (VLM orchestrator, ~15GB VRAM)
- `rl_train`: Isaac Sim 5.0.0.0 (headless) + Isaac Lab v2.2.0 + skrl 1.4.3

### 3-4. Isaac Lab 설치 (스크립트에 포함)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout v2.2.0
./isaaclab.sh --install
```

**주의**: IsaacLab을 clone하면 `~/IsaacLab/scripts/lekiwi_nav_env/` 가 덮어써질 수 있음.
프로젝트 코드를 먼저 백업하거나, clone 후 프로젝트 코드를 다시 복사할 것.

### 3-5. 물체 USD 심볼릭 링크

`object_catalog.json`의 USD 경로가 `/home/yubin/isaac-objects/`를 참조함.
서버에서는 `/home/jovyan/isaac-objects/`에 있으므로 심링크 생성 필요:

```bash
sudo mkdir -p /home/yubin
sudo ln -s /home/jovyan/isaac-objects /home/yubin/isaac-objects
```

sudo 권한 없으면 `object_catalog.json`의 경로를 직접 수정:
```bash
sed -i 's|/home/yubin/|/home/jovyan/|g' ~/IsaacLab/scripts/lekiwi_nav_env/object_catalog.json
sed -i 's|/home/yubin/|/home/jovyan/|g' ~/IsaacLab/scripts/lekiwi_nav_env/object_catalog_all.json
```

### 3-6. 로봇 USD 환경변수

코드가 `LEKIWI_USD_PATH` 환경변수를 참조함 (기본값: `/home/yubin11/Downloads/lekiwi_robot.usd`).
서버에서는 경로가 다르므로 설정 필요:

```bash
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd
# .bashrc에 추가 권장
echo 'export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd' >> ~/.bashrc
```

> **주의**: `.bashrc`에 추가해도 `nohup`/`ssh remote command`(non-interactive)에서는 `.bashrc`가 소싱되지 않는다. 학습 명령에서 `export LEKIWI_USD_PATH=...`를 **반드시 명시**해야 한다.

---

## 4. 서버 conda 환경 구성 (최종)

| 환경 | 용도 | Phase | Conda 경로 |
|------|------|-------|-----------|
| `rl_train` | BC/RL 학습 (Isaac Sim headless) | Phase 1 | `~/miniconda3/` |
| `inference` | Qwen2.5-VL-7B 추론 | Phase 5 | `~/miniconda3/` |
| `lerobotpi0` | pi0-FAST + LeRobot (VLA 파인튜닝) | Phase 4 | `~/miniconda3/` |
| `groot` | GR00T N1.6 (VLA 파인튜닝) | Phase 4 | `~/miniconda3/` |

---

## 5. Desktop/Server 역할 분담

| 단계 | 장비 | 이유 |
|------|------|------|
| Phase 0: 캘리브레이션 | 3090 Desktop | 실로봇 USB 연결, GUI |
| Phase 1: 텔레옵 | 3090 Desktop | 리더암 TCP, GUI 확인 |
| Phase 1: BC/RL 학습 | A100 서버 | state-only, VRAM 40GB, 렌더링 불필요 |
| Phase 2: 데이터 수집 | 3090 Desktop | 카메라 렌더링, RT Core |
| Phase 3: 데이터 변환 | 어디든 | CPU 작업 |
| Phase 4: VLA 파인튜닝 | A100 서버 | VRAM 40GB |
| Phase 5: 배포 | A100 서버 + Jetson | VLM + VLA 추론 |

### 서버 하드웨어 스펙

| 항목 | 스펙 |
|------|------|
| GPU | NVIDIA A100-SXM4-40GB |
| CPU | 2× AMD EPYC 7343 16-Core (64 threads) |
| RAM | 1TB |
| CUDA | 12.4 |
| Driver | 550.127.08 |

---

## 6. 데이터 전송 워크플로우

```
[Phase 1]
Desktop → 서버: 프로젝트 코드 전체
  rsync -avz --progress -e "ssh -i ~/.ssh/private.pem -p 30179" \
      --exclude='*.hdf5' --exclude='logs/' --exclude='__pycache__' --exclude='outputs/' \
      ~/IsaacLab/scripts/lekiwi_nav_env/ \
      jovyan@218.148.55.186:~/IsaacLab/scripts/lekiwi_nav_env/

Desktop → 서버: 텔레옵 HDF5
  scp -P 30179 -i ~/.ssh/private.pem -r demos/ jovyan@218.148.55.186:~/IsaacLab/scripts/lekiwi_nav_env/

서버 → Desktop: RL checkpoint
  scp -P 30179 -i ~/.ssh/private.pem jovyan@218.148.55.186:~/IsaacLab/scripts/lekiwi_nav_env/logs/*/checkpoints/best_agent.pt ./

[Phase 2]
Desktop → 서버: 수집 HDF5 + 이미지
  scp -P 30179 -i ~/.ssh/private.pem -r outputs/ jovyan@218.148.55.186:~/IsaacLab/scripts/lekiwi_nav_env/

[Phase 4]
서버에서 LeRobot v3 변환 + VLA 파인튜닝 진행
```

---

## 7. 학습 명령어

### 서버 학습 공통 프리앰블

`nohup`/`ssh remote command`에서는 conda와 환경변수가 자동 설정되지 않으므로, **모든 학습 명령 앞에 반드시 포함**:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env
```

### BC 학습 (서버)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \

# Skill-2
python train_bc.py --demo_dir demos/ --epochs 200 --expected_obs_dim 30

# Skill-3
python train_bc.py --demo_dir demos_skill3/ --epochs 200 --expected_obs_dim 29
```

### RL 학습 (서버)

#### Skill-1 Navigate (BC 없이 from scratch) — 학습 완료 (2026-02-22)

> **결과**: 8192 envs, ~10500 steps (~20분)에 수렴. direction_compliance 93.8% (평균), collision_rate 1.1%.
> best_agent.pt: `logs/ppo_navigate/ppo_navigate_scratch/checkpoints/best_agent.pt`

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
mkdir -p logs && \
nohup python train_lekiwi.py \
    --skill navigate \
    --num_envs 8192 \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json \
    --headless > logs/train_navigate.log 2>&1 &
```

**Early stopping** (navigate는 자동 활성): `direction_compliance` rolling avg >= 0.93 (window=500)이면 자동 중단.
수동 설정:
```bash
--early_stop_metric direction_compliance --early_stop_threshold 0.93 --early_stop_window 500
```

#### Skill-2 ApproachAndGrasp (BC warm-start)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
mkdir -p logs && \
nohup python train_lekiwi.py \
    --skill approach_and_grasp \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_skill2.pt \
    --multi_object_json object_catalog.json \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json \
    --headless > logs/train_skill2.log 2>&1 &
```

#### Skill-3 CarryAndPlace (BC warm-start + handoff buffer)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
mkdir -p logs && \
nohup python train_lekiwi.py \
    --skill carry_and_place \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_skill3.pt \
    --handoff_buffer handoff_buffer.pkl \
    --multi_object_json object_catalog.json \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json \
    --headless > logs/train_skill3.log 2>&1 &
```

### 학습 모니터링

```bash
# 프로세스 확인
ps aux | grep train_lekiwi | grep -v grep

# GPU 상태
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# 로그 진행 확인 (tqdm은 버퍼링되어 실시간 안 보일 수 있음)
tail -5 logs/train_navigate.log

# 체크포인트 확인 (500 step 간격으로 저장됨)
ls -lt logs/ppo_navigate/ppo_navigate_scratch/checkpoints/ | head -5
```

> **tqdm 버퍼링 주의**: `nohup`으로 실행 시 Python stdout이 fully buffered되어, tqdm 진행률이 로그 파일에 실시간 반영되지 않을 수 있다. 체크포인트 파일의 타임스탬프로 진행을 확인하는 것이 더 정확하다.

### TensorBoard 확인

#### TensorBoard 파일 위치

skrl이 자동 저장하며, 학습 시작 시 생성된다:

```
logs/ppo_navigate/ppo_navigate_scratch/
├── checkpoints/          # agent_500.pt, agent_1000.pt, ..., best_agent.pt
└── events.out.tfevents.* # TensorBoard events 파일
```

Skill별 디렉토리:
- Navigate: `logs/ppo_navigate/ppo_navigate_scratch/`
- Skill-2: `logs/ppo_lekiwi/skill2_bc_finetune/` (또는 `skill2_scratch/`)
- Skill-3: `logs/ppo_lekiwi/skill3_bc_finetune/`

#### 기록되는 메트릭 (Navigate 기준)

| 카테고리 | 메트릭 | 설명 |
|---------|--------|------|
| **Info** | `direction_compliance` | 방향 명령 추종률 (목표: 95%+) |
| **Info** | `collision_rate` | 장애물 충돌 비율 (목표: <5%) |
| **Info** | `avg_speed` | 평균 이동 속도 (m/s) |
| **Info** | `min_obstacle_dist` | 최소 장애물 거리 |
| **Info** | `rew_direction/collision/proximity/smooth` | 개별 보상 항목 |
| **Reward** | `Total reward (mean/max/min)` | 에피소드 누적 보상 |
| **Loss** | `Policy/Value/Entropy loss` | PPO 손실 |
| **Policy** | `Standard deviation` | 탐색 분산 (수렴 시 감소) |
| **Learning** | `Learning rate` | KLAdaptiveLR 현재값 |

#### Desktop에서 TensorBoard 보기 (SSH 터널링)

서버에서 직접 TensorBoard를 실행하고, SSH 터널로 Desktop 브라우저에서 확인:

```bash
# 1. Desktop에서 SSH 터널 + TensorBoard 실행 (한 줄)
ssh -i ~/.ssh/private.pem -p 30179 -L 6006:localhost:6006 jovyan@218.148.55.186 \
    "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
     tensorboard --logdir ~/IsaacLab/scripts/lekiwi_nav_env/logs/ --port 6006 --bind_all"

# 2. Desktop 브라우저에서 열기
#    http://localhost:6006
```

또는 TensorBoard events 파일을 Desktop으로 복사하여 로컬에서 확인:

```bash
# 서버 → Desktop: TensorBoard 파일만 복사 (가벼움, 수십~수백KB)
scp -P 30179 -i ~/.ssh/private.pem -r \
    jovyan@218.148.55.186:~/IsaacLab/scripts/lekiwi_nav_env/logs/ppo_navigate/ \
    ~/IsaacLab/scripts/lekiwi_nav_env/logs/ppo_navigate_server/

# Desktop에서 TensorBoard 실행
tensorboard --logdir ~/IsaacLab/scripts/lekiwi_nav_env/logs/ --port 6006
```

#### Python으로 메트릭 직접 읽기 (SSH 원격)

```bash
ssh -i ~/.ssh/private.pem -p 30179 jovyan@218.148.55.186 \
    "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
     python3 -c \"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
files = sorted(glob.glob('logs/ppo_navigate/ppo_navigate_scratch/events.out.tfevents*'))
ea = EventAccumulator(files[-1])
ea.Reload()
for t in ['Info / direction_compliance', 'Info / collision_rate', 'Reward / Total reward (mean)']:
    vals = ea.Scalars(t)
    if vals: print(f'{t}: {vals[-1].value:.4f} @ step {vals[-1].step}')
\""
```

---

### GUI 시각 검증 (3090 Desktop)

A100 서버는 headless 전용 (RT Core 없음, 카메라 렌더링 불가). 학습 결과를 **시각적으로 확인**하려면 체크포인트를 Desktop으로 가져와서 GUI 모드로 실행해야 한다.

#### 체크포인트 가져오기

```bash
# 서버 → Desktop: best_agent.pt 복사
scp -P 30179 -i ~/.ssh/private.pem \
    jovyan@218.148.55.186:~/IsaacLab/scripts/lekiwi_nav_env/logs/ppo_navigate/ppo_navigate_scratch/checkpoints/best_agent.pt \
    ~/IsaacLab/scripts/lekiwi_nav_env/logs/ppo_navigate/ppo_navigate_scratch/checkpoints/
```

#### Desktop에서 GUI 모드 실행

```bash
# Desktop (3090)
conda activate env_isaaclab
source ~/isaacsim/setup_conda_env.sh
cd ~/IsaacLab/scripts/lekiwi_nav_env

# Navigate: GUI로 1~4 envs (--headless 제거)
python train_lekiwi.py \
    --skill navigate \
    --num_envs 4 \
    --max_iterations 10 \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json
# → Isaac Sim GUI 창에서 로봇의 방향 이동 + 장애물 회피 행동을 시각적으로 확인

# 또는 collect_demos.py로 렌더링 확인 (카메라 포함)
python collect_demos.py \
    --checkpoint logs/ppo_navigate/ppo_navigate_scratch/checkpoints/best_agent.pt \
    --skill navigate \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json \
    --num_envs 1 --num_demos 5
```

#### GUI 검증 체크리스트

- [ ] 로봇이 방향 명령대로 이동하는가 (forward → +Y, turn_left → 반시계)
- [ ] 장애물 근처에서 회피 기동이 보이는가
- [ ] arm이 TUCKED_POSE에 고정되어 있는가
- [ ] gripper가 open 상태 유지인가
- [ ] 로봇 메쉬가 정상 렌더링되는가 (빈 깡통 아닌지)
- [ ] 카메라 뷰(base_cam, wrist_cam)가 정상인가

### 데이터 수집 (Desktop)
```bash
conda activate env_isaaclab
source ~/isaacsim/setup_conda_env.sh
cd ~/IsaacLab/scripts/lekiwi_nav_env

python collect_demos.py \
    --checkpoint logs/ppo_lekiwi/skill2/checkpoints/best_agent.pt \
    --skill approach_and_grasp \
    --multi_object_json object_catalog.json \
    --dynamics_json calibration/tuned_dynamics.json \
    --arm_limit_json calibration/arm_limits_measured.json \
    --num_envs 4 --num_demos 1000 --headless
```

**주의**: Desktop에서 `--dynamics_json`, `--arm_limit_json` 경로는 상대경로 `calibration/`을 쓰는데, 이 파일들은 `~/IsaacLab/calibration/`에 있음. Desktop에서는 심링크를 만들거나 절대경로 사용:
```bash
ln -s ~/IsaacLab/calibration/tuned_dynamics.json calibration/tuned_dynamics.json
ln -s ~/IsaacLab/calibration/arm_limits_measured.json calibration/arm_limits_measured.json
```

---

## 8. num_envs 스케일링 가이드

### A100 성능 벤치마크 (Navigate, 2026-02-22 측정)

| num_envs | VRAM 사용 | GPU Util | 속도 | 72000 step ETA | 비고 |
|----------|----------|----------|------|---------------|------|
| 2048 | ~5 GB | 83% | ~16 it/s | ~75분 | 3090/A100 공통 안전값 |
| 8192 | ~11 GB | 78% | ~8 it/s | ~150분 | A100 전용, 4x 샘플/iter |

- **3090 Desktop (24GB)**: 2048 envs 권장. 8192는 OOM 위험.
- **A100 (40GB)**: 8192 envs까지 안정 (VRAM 11/40GB). 4096도 가능.
- **PhysX 버퍼 주의**: 8192 envs 이상에서는 `PhysxCfg(gpu_max_rigid_patch_count=2**18)` 설정 필수. 기본값(163840)에서 `Patch buffer overflow` 에러 발생. `lekiwi_skill1_env.py`에 이미 적용됨.

### num_envs 선택 기준

- **빠른 wall-clock**: 2048 envs (16 it/s)
- **높은 샘플 효율**: 8192 envs (iter당 8192×24=196K samples vs 2048×24=49K)
- **수렴 속도**: 8192 envs는 wall-clock은 느리지만 gradient 품질이 높아 적은 iteration으로 수렴 가능

---

## 9. 핵심 설계 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **핸드오프 지점** | **0.7m** | Navigate→ApproachAndGrasp 전환 거리 (VLM 판단) |
| Navigate 방식 | Direction-Conditioned RL | VLM 방향 명령 추종 + 장애물 회피 |
| Navigate 방향 명령 | 6가지 cardinal | forward/backward/left/right/turn_left/turn_right |
| Navigate rewards | direction=3.0, collision=-2.0, proximity=-0.5, smooth=-0.005 | |
| Navigate episode | 10초 | 도착 조건 없음, timeout까지 방향 추종 |
| Navigate PPO | entropy=0.005, dead dims 0:6 frozen, clip_values=False | |
| Skill-2 object_dist_min | 0.7m | ApproachAndGrasp curriculum 시작 거리 |
| Skill-2 curriculum_current_max_dist | 0.7m | Curriculum 런타임 시작값 |
| Skill-2 curriculum → max | 2.5m | Curriculum 최대 거리 |

---

## 10. 트러블슈팅

### 10-1. `FileNotFoundError: USD file not found`

```
FileNotFoundError: USD file not found at '/home/yubin11/Downloads/lekiwi_robot.usd'
```

**원인**: `LEKIWI_USD_PATH` 환경변수 미설정. non-interactive shell에서는 `.bashrc`가 로드되지 않음.
**해결**: 학습 명령 앞에 `export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd` 추가.

### 10-2. `Could not open asset @.../lekiwi.usd@`

```
Could not open asset @/home/jovyan/lekiwi/urdf/lekiwi/lekiwi.usd@
```

**원인**: `lekiwi_robot.usd`가 내부적으로 `~/lekiwi/urdf/lekiwi/` 경로의 URDF/mesh 파일을 참조하는데, 서버에 해당 디렉토리가 없음.
**해결**: Desktop에서 `~/lekiwi/` 전체(4.7GB)를 서버로 rsync (섹션 2-1 참조).

### 10-3. `ValueError: No contact sensors added to the prim`

```
ValueError: No contact sensors added to the prim ... no rigid bodies are present
```

**원인**: Navigate 환경(`lekiwi_skill1_env.py`)에서 contact sensor가 불필요한데 기본 활성화되어 있음.
**해결**: 이미 `lekiwi_skill1_env.py`에 `robot_cfg.spawn.activate_contact_sensors = False` 적용됨.

### 10-4. `Patch buffer overflow detected`

```
PhysX error: Patch buffer overflow detected, please increase its size to at least 170478
```

**원인**: 8192 envs 이상에서 PhysX GPU contact patch 버퍼 기본값(163840)이 부족.
**해결**: `SimulationCfg`에 `PhysxCfg(gpu_max_rigid_patch_count=2**18)` 추가. `lekiwi_skill1_env.py`에 이미 적용됨. Skill-2/3에서 8192 envs 사용 시에도 동일 설정 필요.

### 10-5. `EOFError: EOF when reading a line` (EULA)

**원인**: Isaac Sim 첫 실행 시 EULA 동의 프롬프트가 non-interactive에서 EOF.
**해결**: 섹션 3-2 참조 — EULA 파일 사전 생성.

### 10-6. 3090에서 OOM (학습 중 프로세스 무언 종료)

**증상**: 로그에 에러/traceback 없이 프로세스가 사라짐. `dmesg`에 OOM kill 로그.
**원인**: 3090 (24GB VRAM)에서 2048 envs로도 VRAM이 부족할 수 있음.
**해결**: A100 서버에서 학습. 3090은 텔레옵/데이터 수집 전용.

### 10-7. tqdm 진행률이 로그에 안 보임

**원인**: `nohup` 리다이렉트 시 Python stdout이 fully buffered됨. tqdm의 `\r` 업데이트가 파일에 쌓이지만 flush 주기가 길 수 있음.
**해결**: 체크포인트 파일 타임스탬프로 진행 확인. 또는 `PYTHONUNBUFFERED=1` 추가 (로그 크기 증가 주의).

---

## 11. 주의사항

1. **Isaac Lab clone 시 프로젝트 코드 보존**: `./isaaclab.sh --install`은 `scripts/` 안의 기존 파일을 건드리지 않지만, `git clone` 시 덮어쓸 수 있음. 프로젝트 코드(`scripts/lekiwi_nav_env/`)를 별도 백업 후 clone, 다시 복사 권장.

2. **`--dynamics_json` 경로**: 서버에서는 절대경로 `~/IsaacLab/calibration/tuned_dynamics.json` 사용. Desktop에서 프로젝트 내 `calibration/` 안에는 이 파일이 없음.

3. **A100 렌더링 제한**: A100에는 RT Core가 없으므로 카메라 렌더링이 필요한 데이터 수집(Phase 2)은 반드시 Desktop(3090)에서 수행.

4. **물체 USD 경로**: `object_catalog.json`이 `/home/yubin/isaac-objects/`를 참조. 서버에서 심링크 또는 sed 수정 필요 (섹션 3-5 참조).

5. **non-interactive shell**: `nohup`, `ssh user@host "command"` 모두 non-interactive. conda, 환경변수, `.bashrc` 내용이 자동 로드되지 않으므로 명령에 직접 포함해야 한다.

6. **logs 디렉토리**: 학습 전 `mkdir -p logs` 필요. 없으면 nohup 리다이렉트 실패.
