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
| `object_catalog.json` | `~/IsaacLab/scripts/lekiwi_nav_env/` | 5KB | 대표 22종 물체 |
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
- `rl_train`: Isaac Sim 5.0.0.0 (headless) + Isaac Lab v2.2.0 + skrl 1.4.3
- `vllm`: vLLM 0.17.0 + Qwen2.5-VL-7B-Instruct (VLM 추론, OpenAI-compatible API)
- `lerobotpi0`: π0-FAST + LeRobot (VLA 추론, `~/yes/envs/lerobotpi0`)

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

`object_catalog.json`의 USD 경로가 `/home/yubin11/isaac-objects/`를 참조함.
서버에서는 `/home/jovyan/isaac-objects/`에 있으므로 심링크 생성 필요:

```bash
sudo mkdir -p /home/yubin
sudo ln -s /home/jovyan/isaac-objects /home/yubin11/isaac-objects
```

sudo 권한 없으면 `object_catalog.json`의 경로를 직접 수정:
```bash
sed -i 's|/home/yubin11/|/home/jovyan/|g' ~/IsaacLab/scripts/lekiwi_nav_env/object_catalog.json
sed -i 's|/home/yubin11/|/home/jovyan/|g' ~/IsaacLab/scripts/lekiwi_nav_env/object_catalog_all.json
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

| 환경 | 용도 | Phase | 상태 |
|------|------|-------|------|
| `rl_train` | BC/RL 학습 (Isaac Sim headless) | Phase 1 | ✅ 설치됨 |
| `vllm` | vLLM 0.17.0 + Qwen2.5-VL-7B (VLM 추론, OpenAI API, gpu_util=0.50) | Phase 4.5 + Phase 5 | ✅ 검증 완료 |
| `lerobotpi0` | π0-FAST 2.9B + lerobot 0.4.4 (VLA 파인튜닝 + 추론) | Phase 4 + Phase 4.5 + Phase 5 | ✅ 검증 완료 (`~/yes/envs/lerobotpi0`) |
| `groot` | GR00T N1.6 (VLA 파인튜닝, 선택) | Phase 4 | ⬜ |

**conda 경로 참고:**
- `rl_train`, `vllm`: `~/miniconda3/envs/`
- `lerobotpi0`: `~/yes/envs/lerobotpi0`

**모델 캐시:**
- Qwen2.5-VL-7B-Instruct: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` (다운로드 완료)

**lerobotpi0 환경 설치 (검증 완료 2026-03-10):**
```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git@v0.4.4"
pip install transformers==4.53.3   # Pi0 전용, 최신 버전 NOT 호환
huggingface-cli login              # google/paligemma-3b-pt-224 gated model 접근 필수
```
**필수 패치:**
- `_prepare_attention_masks_4d`에서 `.bool()` 호출 추가 (dtype mismatch 방지)
- `validate_action_token_prefix=False` 설정 (action token prefix 검증 비활성화)

**VLM+VLA 동시 추론 GPU 메모리 분배 (A100 40GB, 검증 완료 2026-03-10):**
```
vLLM (Qwen 7B bf16):  --gpu-memory-utilization 0.50 → ~19.8GB 선점
Pi0-FAST 2.9B (VLA):   ~5.9GB 사용
합계:                  ~25.9GB / 40GB
```
> `--gpu-memory-utilization 0.50`는 vLLM이 KV cache를 위해 GPU 메모리를 선점하는 비율. 0.70 이상이면 VLA가 OOM.

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
| Phase 4.5: Sim Full-System | 3090 Desktop + A100 서버 | Desktop=sim 렌더링+실행, A100=VLM+VLA 추론 |
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

# Skill-2 (GMM 기본, mean regression 방지)
python train_bc.py --demo_dir demos_skill2/ --epochs 300 \
    --expected_obs_dim 30 --filter_active \
    --loss gmm --n_components 5 --eval \
    --save_dir checkpoints/skill2/

# Skill-3 (GMM 기본)
python train_bc.py --demo_dir demos_skill3/ --epochs 300 \
    --expected_obs_dim 29 --filter_active \
    --loss gmm --n_components 5 --eval \
    --save_dir checkpoints/skill3/
```

### RL 학습 (서버)

#### Skill-1 Navigate (ResiP: 텔레옵→DP BC→Residual PPO) — 2026-03-09

> **ResiP 방식 채택**: 순수 RL의 drift 문제 해결. 텔레옵→BC→Residual RL 파이프라인.
> - **BC**: DP BC (diffusion_policy.py), obs=20D, act=9D, epoch250 채택
> - **보상**: lin_track(1.5, std=0.25) + ang_track(1.5, std=0.25) + smooth(-0.005) + time(-0.01)
> - **장애물 회피**: RL에서 제거 — VLM이 2-4Hz로 실시간 판단하여 대체
> - **direction commands**: 6개 (forward/backward/strafe_left/strafe_right/turn_left/turn_right)
> - **action scale**: arm=0, gripper=0, base=0.25 (navigate는 base만 학습)
>
> **체크포인트**:
> - BC: `checkpoints/dp_bc_nav/dp_bc_epoch250.pt`
> - ResiP: `checkpoints/resip_nav/resip_best.pt`

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
mkdir -p logs && \

# 1) BC 학습 (텔레옵 데이터 → DP BC)
nohup python train_diffusion_bc.py \
    --demo_path demos/teleop_navigate.hdf5 \
    --obs_dim 20 --act_dim 9 \
    --epochs 300 --save_every 50 \
    --save_dir checkpoints/dp_bc_nav --save_name dp_bc.pt \
    > logs/train_bc_nav.log 2>&1 &

# 2) Residual RL 학습 (BC frozen + residual PPO)
PYTHONUNBUFFERED=1 nohup python train_resip.py \
    --skill navigate \
    --bc_checkpoint checkpoints/dp_bc_nav/dp_bc_epoch250.pt \
    --num_envs 1024 \
    --num_env_steps 250 \
    --total_timesteps 3000000 \
    --action_scale_base 0.25 \
    --lr_actor 3e-4 --lr_critic 5e-3 \
    --warmup_steps_initial 600 --warmup_steps_final 0 --warmup_decay_iters 30 \
    --eval_interval 3 --eval_first true \
    --save_dir checkpoints/resip_nav \
    --headless > logs/train_resip_nav.log 2>&1 &
```

#### Skill-2 ApproachAndGrasp (BC warm-start + BC auxiliary loss)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
mkdir -p logs && \
nohup python train_lekiwi.py \
    --skill approach_and_grasp \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_skill2.pt \
    --lambda_bc_init 0.5 \
    --bc_anneal_ratio 0.6 \
    --multi_object_json object_catalog.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json \
    --headless > logs/train_skill2.log 2>&1 &
```

#### Skill-3 CarryAndPlace (BC warm-start + BC auxiliary loss + handoff buffer)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl_train && \
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
mkdir -p logs && \
nohup python train_lekiwi.py \
    --skill carry_and_place \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_skill3.pt \
    --lambda_bc_init 0.5 \
    --bc_anneal_ratio 0.6 \
    --handoff_buffer handoff_buffer.pkl \
    --multi_object_json object_catalog.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json \
    --headless > logs/train_skill3.log 2>&1 &
```

### Sim Full-System 평가 (Phase 4.5)

#### A100 서버: 추론 서버 시작

**방법 1: 통합 스크립트 (권장)**
```bash
cd ~/IsaacLab/scripts/lekiwi_nav_env

# VLM + VLA 동시 실행
bash launch_servers.sh all --checkpoint ~/datasets/lekiwi_vla/best_model/

# VLM만
bash launch_servers.sh vlm

# VLA만
bash launch_servers.sh vla --checkpoint ~/datasets/lekiwi_vla/best_model/

# 종료
bash launch_servers.sh stop

# 로그 확인
tail -f logs/vlm_server.log   # VLM (vLLM, port 8000)
tail -f logs/vla_server.log   # VLA (Pi0, port 8002)
```

**방법 2: 수동 실행**

VLM 추론 서버 (vLLM, OpenAI-compatible API):
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
bash run_vllm_server.sh
# → port 8000, gpu-memory-utilization 0.50, ~19.8GB
```

VLA 추론 서버 (Pi0-FAST 2.9B, lerobot 0.4.4):
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobotpi0 && \
cd ~/IsaacLab/scripts/lekiwi_nav_env && \
python vla_inference_server.py \
    --checkpoint ~/datasets/lekiwi_vla/best_model/ \
    --port 8002
# → ~5.9GB VRAM. transformers==4.53.3 필수. HF login 필수.
# → 패치: _prepare_attention_masks_4d .bool(), validate_action_token_prefix=False
```

> lerobotpi0 환경 경로: `~/yes/envs/lerobotpi0` (검증 완료 2026-03-10)

#### USD 카메라 prim 경로 (lekiwi_robot.usd)

로봇 USD에 이미 카메라가 포함되어 있으므로, Isaac Lab `Camera` 센서에 prim_path만 지정하면 된다 (spawn 불필요):

| 카메라 | USD prim 경로 | 용도 |
|--------|---------------|------|
| base RGB | `.../Realsense/RSD455/Camera_OmniVision_OV9782_Color` | VLM instruction + VLA 입력 |
| base depth | `.../Realsense/RSD455/Camera_Pseudo_Depth` | depth safety layer |
| wrist RGB | `.../Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera` | VLA 입력 (정밀 파지) |

Isaac Lab env_pattern 예시:
```
/World/envs/env_.*/Robot/LeKiwi/base_plate_layer1_v5/Realsense/RSD455/Camera_OmniVision_OV9782_Color
/World/envs/env_.*/Robot/LeKiwi/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera
```

> **참고**: `Realsense/RSD455`는 Omniverse 서버의 `rsd455.usd`를 참조한다. Isaac Sim에서 로드 시 자동 resolve되지만, 순수 pxr(로컬)에서는 참조 실패로 빈 Xform으로 보인다.

#### 3090 Desktop: sim 평가 실행

```bash
conda activate env_isaaclab
source ~/isaacsim/setup_conda_env.sh
cd ~/IsaacLab/scripts/lekiwi_nav_env

# VLM + VLA 통합 평가 (전체 task)
python eval_full_system.py \
    --vlm_server http://218.148.55.186:8000 \
    --vla_server http://218.148.55.186:8002 \
    --num_trials 30 \
    --task "find the medicine bottle and place it next to the red cup" \
    --multi_object_json object_catalog.json \
    --arm_limit_json calibration/arm_limits_measured.json

# Navigate만 VLM+BC 평가 (vllm/ 디렉토리)
python vllm/run_vlm_navigate.py \
    --vlm_server http://218.148.55.186:8000 \
    --bc_checkpoint checkpoints/dp_bc_nav/dp_bc_epoch250.pt \
    --target_object "medicine bottle"

# Skill별 단독 평가 (VLM 없이)
python eval_full_system.py \
    --vla_server http://218.148.55.186:8002 \
    --eval_mode skill_only --skill approach_and_grasp \
    --instruction "pick up the red cup" \
    --num_trials 50 \
    --multi_object_json object_catalog.json \
    --arm_limit_json calibration/arm_limits_measured.json
```

### 학습 모니터링

```bash
# 프로세스 확인
ps aux | grep train_lekiwi | grep -v grep

# GPU 상태
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# 로그 진행 확인 (tqdm은 버퍼링되어 실시간 안 보일 수 있음)
tail -5 logs/train_navigate_v6g2_full.log

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
| **Loss** | `BC loss` | BC auxiliary loss (Skill-2/3, λ_bc > 0일 때만) |
| **Loss** | `Lambda BC` | 현재 BC loss weight (λ annealing 추적) |
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
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_measured.json
# → Isaac Sim GUI 창에서 로봇의 방향 이동 + 장애물 회피 행동을 시각적으로 확인

# 또는 collect_demos.py로 렌더링 확인 (카메라 포함)
python collect_demos.py \
    --checkpoint logs/ppo_navigate/ppo_navigate_scratch/checkpoints/best_agent.pt \
    --skill navigate \
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
    --arm_limit_json calibration/arm_limits_measured.json \
    --num_envs 4 --num_demos 1000 --headless
```

**주의**: Desktop에서 `--arm_limit_json` 경로는 상대경로 `calibration/`을 쓰는데, 이 파일은 `~/IsaacLab/calibration/`에 있음. Desktop에서는 심링크를 만들거나 절대경로 사용:
```bash
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
| Navigate 방식 | ResiP (DP BC + Residual PPO) | 텔레옵→BC→Residual RL. VLM이 장애물 판단 |
| Navigate 방향 명령 | 6가지 cardinal | forward/backward/left/right/turn_left/turn_right |
| Navigate rewards | lin_track=1.5(std=0.25), ang_track=1.5(std=0.25), smooth=-0.005, time=-0.01 | tracking only, 장애물 보상 없음 (VLM이 대체) |
| VLM Navigate | vLLM + Qwen2.5-VL-7B (2-4Hz) | D455 base_cam 640×400 → 방향 command |
| Navigate episode | 10초 | 도착 조건 없음, timeout까지 방향 추종 |
| Navigate PPO | entropy=0.005, dead dims 0:6 frozen, clip_values=False | |
| Skill-2 object_dist_min | 0.8m | ApproachAndGrasp 스폰 최소 거리 |
| Skill-2 object_dist_max | 1.2m | 스폰 최대 거리 |
| Skill-2 curriculum_current_max_dist | 1.2m | 처음부터 전체 범위(0.8~1.2m) 사용 |

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

2. **A100 렌더링 제한**: A100에는 RT Core가 없으므로 카메라 렌더링이 필요한 데이터 수집(Phase 2)은 반드시 Desktop(3090)에서 수행.

3. **물체 USD 경로**: `object_catalog.json`이 `/home/yubin11/isaac-objects/`를 참조. 서버에서 심링크 또는 sed 수정 필요 (섹션 3-5 참조).

4. **non-interactive shell**: `nohup`, `ssh user@host "command"` 모두 non-interactive. conda, 환경변수, `.bashrc` 내용이 자동 로드되지 않으므로 명령에 직접 포함해야 한다.

5. **logs 디렉토리**: 학습 전 `mkdir -p logs` 필요. 없으면 nohup 리다이렉트 실패.
