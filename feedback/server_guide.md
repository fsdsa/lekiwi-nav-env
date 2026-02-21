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
| `tuned_dynamics.json` | `~/IsaacLab/calibration/` | 141KB | dynamics DR 파라미터 |
| `arm_limits_real2sim.json` | `~/IsaacLab/calibration/` | 2.1KB | arm joint limits |
| `calibration_latest.json` | `~/IsaacLab/calibration/` | 974KB | 실로봇 캘리브레이션 참조 |
| 프로젝트 코드 (41 py) | `~/IsaacLab/scripts/lekiwi_nav_env/` | - | 전체 학습/수집 코드 |
| 텔레옵 HDF5 (10개) | `~/IsaacLab/scripts/lekiwi_nav_env/demos/` | 92KB | BC 학습 데이터 |
| `object_catalog.json` | `~/IsaacLab/scripts/lekiwi_nav_env/` | 5KB | 대표 12종 물체 |
| `object_catalog_all.json` | `~/IsaacLab/scripts/lekiwi_nav_env/` | 445KB | 전체 1030종 물체 |
| 물체 USD (1030종) | `~/isaac-objects/` | 3.8GB | 다중 물체 RL 학습 |

---

## 3. 서버 환경 설치

### 3-1. 환경 설치 스크립트

```bash
cd ~/IsaacLab/scripts/lekiwi_nav_env
bash feedback/setup_server_env.sh          # 전체 (VLM + RL)
bash feedback/setup_server_env.sh vlm      # VLM만
bash feedback/setup_server_env.sh rl       # BC/RL만
```

스크립트가 생성하는 conda 환경:
- `inference`: Qwen2.5-VL-7B-Instruct (VLM orchestrator, ~15GB VRAM)
- `rl_train`: Isaac Sim 5.0.0.0 (headless) + Isaac Lab v2.2.0 + skrl 1.4.3

### 3-2. Isaac Lab 설치 (스크립트에 포함)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout v2.2.0
./isaaclab.sh --install
```

**주의**: IsaacLab을 clone하면 `~/IsaacLab/scripts/lekiwi_nav_env/` 가 덮어써질 수 있음.
프로젝트 코드를 먼저 백업하거나, clone 후 프로젝트 코드를 다시 복사할 것.

### 3-3. 물체 USD 심볼릭 링크

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

### 3-4. 로봇 USD 환경변수

코드가 `LEKIWI_USD_PATH` 환경변수를 참조함 (기본값: `/home/yubin11/Downloads/lekiwi_robot.usd`).
서버에서는 경로가 다르므로 설정 필요:

```bash
export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd
# .bashrc에 추가 권장
echo 'export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd' >> ~/.bashrc
```

---

## 4. 서버 conda 환경 구성 (최종)

| 환경 | 용도 | Phase |
|------|------|-------|
| `rl_train` | BC/RL 학습 (Isaac Sim headless) | Phase 1 |
| `inference` | Qwen2.5-VL-7B 추론 | Phase 5 |
| `lerobotpi0` | pi0-FAST + LeRobot (VLA 파인튜닝) | Phase 4 |
| `groot` | GR00T N1.6 (VLA 파인튜닝) | Phase 4 |

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

---

## 6. 데이터 전송 워크플로우

```
[Phase 1]
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

### BC 학습 (서버)
```bash
conda activate rl_train
cd ~/IsaacLab/scripts/lekiwi_nav_env

# Skill-2
python train_bc.py --demo_dir demos/ --epochs 200 --expected_obs_dim 30

# Skill-3
python train_bc.py --demo_dir demos_skill3/ --epochs 200 --expected_obs_dim 29
```

### RL 학습 (서버)
```bash
conda activate rl_train
cd ~/IsaacLab/scripts/lekiwi_nav_env

# Skill-1 Navigate (BC 없이 from scratch)
python train_lekiwi.py \
    --skill navigate \
    --num_envs 2048 \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_real2sim.json \
    --headless

# Skill-2 ApproachAndGrasp (BC warm-start)
python train_lekiwi.py \
    --skill approach_and_grasp \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_skill2.pt \
    --multi_object_json object_catalog.json \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_real2sim.json \
    --headless

# Skill-3 CarryAndPlace (BC warm-start + handoff buffer)
python train_lekiwi.py \
    --skill carry_and_place \
    --num_envs 2048 \
    --bc_checkpoint checkpoints/bc_skill3.pt \
    --handoff_buffer handoff_buffer.pkl \
    --multi_object_json object_catalog.json \
    --dynamics_json ~/IsaacLab/calibration/tuned_dynamics.json \
    --arm_limit_json ~/IsaacLab/calibration/arm_limits_real2sim.json \
    --headless
```

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
    --arm_limit_json calibration/arm_limits_real2sim.json \
    --num_envs 4 --num_demos 1000 --headless
```

**주의**: Desktop에서 `--dynamics_json`, `--arm_limit_json` 경로는 상대경로 `calibration/`을 쓰는데, 이 파일들은 `~/IsaacLab/calibration/`에 있음. Desktop에서는 심링크를 만들거나 절대경로 사용:
```bash
ln -s ~/IsaacLab/calibration/tuned_dynamics.json calibration/tuned_dynamics.json
ln -s ~/IsaacLab/calibration/arm_limits_real2sim.json calibration/arm_limits_real2sim.json
```

---

## 8. 주의사항

1. **Isaac Lab clone 시 프로젝트 코드 보존**: `./isaaclab.sh --install`은 `scripts/` 안의 기존 파일을 건드리지 않지만, `git clone` 시 덮어쓸 수 있음. 프로젝트 코드(`scripts/lekiwi_nav_env/`)를 별도 백업 후 clone, 다시 복사 권장.

2. **`--dynamics_json` 경로**: 서버에서는 절대경로 `~/IsaacLab/calibration/tuned_dynamics.json` 사용. Desktop에서 프로젝트 내 `calibration/` 안에는 이 파일이 없음.

3. **A100 렌더링 제한**: A100에는 RT Core가 없으므로 카메라 렌더링이 필요한 데이터 수집(Phase 2)은 반드시 Desktop(3090)에서 수행.

4. **물체 USD 경로**: `object_catalog.json`이 `/home/yubin/isaac-objects/`를 참조. 서버에서 심링크 또는 sed 수정 필요 (섹션 3-3 참조).
