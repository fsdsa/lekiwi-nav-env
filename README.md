# LeKiwi Fetch-Navigation RL Pipeline (v7 — Physics Grasp)

Isaac Lab `DirectRLEnv` 기반 LeKiwi Fetch task 파이프라인.
핵심: **Privileged RL Teacher (37D) → 물리 기반 다중 물체 Grasp → VLA Student 증류**.

## 1) v6 → v7 변경점

- **Grasp 판정**: proximity (거리+저속) → **physics-based (gripper 닫힘 + contact force + 적응적 거리)**
- **Observation**: `33D` → **`37D`** (multi-object 모드)
  - `[33:36]` object_bbox_normalized (x, y, z)
  - `[36]` object_category_normalized
  - 기존 33D 위치는 변동 없음 → robot_state 추출 호환
- **다중 물체 RL 학습**: 대표 N종 RigidBody를 pre-spawn, 에피소드마다 랜덤 선택
  - Teacher가 물체 크기/형상을 보고 물체별 다른 grasp 전략 학습
- **object_catalog.json 자동 생성**: `build_object_catalog.py`로 1030종 USD에서 bbox 추출 + 클러스터링 → 대표 선별
- **VLA 입력은 변동 없음**: `image + instruction + robot_state(9D)` → `action(9D)`
  - obs[33:37]은 privileged info로 VLA에 전달하지 않음
- **기존 33D 호환**: `--multi_object_json` 미지정 시 기존 33D proximity 모드 동작

## 2) 파일 구조

```text
scripts/lekiwi_nav_env/
├── __init__.py                  # Gymnasium 등록 (Isaac-LeKiwi-Fetch-Direct-v0)
├── lekiwi_robot_cfg.py          # ArticulationCfg, Kiwi IK (LEKIWI_USD_PATH 환경변수 지원)
├── lekiwi_nav_env.py            # 37D obs, multi-object RigidBody, contact grasp, GRASP timeout
├── models.py                    # ★ 공유 RL 모델 (PolicyNet, ValueNet) — train/collect 공통
├── build_object_catalog.py      # USD bbox 추출 + 대표 물체 선별
├── calibrate_real_robot.py
├── replay_in_sim.py
├── tune_sim_dynamics.py
├── compare_real_sim.py
├── build_arm_limits_real2sim.py
├── record_teleop.py
├── teleop_dual_logger.py
├── sim_action_receiver_logger.py
├── train_bc.py
├── train_lekiwi.py              # skrl PPO (models.py import, --multi_object_json 지원)
├── collect_demos.py             # Expert demo 수집 (models.py import, physics grasp 메타 저장)
├── spawn_manager.py
├── convert_hdf5_to_lerobot_v3.py  # obs 37D 처리, tasks.jsonl 안정적 출력
├── test_env.py                  # 환경 검증 (--dynamics_json, --arm_limit_json 지원)
├── object_catalog.json          # build_object_catalog.py 출력물
├── demos/
├── checkpoints/
├── logs/ppo_lekiwi/
└── outputs/rl_demos/
```

## 3) 환경 스펙

### Observation

기본 33D (multi-object 미사용 시):

| 인덱스 | 항목 | 차원 |
|--------|------|------|
| 0:2 | target_xy_body | 2 |
| 2:4 | object_xy_body | 2 |
| 4:6 | home_xy_body | 2 |
| 6:10 | phase_onehot | 4 |
| 10 | object_visible | 1 |
| 11 | object_grasped | 1 |
| 12:15 | base_lin_vel_body | 3 |
| 15:18 | base_ang_vel_body | 3 |
| 18:24 | arm_joint_pos | 6 |
| 24:30 | arm_joint_vel | 6 |
| 30:33 | wheel_vel | 3 |

Multi-object 37D (추가분):

| 인덱스 | 항목 | 차원 | 비고 |
|--------|------|------|------|
| 33:36 | object_bbox_normalized | 3 | privileged (VLA 미전달) |
| 36 | object_category_normalized | 1 | privileged (VLA 미전달) |

### Action (9D, 변동 없음)

- `[0:3]` body_vel_cmd (vx, vy, wz)
- `[3:9]` arm_pos_target (6 joints, gripper 포함)

### Phase 상태머신

```
SEARCH → APPROACH → GRASP → RETURN → success
                       ↑       │
                       └───────┘ timeout (grasp_timeout_steps=75, ~3초@25Hz)
```

v7 GRASP 판정 (physics mode):
- gripper joint position < threshold (닫힘)
- contact force > threshold (접촉)
- object_dist < adaptive threshold (물체 bbox 크기 반영)
- GRASP timeout: `grasp_timeout_steps` (기본 75 step) 초과 시 APPROACH로 복귀하여 재시도

## 4) 실행 순서

```bash
conda activate env_isaaclab
source ~/isaacsim/setup_conda_env.sh
cd ~/IsaacLab/scripts/lekiwi_nav_env
```

### Step 0. 사전 준비: gripper 정보 확인

lekiwi_robot_cfg.py에 아래 상수가 정확한지 확인:

```python
GRIPPER_JOINT_NAME = "STS3215_03a_v1_4_Revolute_57"
GRIPPER_JOINT_IDX_IN_ARM = 5
```

Isaac Sim에서 lekiwi USD를 열어 gripper finger의 rigid body prim 경로 확인.
이 경로가 `--gripper_contact_prim_path`에 들어감.

### Step 1. 대표 물체 카탈로그 생성

1030종 USD에서 bbox를 자동 추출하고 k-means로 대표 물체를 선별:

```bash
python build_object_catalog.py \
  --index_jsonl ~/isaac-objects/mujoco_obj_usd_index_all.jsonl \
  --output_json object_catalog.json \
  --all_objects_json object_catalog_all.json \
  --num_representatives 12 \
  --scale 1.0 \
  --mass_mode volume_density \
  --density_kg_m3 350
```

출력: `object_catalog.json` (대표 12종, RL 학습용), `object_catalog_all.json` (전체 메타데이터).
pxr 환경(Isaac Sim Python)에서 실행해야 함.

### Step 2. ★ Sim2Real 캘리브레이션 (RL 학습 전 필수)

**Step 3~5의 모든 학습/수집 명령어가 `--dynamics_json`과 `--arm_limit_json`을 사용하므로, 이 단계를 먼저 완료해야 한다.** 캘리브레이션 없이 학습된 정책은 실제 로봇의 dynamics와 불일치하여 Sim2Real 전이 성능이 저하된다.

#### 2-1. 실로봇 측정

```bash
python scripts/lekiwi_nav_env/calibrate_real_robot.py \
  --mode all --connection_mode direct --robot_port /dev/ttyACM0 --sample_hz 20

python scripts/lekiwi_nav_env/calibrate_real_robot.py \
  --mode arm_sysid --sample_hz 50
```

#### 2-2. Arm Joint Limit JSON 생성

```bash
python scripts/lekiwi_nav_env/build_arm_limits_real2sim.py \
  --calibration_json calibration/calibration_latest.json \
  --encoder_calibration_json ~/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_lekiwi.json \
  --output calibration/arm_limits_real2sim.json
```

#### 2-3. Sim 파라미터 튜닝

```bash
python scripts/lekiwi_nav_env/tune_sim_dynamics.py \
  --calibration calibration/calibration_latest.json \
  --iterations 60 --output calibration/tuned_dynamics.json --headless
```

#### 2-4. Replay 검증

```bash
python scripts/lekiwi_nav_env/replay_in_sim.py \
  --calibration calibration/calibration_latest.json \
  --mode command \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --report_path calibration/replay_command_report.json --headless

python scripts/lekiwi_nav_env/replay_in_sim.py \
  --calibration calibration/calibration_latest.json \
  --mode arm_command \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --report_path calibration/replay_arm_report.json --headless

python scripts/lekiwi_nav_env/compare_real_sim.py \
  --input calibration/replay_command_series.json \
  --output_dir calibration/plots

python scripts/lekiwi_nav_env/compare_real_sim.py \
  --input calibration/replay_arm_series.json \
  --output_dir calibration/plots
```

이 단계가 완료되면 `calibration/tuned_dynamics.json`과 `calibration/arm_limits_real2sim.json`이 생성된다. 이후 모든 Step에서 이 파일들을 `--dynamics_json`과 `--arm_limit_json`으로 전달한다.

### Step 3. 환경 검증

```bash
# 기존 proximity 모드 (33D)
python test_env.py --num_envs 4 --headless

# multi-object physics grasp 모드 (37D) + 캘리브레이션 반영
python test_env.py --num_envs 4 \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --headless
```

확인 사항: obs shape (37,), bbox/category 출력, contact force 값, grasp 판정.

### Step 4. RL 학습

```bash
# Multi-object physics grasp (37D, 권장)
python train_lekiwi.py \
  --num_envs 2048 \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --headless

# 단일 물체 physics grasp (33D)
python train_lekiwi.py \
  --num_envs 2048 \
  --object_usd /path/to/grasp_cube.usd \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --headless

# 기존 proximity 모드 (33D, BC warm-start 가능)
python train_lekiwi.py \
  --num_envs 2048 \
  --bc_checkpoint checkpoints/bc_nav.pt \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --headless
```

참고:
- multi-object(37D)는 기존 33D BC 체크포인트와 호환 안 됨 → 자동으로 from scratch 전환
- 학습 완료 후: `logs/ppo_lekiwi/ppo_lekiwi_scratch/checkpoints/best_agent.pt`

### Step 5. RL Expert 데모 수집

```bash
# camera + multi-object physics grasp + v6 annotation (권장)
python collect_demos.py \
  --checkpoint logs/ppo_lekiwi/ppo_lekiwi_scratch/checkpoints/best_agent.pt \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --num_envs 4 --num_demos 200 --headless \
  --annotate_subtasks

# camera + SpawnManager 1030종 시각 다양성 (proximity 모드)
python collect_demos.py \
  --checkpoint logs/ppo_lekiwi/ppo_lekiwi_bc_finetune/checkpoints/best_agent.pt \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --objects_index ~/isaac-objects/mujoco_obj_usd_index_all.jsonl \
  --num_envs 4 --num_demos 200 --headless \
  --annotate_subtasks --object_cap 5 --min_steps 30

# state-only 빠른 수집
python collect_demos.py \
  --checkpoint logs/ppo_lekiwi/ppo_lekiwi_scratch/checkpoints/best_agent.pt \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
  --dynamics_json calibration/tuned_dynamics.json \
  --no_camera --num_envs 64 --num_demos 200 --headless
```

참고:
- physics grasp 모드에서는 SpawnManager(`--objects_index`)가 자동 비활성화됨
- HDF5 attrs에 `object_bbox_xyz`, `object_category_id`, `active_object_type_idx` 저장됨

### Step 6. HDF5 → LeRobot v3 변환

```bash
python convert_hdf5_to_lerobot_v3.py \
  --input outputs/rl_demos/*.hdf5 \
  --output_root outputs/lerobot_v3/lekiwi_fetch_v7 \
  --repo_id local/lekiwi_fetch_v7 \
  --fps 25 \
  --include_subtask_index \
  --skip_episodes_without_images
```

참고:
- obs 37D → `robot_state 9D` 추출은 기존과 동일 위치 (obs[18:24] + obs[30:33])
- obs[33:37] (물체 bbox/category)은 VLA에 전달하지 않음
- `lerobot` 패키지 설치 환경에서 실행

### Step 7. VLA 파인튜닝

LeRobot v3 데이터셋으로 π0Fast 또는 GR00T 파인튜닝 (A100 서버).
VLA 입력: `image + instruction + robot_state(9D)` → `action(9D)`.

## 5) 데이터 흐름 요약

```
build_object_catalog.py
  1030종 USD → bbox 추출 → k-means → 대표 12종 object_catalog.json

calibrate → tune → replay → compare  (★ RL 학습 전 필수)
  실로봇 측정 → tuned_dynamics.json + arm_limits_real2sim.json 생성
  이후 모든 학습/수집에 --dynamics_json --arm_limit_json 전달

train_lekiwi.py (RL 학습)
  Teacher obs 37D = 33D + bbox(3) + category(1)
  대표 12종 RigidBody, contact-based grasp
  → Teacher가 물체별 다른 arm trajectory + gripper timing 학습

collect_demos.py (데이터 수집)
  학습된 Teacher 실행, camera 이미지 저장
  물체별로 다른 action이 기록됨

convert_hdf5_to_lerobot_v3.py
  obs 37D → robot_state 9D 추출 (물체 정보 제외)
  LeRobot v3: (image, instruction, robot_state 9D) → action 9D

VLA 파인튜닝
  Student는 이미지에서 물체를 보고 적절한 action 예측
  추가 privileged info 없이 동작
```

## 6) 호환성

| 항목 | 33D (proximity) | 37D (multi-object) |
|------|----------------|-------------------|
| `--multi_object_json` 미지정 | 기존 동작 | N/A |
| `--multi_object_json` 지정 | N/A | 37D 활성 |
| 기존 BC/RL 체크포인트 | 호환 | 비호환 (재학습 필요) |
| robot_state 추출 | obs[18:24]+obs[30:33] | 동일 |
| VLA 학습 입력 | 9D | 9D (동일) |
| SpawnManager | 사용 가능 | 자동 비활성 |

## 7) 주의사항

- **★ `--dynamics_json`은 Step 2(캘리브레이션) 완료 후 생성됨** — Step 3~5의 학습/수집 전에 반드시 캘리브레이션을 완료할 것
- `--dynamics_json`은 학습/수집 모두 동일한 파일을 사용해야 Sim2Real 정합 유지
- 37D 체크포인트는 33D 환경에서 로드 불가 (역도 마찬가지)
- camera 수집 시 VRAM에 따라 `num_envs`를 1~8로 제한
- `--gripper_contact_prim_path`는 Isaac Sim에서 gripper finger body prim을 직접 확인해서 설정
- `build_object_catalog.py`는 pxr(Isaac Sim Python) 환경에서만 실행 가능
- **USD 경로**: `lekiwi_robot_cfg.py`는 환경변수 `LEKIWI_USD_PATH`를 우선 참조. 미설정 시 기본 경로 사용.
  다른 머신에서 실행 시: `export LEKIWI_USD_PATH=/path/to/lekiwi_robot.usd`
- **모델 공유**: `PolicyNet`/`ValueNet`은 `models.py`에 단일 정의. `train_lekiwi.py`와 `collect_demos.py`가 공통 import하므로 모델 구조 변경 시 `models.py`만 수정
- **GRASP timeout**: `grasp_timeout_steps=75` (기본 ~3초@25Hz). GRASP phase에서 지정 step 내 grasp 미성공 시 APPROACH로 복귀하여 재시도. 0으로 설정하면 무제한
