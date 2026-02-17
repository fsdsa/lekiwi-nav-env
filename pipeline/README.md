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
├── check_calibration_gate.py   # ★ 캘리브레이션 품질 게이트 (RMSE 임계값 pass/fail)
├── sim_real_calibration_test.py # ★ Isaac Sim Script Editor용 sim-real 이동/회전 정합 테스트
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

중요:
- `sim_real_calibration_test.py`(Script Editor)는 **보조 검증**이다.
- 이 스크립트는 base의 pose-level(총 이동거리/총 회전각) 정합을 빠르게 확인하는 용도이며, PRE-3~6(SysID/replay/gate)를 대체하지 않는다.
- VLA 파인튜닝 데이터 수집/실배포 판단은 반드시 PRE-3~6 결과(`tuned_dynamics.json`, replay report, gate pass)로 최종 확정한다.

#### 2-1. 실로봇 측정

```bash
# 전체 측정 (wheel/base/arm/rest/sysid)
python scripts/lekiwi_nav_env/calibrate_real_robot.py \
  --mode all --connection_mode direct --robot_port /dev/ttyACM0 --sample_hz 20

# arm 6축 range만 재측정 (권장: 수동 시작/종료)
python scripts/lekiwi_nav_env/calibrate_real_robot.py \
  --mode joint_range --connection_mode direct --robot_port /dev/ttyACM0 \
  --client_id my_awesome_kiwi --joint_range_duration 0 --sample_hz 20

# 단일 관절만 재측정 (예: gripper)
python scripts/lekiwi_nav_env/calibrate_real_robot.py \
  --mode joint_range_single --connection_mode direct --robot_port /dev/ttyACM0 \
  --client_id my_awesome_kiwi --joint_key arm_gripper.pos \
  --joint_range_duration 0 --sample_hz 20

# arm sysid만 별도 측정
python scripts/lekiwi_nav_env/calibrate_real_robot.py \
  --mode arm_sysid --sample_hz 50
```

메모:
- `--joint_range_duration 0` 또는 음수면 고정 시간 대신 Enter로 시작/종료한다.
- `joint_range`/`joint_range_single` 측정 중에는 arm torque가 자동으로 OFF되고, 종료 시 ON으로 복구된다.
- `joint_range_single`은 기존 `wheel_radius`/`base_radius`를 유지한 채 지정 관절 range만 갱신한다.
- direct 모드에서 모터 캘리브레이션 파일을 쓰려면 현재 로봇 ID에 맞게 `--client_id`를 지정한다(예: `my_awesome_kiwi`).

#### 2-1-1. 현재 진행 현황 (업데이트: 2026-02-17)

- 상태: arm 6축 joint range 재측정 완료, sim-real 이동/회전 정합용 보정값 확정.
- 기준 파일: `calibration/calibration_latest.json` (`timestamp`: `2026-02-15 16:51:18`, `connection_mode`: `direct`)
- arm joint ranges (실측 반영 완료):
  - `arm_shoulder_pan.pos = [-100.0, 98.59623199113409]`
  - `arm_shoulder_lift.pos = [-100.0, 100.0]`
  - `arm_elbow_flex.pos = [-98.39572192513369, 99.37611408199643]`
  - `arm_wrist_flex.pos = [-99.83079526226734, 97.80033840947547]`
  - `arm_wrist_roll.pos = [-96.28815628815629, 89.84126984126982]`
  - `arm_gripper.pos = [0.3762227238525207, 100.0]`
- 참고: `calibration_latest.json`의 1회 추정 wheel/base 값
  - `wheel_radius_m = 0.06550005957508184`
  - `base_radius_m = 0.01620267362424141`
  - 위 값은 수집 조건/축 분리 영향으로 일관성이 낮아, sim 기하 상수로 직접 사용하지 않음.
- sim-real 정합에 사용 중인 확정값:
  - 기하 상수: `WHEEL_RADIUS = 0.049m`, `BASE_RADIUS = 0.1085m`
  - 보정 상수 (`scripts/lekiwi_nav_env/sim_real_calibration_test.py`):
    - `LIN_SCALE = 1.0166`
    - `ANG_SCALE = 1.2360`
    - `WZ_SIGN = -1.0` (real `+CCW`, sim `+CW` 부호 차이 보정)
    - `SIM_FORWARD_AXIS = "y"`, `SIM_FORWARD_SIGN = +1.0`
    - `AUTO_SELECT_LINEAR_MAP = True` (최근 실행 자동 선택: `identity`)
- 최근 Script Editor 검증 결과 (2026-02-17 실행 로그):
  - linear: `SIM 75.03 cm` vs `REAL 75.00 cm` (`R/S = 0.9996`)
  - angular magnitude: `SIM 295.39 deg` vs `REAL 298.00 deg` (`R/S = 1.0088`)
- 중요:
  - 위 보정 상수를 변경했으면 PRE-3(tune) → PRE-4(replay) → PRE-5(compare) → PRE-6(gate)를 다시 실행해 최신 리포트를 갱신한다.

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
  --iterations 60 \
  --cmd_transform_mode real_to_sim \
  --cmd_linear_map identity \
  --cmd_lin_scale 1.0166 \
  --cmd_ang_scale 1.2360 \
  --cmd_wz_sign -1.0 \
  --output calibration/tuned_dynamics.json --headless
```

메모:
- `tuned_dynamics.json`에는 `best_params`(wheel/arm dynamics + `lin_cmd_scale`/`ang_cmd_scale`)와 함께 `command_transform`이 저장된다.
- 이후 replay 단계에서 `--dynamics_json`을 주면 `lin_cmd_scale`/`ang_cmd_scale`는 자동으로 그 값을 기본 사용한다(명시 인자 전달 시 override).

#### 2-4. Replay 검증

```bash
python scripts/lekiwi_nav_env/replay_in_sim.py \
  --calibration calibration/calibration_latest.json \
  --mode command \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --report_path calibration/replay_command_report.json \
  --series_path calibration/replay_command_series.json \
  --headless

python scripts/lekiwi_nav_env/replay_in_sim.py \
  --calibration calibration/calibration_latest.json \
  --mode arm_command \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --report_path calibration/replay_arm_report.json \
  --series_path calibration/replay_arm_series.json \
  --headless

python scripts/lekiwi_nav_env/compare_real_sim.py \
  --input calibration/replay_command_series.json \
  --output_dir calibration/plots

python scripts/lekiwi_nav_env/compare_real_sim.py \
  --input calibration/replay_arm_series.json \
  --output_dir calibration/plots
```

메모:
- `replay_in_sim.py`는 기본적으로 `--dynamics_json`에서 `best_params.lin_cmd_scale`, `best_params.ang_cmd_scale`, `command_transform`을 읽어 적용한다.
- 필요 시 아래 인자로 override 가능:
  - `--lin_cmd_scale`, `--ang_cmd_scale`
  - `--cmd_transform_mode`, `--cmd_linear_map`, `--cmd_lin_scale`, `--cmd_ang_scale`, `--cmd_wz_sign`

이 단계가 완료되면 `calibration/tuned_dynamics.json`과 `calibration/arm_limits_real2sim.json`이 생성된다. 이후 모든 Step에서 이 파일들을 `--dynamics_json`과 `--arm_limit_json`으로 전달한다.

#### 2-5. 캘리브레이션 품질 게이트

replay 결과가 임계값 이내인지 자동 검사. FAIL 시 캘리브레이션 재수행 필요:

```bash
python check_calibration_gate.py \
  --reports calibration/replay_command_report.json \
           calibration/replay_arm_report.json

# 또는 tune 출력으로 직접 검사
python check_calibration_gate.py \
  --reports calibration/tuned_dynamics.json

# 파이프라인 자동화: 게이트 통과 시에만 학습 진행
python check_calibration_gate.py \
  --reports calibration/replay_command_report.json \
           calibration/replay_arm_report.json && \
  python train_lekiwi.py --num_envs 2048 --headless
```

기본 임계값: wheel RMSE ≤ 0.20 rad, arm RMSE ≤ 0.15 rad. `--wheel_rmse_threshold`, `--arm_rmse_threshold`로 조정 가능.

#### 2-6. Isaac Sim Script Editor 정합 검증 (권장)

`scripts/lekiwi_nav_env/sim_real_calibration_test.py`는 Script Editor 전용 검증 스크립트다.
headless CLI가 아니라 Isaac Sim UI에서 실행한다.

```text
1) Isaac Sim에서 LeKiwi USD 로드
2) Play 상태로 전환
3) Script Editor에서 sim_real_calibration_test.py 실행
4) TEST 1(linear), TEST 2(angular) 결과의 ratio(R/S) 확인
```

권장 해석:
- `ratio(R/S) = real_output / sim_output`가 1.0에 가까우면 정합 양호
- 실사용 기준: linear / angular 모두 `0.95 ~ 1.05` 범위
- 회전 부호는 좌표계 차이를 고려해 `WZ_SIGN`으로 맞춘다
  - 현재 LeKiwi 설정: `WZ_SIGN = -1.0`
- 스크립트는 `AUTO_LOAD_COMP_FROM_DYNAMICS=True`일 때 `calibration/tuned_dynamics.json`의
  `command_transform`(`lin_scale`, `ang_scale`, `wz_sign`, `linear_map`)를 우선 로드하고,
  파일이 없으면 코드 상수(`LIN_SCALE`, `ANG_SCALE`, `WZ_SIGN`)를 fallback으로 사용한다.
- `wheel joint delta = n/a`가 나와도 pose 기반 거리/각도 검증은 정상 수행 가능

적용 범위와 한계:
- 확인 가능한 것:
  - base 직진/회전 명령의 총량 비율(거리, 각도)
  - 좌표계/부호 정합(`WZ_SIGN`, `LINEAR_MAP`, forward axis)
- 확인 불가한 것:
  - wheel/arm의 과도응답(상승시간, 오버슈트, 감쇠)
  - arm 6축 command-response 정합(시간축 RMSE)
- 따라서 이 테스트 단독 통과만으로는 RL/VLA 실배포 적합 판정을 내리지 않는다.

#### 2-7. 스케일값 사용 규칙 (데이터 수집 / 실배포)

`sim_real_calibration_test.py`에서 확정한 base 보정은 아래 변환으로 사용한다.

정의:
- `u_real = [vx_real, vy_real, wz_real]` (실로봇 기준 명령)
- `u_sim = [vx_sim, vy_sim, wz_sim]` (시뮬레이터 기준 명령)
- `M = ACTIVE_LINEAR_MAP` (현재 자동 선택: `identity`)

변환식:
- real -> sim
  - `[vx_sim, vy_sim]^T = LIN_SCALE * M * [vx_real, vy_real]^T`
  - `wz_sim = WZ_SIGN * ANG_SCALE * wz_real`
- sim -> real (배포 시 역변환)
  - `[vx_real, vy_real]^T = (1 / LIN_SCALE) * M^{-1} * [vx_sim, vy_sim]^T`
  - `wz_real = wz_sim / (WZ_SIGN * ANG_SCALE)`

현재 확정 상수:
- `LIN_SCALE = 1.0166`
- `ANG_SCALE = 1.2360`
- `WZ_SIGN = -1.0`
- `M = identity` (최근 auto-select 결과)

현재 상수로 단순화된 식 (`M=identity`):
- real -> sim
  - `vx_sim = 1.0166 * vx_real`
  - `vy_sim = 1.0166 * vy_real`
  - `wz_sim = -1.2360 * wz_real`
- sim -> real
  - `vx_real = vx_sim / 1.0166`
  - `vy_real = vy_sim / 1.0166`
  - `wz_real = wz_sim / (-1.2360)`

운영 규칙:
- 데이터 수집(sim):
  - `collect_demos.py` 기본 파이프라인은 sim action space 기준으로 수집한다.
  - 이 경우 수집 시점에 추가 역변환을 넣지 않는다.
- 실배포(real):
  - 모델 출력(sim 기준 base 명령)을 실로봇 전송 직전에 `sim -> real` 역변환 1회 적용한다.
- PRE-3/4 (SysID/replay):
  - 실측 command 로그를 sim에서 재생해야 하므로 `real -> sim` 변환을 적용한 상태로 튜닝/검증한다.
  - 현재 파이프라인은 `tune_sim_dynamics.py` 인자와 `tuned_dynamics.json.command_transform`으로 이를 고정한다.
  - `replay_in_sim.py`는 `--cmd_transform_mode auto` 기본값에서 `dynamics_json`의 변환 설정을 자동 상속한다.
- 주의:
  - 같은 신호 경로에 `dynamics_json` 명령 스케일과 위 보정을 중복 적용하지 않는다.
  - 어느 단계에서 적용했는지(run config/로그)에 명시한다.
  - 실배포 송신 경로에는 `sim -> real` 역변환이 실제 코드로 구현되어 있어야 한다(문서 수식만 두고 누락하지 않음).

#### 2-8. VLA 파인튜닝/실배포 Go-NoGo 기준

- NoGo:
  - `sim_real_calibration_test.py`만 통과했고 PRE-3~6을 수행하지 않은 상태
  - replay report 없이 `LIN_SCALE/ANG_SCALE/WZ_SIGN`만으로 배포를 결정한 상태
- Go(최소):
  - PRE-3 완료: `calibration/tuned_dynamics.json` 생성
  - PRE-4/5 완료: command + arm replay/plot 확인
  - PRE-6 완료: `check_calibration_gate.py` 기준 wheel/arm RMSE 임계값 통과
- Go(권장):
  - 보정 상수(`LIN_SCALE`, `ANG_SCALE`, `WZ_SIGN`, `LINEAR_MAP`) 변경 시 PRE-3~6 재실행
  - 실배포 전, 송신 코드에서 `sim -> real` 역변환 단위 테스트 로그를 보관

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
- 저장되는 `action`은 sim action space 기준이므로, 실배포 시 base 명령은 `2-7`의 `sim -> real` 역변환을 적용

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
실배포 시 `action[0:3]`(base)는 `2-7`의 `sim -> real` 역변환 후 실로봇에 송신.

## 5) 데이터 흐름 요약

```
build_object_catalog.py
  1030종 USD → bbox 추출 → k-means → 대표 12종 object_catalog.json

calibrate → tune → replay → compare → gate  (★ RL 학습 전 필수)
  실로봇 측정 → tuned_dynamics.json + arm_limits_real2sim.json 생성
  check_calibration_gate.py로 RMSE 임계값 통과 확인
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

- **★ `--dynamics_json`은 Step 2(캘리브레이션) 완료 후 생성됨** — Step 3~5의 학습/수집 전에 반드시 캘리브레이션을 완료하고, `check_calibration_gate.py`로 품질 게이트 통과를 확인할 것
- `--dynamics_json`은 학습/수집 모두 동일한 파일을 사용해야 Sim2Real 정합 유지
- 37D 체크포인트는 33D 환경에서 로드 불가 (역도 마찬가지)
- camera 수집 시 VRAM에 따라 `num_envs`를 1~8로 제한
- `--gripper_contact_prim_path`는 Isaac Sim에서 gripper finger body prim을 직접 확인해서 설정
- `build_object_catalog.py`는 pxr(Isaac Sim Python) 환경에서만 실행 가능
- **USD 경로**: `lekiwi_robot_cfg.py`는 환경변수 `LEKIWI_USD_PATH`를 우선 참조. 미설정 시 기본 경로 사용.
  다른 머신에서 실행 시: `export LEKIWI_USD_PATH=/path/to/lekiwi_robot.usd`
- **모델 공유**: `PolicyNet`/`ValueNet`은 `models.py`에 단일 정의. `train_lekiwi.py`와 `collect_demos.py`가 공통 import하므로 모델 구조 변경 시 `models.py`만 수정
- **GRASP timeout**: `grasp_timeout_steps=75` (기본 ~3초@25Hz). GRASP phase에서 지정 step 내 grasp 미성공 시 APPROACH로 복귀하여 재시도. 0으로 설정하면 무제한
