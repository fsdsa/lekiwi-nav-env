# LeKiwi Simulation 데이터 수집 파이프라인

이 문서는 Isaac Sim에서 Skill별 RL Expert를 학습하고 VLA 학습 데이터를 대량 생성하는 전 과정을 상세히 기술한다. 전체 파이프라인의 Phase 0~3에 해당한다.

---

## 1. 왜 Simulation이 필요한가

VLA를 LeKiwi에 파인튜닝하려면 skill당 수백~수천 에피소드의 (카메라 이미지, instruction, robot_state 9D, action 9D) 쌍이 필요하다. 사람 텔레옵으로 모으는 것은 비현실적이다.

sim 데이터가 real에서 통하려면 두 가지가 보장되어야 한다. 첫째, sim의 물리가 real과 충분히 비슷해야 한다(Calibration). 둘째, sim에서 모은 이미지와 행동 패턴이 real의 다양한 상황을 커버해야 한다(Domain Randomization). 이 두 가지를 Phase 0과 Phase 2에서 각각 해결한다.

---

## 2. Phase 0: Sim-Real 일치

학습 환경: RTX 3090 24GB, Isaac Sim 5.0 + Isaac Lab 0.44.9, conda env_isaaclab (Python 3.11, PyTorch 2.7.0+cu128), skrl 1.4.3 / rsl_rl.

```bash
conda activate env_isaaclab
source ~/isaacsim/setup_conda_env.sh
cd ~/IsaacLab/scripts/lekiwi_nav_env
```

### 2-1. Dynamics 캘리브레이션 ✅ 완료

VLA는 joint-level action을 학습한다. sim에서 "shoulder_lift를 -0.5rad으로 보내라"는 action을 학습했는데, 같은 명령을 real에서 보냈을 때 다른 위치로 가면 전이가 깨진다.

#### 2-1-1. 실로봇 측정

```bash
# 전체 측정 (wheel/base/arm/rest/sysid)
python calibrate_real_robot.py \
  --mode all --connection_mode direct --robot_port /dev/ttyACM0 \
  --client_id my_awesome_kiwi --sample_hz 20 --encoder_unit m100

# geometry config 유지, arm/rest/sysid만 갱신
python calibrate_real_robot.py \
  --mode all --skip wheel_radius,base_radius \
  --connection_mode direct --robot_port /dev/ttyACM0 \
  --client_id my_awesome_kiwi --sample_hz 20 --encoder_unit m100

# arm 6축 range만 재측정 (수동 시작/종료)
python calibrate_real_robot.py \
  --mode joint_range --connection_mode direct --robot_port /dev/ttyACM0 \
  --client_id my_awesome_kiwi --joint_range_duration 0 --sample_hz 20

# 단일 관절만 재측정
python calibrate_real_robot.py \
  --mode joint_range_single --connection_mode direct --robot_port /dev/ttyACM0 \
  --client_id my_awesome_kiwi --joint_key arm_gripper.pos \
  --joint_range_duration 0 --sample_hz 20
```

direct 모드는 로봇 USB가 연결된 머신(192.168.0.104)에서 실행해야 한다. `--joint_range_duration 0`이면 Enter로 시작/종료. 측정 중 arm torque 자동 OFF, 종료 시 ON 복구.

#### 2-1-2. 현재 상태 (2026-02-18)

- 기준 파일: `calibration/calibration_latest.json` (timestamp: 2026-02-18 10:12:18)
- Geometry: `lekiwi_robot_cfg.py`의 WHEEL_RADIUS=0.049m, BASE_RADIUS=0.1085m (config 기준, 실측 생략)
- Arm joint ranges (실측 완료):
  - shoulder_pan: [-99.48, 100.0], shoulder_lift: [-100.0, 100.0]
  - elbow_flex: [-100.0, 98.93], wrist_flex: [-100.0, 99.92]
  - wrist_roll: [-95.80, 90.43], gripper: [0.075, 100.0]
- arm_sysid 포함, base_radius 미포함 (skip 사용)

#### 2-1-3. Arm Joint Limit JSON 생성

```bash
python build_arm_limits_real2sim.py \
  --calibration_json calibration/calibration_latest.json \
  --encoder_calibration_json ~/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_awesome_kiwi.json \
  --output calibration/arm_limits_real2sim.json
```

#### 2-1-4. Sim 파라미터 튜닝

```bash
python tune_sim_dynamics.py \
  --calibration calibration/calibration_latest.json \
  --encoder_unit m100 \
  --optimizer cem \
  --iterations 60 \
  --analytical_init \
  --refine \
  --refine_iters 30 \
  --cmd_transform_mode real_to_sim \
  --cmd_linear_map auto \
  --cmd_lin_scale 1.0166 \
  --cmd_ang_scale 1.2360 \
  --cmd_wz_sign -1.0 \
  --freeze_cmd_scales \
  --output calibration/tuned_dynamics.json --headless
```

`tuned_dynamics.json`에는 `best_params`(wheel/arm dynamics + 관절별 스케일)와 `command_transform`(LIN_SCALE, ANG_SCALE, WZ_SIGN, LINEAR_MAP)이 저장된다.

튜닝 결과 (2026-02-18): wheel RMSE=0.117299, arm RMSE=0.086990.

#### 2-1-5. Replay 검증

```bash
python replay_in_sim.py \
  --calibration calibration/calibration_latest.json \
  --mode command \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --report_path calibration/replay_command_report.json \
  --series_path calibration/replay_command_series.json --headless

python replay_in_sim.py \
  --calibration calibration/calibration_latest.json \
  --mode arm_command \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --report_path calibration/replay_arm_report.json \
  --series_path calibration/replay_arm_series.json --headless

python compare_real_sim.py \
  --input calibration/replay_command_series.json \
  --output_dir calibration/plots

python compare_real_sim.py \
  --input calibration/replay_arm_series.json \
  --output_dir calibration/plots
```

Replay 결과 (2026-02-18): wheel RMSE=0.145975, arm RMSE=0.087000.

#### 2-1-6. 캘리브레이션 품질 게이트

```bash
python check_calibration_gate.py \
  --reports calibration/replay_command_report.json \
           calibration/replay_arm_report.json \
  --wheel_rmse_threshold 0.15 \
  --arm_rmse_threshold 0.09
```

현재 상태: **PASS** (wheel 0.146 < 0.15, arm 0.087 < 0.09).

#### 2-1-7. Isaac Sim Script Editor 정합 검증 (보조)

`sim_real_calibration_test.py`는 Script Editor에서 base의 pose-level 총량 비율을 확인하는 보조 도구다. `tuned_dynamics.json`의 command_transform을 자동 로드한다.

적용 범위: base 직진/회전 총량 비율, 좌표계/부호 정합.
한계: wheel/arm 과도응답, arm 시간축 RMSE는 확인 불가. 이 테스트 단독으로 Go 판정하지 않는다.

#### 2-1-8. 스케일값 사용 규칙

확정 상수: `LIN_SCALE=1.0166, ANG_SCALE=1.2360, WZ_SIGN=-1.0, M=identity`

- **RL 학습 / 데이터 수집(sim)**: sim action space 기준으로 직접 학습/수집. 추가 역변환 불필요.
- **실배포(real)**: 모델 출력을 실로봇 전송 직전에 `sim → real` 역변환 1회 적용.
  - `vx_real = vx_sim / 1.0166`, `vy_real = vy_sim / 1.0166`, `wz_real = wz_sim / (-1.2360)`
- **SysID/replay**: 실측 command 로그를 sim에서 재생할 때 `real → sim` 변환 적용.

주의: 같은 경로에 dynamics_json 명령 스케일과 위 보정을 중복 적용하지 않는다.

```bash
# 빠른 검증
python sim_real_command_transform.py --mode sim_to_real --vx 0.20 --vy 0.00 --wz -1.00
```

#### 2-1-9. Go-NoGo 기준

- **NoGo**: Script Editor만 통과하고 tune/replay/gate 미수행
- **Go(최소)**: tuned_dynamics.json 생성 + replay 확인 + gate 통과
- **Go(권장)**: 보정 상수 변경 시 재실행, 실배포 전 sim→real 역변환 단위 테스트 로그 보관

### 2-2. 카메라 캘리브레이션 ⬜ 미완

base_cam(D455)의 extrinsic 실측 필요. D455 factory calibration으로 intrinsic은 SDK에서 읽을 수 있다. wrist_cam(USB 웹캠)은 intrinsic/extrinsic 모두 실측 필요. RGB FOV 약 86°(H)를 sim 카메라에도 맞춘다.

**USD 카메라 prim path (Isaac Sim 5.0.0 확인)**:
- base_cam: `.../Realsense/RSD455/Camera_OmniVision_OV9782_Color`
- wrist_cam: `.../Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera`
- 참고: D455에는 Depth/Left/Right 카메라도 있으나, VLA 학습에는 Color만 사용.

D455의 depth/IMU는 VLA 학습 데이터에 포함되지 않고, Safety Layer와 VIO에서만 사용. sim에서 이 센서를 시뮬레이션할 필요 없다.

### 2-3. Joint Limits — JSON 생성 ✅, USD PhysX 반영 ✅ (코드에서 `arm_limit_write_to_sim=True`)

`arm_limits_real2sim.json`은 이미 생성되어 있다. RL 학습 시 **제어 target clamp와 USD PhysX joint limit 양쪽 모두에 적용**한다 (`arm_limit_write_to_sim=True`).

**검증 결과 (Isaac Sim 5.0.0, 2026-02-19)**: LeKiwi USD의 전체 39개 revolute joint이 모두 `(-inf, +inf)` 확인됨. arm 6개 + gripper 1개는 반드시 캘리브레이션 범위로 덮어써야 팔-몸체 관통 방지.

### 2-4. REST_POSE — 기존 값 존재, 안전 자세 재정의 필요 ⬜

현재 `leader_to_home_tcp_rest_matched_with_keyboard_base.py`에 정의된 `SIM_REST_RAD6`:
```python
SIM_REST_RAD6 = [
    -0.001634,  # shoulder_pan
    -0.002328,  # shoulder_lift
     0.098572,  # elbow_flex
     0.004954,  # wrist_flex
     0.009319,  # wrist_roll
    -0.000285,  # gripper
]
```

이 값은 거의 all-zeros(팔이 완전히 펴진 상태)다. 토크가 크고 실로봇에서 위험하다. 팔이 접힌 안전한 tucked pose를 새로 정의해야 한다. 이 값은 Navigate skill에서 arm 고정 target, RL 에피소드 초기 arm 자세로 사용된다.

### 2-5. ~~lekiwi_v6 데이터 형식 확인~~ ✅ 확정

실제 로봇 데이터(`yubinnn11/lekiwi3`, v3.0) 확인 완료. base state는 body-frame velocity(m/s, rad/s). 채널명: `x.vel, y.vel, theta.vel`. 초기에 참조한 `theo-michel/lekiwi_v6`(v2.1)의 `x_mm, y_mm, theta`(displacement, mm)가 아님. 단위 변환(m→mm) 불필요.

### 2-6. Isaac Sim 5.0.0 API 검증 ✅ 완료 (2026-02-19)

코드 작성에 필요한 Isaac Sim/Lab API를 실제 환경에서 검증했다.

| API / 항목 | 결과 | 비고 |
|------------|------|------|
| `robot.data.root_lin_vel_b` | ✅ `(N, 3)` | body-frame velocity 직접 사용 가능 |
| `robot.data.root_ang_vel_b` | ✅ `(N, 3)` | body-frame angular velocity |
| `robot.data.body_pos_w` | ✅ `(N, 40, 3)` | 3D tensor, `[:, body_idx]` |
| `find_bodies("Moving_Jaw_08d_v1")` | ✅ `ids=[39]` | USD Inspector(idx=6)와 다름, 동적 취득 필수 |
| `UsdPhysics.Joint.CreateBreakForceAttr()` | ✅ | v8 L870에서 이미 사용 중 |
| `PhysxSchema.PhysxJointAPI.CreateBreakForceAttr()` | ❌ | 존재하지 않는 API |
| Contact sensor in USD | ❌ | 코드에서 `ContactSensorCfg`로 동적 생성 |
| 전체 revolute joint limits | 39개 모두 `(-inf, inf)` | arm 7개는 반드시 덮어쓰기 필요 |

---

## 3. Phase 1: RL Expert 학습

### 3-0. 대표 물체 카탈로그 생성

기존 `build_object_catalog.py`로 1030종 USD에서 bbox 추출 + k-means → 대표 12종 선별:

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

pxr(Isaac Sim Python) 환경에서 실행해야 함.

### 3-1. 전체 구조

Skill-2(ApproachAndGrasp)와 Skill-3(CarryAndPlace) 각각에 대해 텔레옵 → BC → RL 순서로 진행한다. Navigate는 RL을 하지 않으므로 Phase 2에서 스크립트 정책으로 데이터를 직접 생성한다.

**기존 v8 코드와의 관계**: v8의 `lekiwi_nav_env.py`는 37D obs + 4-phase FSM(SEARCH→APPROACH→GRASP→RETURN)으로 전체 task를 단일 환경에서 처리한다. 3-Skill 분리는 이 환경을 Skill-2 env(ApproachAndGrasp)와 Skill-3 env(CarryAndPlace)로 분리 리팩토링하는 것이다. v8의 핵심 컴포넌트들은 재사용한다:
- 물리 grasp (FixedJoint attach/detach) → break_force 조정
- Dynamics DR (reset-time wheel/arm/object randomization)
- 캘리브레이션 연동 (`--dynamics_json`, `--arm_limit_json`)
- `models.py`의 PolicyNet/ValueNet/CriticNet 구조 + AAC 파일(`aac_wrapper.py`, `aac_ppo.py`, `aac_trainer.py`)
- Contact sensor 기반 grasp 판정
- GRASP timeout 메커니즘 (75 steps)

**모든 학습/수집 명령에서 `--dynamics_json`과 `--arm_limit_json`을 반드시 사용한다.**

---

### 3-2. Skill-2: ApproachAndGrasp

#### 3-2-1. 텔레옵 수집

sim에서 사람이 직접 10~20개 시범을 보인다. 키보드로 base를 조종하면서 동시에 리더암으로 팔을 움직여서 물체를 잡는다.

한 에피소드의 범위: "물체가 보이는 위치에서 시작 → base+arm 접근 → 물체 잡기 → 살짝 들어올리기". 성공한 에피소드만 저장.

기존 도구 사용:
```bash
# Windows 측: SO-100 리더 + 키보드 TCP 전송
python leader_to_home_tcp_rest_matched_with_keyboard_base.py \
  --host <desktop_ip> --port 15002 --leader_port COM8

# Desktop 측: sim에서 받아서 기록
python record_teleop.py --num_demos 20 \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/Moving_Jaw_08d_v1" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json
```

기록 데이터: (카메라 이미지 2장, 9D state, 9D action) 프레임 단위. HDF5 형식.

**핵심: 텔레옵 시 sim에서 privileged obs(rel_object, contact)를 동시에 기록한다.** 이렇게 해야 BC의 obs가 RL Actor와 동일한 형태가 되어 weight transfer가 매끄럽다. "사후 추출" 방식은 replay 파이프라인이 추가로 필요하므로, sim 텔레옵 시 동시 기록을 기본으로 한다.

#### 3-2-2. BC 학습

텔레옵 데이터로 Behavior Cloning.

**Observation 구성** (30D):

```
BC obs = arm(5) + grip(1) + base_body_vel(3) + base_vel(6) + arm_vel(6) + rel_object(3) + contact(2) + obj_bbox(3) + obj_category(1) = 30D
```

| 채널 | 차원 | 출처 |
|------|------|------|
| arm | 5D | 모터 엔코더 (real 가능) |
| grip | 1D | 모터 엔코더 (real 가능) |
| base_body_vel | 3D | root_lin_vel_b, root_ang_vel_b (sim) / 휠 FK (real) |
| base_lin_vel | 3D | sim IMU / 휠 FK (real 가능) |
| base_ang_vel | 3D | sim IMU / 휠 FK (real 가능) |
| arm_joint_vel | 6D | 모터 엔코더 미분 (real 가능) |
| rel_object | 3D | sim ground truth (real 불가) |
| contact | 2D | sim contact sensor (real 불가) |
| obj_bbox | 3D | sim ground truth (real 불가) — 물체 크기 (x, y, z), normalized |
| obj_category | 1D | sim ground truth (real 불가) — 물체 종류 인덱스, normalized |

왜 9D가 아니라 30D로 만드는가: 첫째, BC의 가중치를 RL Actor에 직접 복사해서 초기화하기 때문이다. RL Actor는 30D obs를 받으므로 BC도 30D를 받아야 weight transfer가 가능하다. 둘째, obj_bbox/obj_category가 있어야 Actor가 **물체별 차별화된 grasp 전략**을 학습할 수 있다. 셋째, **속도 정보(base_vel 6D + arm_vel 6D = 12D)가 동적 제어에 필수**다. v8의 37D obs에도 base_lin_vel(3) + base_ang_vel(3) + arm_joint_vel(6) + wheel_vel(3) = 15D 속도 정보가 포함되어 있었다. 속도 정보 없이는 Actor가 현재 운동 상태를 알 수 없어서 과도한 가감속, 오버슈트가 발생한다.

**네트워크 구조**: RL Actor와 정확히 동일해야 한다. 기존 `models.py`의 `PolicyNet`을 사용한다. hidden dims, activation이 다르면 가중치 복사가 안 된다.

**Action 출력**: 9D (arm target 5D + gripper cmd 1D + base cmd 3D).

**학습 설정**: 기존 `train_bc.py` 사용. `--normalize` OFF가 기본(PPO RunningStandardScaler와 호환). lr=1e-3, batch_size=256, 200 epochs. 목표 성공률 30~40%.

```bash
python train_bc.py --demo_dir demos/ --epochs 200 --expected_obs_dim 30
```

#### 3-2-3. RL 학습

BC checkpoint로 PPO의 Actor를 초기화하고 RL 학습을 시작한다.

**알고리즘**: PPO + Asymmetric Actor-Critic (AAC). skrl 1.4.3는 native AAC를 지원하지 않으므로 3개 파일로 구현한다: `aac_wrapper.py`가 IsaacLabWrapper에 `state()` 메서드를 monkey-patch하여 critic obs를 노출, `aac_ppo.py`가 PPO를 상속하여 `critic_states` memory tensor 관리, `aac_trainer.py`가 SequentialTrainer를 상속하여 critic states 매 step 추적. 환경의 `_get_observations()`가 `self._critic_obs`에 critic obs를 저장하고, wrapper의 `state()`가 이를 읽어 agent에 전달한다.

**Actor Observation** (30D, state vector only):

| 채널 | 차원 | 출처 | 설명 |
|------|------|------|------|
| arm | 5D | 모터 엔코더 (real 가능) | 관절 위치 |
| grip | 1D | 모터 엔코더 (real 가능) | 그리퍼 위치 |
| base_body_vel | 3D | root_lin_vel_b, root_ang_vel_b (sim) / 휠 FK (real) | body-frame velocity (m/s, rad/s) |
| base_lin_vel | 3D | sim (real: IMU/휠 FK) | body-frame 선속도 |
| base_ang_vel | 3D | sim (real: IMU/휠 FK) | body-frame 각속도 |
| arm_joint_vel | 6D | sim (real: 엔코더 미분) | 관절 각속도 |
| rel_object | 3D | sim ground truth (real 불가) | 물체의 body-frame 상대 위치 |
| contact | 2D | sim contact sensor (real 불가) | 그리퍼 좌/우 접촉 여부 |
| obj_bbox | 3D | sim ground truth (real 불가) | 물체 크기 (normalized) |
| obj_category | 1D | sim ground truth (real 불가) | 물체 종류 (normalized) |

**카메라 이미지는 Actor에 입력하지 않는다.** privileged state가 충분한 정보를 제공하고, 이미지를 넣으면 병렬 환경 수가 2048 → 1~8로 급감한다. 이미지는 Phase 2 수집 시에만 렌더링.

**속도 정보(12D)가 필수인 이유**: v8의 37D obs에도 base_lin_vel(3) + base_ang_vel(3) + arm_joint_vel(6) + wheel_vel(3) = 15D 속도 정보가 포함되어 있었다. 속도가 없으면 Actor가 현재 운동 상태를 모르기 때문에, 접근 중 감속 타이밍, arm 도달 후 안정화, grasp 직전 미세 조정 등의 동적 제어가 불가능하다. 속도는 privileged가 아닌 real에서도 얻을 수 있는 정보(IMU, 휠 FK, 엔코더 미분)이지만, VLA 9D state에는 포함하지 않는다 — VLA는 이미지의 optical flow에서 암묵적으로 속도를 추론한다.

obj_bbox/obj_category가 Actor에 들어가는 이유: 기존 v8에서 검증된 설계다. 12종 다중 물체를 학습할 때, 물체 크기와 형태를 알아야 물체별로 다른 접근 각도, arm trajectory, gripper timing을 학습할 수 있다.

**Critic Observation** (37D): Actor obs 30D + obj_bbox(3D, 비정규화) + obj_mass(1D) + obj_dist(1D) + heading_object(1D) + vel_toward_object(1D) = 37D.

**Action** (9D): arm_target 5D + gripper_cmd 1D(continuous position target) + base_cmd 3D. **순서 주의**: 새 skill 환경에서는 VLA/yubinnn11/lekiwi3 포맷 `[arm5, grip1, base3]`으로 통일한다. 기존 v8의 `[base3, arm6]` 순서와 **반대**이므로, v8 코드 재사용 시 인덱싱을 변경해야 한다. gripper는 RL 학습 시 continuous로 유지하고, VLA 데이터 저장 시에만 0.5 threshold로 binary 변환한다. base_cmd는 sim/real 모두 m/s, rad/s 단위 — 단위 변환 불필요.

**Grasp 메커니즘**: 기존 v8의 physics-based grasp 재사용. gripper close + contact force + **adaptive distance** → FixedJoint attach. grasp 판정 거리와 gripper threshold가 물체의 bbox 크기에 따라 자동 조정된다 (큰 물체는 더 먼 거리에서 grasp 시도 허용, gripper threshold도 bbox에 비례 조정). **break_force 기본값 30N, DR로 15~45N 랜덤화** (`dr_grasp_break_force_range`). 기존 1e8에서 변경. 이유: (1) Skill-3에서 조심스러운 운반 행동 학습, (2) 고정값 과적합 방지.

**GRASP timeout**: 기존 v8의 `grasp_timeout_steps=75` (~3초@25Hz) 유지. timeout 내 grasp 미성공 시 APPROACH로 복귀 재시도.

**Grasp Break 감지**: FixedJoint 파손 시 `object_grasped`가 자동으로 False가 되지 않으므로, 매 step gripper-object 거리를 체크하여 `grasp_drop_detect_dist`(0.15m) 초과 시 drop 판정 (`just_dropped=True`). Skill-3에서 drop → terminated(에피소드 즉시 종료) + `rew_drop_penalty=-10`. 의도적 place(home 근처 gripper 열기)와 비의도적 drop(break_force 초과)은 `just_dropped` 플래그로 구분.

**Reward 설계**:

| 항목 | 값 | 설명 |
|------|-----|------|
| approach | -‖robot − object‖₂ | 물체까지 거리 줄이기 |
| grasp | +10 | 그리퍼 접촉 + close 성공 |
| lift | +5 | 물체 z > target_height |
| collision | -1 | 환경 충돌 |
| time | -0.01/step | 시간 초과 방지 |

**다중 물체 관리 (기존 v8 방식 유지)**: object_catalog.json의 12종 대표 물체를 환경 초기화 시 모두 pre-spawn한다. 매 에피소드 reset 시 12종 중 1종을 랜덤 선택하고, 선택된 물체만 로봇 근처에 배치하고 나머지는 z=-10에 숨긴다. 물체는 **바닥 위**에 배치되며(object_height = bbox_z × 0.5, 물체 크기에 맞게 지면 안착), 로봇 home에서 1.0~2.5m 거리에 360° 랜덤 방향으로 놓인다. 물체 yaw도 랜덤. 이렇게 하면 Teacher가 물체별로 다른 크기/형상에 적응한 grasp 전략을 학습한다.

**Curriculum Learning**: 처음에는 물체를 로봇 앞 0.5m에 놓고, 성공률 70% 초과 시 거리를 점진적으로 2.5m까지 늘린다.

**Domain Randomization (Dynamics, reset-time)**: 기존 v8 코드의 `enable_domain_randomization=True`를 그대로 사용.
- Wheel: stiffness(0.75~1.5x), damping(0.3~3.0x), friction(0.7~1.3x)
- Arm: stiffness(0.8~1.25x), damping(0.5~2.0x)
- Object: mass(0.5~2.0x), friction(0.6~1.5x)
- Base values: `tuned_dynamics.json`의 `best_params`
- Observation noise: joint_pos(0.01 rad), base_vel(0.02 m/s), object_rel(0.02 m)
- Action delay: 1 step (10-50ms 통신 지연 시뮬레이션)

**PPO 하이퍼파라미터**: lr=3e-4, gamma=0.99, GAE lambda=0.95, ratio_clip=0.15, grad_norm_clip=0.5, entropy_coef=0.01, mini_batches=4. 병렬 환경 **2048개** (state-only Actor, 이미지 렌더링 불필요 — v8 기본값과 동일).

**BC → RL weight transfer**: BC의 state_dict를 RL Actor에 key-by-key 복사. 네트워크 구조 동일하므로 shape 동일. Critic은 랜덤 초기화. 기존 `train_lekiwi.py`의 BC warm-start 로직 재사용 (obs dim mismatch 시 net.0.weight 자동 어댑트).

```bash
python train_lekiwi.py \
  --num_envs 2048 \
  --bc_checkpoint checkpoints/bc_skill2.pt \
  --skill approach_and_grasp \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/Moving_Jaw_08d_v1" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --headless
```

**수렴 기준**: 성공률 80% 이상. 정체 시 approach bonus 0.1 활성화.

---

### 3-3. Handoff Buffer 생성

Skill-2 학습이 충분히 진행되면(성공률 80%+), Skill-2 expert를 수백 번 돌려서 성공 에피소드 종료 시점의 전체 상태를 저장한다. `generate_handoff_buffer.py`는 AAC 체크포인트를 로드할 수 있도록 `wrap_env_aac` + `CriticNet` + `AAC_PPO`를 사용한다:

```python
handoff_entry = {
    "base_pos": [x, y, z],           # world position
    "base_ori": [qw, qx, qy, qz],   # world orientation
    "arm_joints": [5D],              # 관절 위치
    "gripper_state": float,          # 그리퍼 위치
    "object_pos": [x, y, z],         # 물체 world position
    "object_ori": [qw, qx, qy, qz], # 물체 orientation
    "object_type_idx": int,          # 물체 종류 인덱스
}
```

world-frame 절대 좌표를 포함하지만, sim 내부 reset 용도이므로 괜찮다. VLA에 전달되지 않는다.

다양한 물체(종류, 크기, 무게), 다양한 그립 자세, 다양한 로봇 위치에서 성공한 상태를 충분히 모아야 한다. **최소 200~500개** handoff entry 확보.

**Handoff Noise Injection**: 버퍼 생성 시 각 entry에 노이즈를 추가한다:
- arm joint: ±0.05 rad (`--noise_arm_std`)
- object xy: ±0.02 m (`--noise_obj_xy_std`)
- base xy: ±0.03 m (`--noise_base_xy_std`)
- base yaw: ±0.1 rad (`--noise_base_yaw_std`)

이유: 실제 배포 시 VLA의 Skill-2→3 전환 상태는 RL Expert보다 부정확하다. Skill-3가 "약간 비틀어진 grasp 상태"에서도 복구/운반할 수 있도록 학습시킨다.

---

### 3-4. Skill-3: CarryAndPlace

#### 3-4-1. 텔레옵 수집

Handoff Buffer의 상태 중 하나를 sim에 로드하고, 사람이 10~20개 시범을 보인다. 물체를 이미 잡은 상태에서 시작하여 home으로 이동하고 내려놓는다.

#### 3-4-2. BC 학습

Skill-2와 동일한 방식이되, obs 구성이 다르다.

```
BC obs = arm(5) + grip(1) + base_body_vel(3) + base_vel(6) + arm_vel(6) + home_rel(3) + grip_force(1) + obj_bbox(3) + obj_category(1) = 29D
```

| 채널 | 차원 | 출처 |
|------|------|------|
| arm | 5D | 모터 엔코더 (real 가능) |
| grip | 1D | 모터 엔코더 (real 가능) |
| base_body_vel | 3D | root_lin_vel_b, root_ang_vel_b (sim) / 휠 FK (real) |
| base_lin_vel | 3D | sim (real: IMU/휠 FK) |
| base_ang_vel | 3D | sim (real: IMU/휠 FK) |
| arm_joint_vel | 6D | sim (real: 엔코더 미분) |
| home_rel | 3D | sim ground truth (real 불가) |
| grip_force | 1D | sim force sensor (real 불가) |
| obj_bbox | 3D | sim ground truth (real 불가) — 물체 크기 (normalized) |
| obj_category | 1D | sim ground truth (real 불가) — 물체 종류 (normalized) |

home_rel: home 위치의 body-frame 상대 벡터. grip_force: 그리퍼가 물체를 누르는 힘. obj_bbox/obj_category: 물체 크기와 종류를 알아야 운반 중 적절한 속도/자세 조절을 학습할 수 있다.

네트워크 구조는 Skill-3 RL Actor와 동일 (`models.py` PolicyNet, 입력 차원 29D).

```bash
python train_bc.py --demo_dir demos_skill3/ --epochs 200 --expected_obs_dim 29
```

#### 3-4-3. RL 학습

**Actor Observation** (29D, state vector only):

| 채널 | 차원 | 출처 | 설명 |
|------|------|------|------|
| arm | 5D | 모터 엔코더 (real 가능) | 관절 위치 |
| grip | 1D | 모터 엔코더 (real 가능) | 그리퍼 위치 |
| base_body_vel | 3D | root_lin_vel_b, root_ang_vel_b (sim) / 휠 FK (real) | body-frame velocity (m/s, rad/s) |
| base_lin_vel | 3D | sim (real: IMU/휠 FK) | body-frame 선속도 |
| base_ang_vel | 3D | sim (real: IMU/휠 FK) | body-frame 각속도 |
| arm_joint_vel | 6D | sim (real: 엔코더 미분) | 관절 각속도 |
| home_rel | 3D | sim ground truth (real 불가) | home의 body-frame 상대 벡터 |
| grip_force | 1D | sim force sensor (real 불가) | 그리퍼 힘 |
| obj_bbox | 3D | sim ground truth (real 불가) | 물체 크기 (normalized) |
| obj_category | 1D | sim ground truth (real 불가) | 물체 종류 (normalized) |

**Critic Observation** (36D): Actor obs 29D + obj_dimensions(3D) + obj_mass(1D) + gripper_rel_pos(3D) = 36D.

**Action**: 9D `[arm5, grip1, base3]`. gripper_cmd는 continuous — 대부분 close 유지, place 단계에서 서서히 open 전환. VLA 데이터 저장 시에만 binary 변환.

**Reset**: 매 에피소드 Handoff Buffer에서 상태 랜덤 샘플.

**Reward**:

| 항목 | 값 | 설명 |
|------|-----|------|
| carry | -‖robot − home‖₂ | home까지 거리 줄이기 |
| hold | +0.1/step | 물체가 그리퍼에 있으면 |
| place | +20 | home 근처에서 **의도적으로** 놓음 (gripper open) |
| drop | -10 | break_force 초과로 물체 낙하 (`just_dropped`) |
| collision | -1 | 환경 충돌 |

**Termination**: drop 발생 시 에피소드를 즉시 `terminated`로 종료한다 (Skill-2와의 핵심 차이). 의도적 place와 비의도적 drop은 Skill-3의 `_update_grasp_state()` 오버라이드로 구분한다: gripper가 `place_gripper_threshold`(0.3) 이상 열리고 home 근처(`return_thresh` 내)이면 `intentional_placed=True`로 설정하여 FixedJoint를 해제하고 `just_dropped=False`를 유지한다. 반면 break_force 초과로 물체가 떨어진 경우 `just_dropped=True`가 된다. place는 `truncated`(성공, +20 보상), drop은 `terminated`(실패, -10 페널티)로 처리. `_get_dones()`는 부모의 lift 기반 task_success 대신 `place_success`를 직접 계산하여 오버라이드한다 (Skill-3는 handoff buffer에서 이미 grasped+lifted 상태로 시작하므로).

**Dynamics DR**: Skill-2와 동일 + **break_force DR** (`dr_grasp_break_force_range: 15~45N`). 주의: `_reset_idx()`에서 `_apply_domain_randomization()`을 `_attach_grasp_fixed_joint_for_envs()` **이전**에 호출.

```bash
python train_lekiwi.py \
  --num_envs 2048 \
  --bc_checkpoint checkpoints/bc_skill3.pt \
  --skill carry_and_place \
  --handoff_buffer handoff_buffer.pkl \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/Moving_Jaw_08d_v1" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --headless
```

---

## 4. Phase 2: VLA 학습 데이터 대량 수집

### 4-1. 수집의 핵심 원칙

RL Expert를 sim에서 실행하면서 **VLA가 실제로 받게 될 정보만 저장한다.**

**저장하는 것:**
- observation.images.base: base_cam 렌더 (1280×720, MP4)
- observation.images.wrist: wrist_cam 렌더 (640×480, MP4)
- observation.state: [arm_shoulder_pan.pos, arm_shoulder_lift.pos, arm_elbow_flex.pos, arm_wrist_flex.pos, arm_wrist_roll.pos, arm_gripper.pos, x.vel, y.vel, theta.vel] = 9D
- action: [shoulder_pan_target, ..., gripper_cmd, base_x_cmd, base_y_cmd, base_theta_cmd] = 9D
- instruction: 자연어 텍스트

**저장하지 않는 것:**
- rel_object, contact, home_rel, grip_force (Actor privileged obs)
- obj_bbox, obj_mass, goal_dist (Critic 전용)

기존 `collect_demos.py`의 robot_state 추출 로직(obs[18:24] + obs[30:33] → 9D)은 환경 obs 차원에 따라 재매핑이 필요하다.

### 4-2. Base Displacement 계산 (Sim에서)

sim에서 Isaac Sim의 `root_lin_vel_b`와 `root_ang_vel_b`로 body-frame velocity를 직접 읽는다. **단위 변환 불필요**: sim과 real 모두 m/s(vx, vy), rad/s(wz). 이전 설계의 `_compute_body_displacement()` + m→mm 변환은 삭제됨.

---

### 4-3. Navigate 데이터 수집 (스크립트 정책)

#### 4-3-1. 왜 스크립트 정책인가

Navigate 핵심 행동: "목표 방향 이동"과 "탐색 회전". 둘 다 base만 움직이고 arm은 rest pose. RL 필요 없을 만큼 단순하지만, VLA 학습용 수천 에피소드가 필요하므로 스크립트로 생성.

#### 4-3-2. 목표 방향 이동 (Directed Navigation)

```python
# 매 프레임:
direction = target_pos - robot_pos  # sim ground truth
angle_to_target = atan2(direction.y, direction.x) - robot_heading
base_cmd = [K_lin * cos(angle_to_target), K_lin * sin(angle_to_target), K_ang * angle_to_target]
arm_cmd = REST_POSE  # 고정
gripper_cmd = 1.0    # open 유지
action = [arm_cmd(5D), gripper_cmd(1D), base_cmd(3D)]  # 9D
```

**노이즈 주입**: (1) 조향 흔들림 σ=0.05 rad, (2) 속도 양자화 3~5단계, (3) 5% 확률로 1~3프레임 이전 action 반복.

#### 4-3-3. 탐색 회전 (Search Rotation)

제자리 회전 또는 전진+회전. instruction: "turn right slowly to search for the red cup".

**instruction의 가시성 조건**: 물체가 base_cam FOV 안에 10×10 pixel 이상으로 보이면 "navigate toward~", 안 보이면 "turn to search for~". 물체가 보이는데 "search"가 붙으면 시각-언어 정합성이 깨진다.

#### 4-3-4. 환경 요구사항

Navigate 데이터는 시각적 다양성이 중요하다. 빈 평면에서 proportional controller를 돌리면 VLA가 학습할 실내 시각 정보가 없다. 최소한:
- 바닥/벽 텍스쳐 랜덤 (Wood, Tile, Carpet 등)
- Distractor 가구 1~3개 (테이블, 의자, 선반 등)
- 조명 DR (색온도, 강도, 방향)

Skill-2/3 수집 환경(바닥 위 물체 + 로봇)에 추가 배경을 넣어 Navigate 환경으로 재사용할 수 있다.

#### 4-3-5. 저장 원칙

스크립트가 목표 좌표를 알고 있었다는 사실은 저장하지 않는다. 저장하는 것은 오직 (이미지, 9D state, 9D action, instruction). 9D action에서 arm 5D = REST_POSE, gripper = 1.0, base 3D = command.

목표: 1K~2K개. Directed Navigation : Search Rotation ≈ 7:3.

---

### 4-4. Skill-2 데이터 수집 (RL Expert Rollout)

학습된 Skill-2 RL Expert를 sim에서 반복 실행한다. 카메라 렌더링이 필요하므로 `collect_demos.py` 내부의 `Skill2EnvWithCam`(Skill2Env를 상속하여 TiltedCamera 2대를 추가한 서브클래스)을 사용한다.

```bash
python collect_demos.py \
  --checkpoint logs/ppo_lekiwi/skill2_bc_finetune/checkpoints/best_agent.pt \
  --skill approach_and_grasp \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/Moving_Jaw_08d_v1" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --num_envs 4 --num_demos 1000 --headless \
  --annotate_subtasks
```

#### 4-4-1. Domain Randomization

**Dynamics DR (기본, 항상 적용)**:
- Wheel: stiffness 0.75~1.5x, damping 0.3~3.0x, friction 0.7~1.3x
- Arm: stiffness 0.8~1.25x, damping 0.5~2.0x
- Object: mass 0.5~2.0x, friction 0.6~1.5x
- 기존 v8 코드 `enable_domain_randomization=True` 사용
- Observation noise: joint_pos(0.01 rad), base_vel(0.02 m/s), object_rel(0.02 m)
- Action delay: 1 step (10-50ms 통신 지연 시뮬레이션)

**Weak Visual DR (기본)**:
- 물체 종류: object_catalog.json 12종 중 에피소드마다 1종 랜덤 선택 (나머지는 z=-10에 숨김)
- 물체 위치: 로봇 home에서 1.0~2.5m 거리, 360° 랜덤 방향, **바닥 위** (object_height = bbox_z × 0.5)
- 물체 회전(yaw 랜덤) / 스케일(±10%)
- 로봇 시작 위치/방향: home 기준 랜덤

**Strong Visual DR (추가, C5/C6/C7용)**:
- 물체 종류: 1030개 USD 전체 (object_catalog_all.json 활용)
- 물체 높이 다양화: 바닥(기본) + 선반/테이블 등 elevated surface 추가 (0.0~0.8m)
- 조명 DR, 카메라 노이즈, 바닥/벽 텍스쳐, Distractor 물체 1~3개

C4(dynamics + weak visual) vs C5(dynamics + strong visual)로 효과 비교.

#### 4-4-2. 성공 에피소드만 저장 + Visibility Gate

RL Expert 성공률 90%+라 해도 실패 에피소드는 VLA에 나쁜 행동을 가르칠 수 있으므로 제외. 성공 판정: 물체가 그리퍼에 결합 + z > target_height.

**Visibility Gate**: 에피소드 시작부에서 물체가 카메라에 아직 안 보이는 구간을 **앞에서 잘라낸다**. 첫 visible frame(base_cam 또는 wrist_cam에서 물체가 10×10 pixel 이상)부터 저장을 시작한다.

중요: **에피소드 중간의 프레임을 제거하지 않는다.** VLA는 action chunk(π0-FAST 10 steps, GR00T 16 steps)를 연속 미래 action으로 학습하므로, 중간에 구멍이 나면 temporal continuity가 깨져서 chunk label이 엉뚱한 미래를 가리킨다. Visibility Gate는 반드시 에피소드 **앞부분 trim**으로만 적용한다.

#### 4-4-3. Instruction 생성

물체 종류와 위치에 맞게 템플릿:

```
"approach the {color} {object} and grasp it"
"pick up the {object} from the {surface}"
"grab the {color} {object} on the {location}"
```

기존 `collect_demos.py`의 physics multi-object instruction 생성(활성 물체 이름 기반) 패턴을 확장.

#### 4-4-4. 수집 규모

C1~C5: 1K개, C6~C7: 10K개. Camera 수집 시 VRAM에 따라 num_envs 1~8, state-only는 64+.

---

### 4-5. Skill-3 데이터 수집 (RL Expert Rollout)

학습된 Skill-3 RL Expert를 sim에서 반복 실행. 카메라 렌더링이 필요하므로 `collect_demos.py` 내부의 `Skill3EnvWithCam`(Skill3Env를 상속하여 TiltedCamera 2대를 추가한 서브클래스)을 사용한다.

```bash
python collect_demos.py \
  --checkpoint logs/ppo_lekiwi/skill3_bc_finetune/checkpoints/best_agent.pt \
  --skill carry_and_place \
  --handoff_buffer handoff_buffer.pkl \
  --multi_object_json object_catalog.json \
  --gripper_contact_prim_path "/World/envs/env_.*/Robot/Moving_Jaw_08d_v1" \
  --dynamics_json calibration/tuned_dynamics.json \
  --arm_limit_json calibration/arm_limits_real2sim.json \
  --num_envs 4 --num_demos 1000 --headless \
  --annotate_subtasks
```

초기 상태: Handoff Buffer에서 샘플. DR은 Skill-2와 동일 (물체 위치 DR 불필요, home 위치/방향 랜덤화).

Instruction: "carry the mug back to the basket and place it", "bring the bottle to the home position".

성공 판정: 물체가 목표 위치 5cm 이내 안착 + gripper open.

---

### 4-6. 데이터 생성 경로 비교

**경로 A (Mimic):** 텔레옵 10~20개를 keypoint 보간으로 1K~10K개 생성. 장점: 빠르고 사람 행동 패턴 보존. 단점: 품질 상한 = 텔레옵 수준.

**경로 B (RL Rollout):** BC→RL expert 자동 실행. 장점: 사람보다 효율적인 행동 발견 가능. 단점: RL 학습 시간.

**경로 C (A + B 합산):** 실험 C2(Mimic only) vs C3(RL only) vs C4(합산)으로 비교.

---

## 5. Phase 3: 데이터 변환

### 5-1. HDF5 → LeRobot v3

기존 `convert_hdf5_to_lerobot_v3.py` 사용.

```bash
python convert_hdf5_to_lerobot_v3.py \
  --input outputs/rl_demos/*.hdf5 \
  --output_root outputs/lerobot_v3/lekiwi_skill2 \
  --repo_id local/lekiwi_skill2 \
  --fps 25 \
  --include_subtask_index \
  --skip_episodes_without_images
```

**LeRobot v3 Parquet 주요 columns:**

| Column | Type | Shape | 설명 |
|--------|------|-------|------|
| observation.state | float32 | (9,) | arm 5 + grip 1 + base_body_vel 3 |
| action | float32 | (9,) | arm_target 5 + grip_cmd 1 + base_cmd 3 |
| observation.images.base | VideoFrame | — | base_cam MP4 |
| observation.images.wrist | VideoFrame | — | wrist_cam MP4 |
| episode_index | int64 | — | 에피소드 번호 |
| task_index | int64 | — | tasks.parquet의 instruction 참조 |

**robot_state 추출**: Skill-2/3의 RL obs는 각각 30D/29D이지만, VLA에 전달되는 것은 공통 9D (arm 5 + grip 1 + base_body_vel 3). 두 skill 모두 obs의 앞 9D가 [arm(5) + grip(1) + base_body_vel(3)]이므로, obs[0:9]를 그대로 추출하면 된다. base_body_vel은 `root_lin_vel_b`(vx, vy)와 `root_ang_vel_b`(wz)에서 직접 읽는다. 단위 변환 불필요.

**⚠ v8의 obs[18:24]+obs[30:33] 추출은 사용하지 않는다.** 기존 `convert_hdf5_to_lerobot_v3.py`의 `infer_robot_state_from_obs()`와 `collect_demos.py`의 `extract_robot_state_9d()`는 v8의 37D obs 구조에 맞춰져 있고, 마지막 3D가 wheel_angular_vel(개별 휠 각속도)이다. 새 skill env에서는 collect 시점에 body-frame velocity를 직접 읽어 HDF5의 `robot_state` 필드에 저장하고, 변환 스크립트는 이 필드를 그대로 읽는다.

**Action 순서 주의**: 새 skill 환경의 action은 `[arm5, grip1, base3]` (yubinnn11/lekiwi3 v3.0 호환). 기존 v8의 `[base3, arm6]` 순서와 **반대**이므로, v8의 `_apply_action()` 코드 재사용 시 인덱스 매핑을 변경해야 한다: `base_cmd = action[:, 6:9]`, `arm_target = action[:, 0:6]`.

**Gripper 값**: RL 학습 시에는 continuous position target(v8과 동일). VLA 데이터 저장 시에만 threshold 0.5 → binary 0/1 변환.

**채널명과 순서는 yubinnn11/lekiwi3 (v3.0)과 동일하게 맞춘다.** observation.state[0]=arm_shoulder_pan.pos, [5]=arm_gripper.pos, [6]=x.vel, [8]=theta.vel.

**단위 변환 불필요**: sim velocity(m/s, rad/s) = real velocity(m/s, rad/s). 이전 설계의 m→mm 변환(×1000)은 삭제됨.

**Gripper 값 변환 (필수)**: RL의 continuous gripper 값을 threshold 0.5 → binary 0/1 변환. 0=closed, 1=open.

### 5-2. LeRobot v3 → GR00T LeRobot v2 (GR00T 사용 시에만)

Isaac-GR00T repo의 convert_v3_to_v2.py로 변환 후 modality.json 추가.

**LeKiwi용 modality config:**
- state: single_arm(5D) + gripper(1D) + base_body_vel(3D)
- action: single_arm(5D, RELATIVE) + gripper(1D, ABSOLUTE) + base_vel_cmd(3D, RELATIVE)
- action_horizon: 16 steps
- video: base + wrist

### 5-3. 정규화 통계 (stats.json)

π0-FAST: 1st/99th percentile quantile → [-1, 1]. 이상치가 통계를 왜곡하지 않는지 확인.

---

## 6. 데이터 품질 검증

**채널 정합성**: observation.state[0]=arm_shoulder_pan.pos, action[8]=base_theta_vel_cmd. yubinnn11/lekiwi3 v3.0 일치.

**값 범위**: arm joint이 joint limits 내, gripper가 정확히 0 또는 1, base velocity가 m/s 스케일(~0.01~0.3 m/s).

**단위 확인**: 프레임당 값이 0.003~0.01이면 m 단위가 미변환된 것.

**시간 동기화**: 이미지 frame_index와 state/action 매칭.

**Instruction-Skill 정합**: ApproachAndGrasp 에피소드에 "navigate to the table"이 붙어있으면 안 된다.

**Action chunk 호환성**: π0-FAST(chunk 10)와 GR00T(chunk 16)는 같은 데이터를 쓴다. 데이터는 프레임 단위 저장, 모델이 학습 시 자기 chunk size만큼 미래 action을 label로 사용.

---

## 7. 전체 데이터 흐름 요약

```
Phase 0: Sim-Real 일치 ★ Phase 1 시작 전 필수 (Hard Gate) ★
  [✅] 실로봇 모터 특성 → sim 물리 파라미터 일치 (tuned_dynamics.json)
  [✅] 실로봇 관절 한계 → arm_limits_real2sim.json 생성
  [✅] Calibration gate 통과 (wheel=0.146, arm=0.087)
  [⬜] 실제 카메라 특성 → sim 카메라 일치
  [✅] Joint limits → `arm_limit_write_to_sim=True`로 코드에서 자동 적용
  [⬜] REST_POSE 안전 자세 재정의
  [✅] 데이터셋 형식 확인 — yubinnn11/lekiwi3 v3.0, velocity(m/s, rad/s)

Phase 1: RL Expert 학습 (RTX 3090)
  Skill-2 (ApproachAndGrasp):
    텔레옵 10~20개 → BC (30D obs, 성공률 ~30%) → RL (PPO+AAC, 성공률 90%+)
  Handoff Buffer:
    Skill-2 성공 상태 200~500개 저장
  Skill-3 (CarryAndPlace):
    텔레옵 10~20개 (Handoff에서) → BC (29D obs) → RL (PPO+AAC, 성공률 90%+)

Phase 2: VLA 데이터 대량 수집 (RTX 3090)
  Navigate: 스크립트 정책 → 1K~2K 에피소드
  Skill-2: RL Expert rollout × (Dynamics + Visual) DR → 1K~10K (성공 + visibility trim)
  Skill-3: RL Expert rollout × (Dynamics + Visual) DR → 1K~10K (성공 + visibility trim)

  저장: 이미지 + 9D state + 9D action + instruction (privileged obs 저장하지 않음)

Phase 3: 포맷 변환
  HDF5 → LeRobot v3 (Parquet + MP4)
  단위 변환 불필요 확인 (sim velocity = real velocity)
  gripper continuous → binary 0/1 변환
  (GR00T) → v2 + modality.json
  채널명/순서: yubinnn11/lekiwi3 v3.0 일치 (x.vel, y.vel, theta.vel)

  → Navigate + Skill-2 + Skill-3 데이터 통합 → VLA 파인튜닝 데이터셋 완성 → A100 전송
```
