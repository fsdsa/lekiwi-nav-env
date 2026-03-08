# LeKiwi 3-Skill Pipeline (v9 — Navigate + ApproachGrasp + CarryPlace)

Isaac Lab `DirectRLEnv` 기반 LeKiwi 3-Skill 파이프라인.
핵심: **Navigate(scripted) → Skill-2 RL(ApproachGrasp, AAC) → Skill-3 RL(CarryPlace, AAC) → VLA 증류**.

## 1) v8 → v9 변경점

- **3-Skill 분리 아키텍처**: 기존 unified 4-phase FSM을 3개 독립 스킬로 분리
  - **Navigate (Skill-1)**: Scripted proportional controller, VLA 학습 데이터 생성용
  - **ApproachGrasp (Skill-2)**: RL 환경, 14D actor obs / 21D critic obs (AAC)
  - **CarryPlace (Skill-3)**: RL 환경, 13D actor obs / 20D critic obs (AAC)
- **Asymmetric Actor-Critic (AAC)**: Actor는 real-deployable obs, Critic은 privileged info 추가
- **Handoff Buffer**: Skill-2 성공 상태를 저장 → Skill-3 초기화에 사용 (200-500 entries)
- **Curriculum Learning**: Skill-2 object distance 0.5m→2.5m, 70% 성공률 시 증가
- **break_force 변경**: mass × g × 10 (기존 1e8에서 물리적으로 현실적인 값으로)
- **Visibility Gate**: 에피소드 시작부 카메라에 물체 안 보이는 구간 제거
- **VLA 호환 변환**: base displacement m→mm (×1000), gripper binary 0/1

## 2) 파일 구조

```text
scripts/lekiwi_nav_env/
├── __init__.py                    # Gymnasium 등록 (fetch + skill2 + skill3)
├── lekiwi_robot_cfg.py            # ArticulationCfg, Kiwi IK
├── lekiwi_nav_env.py              # 기존 unified 4-phase env (backward compat)
├── env_common.py                  # ★ 공통 base environment class
├── skill2_approach_grasp_env.py   # ★ Skill-2 ApproachGrasp RL 환경
├── skill3_carry_place_env.py      # ★ Skill-3 CarryPlace RL 환경
├── handoff_buffer.py              # ★ Handoff Buffer save/load/sample
├── models.py                      # PolicyNet, ValueNet, AsymmetricValueNet
├── collect_navigate_data.py       # ★ Navigate scripted policy 데이터 수집
├── train_lekiwi.py                # PPO 학습 (--skill fetch/skill2/skill3)
├── train_bc.py                    # BC 학습 (--skill skill2/skill3)
├── collect_demos.py               # RL expert 데모 수집 (--skill, --visibility_gate)
├── convert_hdf5_to_lerobot_v3.py  # HDF5→LeRobot v3 (--base_disp_m_to_mm, --gripper_binary)
├── build_object_catalog.py        # USD bbox + 대표 물체 선별
├── calibrate_real_robot.py
├── tune_sim_dynamics.py
├── spawn_manager.py
├── record_teleop.py
├── deploy_vla_action_bridge.py
└── pipeline/README.md             # 이 문서
```

## 3) Observation / Action 정의

### Skill-2 ApproachAndGrasp

| 구분 | Dim | 구성 |
|------|-----|------|
| Actor obs | 14D | arm_pos(5) + grip_pos(1) + base_disp(3) + rel_object(3) + contact(2) |
| Critic obs | 21D | Actor 14D + obj_bbox(6D) + obj_mass(1D) |
| Action | 9D | arm_target(5) + gripper_cmd(1) + base_cmd(3) |

### Skill-3 CarryAndPlace

| 구분 | Dim | 구성 |
|------|-----|------|
| Actor obs | 13D | arm_pos(5) + grip_pos(1) + base_disp(3) + home_rel(3) + grip_force(1) |
| Critic obs | 20D | Actor 13D + obj_dims(3D) + obj_mass(1D) + grip_rel_pos(3D) |
| Action | 9D | arm_target(5) + gripper_cmd(1) + base_cmd(3) |

## 4) 파이프라인 실행

### Step 1: Navigate 데이터 수집 (scripted)
```bash
python collect_navigate_data.py \
    --num_demos 200 --num_envs 4 \
    --multi_object_json object_catalog.json \
    --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body>" \
    --dynamics_json calibration/tuned_dynamics.json
```

### Step 2: Skill-2 RL 학습 (ApproachAndGrasp)
```bash
python train_lekiwi.py --skill skill2 --num_envs 2048 \
    --multi_object_json object_catalog.json \
    --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body>" \
    --dynamics_json calibration/tuned_dynamics.json --headless
```

### Step 3: Skill-2 데모 수집 + Handoff Buffer 생성
```bash
python collect_demos.py --skill skill2 \
    --checkpoint logs/ppo_lekiwi/skill2_scratch/checkpoints/best_agent.pt \
    --num_demos 50 --visibility_gate \
    --multi_object_json object_catalog.json \
    --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body>"
```

### Step 4: Skill-3 RL 학습 (CarryAndPlace)
```bash
python train_lekiwi.py --skill skill3 --num_envs 2048 \
    --handoff_buffer_path handoff_buffer.pt \
    --multi_object_json object_catalog.json \
    --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body>" \
    --dynamics_json calibration/tuned_dynamics.json --headless
```

### Step 5: HDF5 → LeRobot v3 변환
```bash
python convert_hdf5_to_lerobot_v3.py \
    --input outputs/skill2_demos/*.hdf5 \
    --output_root lerobot_skill2/ \
    --base_disp_m_to_mm --gripper_binary \
    --include_subtask_index
```

### Step 6: VLA 학습
- π0-FAST (chunk 10) 또는 GR00T N1.6 (chunk 16) target

## 5) Reward 설계

### Skill-2 Rewards
- `approach`: -‖robot−object‖₂ (거리 기반)
- `grasp_success`: +10 (파지 성공)
- `lift`: +5 (물체 들어올림)
- `collision`: -1 (충돌)
- `time_penalty`: -0.01/step

### Skill-3 Rewards
- `carry`: -‖robot−home‖₂ (홈 방향 거리)
- `hold`: +0.1/step (물체 유지)
- `place_success`: +20 (배치 성공)
- `drop`: -10 (물체 낙하)
- `collision`: -1 (충돌)
- `time_penalty`: -0.01/step

## 6) Curriculum (Skill-2)

- 시작: object_dist_max = 0.5m
- 성공률 > 70% 시: object_dist_max += 0.2m
- 최종: object_dist_max = 2.5m

## 7) Handoff Buffer

Skill-2 성공 시 터미널 상태를 저장:
- robot root state, joint positions/velocities
- object position/orientation, bbox, mass
- home position

Skill-3 `_reset_idx()`에서 이 상태로 초기화.
Target buffer size: 200-500 entries.

## 8) 기존 호환성

- `--skill fetch` (기본값): 기존 unified 4-phase 환경 사용
- 기존 33D/37D 모드 그대로 동작
- BC → RL warm-start도 `--skill fetch`에서 그대로 지원
