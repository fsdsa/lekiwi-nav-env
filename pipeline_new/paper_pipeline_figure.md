# Paper Figure: Full Pipeline Architecture

> 논문에 넣을 그림 설명용. ASCII diagram + 텍스트로 각 그림의 구성 요소를 정리한다.

---

## Figure 1. System Overview — VLM + VLA Hierarchical Control

전체 시스템의 실행 시점(inference-time) 아키텍처. "약병 찾아서 빨간 컵 옆에 놓아" 같은 자연어 명령을 받아 로봇이 자율적으로 수행하는 전체 흐름.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        User Command (Natural Language)                    │
│               "Find the medicine bottle and place it next to             │
│                            the red cup"                                  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     VLM: Qwen3-VL-8B-Instruct                         │
│                     (High-Level Planner, "What")                         │
│                                                                          │
│  ① Classify (1회, text-only)                                             │
│     "약병 찾아서 빨간 컵 옆에 놓아"                                        │
│     → { source: "medicine bottle", dest: "red cup" }                     │
│                                                                          │
│  ② Orchestrate (반복, 0.3 Hz, base_cam RGB)                              │
│     상황 판단 → Skill Phase 전환 → VLA Instruction 생성                    │
│     "turn right to search for the medicine bottle"                       │
│     "approach and pick up the medicine bottle"                           │
│     "move toward the red cup and place the bottle"                       │
│     "done"                                                               │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │  instruction (text)
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      VLA: Pi0-FAST 2.9B                                  │
│                   (Low-Level Controller, "How")                          │
│                                                                          │
│  Input:  base_cam RGB + wrist_cam RGB + 9D state + instruction           │
│  Output: 9D action chunk [arm(5), gripper(1), base(3)]                   │
│  Freq:   5~10 Hz (동기)                                                  │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │  9D action
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Safety Layer (Local, 100 Hz)                           │
│                                                                          │
│  base_cam depth → 전방 중앙 1/3 min depth < 0.3m → base 정지              │
│  서버 응답 timeout → gentle stop                                          │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  LeKiwi Robot    │
                     │  arm(5) + grip(1)│
                     │  + base(3)       │
                     │  = 9D action     │
                     └─────────────────┘
```

**핵심 포인트 (캡션용)**:
- VLM = "What to do" (상위 계획), VLA = "How to do it" (하위 제어)
- VLM은 0.3Hz로 상황 판단 + instruction 생성, VLA는 5~10Hz로 연속 action 생성
- Depth-based safety layer가 100Hz로 장애물 긴급 정지
- 전체 루프 6.4Hz 실측 (A100 서버)

---

## Figure 2. 3-Skill Decomposition

하나의 long-horizon task를 3개 skill로 분리하여 RL 학습 효율과 데이터 수집 유연성을 확보.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                            Full Task (Long-Horizon)                            │
│        "Find object → Pick up → Carry to destination → Place down"             │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
     ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
     │   Skill 1         │  │   Skill 2         │  │   Skill 3         │
     │   Navigate         │  │   Approach &      │  │   Carry &         │
     │                    │  │   Grasp            │  │   Place           │
     ├────────────────────┤  ├────────────────────┤  ├────────────────────┤
     │ VLM 방향 명령 추종  │  │ Base+Arm 동시 접근 │  │ 물체 운반 + 배치   │
     │ + 장애물 회피       │  │ → 물체 파지 + Lift │  │ dest 물체 옆에 놓기│
     ├────────────────────┤  ├────────────────────┤  ├────────────────────┤
     │ Obs: 20D (actor)   │  │ Obs: 30D (actor)   │  │ Obs: 29D (actor)   │
     │       25D (critic) │  │       37D (critic)  │  │       36D (critic) │
     │ Act: base 3D only  │  │ Act: arm5+grip1     │  │ Act: arm5+grip1    │
     │  (arm=tucked)      │  │      +base3 = 9D    │  │      +base3 = 9D   │
     ├────────────────────┤  ├────────────────────┤  ├────────────────────┤
     │ Method: ResiP      │  │ Method: ResiP       │  │ Method: ResiP      │
     │ (DP BC + Residual  │  │ (DP BC + Residual   │  │ (DP BC + Residual  │
     │  PPO)              │  │  PPO)               │  │  PPO)              │
     └────────┬───────────┘  └────────┬───────────┘  └────────────────────┘
              │                       │                        ▲
              │                       │  Handoff Buffer        │
              │                       │  (success states       │
              │                       │   → pickle)            │
              │                       └────────────────────────┘
              │
              │  VLM이 skill 전환 판단:
              │  물체가 카메라 FOV 내 + 0.7m 이내 → Skill 2
              │  물체 파지 완료 + dest 물체 근접    → Skill 3
              ▼
```

**핵심 포인트 (캡션용)**:
- 3-Skill 분리: 각 skill은 짧은 horizon + dense reward로 RL 학습 수렴 보장
- Skill 간 전환은 VLM이 카메라 이미지로 판단 (handcrafted trigger 없음)
- Handoff Buffer: Skill-2 성공 상태를 Skill-3 초기 상태로 연결
- 모든 skill이 동일한 9D action space → 단일 VLA에 통합 학습 가능

---

## Figure 3. Training Pipeline — Teleop → BC → RL → VLA Distillation

Sim에서 RL expert를 학습하고, expert rollout으로 VLA 학습 데이터를 대량 생산하는 전체 학습 파이프라인.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                      │
│                                                                                     │
│  Phase 0: Sim-Real Calibration                                                      │
│  ┌─────────────────────────────────────────────────┐                                │
│  │ Real Robot Measurement                           │                                │
│  │  · Arm joint limits (leader arm TCP)             │                                │
│  │  · Tucked pose (self-collision boundary)         │                                │
│  │  · Wheel/arm dynamics (SysID)                    │                                │
│  │  · Command transform (LIN_SCALE, ANG_SCALE)     │                                │
│  └──────────────────────┬──────────────────────────┘                                │
│                         │ calibrated sim                                             │
│                         ▼                                                            │
│  Phase 1: RL Expert Training (Isaac Sim, A100)                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐       │
│  │                                                                           │       │
│  │  ┌────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────┐ │       │
│  │  │ Teleoperate │───▶│ BC (DP)    │───▶│ Residual    │───▶│ RL Expert    │ │       │
│  │  │ in Sim      │    │ (DDPM,     │    │ PPO (frozen │    │ Policy       │ │       │
│  │  │ (10~20 eps) │    │  ~5.3M)    │    │ DP + ~154K  │    │ (90%+ SR)    │ │       │
│  │  │             │    │ SR ~30%    │    │ residual)   │    │              │ │       │
│  │  └────────────┘    └────────────┘    └─────────────┘    └──────┬───────┘ │       │
│  │                                                                │         │       │
│  │  Navigate: DP BC + Residual PPO (base only, arm=tucked)        │         │       │
│  │  Approach&Grasp: DP BC + Residual PPO (arm+base simultaneous)  │         │       │
│  │  Carry&Place: DP BC + Residual PPO (carry + place at dest)     │         │       │
│  │                                                                │         │       │
│  └────────────────────────────────────────────────────────────────┘         │       │
│                                                                   │         │       │
│  Phase 2: Expert Rollout Data Collection (Isaac Sim, 3090)        │         │       │
│  ┌────────────────────────────────────────────────────────────────┘         │       │
│  │                                                                           │       │
│  │  RL Expert × 1K~10K episodes                                             │       │
│  │  ┌──────────────────────────────────────────────┐                        │       │
│  │  │  Per frame:                                   │                        │       │
│  │  │   · base_cam RGB (640×400)                    │                        │       │
│  │  │   · wrist_cam RGB (640×480)                   │                        │       │
│  │  │   · 9D robot_state [arm5, grip1, base_vel3]   │                        │       │
│  │  │   · 9D action [arm5, grip1, base_cmd3]        │                        │       │
│  │  │   · text instruction (VLM-style)              │                        │       │
│  │  └──────────────────────────────────────────────┘                        │       │
│  │  + Domain Randomization                                                   │       │
│  │    · Dynamics: wheel damping, object mass, obs noise, action delay        │       │
│  │    · Visual: lighting, textures, distractor objects                        │       │
│  │  + Visibility Gate: 물체가 카메라에 안 보이는 앞부분 제거                    │       │
│  │                                                                           │       │
│  └───────────────────────────────┬───────────────────────────────────────────┘       │
│                                  │  HDF5 → LeRobot v3 format                        │
│                                  ▼                                                   │
│  Phase 3: VLA Fine-Tuning (A100)                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────┐       │
│  │                                                                           │       │
│  │  Pi0-FAST 2.9B (PaliGemma VLM + FAST tokenizer)                          │       │
│  │  Input:  2× RGB + 9D state + text instruction                            │       │
│  │  Output: 9D action chunk (10 steps)                                       │       │
│  │                                                                           │       │
│  │  All 3 skills → single unified VLA                                        │       │
│  │  (instruction text가 skill을 자연스럽게 구분)                               │       │
│  │                                                                           │       │
│  └───────────────────────────────────────────────────────────────────────────┘       │
│                                                                                     │
│  Phase 4.5: Sim Full-System Validation                                               │
│  ┌───────────────────────────────────────────────────────────────────────────┐       │
│  │  Isaac Sim (3090) ←──HTTP──→ VLM + VLA (A100)                             │       │
│  │  Closed-loop: VLM instruction → VLA action → sim.step() → camera → VLM   │       │
│  │  End-to-end task completion test before real deployment                    │       │
│  └───────────────────────────────────────────────────────────────────────────┘       │
│                                                                                     │
│  Phase 5: Real-World Deployment                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────┐       │
│  │  Jetson Orin Nano (onboard) ←──WiFi──→ VLM + VLA (A100 server)            │       │
│  │  + D455 VIO + Wheel Odometry + EKF → pose estimation                      │       │
│  │  + Sim→Real action transform (calibrated scales)                          │       │
│  │  + Safety layer (depth 100Hz + timeout watchdog)                          │       │
│  └───────────────────────────────────────────────────────────────────────────┘       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**핵심 포인트 (캡션용)**:
- 사람 텔레옵은 skill당 10~20개만 필요 (Navigate는 0개)
- DP BC(30~40% SR) → Residual PPO(90%+ SR)로 bootstrapping
- RL Expert rollout으로 VLA 학습 데이터 1K~10K 자동 생산
- 3 skill 데이터를 single VLA에 통합 학습 (instruction이 skill 구분)

---

## Figure 4. ResiP (Residual Policy Learning) Architecture

Diffusion Policy BC를 frozen base로 두고, 경량 residual MLP를 PPO로 학습하는 구조.

```
                        Observation (30D)
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
    ┌─────────────────────┐   ┌─────────────────────────┐
    │  Diffusion Policy   │   │  Residual Policy         │
    │  (Frozen, ~5.3M)    │   │  (Trainable, ~154K)      │
    │                     │   │                           │
    │  ConditionalUnet1D  │   │  Actor: 256-256-ReLU      │
    │  DDPM train /       │   │  Critic: 256-256-ReLU     │
    │  DDIM 16-step infer │   │  Input: obs + base_action │
    │                     │   │                           │
    │  pred_horizon = 16  │   │  Per-dim scale:           │
    │  obs → action chunk │   │   arm=0.15, grip=0.20,    │
    │                     │   │   base=0.25               │
    └──────────┬──────────┘   └────────────┬──────────────┘
               │                           │
               │  base_action (normalized) │  residual_action
               │                           │
               └──────────┬────────────────┘
                          │  combined = base + residual × scale
                          ▼
                    9D Action Output
                   [arm5, grip1, base3]
```

**핵심 포인트 (캡션용)**:
- Frozen DP가 대략적 행동 제공, 경량 residual이 PPO로 보정
- Residual은 DP 파라미터의 ~3%만 학습 → sample efficient
- Per-dimension action scale로 arm/gripper/base 독립 조절
- Warmup: 초기에는 DP만 실행, 점진적으로 residual 활성화

---

## Figure 5. Asymmetric Actor-Critic (AAC)

RL 학습 시 Actor와 Critic이 서로 다른 observation을 받는 구조. Actor의 privileged obs로 최적 행동을 학습하되, VLA distillation 시에는 카메라+9D state만 사용.

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                        Isaac Sim Environment                             │
 │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────┐ │
 │  │ Robot State   │  │ Privileged   │  │ Camera Rendering               │ │
 │  │ (real에서도   │  │ (sim only)   │  │ (Phase 2 수집 시에만 활성)      │ │
 │  │  획득 가능)   │  │              │  │                                │ │
 │  │ arm(5)        │  │ rel_obj(3)   │  │ base_cam RGB                   │ │
 │  │ grip(1)       │  │ contact(2)   │  │ wrist_cam RGB                  │ │
 │  │ base_vel(3)   │  │ obj_bbox(3)  │  │                                │ │
 │  │ velocities(12)│  │ obj_cat(1)   │  │                                │ │
 │  └──────┬───────┘  └──────┬───────┘  └────────────┬───────────────────┘ │
 └─────────┼──────────────────┼───────────────────────┼────────────────────┘
           │                  │                       │
      ┌────┴──────────────────┴────┐                  │
      ▼                            ▼                  │
 ┌──────────┐              ┌──────────────┐           │
 │  Actor    │              │   Critic      │           │
 │  (30D)    │              │   (37D)       │           │
 │           │              │               │           │
 │ 9D+vel+   │              │ Actor 30D +   │           │
 │ rel_obj+  │              │ mass + dist + │           │
 │ contact+  │              │ heading +     │           │
 │ bbox+cat  │              │ vel_toward    │           │
 └─────┬─────┘              └───────────────┘           │
       │ 9D action                                      │
       │                                                │
       ▼                                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │           Expert Rollout Data (Phase 2)                      │
 │                                                              │
 │  저장: base_cam + wrist_cam + 9D state + 9D action + text   │
 │  삭제: privileged obs (rel_obj, contact, bbox, cat)          │
 │                                                              │
 │  → VLA는 카메라 이미지에서 privileged 정보를 스스로 추론       │
 │    (distillation의 핵심)                                     │
 └─────────────────────────────────────────────────────────────┘
```

**핵심 포인트 (캡션용)**:
- Actor: sim-only privileged obs로 최적 행동 생성 (카메라 이미지 불필요)
- Critic: Actor obs + 추가 privileged (mass, distance, heading)
- 수집 시: 카메라 이미지 렌더링 + 9D state/action만 저장
- VLA Student는 (이미지 + 9D state + instruction)으로 Teacher 행동을 모방

---

## Figure 6. Deployment Architecture

Sim 검증(Phase 4.5)과 실배포(Phase 5)의 하드웨어 구성.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Phase 4.5: Sim Full-System Validation                                  │
 │                                                                         │
 │  ┌─────────────────────┐         ┌──────────────────────────────────┐  │
 │  │  3090 Desktop        │  HTTP   │  A100 Server (40GB)              │  │
 │  │                      │◄───────▶│                                  │  │
 │  │  Isaac Sim (1 env)   │         │  VLM: Qwen3-VL-8B (~29.8GB)  │  │
 │  │  Camera rendering    │         │  VLA: Pi0-FAST 2.9B (~8.1GB)    │  │
 │  │  9D state reading    │         │  Total: ~37.9GB / 40GB           │  │
 │  │  Action execution    │         │                                  │  │
 │  └─────────────────────┘         └──────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Phase 5: Real-World Deployment                                         │
 │                                                                         │
 │  ┌─────────────────────┐         ┌──────────────────────────────────┐  │
 │  │  Jetson Orin Nano    │  WiFi   │  A100 Server (40GB)              │  │
 │  │  (onboard robot)     │◄───────▶│                                  │  │
 │  │                      │         │  VLM: Qwen3-VL-8B (~29.8GB)  │  │
 │  │  D455 RGB-D + IMU    │         │  VLA: Pi0-FAST 2.9B (~8.1GB)    │  │
 │  │  Wrist cam USB       │         │                                  │  │
 │  │  VIO + Wheel Odom    │         │                                  │  │
 │  │  + EKF fusion        │         │                                  │  │
 │  │  Safety layer (100Hz)│         │                                  │  │
 │  │  Sim→Real transform  │         │                                  │  │
 │  └─────────────────────┘         └──────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Figure 7. LeKiwi Robot — Hardware & Observation/Action Space

```
              ┌──────────────────────────────────────┐
              │          LeKiwi Mobile Manipulator     │
              │                                        │
              │  Arm: SO-100 (5 joints + 1 gripper)    │
              │  Base: Kiwi Drive (3 omni-wheels)      │
              │  Sensors:                               │
              │   · base_cam: Intel D455 (RGB-D + IMU) │
              │   · wrist_cam: USB webcam               │
              │   · 9× motor encoders                   │
              │  Compute: Raspberry Pi 5 (onboard)      │
              │           + Jetson Orin Nano (deploy)    │
              └──────────────────────────────────────┘

 State / Action Space (9D, unified across all skills):
 ┌────────────────────────────────────────────────────────────┐
 │  [0:5]  arm joint position target (5D, rad)                │
 │  [5]    gripper position target (1D, continuous)           │
 │  [6:8]  base linear velocity vx, vy (2D, m/s, body frame) │
 │  [8]    base angular velocity wz (1D, rad/s, CCW+)        │
 └────────────────────────────────────────────────────────────┘
```

---

## Figure 8. Experimental Comparison Matrix

```
┌──────────────────────┬──────────┬────────────┬──────────────────────────┐
│                       │ A1 (E2E) │ B1/B2      │ C1/C2 (Ours)            │
│                       │ Baseline │ VLM+VLA    │ VLM+VLA+RL              │
├──────────────────────┼──────────┼────────────┼──────────────────────────┤
│ VLM Orchestrator      │    ✗     │     ✓      │          ✓              │
│ Skill Decomposition   │    ✗     │     ✓      │          ✓              │
│ RL Expert             │    ✗     │     ✗      │          ✓              │
│ Human Teleop          │  50 eps  │  20~40 eps │    10~20 eps/skill      │
│ VLA Training Data     │  50 eps  │  20~1K eps │    1K~10K eps           │
│ Navigate Data Source  │  teleop  │   teleop   │  RL (0 human demos)     │
├──────────────────────┼──────────┼────────────┼──────────────────────────┤
│ Metric: Task SR (%)   │          │            │                          │
│ Metric: Avg Steps     │          │            │                          │
│ Metric: Grasp SR (%)  │          │            │                          │
└──────────────────────┴──────────┴────────────┴──────────────────────────┘

A1: Single VLA, full-task teleop 50개 → π0-FAST (no VLM, no skill split)
B1: VLM+VLA, skill별 teleop 20~40개 → π0-FAST
B2: VLM+VLA, teleop + Isaac Lab Mimic augmentation (1K) → π0-FAST
C1: VLM+VLA+RL, RL expert rollout 1K → π0-FAST (Ours)
C2: VLM+VLA+RL+Mimic, RL 1K + Mimic 1K → π0-FAST (Ours)
```

---

## Summary Table: Key Numbers

| Component | Spec |
|-----------|------|
| VLM | Qwen3-VL-8B-Instruct, ~29.8GB VRAM |
| VLA | Pi0-FAST 2.9B (LeRobot 0.5.0), ~8.1GB VRAM |
| BC Model | ConditionalUnet1D [64,128,256], ~5.3M params |
| Residual Policy | 256-256 MLP, ~154K params |
| State/Action | 9D unified [arm5, grip1, base3] |
| Sim Envs | 1024~8192 parallel (Isaac Sim, A100) |
| RL FPS | ~8,000~9,000 steps/s |
| Inference Loop | 6.4 Hz end-to-end |
| VLM Latency | ~191ms (async) |
| VLA Latency | 27~58ms (sync) |
| Human Demos | 10~20 per skill (Navigate: 0) |
| RL Expert Data | 1K~10K episodes auto-generated |
| Object Catalog | 22 categories (from 1030 USD) |
| Robot | LeKiwi (SO-100 arm + Kiwi Drive base) |
| Sensors | D455 RGB-D + USB wrist cam + 9 encoders |
