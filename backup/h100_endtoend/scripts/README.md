# Pi0.5 H100 End-to-End Training — 배포 패키지

A100에서 학습 중인 기존 모델과 **별개**의 end-to-end (navigate+grasp+carry+place) fine-tuning.
H100 80GB 클라우드 서버 대상 (24 CPU core, 240 GB RAM).

## 📦 패키지 내용

```
h100_deploy/
├── README.md                 # 이 파일
├── setup_env.sh              # conda env + PyTorch + lerobot 설치
├── train_h100.sh             # 학습 런처 (H100 튜닝 완료)
├── base_model/               # pi05_base (PaliGemma-3B+300M, 14 GB)
└── dataset/                  # combined_aug4x_middle (96 eps, 519 MB)
    ├── data/                 #   state/action parquet (40 MB)
    ├── videos/               #   h264 mp4 × 4 files (480 MB)
    └── meta/                 #   episodes.parquet, stats.json, tasks.parquet
```

## 🚀 Quick Start (H100 서버에서)

```bash
# 1. 패키지 압축해제 후 해당 디렉터리로 이동
cd h100_deploy/

# 2. 환경 세팅 (최초 1회)
chmod +x setup_env.sh && ./setup_env.sh

# 3. 학습 시작 (기본 batch=16, 80K steps ≈ 3.3 epoch)
conda activate pi05_h100
chmod +x train_h100.sh && ./train_h100.sh

# 4. 모니터링
tail -f train.log
```

---

## 🛠 최적화 가이드

### 설정 변수 (환경변수로 override)

```bash
BATCH=32 STEPS=40000 LR=1e-4 DECAY=30000 ./train_h100.sh  # 더 공격적
BATCH=16 STEPS=80000 LR=7e-5 DECAY=60000 ./train_h100.sh  # 안전 (default)
```

| 변수 | Default (batch=16) | Alt (batch=32) | Note |
|---|---|---|---|
| BATCH | 16 | 32 | H100 80 GB 가용 |
| STEPS | 80,000 | 40,000 | 3.3 epoch 기준 |
| LR | 7e-5 | 1e-4 | sqrt 스케일링 |
| DECAY | 60,000 | 30,000 | 75% 지점 |
| SAVE_FREQ | 8,000 | 4,000 | 10%마다 |
| NUM_WORKERS | 8 | 12 | 24 core의 1/3 ~ 1/2 |
| COMPILE | false | true | 첫 실행은 false, 안정적이면 true |
| GRAD_CKPT | false | false | VRAM 충분, 속도 +30% |

### 속도 측정 → 역산 레시피

H100 받자마자 **500 step 테스트 런**으로 it/s 측정:

```bash
STEPS=500 ./train_h100.sh
# → train.log에서 마지막 it/s 확인
# 예: 2.5 it/s면 80000/2.5/3600 = 8.9h @ batch=16
```

이후 체감 필요 시 batch 32로 올려 재시작.

### VRAM 여유 보고 gradient_checkpointing 토글

```bash
nvidia-smi  # VRAM 70+ GB 쓰고 있으면 grad_ckpt=true로 낮춰
```

- batch=16 + grad_ckpt=false → 예상 ~45 GB
- batch=32 + grad_ckpt=false → 예상 ~65 GB
- 80 GB 가까이 차면 위험 → GRAD_CKPT=true로 전환

---

## 🔑 튜닝 결정 근거 (왜 이 값?)

### LR 스케일링 (sqrt rule)

원본 A100 config: `batch=2, LR=2.5e-5`

**Square root rule** (fine-tuning에 적합):
- batch=16: LR = 2.5e-5 × √(16/2) = **7.07e-5**
- batch=32: LR = 2.5e-5 × √(32/2) = **1e-4**

Linear rule은 대규모 pre-training에 적합. fine-tuning + 소규모 data(96 eps)는 보수적 스케일링이 안전.

### Steps 계산

```
frames = 383,584
steps_per_epoch = frames / batch
  batch=16: 23,974 steps/epoch
  batch=32: 11,987 steps/epoch

target ≈ 3 epoch:
  batch=16: 72,000 steps → 80,000 여유 포함
  batch=32: 36,000 steps → 40,000 여유 포함
```

### Scheduler decay

cosine_decay_with_warmup. warmup 1000 step, decay 60K로 75% 지점에서 LR이 0에 근접.

### n_action_steps=50

원본 A100 학습과 동일. eval 시 chunk 전체 사용 (receding horizon 전략은 runtime에서 skill별 분기).

---

## ⚠️ 주의 사항

### 1. Gradient checkpointing off 안정화

`gradient_checkpointing=false`로 시작했다가 OOM 나면:
```bash
GRAD_CKPT=true ./train_h100.sh
```
그러면 +30% 속도 손해는 있지만 안전.

### 2. compile_model

Pi0.5는 가끔 torch.compile에서 이슈 (custom kernels). 첫 실행은 **false**로 시작. 안정적이면 `COMPILE=true` 재시작.

### 3. DataLoader 병목

H100이 너무 빨라서 `num_workers=8`로도 데이터 로딩이 못 따라오면 GPU util이 80% 이하로 떨어짐. 그 때는 `NUM_WORKERS=12` 또는 16으로 올려.

`nvidia-smi dmon -c 20` 로 실시간 GPU util 관찰.

### 4. Dataset 경로

`dataset/` 디렉터리가 symbolic link일 수 있으니, 압축할 때 `tar --dereference` 쓰거나 미리 `cp -rL` 해서 실제 복사본 만들기.

---

## 📊 학습 모니터링

```bash
# loss 추이
grep -E 'INFO.*ot_train.py.*step:' train.log | tail -10

# 스텝당 시간
grep -oE 'it/s|s/step' train.log | tail -3

# GPU 상태
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
```

정상 지표:
- loss: 처음 ~0.5 → 초반 빠르게 감소 → 0.01~0.05 사이 수렴
- GPU util: 85%+ (낮으면 DataLoader 병목)
- Temp: <80°C

---

## 📂 출력물

`outputs/h100_endtoend/checkpoints/NNNN0000/` 에 save_freq마다 저장:
- `pretrained_model/` — 배포용 모델 (eval/inference용)
- `training_state/` — optimizer state (resume용)

마지막 ckpt는 `checkpoints/last` 심볼릭 링크.

---

## 🔄 Resume

중단 후 재개:
```bash
lerobot-train --config_path=outputs/h100_endtoend/checkpoints/last/pretrained_model/train_config.json --resume=true
```

---

## 📝 Dataset 특성 (참고)

- **96 episodes** = 24 anchor (실제 human teleop) + 72 noise aug (action-only σ scaled from delta)
- **Task**: "find the medicine bottle and place it next to the red cup" (end-to-end)
- **Difficulty**: 48 easy + 48 middle (scene clutter 정도 다름)
- **Frame rate**: 25 Hz
- **Trajectory length**: 1,900 ~ 6,906 frames (매우 long horizon)

노이즈 σ (action 전용, state unchanged):
```
arm_pan=0.00054, arm_lift=0.00260, arm_elbow=0.00235, arm_wristf=0.00305,
arm_wristr=0.00128, gripper=0.00398, x.vel=0.00114, y.vel=0.02654, theta.vel=0.05037
```
→ 새 데모 자체의 delta_std × 0.5 (teleop 자연 smoothness의 절반)
