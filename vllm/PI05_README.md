# Pi0.5 Backup Training Setup

Pi0fast 학습이 실패할 경우 즉시 전환할 수 있도록 미리 준비된 pi0.5 학습 환경.

## 준비 완료된 것

| 항목 | 상태 | 경로/설정 |
|------|------|-----------|
| Pi0.5 base model | 다운로드됨 (14GB) | `./pi05_base/` |
| 학습 스크립트 | 작성됨 | `train_pi05.sh` |
| Stats q01/q99 | 이미 있음 | `lekiwi_viva_v2/meta/stats.json` |
| 데이터셋 | 동일 | `local/lekiwi_fetch_v6` |

## Pi0.5 vs Pi0fast 핵심 차이

| 항목 | Pi0fast | Pi0.5 |
|------|---------|-------|
| Action 표현 | FAST tokens (DCT+BPE 이산) | Continuous (flow matching) |
| 학습 방식 | Autoregressive token prediction | Velocity field regression |
| 정규화 (default) | MEAN_STD | **QUANTILES** |
| Multi-modal | token 표현으로 가능 | flow matching이 자연스럽게 처리 |
| 작은 dataset | 안정성 떨어짐 (issue #1811) | 더 안정적 |
| Inference | ~50 token 생성 | 5-10 denoising step |
| Memory | 적음 (3B params) | **더 많음** (3B + action expert) |

## ⚠ 주요 차이점: Image Key 매핑

**Pi0.5 base config는 3개 이미지 기대:**
- `observation.images.base_0_rgb`
- `observation.images.left_wrist_0_rgb`
- `observation.images.right_wrist_0_rgb`

**우리 데이터셋은 2개:**
- `observation.images.front`
- `observation.images.wrist`

**해결 방법 (자동 처리):**
1. `--rename_map`으로 학습 시 자동 매핑:
   - `front` → `base_0_rgb`
   - `wrist` → `left_wrist_0_rgb`
2. `right_wrist_0_rgb`는 누락 → pi05가 **자동으로 -1 padded image로 채움**
   (modeling_pi05.py line 1199-1201 확인)

## 실행

```bash
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm
./train_pi05.sh
```

## 메모리 주의사항

- A100 40GB에서 batch_size=2 시작 권장
- gradient_checkpointing=true 필수
- compile_model=false (compile + grad_checkpoint 충돌 가능)
- OOM 시: batch_size=1로 낮추기

## 학습 명령어 (수동 실행 시)

```bash
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm

nohup /home/jovyan/yes/envs/lerobotpi0v2/bin/lerobot-train \
    --dataset.repo_id=local/lekiwi_fetch_v6 \
    --dataset.root=/home/jovyan/lerobot_data/lekiwi_viva_v2 \
    --policy.path=./pi05_base \
    --policy.repo_id=local/pi05_lekiwi \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_state_dim=32 \
    --policy.max_action_dim=32 \
    --batch_size=2 \
    --steps=200000 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=0 \
    --num_workers=4 \
    --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}' \
    --output_dir=outputs/train/pi05_lekiwi_$(date +%Y%m%d_%H%M%S) \
    > /home/jovyan/pi05_train.log 2>&1 &
```

## 추론 코드 수정 필요

`vla_inference_server.py`를 pi05 사용하도록 수정 시:
- `PI0FastPolicy.from_pretrained()` → `PI05Policy.from_pretrained()`
- 이미지 key는 학습 시 사용한 base_0_rgb / left_wrist_0_rgb 형식으로 보내야 함
- 또는 inference 측에서도 rename 처리

## Pi0fast 학습 실패 시 전환 절차

1. 현재 pi0fast 학습 중단:
   ```bash
   ps aux | grep lerobot-train | grep -v grep | awk '{print $2}' | xargs -r kill -9
   ```

2. GPU 메모리 정리 확인:
   ```bash
   nvidia-smi
   ```

3. Pi0.5 학습 시작:
   ```bash
   cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env/vllm
   ./train_pi05.sh
   ```

4. 첫 step 통과 확인 (sanity check):
   ```bash
   sleep 60
   tail -30 /home/jovyan/pi05_train_*.log
   ```

   에러 없이 step 1~10 통과하면 정상

5. 200K steps까지 학습 (~3.83 epoch)

## 검증 안 된 위험 요소

- ⚠ Pi05 chunk_size=10 override가 base의 chunk_size=50 weight와 호환되는지
- ⚠ rename_map이 lerobot 0.5.0에서 정상 작동하는지 (dry-run 필요)
- ⚠ 우리 데이터셋의 quantile stats가 pi05 학습에 충분한지
- ⚠ A100 40GB에서 batch_size=2가 OOM 없이 작동하는지

위 항목들은 실제 학습 시작 시 dry-run으로 검증해야 합니다.
