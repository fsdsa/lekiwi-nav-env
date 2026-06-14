# 노트북 실행 명령 (canonical) — viva / only-vla / rl_hybrid

> 세 모델 **모두 서버(A100)에 있음**. 노트북은 sim 실행 + SSH 터널로 서버에 요청.
> **연결**: `ssh A100` 별칭 사용 (포트는 `~/.ssh/config`; pod 재시작마다 바뀌니 `-p 30179`/`30628` 하드코딩 금지).
> **GUI로 실행** (`--headless` 금지 — 카메라 크래시).
> 난이도: **easy**=room_9, **middle**=room_4, **hard**=room_6→room_3. scene_idx 1302 고정, `--difficulty`만 바꾸면 됨.

## 공통 preamble (노트북, 셸마다 1회)
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES LEKIWI_USD_PATH="$HOME/Downloads/lekiwi_robot.usd"
cd ~/IsaacLab/scripts/lekiwi_nav_env
```

---
## ① VIVA (VLM + VLA)
**서버** (VLM+VLA):
```bash
ssh A100 'cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env && bash launch_servers.sh stop && bash launch_servers.sh all --checkpoint vllm/outputs/train/pi05_viva_v5_20260421_071948/checkpoints/250000/pretrained_model'
```
**노트북 터널** (8000+8002) → **실행** (`--difficulty` easy/middle/hard):
```bash
ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 A100
python vllm/run_full_task.py --mode viva --difficulty easy \
  --user_command "find the blue medicine bottle and place it next to the red cup" \
  --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --scene_idx 1302 --scene_scale 0.6 --num_trials 10 --vlm_interval 100
```

---
## ② Only-VLA (end-to-end, VLM 없음)
**서버** (VLA만 — `vla`, `all` 아님):
```bash
ssh A100 'cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env && bash launch_servers.sh stop && bash launch_servers.sh vla --checkpoint /home/jovyan/h100_endtoend_backup/outputs/h100_endtoend/checkpoints/064000/pretrained_model'
```
**노트북 터널** (8002만) → **실행** (`--difficulty` easy/middle/hard):
```bash
ssh -f -N -L 8002:localhost:8002 A100
python vllm/run_only_vla.py --difficulty easy \
  --instruction "find the medicine bottle and place it next to the red cup" \
  --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --scene_idx 1302 --scene_scale 0.6 --num_trials 10
```
> ★ instruction엔 **"blue" 없음** (VLM 정규화 없이 그대로 VLA에 들어가므로 학습 문자열과 정확 일치 필수). 생략하면 기본값이 이 문자열이라 더 안전.

---
## ③ rl_hybrid (S1 nav·S3 carry = 로컬 RL / S2·S4 = viva VLA)
**서버**: ①과 동일 (`launch_servers.sh all --checkpoint <pi05_viva_v5>`) · **터널**: 8000+8002
```bash
python vllm/run_full_task.py --mode rl_hybrid --difficulty easy \
  --nav_dp_checkpoint checkpoints/dp_bc_nav_skill2_v4/dp_bc_epoch300.pt \
  --nav_resip_checkpoint checkpoints/resip_nav_tucked_v4/resip_nav_best.pt \
  --carry_dp_checkpoint checkpoints/dp_bc_carry_v4/dp_bc_epoch300.pt \
  --carry_resip_checkpoint checkpoints/resip_carry_v6/resip_carry_iter240.pt \
  --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --scene_idx 1302 --scene_scale 0.6 \
  --user_command "find the blue medicine bottle and place it next to the red cup"
```

---
## ④ eval_vlm_rl (VLM 지시어 + RL expert만, VLA 없음)
> **4개 스킬 전부 로컬 RL** (S1 nav=lookup · S2 lift · S3 carry · S4 place), VLM은 스킬전환+방향지시어만.
> **VLA 불필요 → 8002 터널 X, 서버는 VLM만.** (구 `eval_viva_pipeline.py` 후속 — 이름 변경됨)
> place(S4)는 `eval_s3.py`의 55D obs + 4-phase(ABCD) 상태머신을 그대로 포팅.

**서버** (VLM만):
```bash
ssh A100 'cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env && bash launch_servers.sh stop && bash launch_servers.sh vlm'
```
**노트북 터널** (8000만) → **실행** (`--difficulty` easy/middle/hard):
```bash
ssh -f -N -L 8000:localhost:8000 A100
python vllm/eval_vlm_rl.py --difficulty easy \
  --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
  --resip_checkpoint "backup/appoachandlift/resip64%.pt" \
  --carry_dp_checkpoint checkpoints/dp_bc_carry_v4/dp_bc_epoch300.pt \
  --carry_resip_checkpoint checkpoints/resip_carry_v6/resip_carry_iter240.pt \
  --place_dp_checkpoint checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt \
  --place_resip_checkpoint checkpoints/resip_s3_v25/resip_iter80.pt \
  --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --scene_idx 1302 --scene_scale 0.6
```
> 키보드: 1/2/3/4=스킬 강제전환, t=TARGET_FOUND, r=리셋, q=종료. place 체크포인트 빼면 그 스킬만 정지(HOLD).

---
## 모델 전환 / 주의
- 포트 8002엔 VLA **1개만** → 모델 바꿀 때 서버에서 `bash launch_servers.sh stop` 후 해당 `--checkpoint`로 재시작.
- ①③ = viva VLA(`pi05_viva_v5/250000`) · ② = e2e VLA(`h100_endtoend/064000`).
- `ssh A100` 안 되면 pod 재시작된 것 → `~/.ssh/config`의 `Host A100` Port를 현재 값으로 갱신.
- 원본 노트의 `-p 30179`는 죽은 포트. only-vla는 VLA(8002)만 필요(VLM 8000 불필요).
- (검증됨) viva 학습 task = nav 6 + carry 6 + `approach and lift the medicine bottle` → place skill은 viva VLA에 미학습.
