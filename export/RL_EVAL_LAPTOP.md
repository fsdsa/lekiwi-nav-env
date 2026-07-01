# 노트북(kswltd) RL eval 명령어 — 로컬 sim 검증용

> **이 4개는 순수 RL/BC sim eval — VLM/VLA 서버·터널 불필요.** 노트북에서 바로 실행.
> 원본(yubin11)→노트북(kswltd) 수정: `/home/yubin11/`→`~/`, `backup/approachandlift`(오타)→실제 `backup/appoachandlift`, `resip64%.pt` 따옴표 처리.
> 모든 파일 전송·검증 완료 (스크립트·체크포인트·backup·USD). GPU RTX 5070 Ti 12GB로 num_envs=1 eval 충분.

## 0. 먼저 (터미널/셸마다 1회)
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
export LEKIWI_USD_PATH="$HOME/Downloads/lekiwi_robot.usd"
cd ~/IsaacLab/scripts/lekiwi_nav_env
```

## 1. Navigate RL  (★ 26D 체크포인트로 수정 — 아래 주의)
> ⚠️ 원래 명령의 `dp_bc_nav`(20D) + `backup/navigate/resip_best`(20D)는 **현재 eval_navigate.py(26D obs)와 안 맞아** `tensor a(26) vs b(20)` 에러남. 현재 env에 맞는 **26D 짝**(rl_hybrid의 nav 체크포인트)을 써야 함:
```bash
python eval_navigate.py --skill navigate \
    --dp_checkpoint checkpoints/dp_bc_nav_skill2_v4/dp_bc_epoch300.pt \
    --resip_checkpoint checkpoints/resip_nav_tucked_v4/resip_nav_best.pt \
    --num_envs 1 --num_episodes 6 \
    2> >(grep -v "omni.physx.tensors.plugin")
```

## 2. Approach & Lift RL  (수정: 경로 ~, backup 철자, % 따옴표)
```bash
python eval_resip.py --skill approach_and_grasp \
    --dp_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    --resip_checkpoint "backup/appoachandlift/resip64%.pt" \
    --num_episodes 5 \
    --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
    2> >(grep -v "omni.physx.tensors.plugin")
```

## 3. Carry RL  (수정: % 따옴표만)
```bash
python eval_carry.py \
    --carry_bc_checkpoint checkpoints/dp_bc_carry_v4/dp_bc_epoch300.pt \
    --carry_resip_checkpoint checkpoints/resip_carry_v6/resip_carry_iter240.pt \
    --s2_bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    --s2_resip_checkpoint "backup/appoachandlift/resip64%.pt" \
    --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
    --num_episodes 6 \
    2> >(grep -v "omni.physx.tensors.plugin")
```

## 4. Approach & Place RL (S3)  (수정: object/dest 경로 ~)
```bash
PYTHONUNBUFFERED=1 python eval_s3.py \
    --s3_bc_checkpoint checkpoints/dp_bc_skill3_55d_fixed_1e-4/dp_bc_epoch500.pt \
    --s3_resip_checkpoint checkpoints/resip_s3_v25/resip_iter80.pt \
    --s2_bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \
    --s2_resip_checkpoint "backup/appoachandlift/resip64%.pt" \
    --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
    --object_scale_phys 0.7 \
    --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
    --dest_object_scale 0.56 \
    --num_envs 1 \
    --num_episodes 20 \
    2>&1 | grep --line-buffered -v "omni.physx.tensors.plugin" | tee ~/eval_s3_v19u_iter110.log
```

## 참고
- 4개 다 **서버 없이** 로컬 sim에서 동작 (RL/BC 체크포인트만, VLM/VLA 미사용).
- `2> >(grep -v ...)` 프로세스 치환은 **bash 필요** (sh로 실행 금지).
- 첫 실행 시 Isaac Sim EULA는 위 `OMNI_KIT_ACCEPT_EULA=YES`로 자동 동의됨.
- backup 폴더 실제 철자는 `appoachandlift`(오타, r 없음) — 원본에서부터 그랬음.
