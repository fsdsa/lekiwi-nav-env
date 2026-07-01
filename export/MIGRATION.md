# LeKiwi 파이프라인 — 새 데스크탑으로 옮기기 (서버는 그대로)

> 서버(A100: VLM `:8000` Qwen3‑VL‑8B / VLA `:8002` Pi0.5)는 **안 건드린다.**
> sim/teleop/eval + 로컬 BC·RL 추론 + VLM/VLA 요청을 돌리는 **데스크탑만 복제**한다.
>
> "IsaacLab 폴더만 옮기면 되나?" → **아니다.** 아래 **5가지**가 다 있어야 동작한다:
> ① Isaac Sim conda env  ② 코드(IsaacLab+프로젝트)  ③ checkpoints(BC/RL)  ④ 외부 자산(object/scene/robot USD)  ⑤ SSH 설정·키
> (③④⑤는 git에 없거나 폴더 밖이라 따로 챙겨야 함)

---

## 0. 한눈에 — 무엇이 어디에 있나 (기존 데스크탑 실측)

| # | 항목 | 경로 | 크기 | git? | 옮기는 법 |
|---|------|------|------|------|-----------|
| ① | **Isaac Sim conda env** | `~/miniconda3/envs/env_isaaclab` | 9.7G | — | `setup_env.sh` (mirror=rsync / install) |
| ② | IsaacLab 프레임워크 + 프로젝트 코드 | `~/IsaacLab` (+ `scripts/lekiwi_nav_env`) | — | ✅ | rsync (또는 git clone) |
| ②' | calibration (tucked/arm limits) | `…/lekiwi_nav_env/calibration` | 12K | ✅ tracked | 코드와 함께 옴 |
| ③ | **checkpoints (BC/RL)** | `…/lekiwi_nav_env/checkpoints` | **39G** | ❌ gitignore | rsync (또는 쓰는 것만) |
| ④a | object USD | `~/isaac-objects` | 3.8G | — | rsync |
| ④b | scene 자산 (ProcTHOR) | `~/molmospaces` | 24M | — | rsync |
| ④c | **로봇 USD** | `~/Downloads/lekiwi_robot.usd` | 7.3M | — | rsync (필수) |
| ⑤ | **SSH 설정 + 키** | `~/.ssh/config`, `~/.ssh/private.pem` | — | — | rsync (서버 접속용) |
| ✗ | ~~demos (텔레옵 데이터)~~ | `…/demos`, `demos_skill2/3` | **220G** | ❌ gitignore | **안 옮겨도 됨** (학습용, 학습은 서버) |

> **viva 모드만** 쓸 거면 ③ checkpoints 불필요 (파지/운반도 서버 VLA가 함).
> **rl_hybrid** 쓰면 ③ 중 4개만 필요: `dp_bc_nav_skill2_v4`, `resip_nav_tucked_v4`, `dp_bc_carry_v4`, `resip_carry_v6` (합 ~44MB).

---

## 1. 사전 준비 (새 데스크탑)

- **RTX GPU + NVIDIA 드라이버** — Isaac Sim 렌더링 필수 (3090/4090 등).
- **Miniconda** 설치 (`~/miniconda3`).
- **★ username을 `yubin11` + 동일 home 경로로 맞추면 제일 쉽다.**
  로봇 USD·isaaclab editable 설치 경로가 `/home/yubin11/...` 로 박혀 있어서, 같은 경로면 그대로 동작.
  다른 username이면 → §7 "다른 username" 참고 (mirror 불가, install 모드 + 경로 수정 필요).
- 두 데스크탑 사이 SSH 가능해야 함 (rsync pull). 안 되면 외장 디스크/중간 서버 경유.

---

## 2. 무엇을 보내나 (rsync) — 새 데스크탑에서 실행

```bash
OLD=yubin11@<기존데스크탑_IP>          # ← 기존 데스크탑 ssh 주소로 교체

# ② 코드 + ③ checkpoints + calibration  (demos 220G 는 제외!)
rsync -avzP --exclude='demos/' --exclude='demos_skill2/' --exclude='demos_skill3/' \
      "$OLD:/home/yubin11/IsaacLab/"  /home/yubin11/IsaacLab/

# ④a object USD (3.8G)
rsync -avzP "$OLD:/home/yubin11/isaac-objects/"  /home/yubin11/isaac-objects/

# ④b scene 자산 (24M)
rsync -avzP "$OLD:/home/yubin11/molmospaces/"  /home/yubin11/molmospaces/

# ④c 로봇 USD (7.3M)
rsync -avzP "$OLD:/home/yubin11/Downloads/lekiwi_robot.usd"  /home/yubin11/Downloads/

# ⑤ SSH 설정 + 키 (서버 접속)
mkdir -p ~/.ssh
rsync -avzP "$OLD:/home/yubin11/.ssh/config" "$OLD:/home/yubin11/.ssh/private.pem"  ~/.ssh/
chmod 700 ~/.ssh; chmod 600 ~/.ssh/config; chmod 600 ~/.ssh/private.pem
```

**checkpoints 39G가 부담이면** ② rsync 에 `--exclude='scripts/lekiwi_nav_env/checkpoints/'` 추가하고, 필요한 것만:

```bash
B=/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/checkpoints
for d in dp_bc_nav_skill2_v4 resip_nav_tucked_v4 dp_bc_carry_v4 resip_carry_v6; do
  rsync -avzP "$OLD:$B/$d/" "$B/$d/"
done
```

> 대안: ② 대신 `git clone https://github.com/yubinws/lekiwi-nav-env.git`(branch `stage1_server`) + IsaacLab 는 공식 `git clone`.
> 단 checkpoints(③)는 gitignore라 **반드시 위 rsync로 별도 전송**.

---

## 3. 환경 셋업 (conda env)

`~/IsaacLab` 가 들어온 뒤 실행:

```bash
# (권장) 기존 env를 그대로 복제 — 버전 100% 일치, 제일 안전
MODE=mirror OLD_HOST=yubin11@<기존데스크탑_IP> bash /home/yubin11/IsaacLab/scripts/lekiwi_nav_env/export/setup_env.sh

# (대안) 새로 설치 — 기존 접근 불가 시 (isaacsim 5.0.0 / isaaclab 0.44.9 에 맞춰짐)
MODE=install bash /home/yubin11/IsaacLab/scripts/lekiwi_nav_env/export/setup_env.sh
```

스크립트가 하는 일: conda env(`env_isaaclab`) 생성/복제 → Isaac Sim·Lab → `diffusers`/`requests` → import 검증.

---

## 4. 설정 (config)

```bash
# 로봇 USD — 같은 username/경로면 기본값(/home/yubin11/Downloads/lekiwi_robot.usd) 그대로 OK.
# 경로가 다르면 환경변수로 지정 (~/.bashrc 에 추가 권장):
export LEKIWI_USD_PATH=$HOME/Downloads/lekiwi_robot.usd

# SSH 포트 — 서버 pod 재시작마다 바뀐다. 안 붙으면 ~/.ssh/config 의 Host A100 Port 갱신.
#   Host A100  /  HostName 218.148.55.186  /  Port <현재포트>  /  User jovyan  /  IdentityFile ~/.ssh/private.pem

# 터널 (VLM/VLA 요청용) — 별도 터미널에서 살려둔다:
ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 A100
```

---

## 5. 검증 (순서대로)

```bash
# (a) 서버 SSH 되나
ssh A100 hostname

# (b) 서버에 모델 떠 있나 (안 떠 있으면 서버에서 launch — §아래)
ssh A100 "curl -s localhost:8000/v1/models | head -c 60; echo; curl -s localhost:8002/health"

# (c) 터널 통해 로컬에서 보이나  (터널 살린 뒤)
curl -s localhost:8000/v1/models | head -c 60; echo
curl -s localhost:8002/health

# (d) 데스크탑 env 정상 (★ 반드시 conda activate 후 — bin/python 직접호출은 실패)
conda activate env_isaaclab
python /home/yubin11/IsaacLab/scripts/lekiwi_nav_env/vllm/run_full_task.py --help | head -5
```

서버에 모델이 안 떠 있으면 (서버에서):
```bash
ssh A100
cd /home/jovyan/IsaacLab/scripts/lekiwi_nav_env
bash launch_servers.sh all --checkpoint vllm/outputs/train/pi05_viva_v5_20260421_071948/checkpoints/250000/pretrained_model
```

---

## 6. 실행

```bash
conda activate env_isaaclab
cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env

# viva (전체 VLM+VLA)
python vllm/run_full_task.py --mode viva --difficulty easy \
  --user_command "find the blue medicine bottle and place it next to the red cup" \
  --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --scene_idx 1302 --scene_scale 0.6 --num_trials 10 --vlm_interval 100

# rl_hybrid (S1/S3=로컬 RL, S2/S4=VLA) — checkpoints(③) 필요
python vllm/run_full_task.py --mode rl_hybrid \
  --nav_dp_checkpoint checkpoints/dp_bc_nav_skill2_v4/dp_bc_epoch300.pt \
  --nav_resip_checkpoint checkpoints/resip_nav_tucked_v4/resip_nav_best.pt \
  --carry_dp_checkpoint checkpoints/dp_bc_carry_v4/dp_bc_epoch300.pt \
  --carry_resip_checkpoint checkpoints/resip_carry_v6/resip_carry_iter240.pt \
  --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --scene_idx 1302 --scene_scale 0.6 --difficulty easy \
  --user_command "find the blue medicine bottle and place it next to the red cup"
```

> `~/isaac-objects/...` 는 `~` 라서 username 달라도 동작(자산만 ~/isaac-objects 에 있으면 됨).

---

## 7. 함정 / 트러블슈팅

- **`conda activate` 필수**: `…/env_isaaclab/bin/python` 직접 호출하면 Isaac Sim 경로 훅(activate.d)이 안 돌아서 `ModuleNotFoundError: isaacsim.simulation_app` / `IndexError` 난다. **항상 `conda activate env_isaaclab` 후 실행.**
- **다른 username** (예: home이 `/home/foo`):
  - mirror 모드 ✗ (env에 `/home/yubin11` 절대경로가 박힘) → **install 모드** 사용.
  - 로봇 USD: `export LEKIWI_USD_PATH=$HOME/Downloads/lekiwi_robot.usd`.
  - object USD: 명령의 `~/isaac-objects/...` 는 OK (~ 확장).
  - isaaclab editable: install 모드의 `./isaaclab.sh -i` 가 새 경로로 재바인딩.
- **SSH 포트 가변**: 서버 pod 재시작 시 포트 바뀜 → `~/.ssh/config` 의 `Host A100` Port 갱신. 코드/명령에 포트 하드코딩 금지(`ssh A100` 별칭 사용).
- **터널 죽음**: `pgrep -af "ssh -f -N -L 8000"` 로 확인, 없으면 §4 명령 다시.
- **checkpoints 없음** → rl_hybrid 가 "checkpoint not found"/phantom: ③ rsync 했는지 확인 (gitignore라 clone으로 안 옴).
- **GPU/드라이버**: Isaac Sim이 안 켜지면 드라이버 버전 확인 (`nvidia-smi`), RTX 필요.
- mirror env가 import 실패하면 → `MODE=install` 로 새로 설치.

---

## 부록 — 기존 데스크탑 실측 스펙 (2026‑06‑11)

- **conda env**: `env_isaaclab` (9.7G), Python **3.11**
  - isaacsim **5.0.0.0** · isaacsim‑kernel 5.0.0.0 · omniverse‑kit **107.3.1.206797**
  - isaaclab **0.44.9** (editable @ `~/IsaacLab/source/isaaclab`) · isaaclab_assets 0.2.2 · isaaclab_mimic 1.0.12 · isaaclab_rl 0.2.3 · isaaclab_tasks 0.10.45
  - diffusers **0.36.0** (diffusion_policy 의존) · requests (VLA client)
- **repo**: 프레임워크 `isaac-sim/IsaacLab`, 프로젝트 `github.com/yubinws/lekiwi-nav-env`(구 fsdsa, 이전됨), branch **stage1_server**
- **서버**: `ssh A100` (포트는 `~/.ssh/config`, 가변) / VLM Qwen3‑VL‑8B `:8000` / VLA Pi0.5 `pi05_viva_v5/250000` `:8002` / launch=`bash launch_servers.sh all --checkpoint <pi05>`
- **크기**: checkpoints 39G(gitignore) · demos 220G(gitignore·생략) · isaac‑objects 3.8G · molmospaces 24M · robot USD 7.3M · env 9.7G
