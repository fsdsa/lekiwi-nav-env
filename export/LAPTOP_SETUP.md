# 노트북(kswltd-Predator) 셋업 Runbook — LeKiwi VLM+VLA+RL 파이프라인

> **이 문서는 이 노트북의 Claude Code가 그대로 실행하도록 작성됨.**
> 다른 데스크탑(user=`yubin11`)에서 이 노트북(user=`kswltd`)으로 파일을 rsync로 보낸 뒤,
> **sim/eval + 로컬 BC·RL 추론 + 서버(A100)의 VLM/VLA 요청**을 돌릴 수 있게 셋업한다.
> 서버(A100)는 그대로 두고, 이 노트북은 SSH 터널로 추론 요청만 한다.

## 이 노트북 확정 사실 (실측 검증됨)
- user=`kswltd`, HOME=`/home/kswltd`
- conda = **anaconda3** (`/home/kswltd/anaconda3`) — ★ miniconda 아님
- GPU = **RTX 5070 Ti Laptop 12GB** (Isaac Sim 충분)
- rsync 설치됨, 여유 ~691G, `~/Downloads` 존재
- ★ 원본 데스크탑 user=`yubin11` → 코드 곳곳 `/home/yubin11` 하드코딩 → §3에서 처리

---

## 0. 이미 전송된 것 (rsync over Tailscale)
`/home/kswltd/` 아래로 도착:
- `~/IsaacLab/` — IsaacLab 프레임워크 + 프로젝트(`scripts/lekiwi_nav_env`) + **checkpoints 39G** (demos 제외)
- `~/isaac-objects/` (3.8G) · `~/molmospaces/` · `~/Downloads/lekiwi_robot.usd` · `~/.ssh/private.pem`
- **conda env는 전송 안 됨** (경로가 달라 복제 불가) → §2에서 새로 설치

---

## 1. 전송 검증 (먼저)
```bash
du -sh ~/IsaacLab ~/isaac-objects ~/molmospaces 2>/dev/null
ls -la ~/Downloads/lekiwi_robot.usd ~/.ssh/private.pem
ls ~/IsaacLab/isaaclab.sh \
   ~/IsaacLab/scripts/lekiwi_nav_env/vllm/run_full_task.py \
   ~/IsaacLab/scripts/lekiwi_nav_env/export/setup_env.sh
# rl_hybrid용 checkpoint 4개
ls -d ~/IsaacLab/scripts/lekiwi_nav_env/checkpoints/{dp_bc_nav_skill2_v4,resip_nav_tucked_v4,dp_bc_carry_v4,resip_carry_v6}
```
기대: IsaacLab 수십 GB, isaac-objects ~3.8G, 위 파일 전부 존재. **하나라도 없으면 전송 미완** → 데스크탑(yubin11)에서 재전송 요청.

---

## 2. conda env 설치 (★ install 모드 + anaconda3)

> ⏱ **전송 중에 미리 시작 가능 (시간 단축).** env 설치는 두 부분 — pip 다운로드(인터넷)는 전송 완료 전에 돌려도 되고, `./isaaclab.sh -i`(editable isaaclab)만 `~/IsaacLab/source/` 전송 완료 후 가능.
>
> **방법 B (병렬·빠름) — 전송 도는 동안 지금 실행:**
> ```bash
> source $HOME/anaconda3/etc/profile.d/conda.sh
> conda create -y -n env_isaaclab python=3.11 && conda activate env_isaaclab
> pip install --upgrade pip
> pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com   # 큰 다운로드(병렬 OK)
> pip install "diffusers>=0.36" requests numpy
> ```
> 전송 끝난 뒤 (`ls ~/IsaacLab/source/isaaclab/setup.py` 확인되면):
> ```bash
> cd ~/IsaacLab && ./isaaclab.sh -i
> ```
> ※ 방법 B를 썼으면 아래 `setup_env.sh`는 안 돌려도 됨 (conda create 충돌).

**방법 A (간단·권장) — 전송 100% 완료 후 한 번에:**
```bash
source $HOME/anaconda3/etc/profile.d/conda.sh        # conda 활성화 (anaconda)
cd ~/IsaacLab/scripts/lekiwi_nav_env
chmod +x export/setup_env.sh
MODE=install CONDA_BASE=$HOME/anaconda3 ENV_NAME=env_isaaclab ISAACLAB_DIR=$HOME/IsaacLab \
  bash export/setup_env.sh
```
하는 일: `env_isaaclab`(python 3.11) 생성 → Isaac Sim 5.0.0(pip) → `./isaaclab.sh -i`(editable isaaclab + RL libs) → `diffusers`/`requests` → import 검증.

검증:
```bash
conda activate env_isaaclab
python -c "import isaacsim, isaaclab, torch, diffusers, requests; print('env OK')"
```
- ⚠️ `isaacsim` import 실패 시 점검: (a) **반드시 `conda activate env_isaaclab` 후** 실행했는지(`bin/python` 직접 호출 ✗), (b) `pip show isaacsim` 버전이 5.0.0인지, (c) Isaac Lab 공식 설치문서 대조.
- 원본 데스크탑 정확한 버전: **isaacsim 5.0.0.0 / omniverse-kit 107.3.1 / isaaclab 0.44.9 / python 3.11 / diffusers 0.36.0**.

---

## 3. ★ 경로 수정 (/home/yubin11 → /home/kswltd) — 가장 중요

**메인 파이프라인(`run_full_task.py`)에 필요한 수정은 딱 2개:**

### 3-1. 로봇 USD (필수)
`lekiwi_robot_cfg.py`는 `LEKIWI_USD_PATH` env가 없으면 `/home/yubin11/Downloads/lekiwi_robot.usd`를 찾는다.
이 파일은 **수정 금지** → 환경변수로 해결:
```bash
echo 'export LEKIWI_USD_PATH=$HOME/Downloads/lekiwi_robot.usd' >> ~/.bashrc
export LEKIWI_USD_PATH=$HOME/Downloads/lekiwi_robot.usd
ls "$LEKIWI_USD_PATH"      # 존재 확인 (필수)
```

### 3-2. object/dest USD (CLI로 해결, 코드수정 불필요)
`lekiwi_skill3_env.py:54`의 dest 기본값이 `/home/yubin11/...`지만, 실행 때 **항상 `--object_usd`/`--dest_object_usd`를 `~`로 넘기면** 무시된다. §6 명령처럼 `~/isaac-objects/...` 사용 (`~`=`/home/kswltd` 자동 확장).

### 3-3. (선택) 메인 파이프라인엔 불필요한 하드코딩
아래는 **run_full_task.py 에는 안 쓰임.** 해당 워크플로 쓸 때만 손대면 됨:
- `run_eval_scene.sh`, `run_train_nav.sh`, `run_stage0_*.sh`, `run_navigate_train.sh`, `monitor_*.sh`
  → `/home/yubin11/miniconda3`, `/home/yubin11/isaacsim`(★ **별도 standalone Isaac Sim**, 이번에 전송 안 됨) 박힘.
  이 스크립트들은 conda env가 아니라 standalone isaacsim 방식이라 그대로는 안 돌아감.
  텔레옵/학습이 필요하면 그냥 `conda activate env_isaaclab && python <스크립트>` 로 직접 실행 권장.
- `object_catalog_19.json` 등 — object 경로가 `/home/yubin11/...`. 멀티오브젝트 catalog 모드 쓸 때만:
  `sed -i 's#/home/yubin11#/home/kswltd#g' object_catalog_*.json`

---

## 3.5 ★ USD 외부 의존성 (이거 빠지면 로봇·물체 안 뜸 — 2026-06-13 추가)
원본 USD들이 외부 트리/절대경로를 참조함. **둘 다** 처리해야 sim이 돈다.

**(A) 로봇 메시 트리 `~/lekiwi/` (4.7G)** — `~/Downloads/lekiwi_robot.usd`는 래퍼라 `../lekiwi/urdf/lekiwi/lekiwi.usd`(상대참조)를 가리킴. 데스크탑에서 별도 rsync로 전송됨.
```bash
ls ~/lekiwi/urdf/lekiwi/lekiwi.usd && du -sh ~/lekiwi    # ~4.7G 나와야 함
```
없으면 데스크탑(yubin11)에서: `rsync -a ~/lekiwi/ kswltd@<tailscale>:~/lekiwi/`

**(B) `/home/yubin` 심볼릭링크 (sudo)** — object `model_clean.usd`가 collider·texture를 **절대경로 `/home/yubin/isaac-objects/...`**로 참조함(원본 환경 잔재). 원본엔 `/home/yubin/{isaac-objects,lekiwi}`→`/home/yubin11/...` 심볼릭링크가 있음. 노트북에도 동일하게 `/home/kswltd`로 만든다:
```bash
sudo mkdir -p /home/yubin
sudo ln -sfn /home/kswltd/isaac-objects /home/yubin/isaac-objects
sudo ln -sfn /home/kswltd/lekiwi        /home/yubin/lekiwi
# 확인 (collider 열려야 함):
ls /home/yubin/isaac-objects/mujoco_scanned_objects/models/5_HTP/colliders/collider_00.usd
```
이거 없으면 물체 collider/texture 미해석 → 파지·접촉 물리가 깨짐.

## 4. 서버(A100) 접속 설정
```bash
chmod 700 ~/.ssh; chmod 600 ~/.ssh/private.pem
# ~/.ssh/config 에 A100 블록 추가 (없을 때만)
grep -q "Host A100" ~/.ssh/config 2>/dev/null || cat >> ~/.ssh/config <<'EOF'

Host A100
    HostName 218.148.55.186
    User jovyan
    Port 30628
    IdentityFile ~/.ssh/private.pem
EOF
chmod 600 ~/.ssh/config
ssh A100 hostname        # 접속 확인
```
- ⚠️ **Port `30628`은 서버 pod 재시작마다 바뀐다.** `ssh A100`가 `Connection refused`면 데스크탑/서버 담당자에게 현재 포트 확인 후 `~/.ssh/config`의 Port만 갱신.

서버 모델 확인 + 터널:
```bash
ssh A100 "curl -s localhost:8000/v1/models | head -c 60; echo; curl -s localhost:8002/health"
# 모델이 안 떠 있으면 (서버에서):
#   cd ~/IsaacLab/scripts/lekiwi_nav_env && bash launch_servers.sh all \
#     --checkpoint vllm/outputs/train/pi05_viva_v5_20260421_071948/checkpoints/250000/pretrained_model
# 터널 (이 노트북, 별도 터미널/백그라운드):
ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 A100
curl -s localhost:8000/v1/models | head -c 60; echo    # 로컬에서 보이면 OK
curl -s localhost:8002/health
```

---

## 5. 최종 검증
```bash
conda activate env_isaaclab
cd ~/IsaacLab/scripts/lekiwi_nav_env
python vllm/run_full_task.py --help | head -5     # argparse 뜨면 코드+env OK
```

---

## 6. 실행
```bash
conda activate env_isaaclab
cd ~/IsaacLab/scripts/lekiwi_nav_env

# viva (전체 VLM+VLA)
python vllm/run_full_task.py --mode viva --difficulty easy \
  --user_command "find the blue medicine bottle and place it next to the red cup" \
  --object_usd ~/isaac-objects/mujoco_scanned_objects/models/5_HTP/model_clean.usd \
  --dest_object_usd ~/isaac-objects/mujoco_scanned_objects/models/ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \
  --scene_idx 1302 --scene_scale 0.6 --num_trials 10 --vlm_interval 100

# rl_hybrid (S1/S3=로컬 RL, S2/S4=VLA — checkpoints 4개 필요)
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

---

## 7. 함정 / 트러블슈팅
- **`conda activate env_isaaclab` 필수** — `bin/python` 직접 호출 시 `ModuleNotFoundError: isaacsim.simulation_app` / `IndexError`. (activate 훅이 Isaac Sim 경로 세팅)
- **로봇이 안 뜸 / USD 못 찾음** → `LEKIWI_USD_PATH` 미설정 or robot USD 없음 (§3-1).
- **dest object 엉뚱/에러** → `--dest_object_usd` 안 넘겨서 `/home/yubin11` 기본값 탔음 (§3-2).
- **`ssh A100` 거부** → 서버 포트 바뀜 → `~/.ssh/config` Port 갱신 (§4).
- **VLA/VLM 응답 없음** → 서버 미기동 or 터널 안 뜸 (§4 재확인).
- **rl_hybrid "checkpoint not found"** → checkpoints 전송 확인 (§1).
- **Isaac Sim 렌더 에러** → `nvidia-smi`로 드라이버 확인 (RTX 5070 Ti면 충분).

---

## 부록 A. 발견된 `/home/yubin11` 하드코딩 전체 + 처리방침
| 파일 | 영향 | 처리 |
|---|---|---|
| `lekiwi_robot_cfg.py:74` (robot USD 기본값) | **메인 ★** | `LEKIWI_USD_PATH` env (파일 수정금지) |
| `lekiwi_skill3_env.py:54` (dest USD 기본값) | 메인 | `--dest_object_usd ~/...` 로 override |
| `run_eval_scene.sh` / `run_train_nav.sh` / `run_stage0_*.sh` / `run_navigate_train.sh` / `monitor_*.sh` | standalone Isaac Sim용(전송X) | 안 쓰면 무시. 쓰면 경로+`~/isaacsim` 별도 필요 |
| `object_catalog_*.json` | catalog 모드만 | 쓸 때 `sed -i 's#/home/yubin11#/home/kswltd#g'` |
| `make_skill3_release_tail_open.py` | demos 가공 | 추론엔 불필요 |

## 부록 B. 참고 스펙
- env: python 3.11 / isaacsim 5.0.0.0 / omniverse-kit 107.3.1 / isaaclab 0.44.9 / diffusers 0.36.0
- 서버: `ssh A100` (HostName 218.148.55.186, **Port 30628=가변**, User jovyan, key `~/.ssh/private.pem`)
  - VLM Qwen3-VL-8B `:8000` / VLA Pi0.5 `pi05_viva_v5/250000` `:8002`
  - 모델 기동: `bash launch_servers.sh all --checkpoint <pi05 path>`
- 프로젝트 repo: `github.com/yubinws/lekiwi-nav-env` (branch `stage1_server`)
- 같은 디렉토리의 `setup_env.sh`(환경설치), `MIGRATION.md`(전송 일반론)도 참고.
