#!/usr/bin/env bash
# =============================================================================
#  LeKiwi Nav Env — 새 데스크탑 환경 셋업 (Isaac Sim + Isaac Lab + 의존성)
# -----------------------------------------------------------------------------
#  서버(A100, VLM:8000 / VLA:8002)는 그대로 두고, sim/teleop/eval + 로컬 BC/RL
#  추론을 돌릴 "데스크탑"을 복제할 때 쓰는 환경 셋업 스크립트.
#
#  ※ 이 스크립트는 "환경(conda env)"만 만든다. 코드/체크포인트/자산/SSH키 전송은
#    export/MIGRATION.md 의 §2 를 먼저 끝낸 뒤 실행할 것.
#
#  사용 순서:
#    1) (먼저) export/MIGRATION.md §2 대로 ~/IsaacLab 등 rsync 완료
#    2) 모드 선택해서 실행:
#         # (권장) 기존 데스크탑 env를 "그대로" 복제 — 버전 100% 일치
#         MODE=mirror OLD_HOST=yubin11@<기존데스크탑_IP> bash export/setup_env.sh
#
#         # (대안) Isaac Sim/Lab 새로 설치 — 기존 데스크탑 접근 불가 시
#         MODE=install bash export/setup_env.sh
#
#  현재(기존 데스크탑) 실측 버전 — 새로 설치 시 이 버전에 맞출 것:
#    python 3.11 · isaacsim 5.0.0.0 · omniverse-kit 107.3.1 · isaaclab 0.44.9 · diffusers 0.36.0
# =============================================================================
set -euo pipefail

# ── 설정 (필요 시 수정) ──────────────────────────────────────────────────────
MODE="${MODE:-mirror}"                              # mirror | install
OLD_HOST="${OLD_HOST:-}"                            # mirror: 기존 데스크탑 ssh 주소 (예: yubin11@192.168.0.10)
ENV_NAME="${ENV_NAME:-env_isaaclab}"
PY_VER="${PY_VER:-3.11}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ISAACSIM_SPEC="${ISAACSIM_SPEC:-isaacsim[all,extscache]==5.0.0}"  # install 모드용. Isaac Lab 문서로 최종 확인
# ────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " setup_env.sh  MODE=$MODE  ENV=$ENV_NAME  PY=$PY_VER"
echo " ISAACLAB_DIR=$ISAACLAB_DIR  CONDA_BASE=$CONDA_BASE"
echo "============================================================"

# ── 0) 사전 점검 ─────────────────────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || { echo "[ERR] conda 없음 → miniconda 먼저 설치"; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "[WARN] nvidia-smi 없음 — Isaac Sim 렌더링엔 RTX GPU + NVIDIA 드라이버 필수"
fi
[ -f "$ISAACLAB_DIR/isaaclab.sh" ] || {
  echo "[ERR] $ISAACLAB_DIR/isaaclab.sh 없음 → 먼저 ~/IsaacLab 를 rsync 하라 (MIGRATION.md §2)"; exit 1; }

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── mirror: 기존 데스크탑 conda env를 통째로 rsync (버전 그대로) ──────────────
mirror_env() {
  [ -n "$OLD_HOST" ] || { echo "[ERR] mirror 모드는 OLD_HOST 필요 (예: OLD_HOST=yubin11@IP)"; exit 1; }
  echo "==> [mirror] $OLD_HOST:$CONDA_BASE/envs/$ENV_NAME → 로컬로 복제"
  echo "    조건: (1) username/home 경로 동일  (2) OS·드라이버 호환  (3) ~/IsaacLab 동일 경로"
  mkdir -p "$CONDA_BASE/envs/$ENV_NAME"
  rsync -avzP "$OLD_HOST:$CONDA_BASE/envs/$ENV_NAME/" "$CONDA_BASE/envs/$ENV_NAME/"
  echo "==> [mirror] 완료 (editable isaaclab는 $ISAACLAB_DIR/source 를 가리킴 → 경로 같아야 함)"
}

# ── install: Isaac Sim/Lab 새로 설치 ────────────────────────────────────────
install_env() {
  echo "==> [install] conda env 생성 + Isaac Sim/Lab 설치"
  conda create -y -n "$ENV_NAME" python="$PY_VER"
  set +u; conda activate "$ENV_NAME"; set -u
  python -m pip install --upgrade pip
  echo "==> Isaac Sim 설치: $ISAACSIM_SPEC  (버전은 기존=5.0.0 과 동일하게)"
  pip install "$ISAACSIM_SPEC" --extra-index-url https://pypi.nvidia.com
  echo "==> Isaac Lab 확장(editable) + RL 라이브러리 설치"
  ( cd "$ISAACLAB_DIR" && ./isaaclab.sh -i )
}

case "$MODE" in
  mirror)  mirror_env  ;;
  install) install_env ;;
  *) echo "[ERR] MODE는 mirror | install 중 하나"; exit 1 ;;
esac

# ── 프로젝트 추가 의존성 ─────────────────────────────────────────────────────
set +u; conda activate "$ENV_NAME"; set -u
echo "==> 프로젝트 추가 의존성 (diffusion_policy=diffusers, VLA client=requests, 키보드=pynput)"
pip install "diffusers>=0.36" requests numpy pynput >/dev/null

# ── 검증 (반드시 conda activate 상태에서. bin/python 직접 호출은 Isaac Sim 경로 훅 누락으로 실패) ──
echo "==> 검증 (import 체크)"
python - <<'PY' || echo "[WARN] 일부 import 실패 — 위 로그 확인 (mirror면 경로/username, install이면 버전)"
import importlib
for m in ["torch","numpy","diffusers","requests","isaaclab","isaacsim"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:10s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:10s} {type(e).__name__}: {str(e)[:80]}")
PY

echo "============================================================"
echo " 환경 셋업 끝. 다음 단계 → export/MIGRATION.md §4(설정)·§5(검증)·§6(실행)"
echo "   - 로봇 USD 경로 다르면: export LEKIWI_USD_PATH=~/Downloads/lekiwi_robot.usd"
echo "   - 터널: ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 A100"
echo "   - 실행: conda activate $ENV_NAME && python vllm/run_full_task.py ..."
echo "============================================================"
