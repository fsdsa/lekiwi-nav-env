#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="A100"
REMOTE_DIR="/home/jovyan/IsaacLab/scripts/lekiwi_nav_env"
REMOTE_LOG="/home/jovyan/resip_s3_36d_v15_stage2_v9a.log"
LOCAL_DIR="/home/yubin11/IsaacLab/scripts/lekiwi_nav_env"
LOCAL_LOG_DIR="${LOCAL_DIR}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MON_LOG="${LOCAL_LOG_DIR}/stage2_v9a_autofix_${STAMP}.log"

mkdir -p "${LOCAL_LOG_DIR}"
exec > >(tee -a "${MON_LOG}") 2>&1

echo "[MON] start $(date '+%F %T %Z')"

read -r START_MTIME START_SIZE <<<"$(ssh "${REMOTE_HOST}" "stat -c '%Y %s' ${REMOTE_LOG} 2>/dev/null || echo '0 0'")"
echo "[MON] initial mtime=${START_MTIME} size=${START_SIZE}"

sleep 1800

echo "[MON] check $(date '+%F %T %Z')"
ITER2_COUNT="$(ssh "${REMOTE_HOST}" "grep -c 'Iter 2/195' ${REMOTE_LOG} 2>/dev/null || true")"
read -r CUR_MTIME CUR_SIZE <<<"$(ssh "${REMOTE_HOST}" "stat -c '%Y %s' ${REMOTE_LOG} 2>/dev/null || echo '0 0'")"
PID="$(ssh "${REMOTE_HOST}" "cat /home/jovyan/.stage2_v9a.pid 2>/dev/null || true")"
PROC_ALIVE=0
if [[ -n "${PID}" ]] && ssh "${REMOTE_HOST}" "ps -p ${PID} >/dev/null 2>&1"; then
  PROC_ALIVE=1
fi

echo "[MON] iter2=${ITER2_COUNT} pid=${PID:-none} alive=${PROC_ALIVE} cur_mtime=${CUR_MTIME} cur_size=${CUR_SIZE}"

if [[ "${ITER2_COUNT}" -ge 1 ]]; then
  echo "[MON] healthy: Iter 2 reached, leaving run untouched"
  exit 0
fi

echo "[MON] unhealthy: Iter 2 not reached within 30 minutes, applying log-throttle patch and restarting"

scp "${LOCAL_DIR}/train_resip.py" "${REMOTE_HOST}:${REMOTE_DIR}/train_resip.py"
scp "${LOCAL_DIR}/launch_stage2_v9a_remote.sh" "${REMOTE_HOST}:${REMOTE_DIR}/launch_stage2_v9a_remote.sh"

ssh "${REMOTE_HOST}" "chmod +x ${REMOTE_DIR}/launch_stage2_v9a_remote.sh && ${REMOTE_DIR}/launch_stage2_v9a_remote.sh"

sleep 15

NEW_PID="$(ssh "${REMOTE_HOST}" "cat /home/jovyan/.stage2_v9a.pid 2>/dev/null || true")"
echo "[MON] relaunched pid=${NEW_PID:-none}"
ssh "${REMOTE_HOST}" "ps -p ${NEW_PID} -o pid,stat,etime,cmd 2>/dev/null || true; ls -l ${REMOTE_LOG} 2>/dev/null || true; tail -n 30 ${REMOTE_LOG} 2>/dev/null || true"
