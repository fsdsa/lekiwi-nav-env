#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/stage0_v9f_monitor.log"
LOCAL_PATTERN='train_resip.py.*stage0_v9f'
SERVER_PATTERN='train_resip.py.*stage0_v9f'
LOCAL_RUNNER="/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/run_stage0_v9f_detached.sh"
SERVER_RUNNER="/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/run_stage0_server_v9f.sh"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG_FILE"
}

start_local() {
  nohup /bin/bash "$LOCAL_RUNNER" >/dev/null 2>&1 < /dev/null &
  sleep 2
  log "restarted local stage0_v9f"
}

start_server() {
  ssh A100 "nohup /bin/bash '$SERVER_RUNNER' >/dev/null 2>&1 < /dev/null &" >/dev/null 2>&1 || true
  sleep 2
  log "restarted server stage0_v9f"
}

while true; do
  local_pid="$(pgrep -af "$LOCAL_PATTERN" | grep -v monitor_stage0_v9f | awk 'NR==1{print $1}')"
  if [[ -z "${local_pid:-}" ]]; then
    log "local stage0_v9f missing"
    start_local
  else
    log "local ok pid=$local_pid"
  fi

  server_pid="$(ssh A100 "pgrep -af \"$SERVER_PATTERN\" | awk 'NR==1{print \$1}'" 2>/dev/null || true)"
  if [[ -z "${server_pid:-}" ]]; then
    log "server stage0_v9f missing"
    start_server
  else
    log "server ok pid=$server_pid"
  fi

  sleep 600
done
