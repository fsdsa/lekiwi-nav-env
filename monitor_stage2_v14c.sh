#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/stage2_v14c_monitor.log"
SERVER_RUNNER="/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/run_stage2_server_v14c.sh"
SERVER_LOG="/home/jovyan/resip_s3_36d_v15_stage2_v14c.log"
STALE_SECS=1800
POLL_SECS=300

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG_FILE"
}

server_pid() {
  ssh -o BatchMode=yes A100 "ps -eo pid,cmd | grep 'python train_resip.py' | grep 'stage2_v14c' | grep -v grep | awk 'NR==1{print \$1}'" 2>/dev/null || true
}

server_log_age() {
  ssh -o BatchMode=yes A100 "if [[ -f '$SERVER_LOG' ]]; then echo \$(( \$(date +%s) - \$(stat -c %Y '$SERVER_LOG') )); else echo 999999; fi" 2>/dev/null || echo 999999
}

start_server() {
  ssh -o BatchMode=yes A100 "nohup /bin/bash '$SERVER_RUNNER' >/dev/null 2>&1 < /dev/null &" >/dev/null 2>&1 || true
  sleep 2
  log "started server v14c"
}

restart_server() {
  ssh -o BatchMode=yes A100 "pkill -f 'python train_resip.py.*stage2_v14c' >/dev/null 2>&1 || true; pkill -f 'run_stage2_server_v14c.sh' >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
  start_server
}

while true; do
  spid="$(server_pid || true)"
  sage="$(server_log_age)"
  if [[ -z "${spid:-}" ]]; then
    log "server missing"
    restart_server
  elif (( sage > STALE_SECS )); then
    log "server stale age=${sage}s pid=$spid"
    restart_server
  else
    log "server ok pid=$spid age=${sage}s"
  fi
  sleep "$POLL_SECS"
done
