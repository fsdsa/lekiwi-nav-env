#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/stage0_v9g_monitor.log"
LOCAL_RUNNER="/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/run_stage0_v9g_detached.sh"
SERVER_RUNNER="/home/jovyan/IsaacLab/scripts/lekiwi_nav_env/run_stage0_server_v9g.sh"
LOCAL_LOG="/home/yubin11/resip_s3_36d_v15_stage0_v9g.log"
SERVER_LOG="/home/jovyan/resip_s3_36d_v15_stage0_v9g.log"
LOCAL_PATTERN='python train_resip.py.*stage0_v9g'
SERVER_PATTERN='python train_resip.py.*stage0_v9g'
STALE_SECS=1800
POLL_SECS=300
MONITOR_LOCAL=0
MONITOR_SERVER=1

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG_FILE"
}

local_pid() {
  pgrep -af "$LOCAL_PATTERN" | awk 'NR==1{print $1}'
}

server_pid() {
  ssh A100 "pgrep -af \"$SERVER_PATTERN\" | awk 'NR==1{print \$1}'" 2>/dev/null || true
}

local_log_age() {
  if [[ -f "$LOCAL_LOG" ]]; then
    echo $(( $(date +%s) - $(stat -c %Y "$LOCAL_LOG") ))
  else
    echo 999999
  fi
}

server_log_age() {
  ssh A100 "if [[ -f '$SERVER_LOG' ]]; then echo \$(( \$(date +%s) - \$(stat -c %Y '$SERVER_LOG') )); else echo 999999; fi" 2>/dev/null || echo 999999
}

start_local() {
  nohup /bin/bash "$LOCAL_RUNNER" >/dev/null 2>&1 < /dev/null &
  sleep 2
  log "started local v9g"
}

start_server() {
  ssh A100 "nohup /bin/bash '$SERVER_RUNNER' >/dev/null 2>&1 < /dev/null &" >/dev/null 2>&1 || true
  sleep 2
  log "started server v9g"
}

restart_local() {
  pkill -f "$LOCAL_PATTERN" >/dev/null 2>&1 || true
  pkill -f "run_stage0_v9g_detached.sh" >/dev/null 2>&1 || true
  start_local
}

restart_server() {
  ssh A100 "pkill -f \"$SERVER_PATTERN\" >/dev/null 2>&1 || true; pkill -f \"run_stage0_server_v9g.sh\" >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
  start_server
}

while true; do
  if (( MONITOR_LOCAL )); then
    lpid="$(local_pid || true)"
    lage="$(local_log_age)"
    if [[ -z "${lpid:-}" ]]; then
      log "local missing"
      restart_local
    elif (( lage > STALE_SECS )); then
      log "local stale age=${lage}s pid=$lpid"
      restart_local
    else
      log "local ok pid=$lpid age=${lage}s"
    fi
  else
    log "local monitor disabled"
  fi

  if (( MONITOR_SERVER )); then
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
  else
    log "server monitor disabled"
  fi

  sleep "$POLL_SECS"
done
