#!/usr/bin/env bash
set -euo pipefail
LOG_LOCAL=/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/resip_s3_36d_v15_stage1g_monitor_$(date +%Y%m%d_%H%M%S).log
REMOTE=jovyan@218.148.55.186
SSH_OPTS='-i /home/yubin11/.ssh/private.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 30179'
LOCAL_COPY=/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/resip_s3_36d_v15_stage1g_server.log
for i in $(seq 1 7); do
  {
    echo "===== $(date '+%F %T %Z') ====="
    ssh $SSH_OPTS $REMOTE "pgrep -af 'python train_resip.py.*resip_s3_36d_v15_stage1g' || true"
    scp $SSH_OPTS $REMOTE:/home/jovyan/resip_s3_36d_v15_stage1g.log $LOCAL_COPY >/dev/null 2>&1 || true
    python - <<'PY'
from pathlib import Path
import re
p = Path('/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/resip_s3_36d_v15_stage1g_server.log')
if not p.exists():
    print('local_copy_missing')
    raise SystemExit
text = p.read_text(errors='ignore').splitlines()
iter_line = next((line for line in reversed(text) if 'Iter ' in line and ('| EVAL |' in line or '| TRAIN |' in line)), None)
print(iter_line or 'iter_line_missing')
summary = next((line for line in reversed(text) if 'S2→S3:' in line and 'place_hold_total=' in line), None)
print(summary or 'summary_missing')
src = []
for line in text[-5000:]:
    m = re.search(r'base→dst.*?src→dst mean=([0-9]+\.[0-9]+)', line)
    if m:
        src.append(float(m.group(1)))
print('recent_src_dst_means=', src[-5:] if src else [])
PY
    echo
  } >> "$LOG_LOCAL" 2>&1
  sleep 600
 done
