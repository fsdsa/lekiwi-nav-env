#!/usr/bin/env bash
set -euo pipefail
SSH_OPTS='-i /home/yubin11/.ssh/private.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 30179'
REMOTE='jovyan@218.148.55.186'
OUT='/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/stage1g_iter1_watch.log'
for i in $(seq 1 40); do
  scp $SSH_OPTS $REMOTE:/home/jovyan/resip_s3_36d_v15_stage1g.log /home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/resip_s3_36d_v15_stage1g_server.log >/dev/null 2>&1 || true
  if rg -n "Iter 1/195 \| EVAL|Iter 2/195|upright=|Trend:|PhB quality:" /home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/resip_s3_36d_v15_stage1g_server.log > /tmp/stage1g_iter1_rg.txt 2>/dev/null; then
    {
      echo "===== $(date '+%F %T %Z') ====="
      cat /tmp/stage1g_iter1_rg.txt
      echo
    } >> "$OUT"
    if rg -q "Iter 2/195|PhB quality:|upright=" /home/yubin11/IsaacLab/scripts/lekiwi_nav_env/logs/resip_s3_36d_v15_stage1g_server.log; then
      exit 0
    fi
  fi
  sleep 60
done
