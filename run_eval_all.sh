#!/bin/bash
cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env
BACKUP_DIR=logs/ppo_navigate/ppo_navigate_scratch/backup
OUT=/home/yubin11/IsaacLab/scripts/lekiwi_nav_env/eval_backup_results.txt

echo "=== Backup Checkpoint Evaluation ===" > "$OUT"
echo "Date: $(date)" >> "$OUT"
echo "" >> "$OUT"

for ckpt in v6c v6e v6f v6g2 v6g3 v8; do
    PT="$BACKUP_DIR/best_agent_${ckpt}.pt"
    echo ">>> Running: $ckpt ..."
    echo "============================================" >> "$OUT"
    echo "  Checkpoint: best_agent_${ckpt}.pt" >> "$OUT"
    echo "============================================" >> "$OUT"

    EXTRA=""
    if [ "$ckpt" = "v8" ]; then
        EXTRA="--no_masking"
    fi

    timeout 180 python eval_navigate.py --checkpoint "$PT" --num_envs 1 --max_steps 3000 --headless $EXTRA 2>&1 >> "$OUT"

    # 시뮬레이터 강제 종료 후 다음 run
    pkill -9 -f "eval_navigate.py" 2>/dev/null
    sleep 3

    echo "" >> "$OUT"
    echo ">>> Done: $ckpt"
done

echo "=== All evaluations complete ===" >> "$OUT"
echo "All done!"
