#!/usr/bin/env bash
# Sequential experiment queue. Run inside tmux:  tmux new-session -d -s eel 'bash experiments/run_queue.sh'
# Each step logs to results/<step>.log; a summary goes to results/queue_summary.txt; results/QUEUE_DONE marks the end.
set -u
cd "$(dirname "$0")/.."
S=work
SUM=results/queue_summary.txt
rm -f results/QUEUE_DONE
: > "$SUM"

step() {  # step <name> <command...>
  local name=$1; shift
  echo "=== $name  start $(date +%H:%M:%S)" | tee -a "$SUM"
  if "$@" > "results/$name.log" 2>&1; then status=ok; else status="FAILED($?)"; fi
  tr '\r' '\n' < "results/$name.log" | grep -vE '^[0-9]+/[0-9]+$' | tail -14 >> "$SUM"
  echo "=== $name  $status $(date +%H:%M:%S)" | tee -a "$SUM"
}

step diag_imgsz640  uv run python experiments/diagnose.py weights/eel_1cls.pt "$S/eel1" results/diag_imgsz640.csv 5 640
step diag_imgsz768  uv run python experiments/diagnose.py weights/eel_1cls.pt "$S/eel1" results/diag_imgsz768.csv 5 768
step ablate_s32     bash -c "uv run python experiments/ablate_stride32.py weights/eel_1cls.pt weights/eel_1cls_no_s32.pt && uv run python experiments/compare_preds.py weights/eel_1cls.pt weights/eel_1cls_no_s32.pt $S/eel1/images/train 200"
step mask_ceiling   uv run python experiments/mask_res_ceiling.py data/labels/train 5

date > results/QUEUE_DONE
