#!/usr/bin/env bash
# File-based job queue: runs results/v2/queue.txt line by line (append lines any time).
# Each line: a shell command. Progress in results/v2/queue.pos; logs in results/v2/queue_logs/<n>.log.
# A line 'STOP' ends the runner. Waits for an optional marker file before starting.
set -u
cd "$(dirname "$0")/.."
Q=results/v2/queue.txt; POS=results/v2/queue.pos; L=results/v2/queue_logs
mkdir -p $L; touch $Q; [ -f $POS ] || echo 0 > $POS
[ -n "${WAIT_FOR:-}" ] && until [ -f "$WAIT_FOR" ]; do sleep 10; done
while true; do
  n=$(cat $POS); total=$(wc -l < $Q)
  if [ "$n" -ge "$total" ]; then sleep 15; continue; fi
  line=$(sed -n "$((n+1))p" $Q)
  [ "$line" = "STOP" ] && { echo "$((n+1))" > $POS; echo "runner stopped $(date +%T)" >> results/v2/queue_status.txt; break; }
  echo "[$(date +%T)] start #$((n+1)): $line" >> results/v2/queue_status.txt
  bash -c "$line" > $L/$((n+1)).log 2>&1; rc=$?
  echo "[$(date +%T)] done  #$((n+1)) rc=$rc: $(tail -1 $L/$((n+1)).log | cut -c1-200)" >> results/v2/queue_status.txt
  echo $((n+1)) > $POS
done
