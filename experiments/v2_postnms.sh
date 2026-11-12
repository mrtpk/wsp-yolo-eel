#!/usr/bin/env bash
# NMS on top of the end-to-end head at several IoU thresholds. Runs after v2_heads.sh.
set -u
cd "$(dirname "$0")/.."
S=work
rm -f results/v2/postnms.done
until [ -f results/v2/heads.done ]; do sleep 5; done
for t in 0.5 0.6 0.7; do
  n=E_postnms_$t
  uv run python experiments/evaluate.py weights/eel_1cls.pt $S/eel1.yaml postnms=$t > results/v2/$n.log 2>&1
  echo "$n $(grep -A4 '"box"' results/v2/$n.log | tr -d ' \n')" >> results/v2/heads_summary.txt
done
touch results/v2/postnms.done
