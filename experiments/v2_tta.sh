#!/usr/bin/env bash
# Test-time augmentation, alone and with NMS on top. Runs after v2_postnms.sh.
set -u
cd "$(dirname "$0")/.."
S=work
rm -f results/v2/tta.done
until [ -f results/v2/postnms.done ]; do sleep 5; done
for spec in "F_tta augment=True" "G_tta_postnms augment=True postnms=0.6"; do
  set -- $spec; n=$1; shift
  uv run python experiments/evaluate.py weights/eel_1cls.pt $S/eel1.yaml "$@" > results/v2/$n.log 2>&1
  echo "$n $(grep -A4 '"box"' results/v2/$n.log | tr -d ' \n') $(grep -m1 -iE 'error|exception' results/v2/$n.log)" >> results/v2/heads_summary.txt
done
touch results/v2/tta.done
