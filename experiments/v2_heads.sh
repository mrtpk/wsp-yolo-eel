#!/usr/bin/env bash
# v2 head experiments; run in tmux after the FP census. Logs: results/v2/<name>.log ; marker results/v2/heads.done
set -u
cd "$(dirname "$0")/.."
S=work
Y=$S/eel1.yaml
rm -f results/v2/heads.done
until [ -f results/v2/fp.done ]; do sleep 5; done
ev() { local name=$1; shift; uv run python experiments/evaluate.py "$@" > results/v2/$name.log 2>&1; echo "$name $(grep -A4 '"box"' results/v2/$name.log | tr -d ' \n')" >> results/v2/heads_summary.txt; }
: > results/v2/heads_summary.txt
ev A_o2m_nms weights/eel_1cls.pt $Y end2end=False
uv run python experiments/blend_heads.py weights/eel_1cls.pt weights/v2_blend_t50.pt 0.5 > /dev/null 2>&1
ev B_blend_t50 weights/v2_blend_t50.pt $Y
uv run python experiments/blend_heads.py weights/eel_1cls.pt weights/v2_blend_t25.pt 0.25 > /dev/null 2>&1
ev C_blend_t25 weights/v2_blend_t25.pt $Y
uv run python experiments/blend_heads.py weights/eel_1cls.pt weights/v2_blend_cls_t50.pt 0.5 cv3 > /dev/null 2>&1
ev D_blend_cls_t50 weights/v2_blend_cls_t50.pt $Y
touch results/v2/heads.done
