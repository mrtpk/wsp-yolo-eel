# wsp-yolo-eel: weight-space programming on a YOLO26n eel detector

This repository holds the code, the logs and the report of a weight-space programming study on a small segmentation model: a YOLO26n-seg detector (3.06M parameters) that finds eels in infrared video. Weight-space programming means that you change a trained network by reading and writing its weights directly, based on what each part computes.

Write-up: [Weight-Space Programming - Editing an eel detector without new data](https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_eel_weight_space) (kaizen blog). Report: [`docs/REPORT.md`](docs/REPORT.md).

## What you can and cannot do with this repository

- **Not included:** the base checkpoint, the frames and the labels. They are not public. Every checkpoint that the scripts make is derived from the base checkpoint, so none of them is included either. You cannot rerun the study as it is.
- **Included:** every experiment script, the evaluation log (`results/v2/runs.jsonl`, one JSON line per evaluation), the summary logs, the report, and six figures (four of them show frame crops).
- **Withheld:** per-frame files that hold label geometry (`results/diag_*.csv`, `results/v2/fp_all_*.csv`, `results/v2/dets_peak2.jsonl`: ground-truth boxes and lengths per frame, per-frame detections). The report gives their aggregates. `experiments/export_blog_numbers.py` needs them, so it does not run from this repository.
- You can run the scripts on your own YOLO26-seg checkpoint and data. They are not bit-exact, and paths in the logs are relative to the repository root (`work/` was a scratch folder for data copies and splits).

## Results (second half of the video, held out from every fit in this study, not from the original training)

| Setup | Box recall | Box mAP50-95 | Mask mAP50-95 | Fitness |
| --- | --- | --- | --- | --- |
| Base model | 0.894 | 0.744 | 0.528 | 1.3346 |
| One-class head + peak suppression (no weight change) | 0.936 | 0.772 | 0.556 | 1.3881 |
| + exact widened merge of the two heads + NMS 0.5 | 0.937 | 0.777 | 0.559 | 1.3960 |
| + WiSE-FT interpolation, alpha 0.25 (`eel_v2.pt`) | 0.944 | 0.779 | 0.561 | 1.4004 |

Fitness = sum over box and mask of (0.1 x mAP50 + 0.9 x mAP50-95). About 81 % of the fitness gain comes from peak suppression, which changes no weights. The settings were chosen on the scored half, so these numbers are optimistic.

## Layout

| Path | Content |
| --- | --- |
| `experiments/` | All scripts. `docs/REPORT.md`, section "Reproducibility", maps each script to its experiment and output. |
| `results/` | v1 logs and summaries |
| `results/v2/` | v2 logs, `runs.jsonl` (all evaluations), job queue |
| `docs/REPORT.md` | The full report, v1 and v2 |
| `docs/figures/` | Recall by length, robustness, and four frame-crop sheets |
| `data/`, `weights/` | Empty. See their README files. |

## Setup

uv, Python 3.12, CPU PyTorch:

```
uv sync
uv run python experiments/runs_table.py        # print every logged evaluation with its fitness
```

Pinned versions: `ultralytics==8.4.22`, `torch 2.15.0`, `torchvision 0.29.2` (`uv.lock`). `experiments/eel_v2_infer.py` patches an internal Ultralytics method; another Ultralytics version can break it without an error.

The study ran with Claude Code as the agent that wrote and ran the experiments, on a CPU only machine (14 cores).

## Reference

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming), Takehi, Teoh, Zhang and Khattab, MIT CSAIL.
