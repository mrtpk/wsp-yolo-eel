"""v2: false-positive census. Every detection >= CONF on every frame is classified:

  tp         IoU >= 0.5 with an eel not yet claimed by a higher-scoring detection
  duplicate  IoU >= 0.5 with an eel already claimed (one-to-one head failed to suppress it)
  mislocal   0.1 <= best IoU < 0.5 (overlaps a real eel, box too poor)
  background best IoU < 0.1 (nothing labelled there)

Writes a CSV of all detections (+ per-detection stride, score, IoU, box) and the per-GT match table.
Usage: uv run python experiments/fp_analysis.py <weights.pt> <dataset_root> <out_prefix> [every] [imgsz]
"""
import csv
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

sys.path.insert(0, str(Path(__file__).parent))
import diagnose_utils as du  # noqa: E402

weights, root, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
imgsz = int(sys.argv[5]) if len(sys.argv) > 5 else 512
CONF = 0.25
du.configure(imgsz)
torch.set_num_threads(14)

model = YOLO(weights).model.float().eval()
head = model.model[-1]
EEL = 0 if head.nc == 1 else 21
imgs = sorted((root / "images/train").glob("*.png"))[::every]
dets, gts = [], []
with torch.inference_mode():
    for bi in range(0, len(imgs), 32):
        batch = imgs[bi : bi + 32]
        (_, _), preds = model(torch.stack([du.load(p) for p in batch]))
        o2o = preds["one2one"]
        dec = head._inference(o2o)
        score_all = o2o["scores"][:, EEL].sigmoid()
        strides = head.strides.view(-1)
        for j, p in enumerate(batch):
            boxes, score = dec[j, :4].T, score_all[j]
            gt, ext = du.gt_boxes(root / "labels/train" / (p.stem + ".txt"))
            keep = (score >= CONF).nonzero().flatten()
            keep = keep[score[keep].argsort(descending=True)]
            iou = box_iou(boxes[keep], gt) if len(gt) else torch.zeros(len(keep), 0)
            claimed = {}
            for k, a in enumerate(keep.tolist()):
                best = float(iou[k].max()) if len(gt) else 0.0
                g = int(iou[k].argmax()) if len(gt) else -1
                if best >= 0.5 and g not in claimed:
                    kind = "tp"
                    claimed[g] = a
                elif best >= 0.5:
                    kind = "duplicate"
                elif best >= 0.1:
                    kind = "mislocal"
                else:
                    kind = "background"
                x0, y0, x1, y1 = boxes[a].tolist()
                dets.append(dict(image=p.stem, frame_idx=bi + j, anchor=a, stride=int(strides[a]), score=float(score[a]),
                                 kind=kind, best_iou=best, gt=g,
                                 x0=x0, y0=y0, x1=x1, y1=y1,
                                 # back to 640x360 frame coords for cropping
                                 fx0=x0 / du.W * 640, fy0=(y0 - du.PAD_Y) / du.RH * 360,
                                 fx1=x1 / du.W * 640, fy1=(y1 - du.PAD_Y) / du.RH * 360))
            for g in range(len(gt)):
                gts.append(dict(image=p.stem, gt=g, found=int(g in claimed), length_px=ext[g][1]))
        print(f"{bi + len(batch)}/{len(imgs)}", flush=True)

for name, rows in (("dets", dets), ("gts", gts)):
    with open(f"{prefix}_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

from collections import Counter
import statistics as st

c = Counter(d["kind"] for d in dets)
print(f"\nframes {len(imgs)}  GT {len(gts)}  found {sum(g['found'] for g in gts)}  detections {len(dets)}  {dict(c)}")
for kind in ("tp", "duplicate", "mislocal", "background"):
    ds = [d for d in dets if d["kind"] == kind]
    if ds:
        print(f"  {kind:10} n={len(ds):5}  score median {st.median(d['score'] for d in ds):.2f}  "
              f"stride {dict(Counter(d['stride'] for d in ds))}  best IoU median {st.median(d['best_iou'] for d in ds):.2f}")
fr = Counter(d["image"] for d in dets if d["kind"] == "background")
print(f"  background FPs spread over {len(fr)} frames; worst frames {fr.most_common(5)}")
