"""Per-object diagnostic: which eels are found, which are missed, and why.

For every ground-truth eel we record its size, the best IoU any anchor box reaches on it,
the score of that anchor, and the stride (8/16/32) of the anchor that detects it.

Usage: uv run python experiments/diagnose.py <weights.pt> <dataset_root> <out.csv> [every_nth] [imgsz]
       (dataset_root contains images/train and labels/train with class-0 or class-21 labels)
"""
import csv
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

EEL = None  # resolved from model
CONF = 0.25
IOU = 0.5

weights, root, out_csv = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
imgsz = int(sys.argv[5]) if len(sys.argv) > 5 else 512
sys.path.insert(0, str(Path(__file__).parent))
import diagnose_utils as du  # noqa: E402
from diagnose_utils import load, gt_boxes  # noqa: E402
du.configure(imgsz)
torch.set_num_threads(14)

model = YOLO(weights).model.float().eval()
head = model.model[-1]
EEL = 0 if head.nc == 1 else 21


imgs = sorted((root / "images/train").glob("*.png"))[::every]
rows, fps, n_det = [], 0, 0
B = 32
with torch.inference_mode():
    for bi in range(0, len(imgs), B):
        batch = imgs[bi : bi + B]
        x = torch.stack([load(p) for p in batch])
        (_, _), preds = model(x)
        o2o = preds["one2one"]
        dec = head._inference(o2o)  # (b, 4+nc+nm, A): xyxy boxes, sigmoid scores, mask coefs
        logits = o2o["scores"][:, EEL]  # (b, A)
        strides = head.strides.view(-1)  # (A,)
        for j, p in enumerate(batch):
            boxes = dec[j, :4].T  # (A,4)
            score = logits[j].sigmoid()
            gt, ext = gt_boxes(root / "labels/train" / (p.stem + ".txt"))
            iou = box_iou(gt, boxes) if len(gt) else torch.zeros(0, len(boxes))  # (G,A)
            det = (score >= CONF).nonzero().flatten()
            det = det[score[det].argsort(descending=True)][:300]
            n_det += len(det)
            # greedy match detections to GT
            taken = set()
            match = {}
            for a in det.tolist():
                if len(gt) == 0:
                    fps += 1
                    continue
                g = int(iou[:, a].argmax())
                if iou[g, a] >= IOU and g not in taken:
                    taken.add(g)
                    match[g] = a
                else:
                    fps += 1
            for g in range(len(gt)):
                best_a = int(iou[g].argmax())
                ok = iou[g] >= IOU
                best_ok_score = float(score[ok].max()) if ok.any() else 0.0
                best_ok_a = int(torch.where(ok, score, torch.tensor(-1.0)).argmax()) if ok.any() else -1
                per_level = {}
                for s in (8, 16, 32):
                    m = ok & (strides == s)
                    per_level[s] = float(score[m].max()) if m.any() else 0.0
                rows.append(
                    dict(
                        image=p.stem,
                        bbox_max_side_px=ext[g][0] * 640,
                        length_px=ext[g][1],
                        area_px=ext[g][2],
                        found=int(g in match),
                        det_stride=int(strides[match[g]]) if g in match else 0,
                        det_score=float(score[match[g]]) if g in match else 0.0,
                        best_iou=float(iou[g, best_a]),
                        best_iou_stride=int(strides[best_a]),
                        best_ok_score=best_ok_score,
                        best_ok_stride=int(strides[best_ok_a]) if best_ok_a >= 0 else 0,
                        s8=per_level[8],
                        s16=per_level[16],
                        s32=per_level[32],
                    )
                )
        print(f"{bi + len(batch)}/{len(imgs)}", end="\r", flush=True)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# ---- summary ----
import statistics as st

N = len(rows)
found = [r for r in rows if r["found"]]
miss = [r for r in rows if not r["found"]]
print(f"\nimgsz {imgsz}  GT eels: {N}  found@conf{CONF}: {len(found)} ({len(found)/N:.1%})  detections: {n_det}  false positives: {fps}")
loc = [r for r in miss if r["best_iou"] < IOU]
cls = [r for r in miss if r["best_iou"] >= IOU]
print(f"misses: {len(miss)}  -> localization (no anchor box IoU>={IOU}): {len(loc)}   score (box ok, score<{CONF}): {len(cls)}")
if cls:
    print(f"  score-misses: median best score {st.median(r['best_ok_score'] for r in cls):.3f}")
print("detections by stride:", {s: sum(r["det_stride"] == s for r in found) for s in (8, 16, 32)})
print("\nrecall by eel length (polygon half-perimeter, px in 640x360 frame):")
edges = [0, 60, 100, 150, 200, 300, 2000]
for lo, hi in zip(edges, edges[1:]):
    b = [r for r in rows if lo <= r["length_px"] < hi]
    if b:
        f_ = sum(r["found"] for r in b)
        strides = {s: sum(r["det_stride"] == s for r in b) for s in (8, 16, 32)}
        mean = {s: st.mean(r[f"s{s}"] for r in b) for s in (8, 16, 32)}
        print(f"  [{lo:4},{hi:4}) n={len(b):5}  recall={f_/len(b):.3f}  det by stride {strides}  "
              f"mean best score s8={mean[8]:.2f} s16={mean[16]:.2f} s32={mean[32]:.2f}")
