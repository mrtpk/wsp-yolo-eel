"""v2: can the (uncropped) predicted mask repair badly placed boxes?

Ultralytics crops each instance mask to its predicted box. The prototype masks themselves are not box-limited,
so for each detection we build the full mask  sigmoid(coef . protos) > 0.5,  keep the connected components that
touch the predicted box, and form a new box from them. Compare IoU with the labelled eel, old vs new.
Variants: 'mask' = box of touching components; 'union' = union of old box and mask box.

Usage: uv run python experiments/mask_refine_boxes.py <weights.pt> <dataset_root> <dets.csv> [n_tp=300]
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

sys.path.insert(0, str(Path(__file__).parent))
import diagnose_utils as du  # noqa: E402

weights, root, dets_csv = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
n_tp = int(sys.argv[4]) if len(sys.argv) > 4 else 300
du.configure(512)
torch.set_num_threads(14)
random.seed(0)

rows = list(csv.DictReader(open(dets_csv)))
sel = [r for r in rows if r["kind"] == "mislocal"] + random.sample([r for r in rows if r["kind"] == "tp"], n_tp)
by_img = defaultdict(list)
for r in sel:
    by_img[r["image"]].append(r)

model = YOLO(weights).model.float().eval()
head = model.model[-1]
results = defaultdict(list)  # kind -> list of (old_iou, mask_iou, union_iou)


def refine(mask, box):
    """mask: HxW bool at proto res; box: xyxy in proto px. Returns (mask box, union box) in proto px or None."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    inside = np.unique(lab[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)])
    inside = [i for i in inside if i != 0]
    if not inside:
        return None
    ys, xs = np.nonzero(np.isin(lab, inside))
    mb = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    ub = [min(mb[0], box[0]), min(mb[1], box[1]), max(mb[2], box[2]), max(mb[3], box[3])]
    return mb, ub


imgs = sorted(by_img)
with torch.inference_mode():
    for bi in range(0, len(imgs), 32):
        names = imgs[bi : bi + 32]
        x = torch.stack([du.load(root / "images/train" / f"{n}.png") for n in names])
        (y, proto), preds = model(x)
        dec = head._inference(preds["one2one"])  # (b, 4+1+32, A)
        for j, name in enumerate(names):
            gt, _ = du.gt_boxes(root / "labels/train" / f"{name}.txt")
            P = proto[j].reshape(proto.shape[1], -1)  # (32, h*w)
            ph, pw = proto.shape[2], proto.shape[3]
            for r in by_img[name]:
                a = int(r["anchor"])
                box = dec[j, :4, a]
                coef = dec[j, 5:, a]
                m = (torch.sigmoid(coef @ P).reshape(ph, pw) > 0.5).numpy()
                out = refine(m, (box / 4).tolist())
                g = gt[int(r["gt"])].unsqueeze(0)
                old = float(box_iou(box.unsqueeze(0), g))
                if out is None:
                    results[r["kind"]].append((old, old, old))
                    continue
                mb, ub = (torch.tensor(b, dtype=torch.float32).unsqueeze(0) * 4 for b in out)
                results[r["kind"]].append((old, float(box_iou(mb, g)), float(box_iou(ub, g))))
        print(f"{bi + len(names)}/{len(imgs)}", flush=True)

for kind, res in results.items():
    arr = np.array(res)
    print(f"\n{kind} (n={len(arr)}): median IoU old {np.median(arr[:, 0]):.3f}  mask-box {np.median(arr[:, 1]):.3f}  union {np.median(arr[:, 2]):.3f}")
    for name, col in (("mask-box", 1), ("union", 2)):
        print(f"  {name:8}: IoU>=0.5 old {np.mean(arr[:, 0] >= .5):.1%} -> new {np.mean(arr[:, col] >= .5):.1%}   "
              f"IoU>=0.75 old {np.mean(arr[:, 0] >= .75):.1%} -> new {np.mean(arr[:, col] >= .75):.1%}   "
              f"improved {np.mean(arr[:, col] > arr[:, 0]+.01):.1%} worse {np.mean(arr[:, col] < arr[:, 0]-.01):.1%}")
