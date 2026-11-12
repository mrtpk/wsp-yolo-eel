"""Contact sheet of detections of one kind from fp_analysis.py output.

Each tile: 120x120 frame crop around the detection (upscaled), predicted box in red, labelled eels in yellow.
Usage: uv run python experiments/fp_sheet.py <dets.csv> <dataset_root> <kind> <out.png> [n=24] [seed=0]
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

dets_csv, root, kind, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 24
random.seed(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
rows = [r for r in csv.DictReader(open(dets_csv)) if r["kind"] == kind]
rows = random.sample(rows, min(n, len(rows)))
tiles = []
H = 70  # half crop
for r in rows:
    im = cv2.imread(str(root / "images/train" / f"{r['image']}.png"))
    assert im is not None
    for line in open(root / "labels/train" / f"{r['image']}.txt"):
        xy = (np.array(line.split()[1:], float).reshape(-1, 2) * [640, 360]).astype(np.int32)
        cv2.polylines(im, [xy], True, (0, 255, 255), 1)
    x0, y0, x1, y1 = (float(r[k]) for k in ("fx0", "fy0", "fx1", "fy1"))
    cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    half = max(H, int(max(x1 - x0, y1 - y0) / 2) + 10)
    pad = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    crop = pad[cy : cy + 2 * half, cx : cx + 2 * half]
    t = cv2.resize(crop, (220, 220), interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, f"{r['image'][6:]} s={float(r['score']):.2f} iou={float(r['best_iou']):.2f} st{r['stride']}",
                (3, 12), 0, 0.36, (0, 255, 0), 1)
    tiles.append(t)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)])
cv2.imwrite(out, sheet)
print(f"{kind}: {len(rows)} tiles -> {out}")
