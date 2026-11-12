"""Dump per-eel crops for a mask boundary refiner: grayscale frame, full-resolution mask logits (before box crop),
label mask and box mask, for every detection matched to a labelled eel (box IoU >= 0.5).

Detections come from eel_v2_infer (peak suppression + NMS 0.5, conf 0.25, retina masks), as deployed.
Crop = predicted box grown by PAD px, clipped to the 640x360 frame.

Usage: uv run python experiments/dump_mask_crops.py <weights.pt> <dataset_root> <out.pt> --from N --to M [--nth 1]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).parent))
from eel_v2_infer import load_eel_v2  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
import ultralytics.models.yolo.segment.predict as sp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("root")
p.add_argument("out")
p.add_argument("--from", dest="lo", type=int, default=0)
p.add_argument("--to", dest="hi", type=int, default=10**9)
p.add_argument("--nth", type=int, default=1)
p.add_argument("--imgsz", type=int, default=512)
a = p.parse_args()
PAD = 8
torch.set_num_threads(14)
model = load_eel_v2(a.weights)

captured: dict = {}
_native = ops.process_mask_native


def _cap(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    lg = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    captured["logits"] = ops.scale_masks(lg[None], shape)[0]
    return _native(protos, masks_in, bboxes, shape)


sp.ops.process_mask_native = _cap
root = Path(a.root)
imgs = [q for q in sorted((root / "images/train").glob("*.png")) if a.lo <= int(q.stem.split("_")[1]) < a.hi][:: a.nth]
items = []
for n, q in enumerate(imgs):
    gts = []
    for line in open(root / "labels/train" / f"{q.stem}.txt"):
        v = line.split()
        if v:
            xy = np.array(v[1:], float).reshape(-1, 2) * [640, 360]
            m = np.zeros((360, 640), np.uint8)
            cv2.fillPoly(m, [np.round(xy * 8).astype(np.int32)], 1, shift=3)  # sub-pixel polygon
            gts.append((m, np.r_[xy.min(0), xy.max(0)]))
    captured.clear()
    r = model.predict_nms(str(q), iou=0.5, imgsz=a.imgsz, conf=0.25, retina_masks=True, verbose=False)[0]
    if r.masks is None or not gts or "logits" not in captured:
        continue
    gray = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    pb = r.boxes.xyxy.float()
    lg = captured["logits"]
    for gm, gb in gts:
        ious = torchvision.ops.box_iou(torch.tensor(np.array([gb]), dtype=torch.float32), pb)[0]
        if len(ious) == 0 or ious.max() < 0.5:
            continue
        j = int(ious.argmax())
        x0, y0, x1, y1 = pb[j].tolist()
        X0, Y0 = max(int(x0) - PAD, 0), max(int(y0) - PAD, 0)
        X1, Y1 = min(int(np.ceil(x1)) + PAD, 640), min(int(np.ceil(y1)) + PAD, 360)
        inbox = np.zeros((360, 640), np.uint8)
        inbox[max(int(y0), 0) : int(np.ceil(y1)), max(int(x0), 0) : int(np.ceil(x1))] = 1
        items.append(dict(
            image=q.stem, box=[x0, y0, x1, y1], crop=[X0, Y0, X1, Y1],
            gray=torch.from_numpy(gray[Y0:Y1, X0:X1].copy()),
            logit=lg[j, Y0:Y1, X0:X1].clone().half(),
            gt=torch.from_numpy(gm[Y0:Y1, X0:X1].copy()),
            inbox=torch.from_numpy(inbox[Y0:Y1, X0:X1].copy()),
        ))
    if n % 200 == 0:
        print(f"{n}/{len(imgs)} frames, {len(items)} eels", flush=True)
torch.save(items, a.out)
print(f"saved {a.out}: {len(items)} eels from {len(imgs)} frames")
