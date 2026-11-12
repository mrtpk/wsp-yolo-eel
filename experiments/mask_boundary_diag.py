"""Diagnose mask boundary errors at full frame resolution (640x360).

For each predicted eel matched to a labelled eel (box IoU >= 0.5), compare the full-resolution mask with the
rasterized label polygon:
  - mask IoU and area ratio (pred / GT): systematic fat or thin masks
  - signed boundary error: for each predicted boundary pixel, distance to the GT boundary, + outside GT, - inside
  - boundary F-score at 1 and 2 px tolerance
  - IoU when the mask logit threshold changes (is the boundary biased?)

Usage: uv run python experiments/mask_boundary_diag.py <weights.pt> <dataset_root> [every_nth=10] [--from N]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).parent))
from eel_v2_infer import load_eel_v2  # noqa: E402
from mask_metrics import guided_filter, score  # noqa: E402

w, root = sys.argv[1], Path(sys.argv[2])
nth = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 10
IMGSZ = int(sys.argv[sys.argv.index("--imgsz") + 1]) if "--imgsz" in sys.argv else 512
start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
torch.set_num_threads(14)
model = load_eel_v2(w)
head = model.model.model[-1]

# capture the raw mask logits at full resolution by wrapping process_mask_native (used with retina_masks=True)
from ultralytics.utils import ops  # noqa: E402

captured: dict = {}
_native = ops.process_mask_native


def _cap(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    lg = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    captured["logits"] = ops.scale_masks(lg[None], shape)[0]
    captured["bboxes"] = bboxes.clone()
    return _native(protos, masks_in, bboxes, shape)


ops.process_mask_native = _cap
import ultralytics.models.yolo.segment.predict as sp  # noqa: E402

sp.ops.process_mask_native = _cap

imgs = sorted((root / "images/train").glob("*.png"))
imgs = [p for p in imgs if int(p.stem.split("_")[1]) >= start][::nth]
TH = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


GF = [(2, 1e-3), (2, 1e-2), (4, 1e-3), (4, 1e-2), (4, 1e-1), (8, 1e-2)]
rows: list[dict] = []
for p in imgs:
    gts = []
    for line in open(root / "labels/train" / f"{p.stem}.txt"):
        v = line.split()
        if v:
            xy = (np.array(v[1:], float).reshape(-1, 2) * [640, 360]).round().astype(np.int32)
            m = np.zeros((360, 640), np.uint8)
            cv2.fillPoly(m, [xy], 1)
            gts.append((m, np.r_[xy.min(0), xy.max(0)]))
    captured.clear()
    r = model.predict_nms(str(p), iou=0.5, imgsz=IMGSZ, conf=0.25, retina_masks=True, verbose=False)[0]
    if r.masks is None or not gts or "logits" not in captured:
        continue
    pb = r.boxes.xyxy.numpy()
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # logits are in letterboxed input space; map to frame like Ultralytics does
    lg = captured["logits"]
    lg = ops.scale_masks(lg[None], (360, 640))[0] if lg.shape[1:] != (360, 640) else lg
    bx = torch.tensor(pb, dtype=torch.float32)
    for k, (gm, gb) in enumerate(gts):
        ious = torchvision.ops.box_iou(torch.tensor(np.array([gb]), dtype=torch.float32), bx)[0].tolist()
        if not ious or max(ious) < 0.5:
            continue
        j = int(np.argmax(ious))
        L = lg[j].numpy()
        inbox = np.zeros_like(gm, bool)
        x0, y0, x1, y1 = pb[j].astype(int)
        inbox[max(y0, 0) : y1 + 1, max(x0, 0) : x1 + 1] = True
        row: dict = dict(iou={t: score((L > t) & inbox, gm)["iou"] for t in TH})
        row["base"] = score((L > 0) & inbox, gm)
        P = 1 / (1 + np.exp(-np.clip(L, -30, 30)))
        for r_, e_ in GF:
            Q = guided_filter(gray, P, r_, e_)
            row[f"gf r{r_} e{e_}"] = score((Q > 0.5) & inbox, gm)
        rows.append(row)

n = len(rows)
print(f"matched eels: {n}")
print("mean IoU by logit threshold:", "  ".join(f"{t:+.1f}:{np.mean([r['iou'][t] for r in rows]):.4f}" for t in TH))
print(f"{'variant':16} {'IoU':>7} {'BF@1':>6} {'BF@2':>6} {'|d|px':>6} {'d px':>6} {'area':>6}")
for key in ["base"] + [f"gf r{r_} e{e_}" for r_, e_ in GF]:
    stats = {f: np.mean([r[key][f] for r in rows]) for f in ("iou", "bf1", "bf2", "absd", "sd", "area")}
    print(f"{key:16} {stats['iou']:7.4f} {stats['bf1']:6.3f} {stats['bf2']:6.3f} {stats['absd']:6.3f} {stats['sd']:+6.3f} {stats['area']:6.3f}")
