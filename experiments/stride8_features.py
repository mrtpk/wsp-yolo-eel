"""Cache the 64-d input features of the stride-8 eel classifier (head.one2one_cv3[0][2]) for every anchor.

The eel logit at stride 8 is   logit = w . f + b   with f the cached feature, so any edit of (w, b)
can be scored exactly on the cache without re-running the network.

Anchor labels:
  1  = target: best-IoU stride-8 anchor of an eel that stride 16 does NOT already detect
  2  = duplicate risk: stride-8 anchor with IoU>=0.5 on an eel stride 16 already detects
  3  = neighbour: other stride-8 anchor with IoU>=0.5 on a target eel (would be a duplicate)
  0  = background
Also stores the target eel's length so recall can be broken down by size.

Usage: uv run python experiments/stride8_features.py <weights.pt> <dataset_root> <out.pt> <every> <offset>
"""
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_utils import load, gt_boxes  # noqa: E402

weights, root, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
every, offset = int(sys.argv[4]), int(sys.argv[5])
CONF, IOU = 0.25, 0.5
torch.set_num_threads(14)

model = YOLO(weights).model.float().eval()
head = model.model[-1]
assert head.nc == 1
cap: dict = {}
head.one2one_cv3[0][2].register_forward_hook(lambda m, i, o: cap.__setitem__("f", i[0]))

imgs = sorted((root / "images/train").glob("*.png"))[offset::every]
feats, labels, lengths, frame_ids = [], [], [], []
B = 32
with torch.inference_mode():
    for bi in range(0, len(imgs), B):
        batch = imgs[bi : bi + B]
        (_, _), preds = model(torch.stack([load(p) for p in batch]))
        o2o = preds["one2one"]
        dec = head._inference(o2o)
        score = o2o["scores"][:, 0].sigmoid()
        strides = head.strides.view(-1)
        n8 = int((strides == 8).sum())
        f8 = cap["f"].flatten(2).transpose(1, 2)  # (b, n8, 64)
        for j, p in enumerate(batch):
            boxes = dec[j, :4].T
            gt, ext = gt_boxes(root / "labels/train" / (p.stem + ".txt"))
            lab = torch.zeros(n8, dtype=torch.int8)
            ln = torch.zeros(n8)
            if len(gt):
                iou = box_iou(gt, boxes)
                for g in range(len(gt)):
                    ok = iou[g] >= IOU
                    s16_hit = bool(((strides == 16) & ok & (score[j] >= CONF)).any())
                    i8 = iou[g, :n8]
                    if s16_hit:
                        lab[(i8 >= IOU) & (lab == 0)] = 2
                    elif i8.max() >= IOU:
                        lab[(i8 >= IOU) & (lab == 0)] = 3
                        a = int(i8.argmax())
                        lab[a] = 1
                        ln[a] = ext[g][1]
            feats.append(f8[j].half())
            labels.append(lab)
            lengths.append(ln)
            frame_ids.append(torch.full((n8,), bi + j, dtype=torch.int32))
        print(f"{bi + len(batch)}/{len(imgs)}", end="\r", flush=True)

conv = head.one2one_cv3[0][2]
torch.save(
    dict(
        feats=torch.cat(feats),
        labels=torch.cat(labels),
        lengths=torch.cat(lengths),
        frame=torch.cat(frame_ids),
        w=conv.weight.detach().flatten().clone(),
        b=conv.bias.detach().clone(),
        n_frames=len(imgs),
    ),
    out,
)
lab = torch.cat(labels)
print(f"\nsaved {out}: anchors={len(lab)} targets={(lab == 1).sum()} dup-risk={(lab == 2).sum()} neighbours={(lab == 3).sum()}")
