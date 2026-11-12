"""Dump per-frame detections (boxes in 640x360 frame pixels + score) at a very low confidence, for offline
post-processing experiments (temporal re-scoring). Supports the deployed head with optional in-graph peak
suppression (same implementation as eval_config.py --peak 2).

Usage: uv run python experiments/dump_dets.py <weights.pt> <dataset_root> <out.jsonl> [--peak 2] [--postnms 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import diagnose_utils as du  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("root")
p.add_argument("out")
p.add_argument("--peak", type=int, default=0)
p.add_argument("--postnms", type=float, default=None)
p.add_argument("--conf", type=float, default=0.001)
a = p.parse_args()
du.configure(512)
torch.set_num_threads(14)

model = YOLO(a.weights).model.float().eval()
head = model.model[-1]


def peak_scores(s, x):
    sizes = [xi.shape[2] * xi.shape[3] for xi in x]
    maps = [m.view(s.shape[0], s.shape[1], xi.shape[2], xi.shape[3]) for m, xi in zip(s.split(sizes, -1), x)]
    new = []
    for i, m in enumerate(maps):
        keep = m >= F.max_pool2d(m, 3, 1, 1)
        if i + 1 < len(maps):
            keep &= m >= F.interpolate(F.max_pool2d(maps[i + 1], 3, 1, 1), size=m.shape[2:], mode="nearest")
        if i > 0:
            f = F.max_pool2d(F.max_pool2d(maps[i - 1], 3, 1, 1), 2, 2)
            keep &= m >= f[..., : m.shape[2], : m.shape[3]]
        new.append(torch.where(keep, m, m - 20.0).flatten(2))
    return torch.cat(new, -1)


imgs = sorted((Path(a.root) / "images/train").glob("*.png"))
out = open(a.out, "w")
with torch.inference_mode():
    for bi in range(0, len(imgs), 32):
        batch = imgs[bi : bi + 32]
        x = torch.stack([du.load(p) for p in batch])
        (_, _), preds = model(x)
        o2o = preds["one2one"]
        if a.peak:
            o2o["scores"] = peak_scores(o2o["scores"], o2o["feats"])
        dec = head._inference(o2o)  # (b, 4+1+nm, A)
        for j, pth in enumerate(batch):
            boxes, sc = dec[j, :4].T, dec[j, 4]
            k = (sc >= a.conf).nonzero().flatten()
            k = k[sc[k].argsort(descending=True)][:300]
            b, s = boxes[k], sc[k]
            if a.postnms is not None and len(b) > 1:
                keep = torchvision.ops.nms(b, s, a.postnms)
                b, s = b[keep], s[keep]
            # letterbox px -> 640x360 frame px
            fb = torch.stack([b[:, 0] / du.W * 640, (b[:, 1] - du.PAD_Y) / du.RH * 360,
                              b[:, 2] / du.W * 640, (b[:, 3] - du.PAD_Y) / du.RH * 360], 1)
            out.write(json.dumps(dict(image=pth.stem, boxes=[[round(v, 2) for v in r] for r in fb.tolist()],
                                      scores=[round(v, 5) for v in s.tolist()])) + "\n")
        print(f"{bi + len(batch)}/{len(imgs)}", flush=True)
out.close()
