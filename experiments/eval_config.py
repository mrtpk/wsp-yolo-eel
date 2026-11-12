"""v2 configurable evaluation harness. Every run appends one JSON line to results/v2/runs.jsonl.

The YOLO26 head holds two complete heads: one-to-one (o2o, deployed, no NMS) and one-to-many (o2m, normally
discarded). In NMS mode (--e2e 0) the forward pass uses cv2/cv3/cv4 (box/cls/mask-coef); we replace each of those
with a Pick module that returns o2o, o2m, or a weighted mix of the two heads' outputs, per part.

Other knobs: NMS IoU, NMS on top of end-to-end output (--postnms), conf, imgsz, mask logit threshold,
per-stride class-logit bias shifts.

Usage examples:
  uv run python experiments/eval_config.py --data SUBSET --tag o2m --e2e 0
  uv run python experiments/eval_config.py --data SUBSET --tag mix --e2e 0 --cls mean --box o2m --mask o2m
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--weights", default=str(ROOT / "weights/eel_1cls.pt"))
p.add_argument("--data", required=True)
p.add_argument("--tag", default="")
p.add_argument("--e2e", type=int, default=1, help="1: deployed end-to-end o2o head (no NMS); 0: NMS mode with Pick heads")
p.add_argument("--cls", default="o2m", help="o2o | o2m | mean")
p.add_argument("--box", default="o2m")
p.add_argument("--mask", default="o2m")
p.add_argument("--w", type=float, default=0.5, help="weight of o2o in 'mean' mixes")
p.add_argument("--iou", type=float, default=0.7, help="NMS IoU (NMS mode)")
p.add_argument("--postnms", type=float, default=None, help="NMS IoU applied on top of end-to-end output")
p.add_argument("--conf", type=float, default=0.001)
p.add_argument("--imgsz", type=int, default=512)
p.add_argument("--mask_t", type=float, default=0.0, help="mask logit threshold (0 = prob 0.5)")
p.add_argument("--b8", type=float, default=0.0, help="class-logit shift at stride 8")
p.add_argument("--b16", type=float, default=0.0)
p.add_argument("--b32", type=float, default=0.0)
p.add_argument("--augment", type=int, default=0)
p.add_argument("--peak", type=int, default=0,
               help="in-graph duplicate suppression on class logits: 1 = 3x3 local-max per scale, "
                    "2 = also cross-scale (stride 8 vs 16), 3 = cross-scale only")
p.add_argument("--peak_k", type=int, default=3, help="local-max window")
p.add_argument("--peak_pen", type=float, default=20.0, help="logit penalty for non-peak anchors (soft suppression if small)")
p.add_argument("--fliptta", type=int, default=0,
               help="head-level flip TTA: average class logits and box distances of image + mirrored image")
p.add_argument("--vote", type=float, default=None, help="box-voting cluster IoU (replaces/extends NMS)")
p.add_argument("--vote_nms", type=float, default=0.5, help="IoU of the NMS that picks the kept boxes when voting")
p.add_argument("--vote_p", type=float, default=1.0, help="score exponent of voting weights")
p.add_argument("--vote_min", type=float, default=0.0, help="only candidates with score >= vote_min * kept score vote")
p.add_argument("--vote_mask", type=int, default=0, help="also average mask coefficients")
p.add_argument("--native", type=int, default=0, help="1: score masks at full input resolution (logits upsampled before threshold; GT at mask_ratio 1)")
p.add_argument("--out", default=str(ROOT / "results/v2/runs.jsonl"))
a = p.parse_args()

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class Pick(nn.Module):
    def __init__(self, o2o, o2m, mode, w, shift=0.0):
        super().__init__()
        self.o2o, self.o2m, self.mode, self.w, self.shift = o2o, o2m, mode, w, shift

    def forward(self, x):
        if self.mode == "o2o":
            y = self.o2o(x)
        elif self.mode == "o2m":
            y = self.o2m(x)
        else:
            y = self.w * self.o2o(x) + (1 - self.w) * self.o2m(x)
        return y + self.shift if self.shift else y


class Shift(nn.Module):
    def __init__(self, m, shift):
        super().__init__()
        self.m, self.shift = m, shift

    def forward(self, x):
        return self.m(x) + self.shift


yolo = YOLO(a.weights)
head = yolo.model.model[-1]
shifts = [a.b8, a.b16, a.b32]
if a.e2e:
    for i, s in enumerate(shifts):
        if s:
            head.one2one_cv3[i] = Shift(head.one2one_cv3[i], s)
else:
    for i in range(head.nl):
        head.cv2[i] = Pick(head.one2one_cv2[i], head.cv2[i], a.box, a.w)
        head.cv3[i] = Pick(head.one2one_cv3[i], head.cv3[i], a.cls, a.w, shifts[i])
        head.cv4[i] = Pick(head.one2one_cv4[i], head.cv4[i], a.mask, a.w)

# head-level flip TTA: the batch is doubled with mirrored copies; per-anchor outputs of the mirrored half are
# un-mirrored (and left/right box distances swapped) and averaged with the original half.
if a.fliptta:
    _po = yolo.model._predict_once
    yolo.model._predict_once = lambda x, *ar, **kw: _po(torch.cat([x, x.flip(3)]), *ar, **kw)
    _proto_fwd = head.proto.forward
    head.proto.forward = lambda x, *ar, **kw: _proto_fwd([xi[: xi.shape[0] // 2] for xi in x], *ar, **kw)
    _fh0 = head.forward_head

    def forward_head_tta(x, box_head=None, cls_head=None, mask_head=None):
        out = _fh0(x, box_head=box_head, cls_head=cls_head, mask_head=mask_head)
        if not out:
            return out
        B = x[0].shape[0] // 2
        sizes = [xi.shape[2] * xi.shape[3] for xi in x]
        shapes = [xi.shape[2:] for xi in x]

        def merge(t, swap_lr=False, average=True):
            parts = []
            for m, (h, w) in zip(t.split(sizes, -1), shapes):
                m = m.view(m.shape[0], m.shape[1], h, w)
                o, f = m[:B], m[B:].flip(-1)
                if swap_lr:
                    f = f[:, [2, 1, 0, 3]]
                parts.append(((o + f) / 2 if average else o).flatten(2))
            return torch.cat(parts, -1)

        res = dict(out)
        res["boxes"] = merge(out["boxes"], swap_lr=True)
        res["scores"] = merge(out["scores"])
        if "mask_coefficient" in out:
            res["mask_coefficient"] = merge(out["mask_coefficient"], average=False)
        res["feats"] = [xi[:B] for xi in out["feats"]]
        return res

    head.forward_head = forward_head_tta

# in-graph peak suppression (CenterNet-style), applied to whichever class head feeds inference
if a.peak:
    _fh = head.forward_head

    def forward_head_peak(x, box_head=None, cls_head=None, mask_head=None):
        out = _fh(x, box_head=box_head, cls_head=cls_head, mask_head=mask_head)
        if not out or "scores" not in out:
            return out
        s = out["scores"]  # (bs, nc, A) logits
        sizes = [xi.shape[2] * xi.shape[3] for xi in x]
        maps = [m.view(s.shape[0], s.shape[1], xi.shape[2], xi.shape[3]) for m, xi in zip(s.split(sizes, -1), x)]
        k = a.peak_k
        new = []
        for i, m in enumerate(maps):
            keep = torch.ones_like(m, dtype=torch.bool)
            if a.peak in (1, 2):
                keep &= m >= F.max_pool2d(m, k, 1, k // 2)
            if a.peak in (2, 3):
                if i + 1 < len(maps):  # coarser neighbour: upsample its local max to this grid
                    c = F.max_pool2d(maps[i + 1], 3, 1, 1)
                    keep &= m >= F.interpolate(c, size=m.shape[2:], mode="nearest")
                if i > 0:  # finer neighbour: max over the 2x2 cells (+ ring) it covers
                    f = F.max_pool2d(F.max_pool2d(maps[i - 1], 3, 1, 1), 2, 2)
                    keep &= m >= f[..., : m.shape[2], : m.shape[3]]
            new.append(torch.where(keep, m, m - a.peak_pen).flatten(2))
        out["scores"] = torch.cat(new, -1)
        return out

    head.forward_head = forward_head_peak

# mask threshold: binarize the full mask first, then crop to the box (a negative threshold must not fill the box)
if a.mask_t != 0.0:
    def process_mask_t(protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        r = torch.tensor([[mw / shape[1], mh / shape[0], mw / shape[1], mh / shape[0]]], device=bboxes.device)
        if upsample:
            masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
            r = torch.ones_like(r)
        binm = (masks > a.mask_t).float()
        return ops.crop_mask(binm, boxes=bboxes * r).gt_(0.5).byte()

    ops.process_mask = process_mask_t

if a.vote is not None:
    # NMS with score-weighted box voting: each kept box is replaced by the score^p weighted mean of all
    # candidates overlapping it with IoU >= a.vote (optionally also averaging mask coefficients).
    # Use with --iou 0.99 (NMS mode) so the inner NMS keeps the candidate cloud.
    import torchvision
    from ultralytics.models.yolo.detect.val import DetectionValidator

    _orig_v = DetectionValidator.postprocess

    def _vote(self, preds):
        outs = _orig_v(self, preds)
        for o in outs:
            b, s = o["bboxes"].float(), o["conf"].float()
            if len(b) < 2:
                continue
            keep = torchvision.ops.nms(b, s, a.vote_nms)
            I = torchvision.ops.box_iou(b[keep], b)
            W = (I >= a.vote).float() * (s[None] ** a.vote_p) * (s[None] >= a.vote_min * s[keep][:, None])
            W = W / W.sum(1, keepdim=True)
            o["bboxes"] = (W @ b).to(o["bboxes"].dtype)
            if a.vote_mask and "extra" in o and o["extra"] is not None and len(o["extra"]) == len(b):
                o["extra"] = (W @ o["extra"].float()).to(o["extra"].dtype)
            else:
                o["extra"] = o["extra"][keep]
            for k in ("conf", "cls"):
                o[k] = o[k][keep]
        return outs

    DetectionValidator.postprocess = _vote

if a.postnms is not None:
    import torchvision  # noqa: F811
    from ultralytics.models.yolo.detect.val import DetectionValidator

    _orig = DetectionValidator.postprocess

    def _pp(self, preds):
        outs = _orig(self, preds)
        for o in outs:
            if len(o["bboxes"]) > 1:
                keep = torchvision.ops.nms(o["bboxes"].float(), o["conf"].float(), a.postnms)
                for k in ("bboxes", "conf", "cls", "extra"):
                    o[k] = o[k][keep]
        return outs

    DetectionValidator.postprocess = _pp

if a.native:
    from ultralytics.models.yolo.segment.val import SegmentationValidator

    _im = SegmentationValidator.init_metrics

    def _init_metrics(self, model):
        _im(self, model)
        self.process = ops.process_mask_native

    SegmentationValidator.init_metrics = _init_metrics

t0 = time.time()
r = yolo.val(data=a.data, imgsz=a.imgsz, batch=32, device="cpu", plots=False, verbose=False, workers=4,
             conf=a.conf, iou=a.iou, end2end=bool(a.e2e), augment=bool(a.augment),
             **(dict(mask_ratio=1) if a.native else {}))
rec = dict(time=time.strftime("%H:%M:%S"), secs=round(time.time() - t0), cfg=vars(a),
           box_m=dict(P=r.box.mp, R=r.box.mr, mAP50=r.box.map50, mAP=r.box.map),
           mask_m=dict(P=r.seg.mp, R=r.seg.mr, mAP50=r.seg.map50, mAP=r.seg.map))
with open(a.out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"{a.tag:28} box mAP50 {r.box.map50:.4f} mAP {r.box.map:.4f} R {r.box.mr:.3f} P {r.box.mp:.3f} | "
      f"mask mAP50 {r.seg.map50:.4f} mAP {r.seg.map:.4f}  ({rec['secs']}s)")
