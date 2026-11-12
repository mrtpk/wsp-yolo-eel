"""Deployable v2 inference: the one-class eel model + in-graph peak suppression (+ optional light NMS).

    from eel_v2_infer import load_eel_v2
    model = load_eel_v2("weights/eel_1cls.pt")            # peak suppression, no NMS (end-to-end)
    results = model.predict("frame.png", imgsz=512, conf=0.25)
    results = model.predict_nms("frame.png", iou=0.5)     # peak suppression + light NMS on top

Peak suppression keeps a class score only where it is a local maximum of its own scale's 3x3 neighbourhood and
not beaten by the neighbouring scales (stride 8 vs 16 vs 32). It removes the duplicate boxes the one-to-one head
fails to suppress (adjacent anchors and cross-scale doubles) while staying NMS-free.

CLI: uv run python experiments/eel_v2_infer.py <weights.pt> <image_or_dir> [--nms 0.5] [--refiner weights/mask_refiner.pt]

With a refiner, use predict_nms (it sets retina_masks=True and refines each mask's boundary).
"""
import sys

import torch
import torch.nn.functional as F
import torchvision
from ultralytics import YOLO


def _peak(scores, feats, k=3):
    sizes = [f.shape[2] * f.shape[3] for f in feats]
    maps = [m.view(scores.shape[0], scores.shape[1], f.shape[2], f.shape[3]) for m, f in zip(scores.split(sizes, -1), feats)]
    out = []
    for i, m in enumerate(maps):
        keep = m >= F.max_pool2d(m, k, 1, k // 2)
        if i + 1 < len(maps):
            keep &= m >= F.interpolate(F.max_pool2d(maps[i + 1], 3, 1, 1), size=m.shape[2:], mode="nearest")
        if i > 0:
            f = F.max_pool2d(F.max_pool2d(maps[i - 1], 3, 1, 1), 2, 2)
            keep &= m >= f[..., : m.shape[2], : m.shape[3]]
        out.append(torch.where(keep, m, m - 20.0).flatten(2))
    return torch.cat(out, -1)


def _attach_refiner(model, refiner_path):
    """Replace each mask with the boundary refiner's output (needs retina_masks=True: full-resolution logits).

    The refiner sees the grayscale frame and the upsampled YOLO mask logits in a window of the box + 8 px and
    predicts corrected logits; the mask stays restricted to the box as in YOLO.
    """
    import numpy as np
    import ultralytics.models.yolo.segment.predict as sp
    from ultralytics.utils import ops

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from train_mask_refiner_net import Refiner

    ck = torch.load(refiner_path, map_location="cpu")
    net = Refiner(ck["width"])
    net.load_state_dict(ck["state"])
    net.eval()
    stash = []
    native = ops.process_mask_native

    def capture(protos, masks_in, bboxes, shape):
        c, mh, mw = protos.shape
        lg = ops.scale_masks((masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)[None], shape)[0]
        stash.append(lg)
        return native(protos, masks_in, bboxes, shape)

    sp.ops.process_mask_native = capture

    def refine(results):
        logits = stash[-len(results):] if stash else []
        stash.clear()
        for r, lg in zip(results, logits):
            if r.masks is None or len(r.boxes) == 0:
                continue
            g = r.orig_img
            g = (g if g.ndim == 2 else g[..., :3].mean(2)).astype(np.float32) / 255.0
            H, W = g.shape
            new = r.masks.data.clone()
            for j, (x0, y0, x1, y1) in enumerate(r.boxes.xyxy.tolist()):
                X0, Y0 = max(int(x0) - 8, 0), max(int(y0) - 8, 0)
                X1, Y1 = min(int(x1 + 0.999) + 8, W), min(int(y1 + 0.999) + 8, H)
                box = torch.zeros(Y1 - Y0, X1 - X0)
                box[max(int(y0), 0) - Y0 : int(y1 + 0.999) - Y0, max(int(x0), 0) - X0 : int(x1 + 0.999) - X0] = 1
                with torch.no_grad():
                    out = net(torch.from_numpy(g[Y0:Y1, X0:X1])[None], lg[j, Y0:Y1, X0:X1][None], box[None])[0]
                new[j] = 0
                new[j, Y0:Y1, X0:X1] = ((out > 0).float() * box).to(new.dtype)
            r.masks.data = new
        return results

    return refine


def load_eel_v2(weights, refiner=None):
    """refiner: path to weights/mask_refiner.pt to sharpen mask boundaries (use predict(..., retina_masks=True))."""
    model = YOLO(weights)
    head = model.model.model[-1]
    orig = head.forward_head

    def forward_head(x, box_head=None, cls_head=None, mask_head=None):
        out = orig(x, box_head=box_head, cls_head=cls_head, mask_head=mask_head)
        if out and "scores" in out:
            out["scores"] = _peak(out["scores"], x)
        return out

    head.forward_head = forward_head

    refine = _attach_refiner(model, refiner) if refiner else None

    def predict_nms(source, iou=0.5, **kw):
        if refine:
            kw["retina_masks"] = True
        res = model.predict(source, **kw)
        if refine:
            res = refine(res)
        for i, r in enumerate(res):
            if r.boxes is not None and len(r.boxes) > 1:
                keep = torchvision.ops.nms(r.boxes.xyxy.float(), r.boxes.conf.float(), iou)
                res[i] = r[keep]
        return res

    model.predict_nms = predict_nms
    return model


if __name__ == "__main__":
    w, src = sys.argv[1], sys.argv[2]
    ref = sys.argv[sys.argv.index("--refiner") + 1] if "--refiner" in sys.argv else None
    m = load_eel_v2(w, refiner=ref)
    nms = float(sys.argv[sys.argv.index("--nms") + 1]) if "--nms" in sys.argv else 1.0
    res = m.predict_nms(src, iou=nms, imgsz=512, conf=0.25, verbose=False)
    for r in res:
        print(r.path.split("/")[-1], [round(float(c), 3) for c in r.boxes.conf])
