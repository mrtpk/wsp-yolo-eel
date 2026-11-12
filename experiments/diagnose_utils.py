"""Shared preprocessing for the diagnostic scripts (mirrors ultralytics val letterbox for 640x360 frames).

Call configure(imgsz) first (default 512). Boxes are returned in letterboxed pixel coordinates.
"""
import math

import cv2
import numpy as np
import torch

W, H, RH, PAD_Y = 512, 320, 288, 16


def configure(imgsz=512):
    """Long side = imgsz; short side resized then padded to a stride-32 multiple like ultralytics rect val."""
    global W, H, RH, PAD_Y
    W = imgsz
    RH = round(360 * imgsz / 640)
    H = math.ceil(RH / 32 + 0.5) * 32
    PAD_Y = (H - RH) // 2


def load(path):
    im = cv2.imread(str(path))
    im = cv2.resize(im, (W, RH), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, PAD_Y, H - RH - PAD_Y, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return torch.from_numpy(im[:, :, ::-1].transpose(2, 0, 1).copy()).float() / 255


def gt_boxes(label_path):
    """Return GT boxes in letterboxed pixel xyxy and (bbox_max_side_frac, length_px, area_px) per eel."""
    boxes, ext = [], []
    for line in open(label_path):
        p = line.split()
        if not p:
            continue
        xy = np.array(p[1:], dtype=np.float32).reshape(-1, 2)
        x0, y0 = xy.min(0)
        x1, y1 = xy.max(0)
        boxes.append([x0 * W, y0 * RH + PAD_Y, x1 * W, y1 * RH + PAD_Y])
        poly = xy * [640, 360]
        per = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1).sum()
        ext.append((max(x1 - x0, y1 - y0), per / 2, cv2.contourArea(poly.astype(np.float32))))
    return torch.tensor(boxes).reshape(-1, 4), ext
