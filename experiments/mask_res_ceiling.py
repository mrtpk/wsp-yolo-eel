"""Mask-resolution ceiling: how much IoU does a half-pixel error cost for eel masks at proto resolution?

Ultralytics computes mask IoU at proto resolution (imgsz/4). Eels are ~10 px wide in the 640x360 frame,
i.e. ~2 proto pixels at imgsz 512. For each GT polygon we rasterize it at proto resolution, and again shifted
by 0.5 proto px (diagonally); the IoU between the two is what a perfect-but-half-pixel-off prediction scores.

Usage: uv run python experiments/mask_res_ceiling.py <labels_dir> [every]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

labels = sorted(Path(sys.argv[1]).glob("*.txt"))[:: int(sys.argv[2]) if len(sys.argv) > 2 else 5]
SS = 8  # supersampling for sub-pixel rasterization


def raster(poly, w, h, shift):
    """Area-weighted rasterization: draw at SS x resolution, then box-downsample (soft mask -> threshold 0.5)."""
    m = np.zeros((h * SS, w * SS), np.uint8)
    pts = np.round((poly + shift) * SS).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.reshape(h, SS, w, SS).mean((1, 3)) >= 0.5


print(f"{'imgsz':>6} {'proto':>9} {'mean IoU':>9} {'p10':>6} {'p50':>6} {'>=0.75':>7} {'>=0.9':>6}   (perfect shape, 0.5 proto-px offset)")
for imgsz in (512, 640, 768, 1024):
    pw, ph = imgsz // 4, round(360 * imgsz / 640) // 4
    ious = []
    for f in labels:
        for line in open(f):
            v = line.split()
            if not v:
                continue
            poly = np.array(v[1:], np.float32).reshape(-1, 2) * [pw, ph]
            a, b = raster(poly, pw, ph, 0.0), raster(poly, pw, ph, 0.5)
            u = (a | b).sum()
            if u:
                ious.append((a & b).sum() / u)
    i = np.array(ious)
    print(f"{imgsz:6} {pw:4}x{ph:<4} {i.mean():9.3f} {np.percentile(i, 10):6.3f} {np.median(i):6.3f} {(i >= .75).mean():7.1%} {(i >= .9).mean():6.1%}")
