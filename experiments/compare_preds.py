"""Compare predictions of two checkpoints image by image.

Usage: uv run python experiments/compare_preds.py <a.pt> <b.pt> <image_dir> [n_images]
"""
import sys
from typing import Any
from pathlib import Path
from ultralytics import YOLO

a_path, b_path, img_dir = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
imgs = sorted(Path(img_dir).glob("*.png"))
imgs = imgs[:: max(1, len(imgs) // n)][:n]

a, b = YOLO(a_path), YOLO(b_path)
kw: dict[str, Any] = dict(imgsz=512, conf=0.001, device="cpu", verbose=False, retina_masks=False)
max_box = max_conf = max_mask = 0.0
count_diff = 0
for img in imgs:
    ra, rb = a.predict(str(img), **kw)[0], b.predict(str(img), **kw)[0]
    assert ra.boxes is not None and rb.boxes is not None
    if len(ra.boxes) != len(rb.boxes):
        count_diff += 1
        continue
    if len(ra.boxes) == 0:
        continue
    max_box = max(max_box, (ra.boxes.xyxy - rb.boxes.xyxy).abs().max().item())
    max_conf = max(max_conf, (ra.boxes.conf - rb.boxes.conf).abs().max().item())
    if ra.masks is not None:
        max_mask = max(max_mask, (ra.masks.data != rb.masks.data).float().mean().item())

print(f"images compared: {len(imgs)}")
print(f"images with different detection count: {count_diff}")
print(f"max |box diff| px: {max_box:.4g}   max |conf diff|: {max_conf:.4g}   max mask pixel disagreement: {max_mask:.4g}")
