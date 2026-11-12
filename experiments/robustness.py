"""Robustness proxy for generalization: evaluate one checkpoint on perturbed copies of a frame subset.

All local frames are training frames, so this cannot measure true generalization; it measures how
brittle the fit is to plausible camera/scene changes. Each perturbation writes a dataset copy
(images + labels) under <work_dir>/<name> and runs ultralytics val on it.

Usage: uv run python experiments/robustness.py <weights.pt> <dataset_root> <work_dir> <out.json> [every] [offset]
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

weights, root, work, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
every = int(sys.argv[5]) if len(sys.argv) > 5 else 5
offset = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rng = np.random.default_rng(0)
model = YOLO(weights)
cls_name = model.names


def jpeg(im, q):
    return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


PERTURB = {
    "clean": lambda im: im,
    "darker_0.6": lambda im: np.clip(im * 0.6, 0, 255).astype(np.uint8),
    "brighter_1.4": lambda im: np.clip(im * 1.4, 0, 255).astype(np.uint8),
    "low_contrast": lambda im: np.clip((im - 128.0) * 0.6 + 128, 0, 255).astype(np.uint8),
    "blur_5": lambda im: cv2.GaussianBlur(im, (5, 5), 0),
    "noise_10": lambda im: np.clip(im + rng.normal(0, 10, im.shape), 0, 255).astype(np.uint8),
    "jpeg_q30": lambda im: jpeg(im, 30),
    "hflip": lambda im: im[:, ::-1].copy(),
    "grayscale": lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
}
IMGSZ = {"imgsz_384": 384, "imgsz_640": 640}

imgs = sorted((root / "images/train").glob("*.png"))[offset::every]
results = {}


def make(name, fn, flip=False):
    d = work / name
    if d.exists():
        shutil.rmtree(d)
    (d / "images/train").mkdir(parents=True)
    (d / "labels/train").mkdir(parents=True)
    for p in imgs:
        cv2.imwrite(str(d / "images/train" / p.name), fn(cv2.imread(str(p))))
        lines = []
        for line in open(root / "labels/train" / (p.stem + ".txt")):
            v = line.split()
            if flip:
                v = [v[0]] + [f"{1 - float(x):.6f}" if i % 2 == 0 else x for i, x in enumerate(v[1:])]
            lines.append(" ".join(v))
        (d / "labels/train" / (p.stem + ".txt")).write_text("\n".join(lines) + "\n")
    names = "\n".join(f"  {k}: {v}" for k, v in cls_name.items())
    (work / f"{name}.yaml").write_text(f"path: {d}\ntrain: images/train\nval: images/train\nnames:\n{names}\n")
    return work / f"{name}.yaml"


def val(yaml, imgsz=512):
    r = model.val(data=str(yaml), imgsz=imgsz, batch=32, device="cpu", plots=False, verbose=False, workers=4)
    return dict(box_P=r.box.mp, box_R=r.box.mr, box_mAP50=r.box.map50, box_mAP=r.box.map,
                mask_mAP50=r.seg.map50, mask_mAP=r.seg.map)


for name, fn in PERTURB.items():
    results[name] = val(make(name, fn, flip=(name == "hflip")))
    print(name, {k: round(v, 3) for k, v in results[name].items()}, flush=True)
clean_yaml = work / "clean.yaml"
for name, sz in IMGSZ.items():
    results[name] = val(clean_yaml, sz)
    print(name, {k: round(v, 3) for k, v in results[name].items()}, flush=True)

json.dump(dict(weights=weights, frames=len(imgs), results=results), open(out, "w"), indent=2)
c = results["clean"]
print(f"\n{'perturbation':14} {'box mAP50':>10} {'Δ':>7} {'box mAP50-95':>13} {'Δ':>7} {'recall':>7} {'mask mAP50-95':>14}")
for k, r in results.items():
    print(f"{k:14} {r['box_mAP50']:10.3f} {r['box_mAP50']-c['box_mAP50']:+7.3f} {r['box_mAP']:13.3f} "
          f"{r['box_mAP']-c['box_mAP']:+7.3f} {r['box_R']:7.3f} {r['mask_mAP']:14.3f}")
