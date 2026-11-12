"""v2 gradient-based comparison track: fine-tune the one-class eel model on the EARLY half of the video,
validate every epoch on the LATE half (no shared clips). CPU only.

Augmentation leans on the model's measured weak spots: brightness/contrast-like jitter (hsv_v) and
scale; flips and mosaic as usual.

Usage: uv run python experiments/finetune.py --data <split_train.yaml> --epochs 6 --name ft_e6 [--imgsz 512]
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--weights", default=str(ROOT / "weights/eel_1cls.pt"))
p.add_argument("--data", required=True)
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--imgsz", type=int, default=512)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr0", type=float, default=0.002)
p.add_argument("--name", default="ft")
p.add_argument("--gentle", type=int, default=0, help="1: mild augmentation (hsv_v 0.4, scale 0.2, no mosaic)")
p.add_argument("--mask_ratio", type=int, default=4, help="mask loss resolution divisor (1 = full input resolution)")
p.add_argument("--freeze", type=int, default=0, help="freeze first N layers (e.g. 23 = train head only)")
a = p.parse_args()

model = YOLO(a.weights)
aug = (dict(hsv_v=0.4, scale=0.2, mosaic=0.0, close_mosaic=0) if a.gentle
       else dict(hsv_v=0.6, scale=0.5, mosaic=1.0, close_mosaic=2))
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device="cpu", workers=6,
    project=str(ROOT / "results/v2/train"), name=a.name, exist_ok=True,
    optimizer="SGD", lr0=a.lr0, lrf=0.1, warmup_epochs=0.5, cos_lr=True,
    hsv_h=0.0, hsv_s=0.0, fliplr=0.5, **aug,
    freeze=a.freeze if a.freeze else None, mask_ratio=a.mask_ratio, plots=False, val=True, patience=100, seed=0,
)
