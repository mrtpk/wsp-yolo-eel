"""Score a trained mask refiner on held-out eel crops under different output regions.

Regions: the predicted box (as YOLO crops masks), the box grown by k px, and the whole crop window (box + 8 px),
so the refiner may extend a mask that the box cut off.

Usage: uv run python experiments/eval_mask_refiner.py <refiner.pt> <crops_test.pt> [--best]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mask_metrics import score  # noqa: E402
from train_mask_refiner_net import Refiner  # noqa: E402

ck = torch.load(sys.argv[1])
items = torch.load(sys.argv[2])
net = Refiner(ck["width"])
net.load_state_dict(ck["best"] if "--best" in sys.argv and ck.get("best") else ck["state"])
net.eval()
torch.set_num_threads(14)

regions = {"box": 0, "box+2": 2, "box+4": 4, "window": None}
rows: dict = {"base (box)": []}
rows.update({f"refined ({k})": [] for k in regions})
with torch.no_grad():
    for it in items:
        g = it["gray"].float() / 255.0
        lg = it["logit"].float()
        ib = it["inbox"].float()
        gt = it["gt"].numpy().astype(np.uint8)
        r = net(g[None], lg[None], ib[None])[0].numpy()
        box = ib.numpy().astype(np.uint8)
        rows["base (box)"].append(score((lg.numpy() > 0) & (box > 0), gt))
        for name, k in regions.items():
            reg = np.ones_like(box) if k is None else (cv2.dilate(box, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) if k else box)
            rows[f"refined ({name})"].append(score((r > 0) & (reg > 0), gt))
print(f"held-out eels: {len(items)}")
print(f"{'variant':20} {'IoU':>7} {'BF@1':>6} {'BF@2':>6} {'|d|px':>6} {'d px':>7} {'area':>6}")
for name, rs in rows.items():
    m = {f: float(np.mean([x[f] for x in rs])) for f in ("iou", "bf1", "bf2", "absd", "sd", "area")}
    print(f"{name:20} {m['iou']:7.4f} {m['bf1']:6.3f} {m['bf2']:6.3f} {m['absd']:6.3f} {m['sd']:+7.3f} {m['area']:6.3f}")
