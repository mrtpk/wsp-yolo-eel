"""Contact sheet of the mask refiner on random held-out eels: label (green), YOLO mask (red), refined mask (cyan).

Eels are drawn at random (seed 0) among crops 60-200 px wide, so the sheet is typical, not best case.

Usage: uv run python experiments/refiner_examples.py <refiner.pt> <crops_test.pt> <out.png> [--n 6]
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
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
net = Refiner(ck["width"])
net.load_state_dict(ck["state"])
net.eval()
pool = [it for it in items if 60 <= it["gray"].shape[1] <= 200 and it["gray"].shape[0] <= 120]
rng = np.random.default_rng(0)
pick = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
S, CW, CH = 4, 200 * 4, 120 * 4
tiles = []
for it in pick:
    g = it["gray"].float() / 255.0
    lg = it["logit"].float()
    ib = it["inbox"].float()
    with torch.no_grad():
        r = net(g[None], lg[None], ib[None])[0].numpy()
    box = ib.numpy() > 0
    base = ((lg.numpy() > 0) & box).astype(np.uint8)
    ref = ((r > 0) & box).astype(np.uint8)
    gt = it["gt"].numpy().astype(np.uint8)
    sb, sr = score(base, gt), score(ref, gt)
    img = cv2.cvtColor(cv2.resize(it["gray"].numpy(), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for m, col in ((gt, (0, 220, 0)), (base, (0, 0, 255)), (ref, (255, 255, 0))):
        big = cv2.resize(m, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, 2)
    img = img[:CH, :CW]
    tile = np.full((img.shape[0] + 50, CW, 3), 30, np.uint8)
    tile[: img.shape[0], : img.shape[1]] = img
    cv2.putText(tile, f"area {sb['area']:.2f} -> {sr['area']:.2f}   |d| {sb['absd']:.2f} -> {sr['absd']:.2f} px",
                (10, img.shape[0] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    tiles.append(tile)


def pad(t, h):
    return np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 30, np.uint8)])


rows = []
for i in range(0, n, 2):
    pair = tiles[i : i + 2]
    h = max(t.shape[0] for t in pair) + 20
    rows.append(np.hstack([pad(t, h) for t in pair] + [np.full((h, CW, 3), 30, np.uint8)] * (2 - len(pair))))
sheet = np.vstack(rows)
head = np.full((70, sheet.shape[1], 3), 30, np.uint8)
cv2.putText(head, "label", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)
cv2.putText(head, "YOLO mask", (170, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
cv2.putText(head, "refined mask", (430, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
cv2.imwrite(sys.argv[3], np.vstack([head, sheet]))
print(f"saved {sys.argv[3]} ({n} eels of {len(pool)} in the pool)")
