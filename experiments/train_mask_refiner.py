"""Train a small mask boundary refiner on the first half of the video, score it on the second half.

The YOLO mask logits come from a 1/4-resolution prototype grid (one cell is about 5 frame px; eels are about
9 px wide), so boundaries are blurred and about 0.8 px too fat. The refiner sees the full-resolution infrared crop
and the upsampled logits and predicts a residual logit:

    refined = logit + net([gray, logit / 8, inbox])

Net: fully convolutional, dilated 3x3 convs (receptive field about 35 px), about 40k parameters. The last layer
starts at zero, so the untrained refiner equals the YOLO mask exactly.

Usage: uv run python experiments/train_mask_refiner.py <crops_fit.pt> <crops_test.pt> <out_refiner.pt> [--epochs 30]
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from mask_metrics import score  # noqa: E402
from train_mask_refiner_net import Refiner  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("fit")
p.add_argument("test")
p.add_argument("out")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--width", type=int, default=32)
p.add_argument("--tile", type=int, default=96)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--eval_every", type=int, default=10)
p.add_argument("--eval_sub", type=int, default=4, help="score every Nth test eel at intermediate evaluations")
a = p.parse_args()
torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(14)


def tensors(it):
    return it["gray"].float() / 255.0, it["logit"].float(), it["gt"].float(), it["inbox"].float()


def sample_tile(it, T):
    g, lg, gt, ib = tensors(it)
    H, W = g.shape
    ph, pw = max(T - H, 0), max(T - W, 0)
    if ph or pw:  # pad small crops (logit pad = strongly negative, outside the box)
        g = F.pad(g, (0, pw, 0, ph))
        lg = F.pad(lg, (0, pw, 0, ph), value=-10.0)
        gt = F.pad(gt, (0, pw, 0, ph))
        ib = F.pad(ib, (0, pw, 0, ph))
        H, W = g.shape
    # bias tile position toward the label boundary
    edge = (gt - F.max_pool2d(-gt[None, None], 3, 1, 1)[0, 0].neg()).abs().nonzero()
    if len(edge) and random.random() < 0.8:
        cy, cx = edge[random.randrange(len(edge))].tolist()
        y = min(max(cy - T // 2, 0), H - T)
        x = min(max(cx - T // 2, 0), W - T)
    else:
        y, x = random.randint(0, H - T), random.randint(0, W - T)
    t = [z[y : y + T, x : x + T] for z in (g, lg, gt, ib)]
    if random.random() < 0.5:
        t = [z.flip(1) for z in t]
    if random.random() < 0.5:
        t = [z.flip(0) for z in t]
    return t


def evaluate(net, items):
    """Mean IoU / BF / boundary distance of base and refined masks, restricted to the predicted box like YOLO."""
    net.eval()
    base, ref = [], []
    with torch.no_grad():
        for it in items:
            g, lg, gt, ib = tensors(it)
            r = net(g[None], lg[None], ib[None])[0]
            gm = gt.numpy().astype(np.uint8)
            m = ib.numpy() > 0
            base.append(score((lg.numpy() > 0) & m, gm))
            ref.append(score((r.numpy() > 0) & m, gm))
    net.train()
    agg = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return agg(base), agg(ref)


fit, test = torch.load(a.fit), torch.load(a.test)
print(f"fit eels {len(fit)}, test eels {len(test)}")
net = Refiner(a.width)
print(f"refiner parameters: {sum(q.numel() for q in net.parameters()):,}")
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
steps_per_epoch = max(len(fit) // 32, 1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_epoch)
b0, r0 = evaluate(net, test[:: a.eval_sub])
print(f"test before training: IoU base {b0['iou']:.4f} refined {r0['iou']:.4f} (must match)")
best: tuple[float, dict | None] = (r0["iou"], None)
for ep in range(a.epochs):
    t0 = time.time()
    random.shuffle(fit)
    tot = 0.0
    for s in range(steps_per_epoch):
        batch = [sample_tile(it, a.tile) for it in fit[s * 32 : (s + 1) * 32]]
        g, lg, gt, ib = (torch.stack(z) for z in zip(*batch))
        out = net(g, lg, ib)
        w = 1.0 + 4.0 * ib  # pixels inside the predicted box matter (outside is cropped away anyway)
        loss = (F.binary_cross_entropy_with_logits(out, gt, reduction="none") * w).mean()
        pr = torch.sigmoid(out) * ib
        loss = loss + (1 - (2 * (pr * gt).sum() + 1) / (pr.sum() + (gt * ib).sum() + 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
        b, r = evaluate(net, test if ep == a.epochs - 1 else test[:: a.eval_sub])
        print(f"epoch {ep + 1:3d} loss {tot / steps_per_epoch:.4f}  test IoU {b['iou']:.4f} -> {r['iou']:.4f}  "
              f"BF@1 {b['bf1']:.3f} -> {r['bf1']:.3f}  BF@2 {b['bf2']:.3f} -> {r['bf2']:.3f}  "
              f"|d| {b['absd']:.3f} -> {r['absd']:.3f}  area {b['area']:.3f} -> {r['area']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if r["iou"] > best[0]:
            best = (r["iou"], {k: v.clone() for k, v in net.state_dict().items()})
    else:
        print(f"epoch {ep + 1:3d} loss {tot / steps_per_epoch:.4f} ({time.time() - t0:.0f}s)", flush=True)
torch.save(dict(state=net.state_dict(), best=best[1], width=a.width), a.out)
print(f"saved {a.out}; best test IoU {best[0]:.4f} (note: 'best' is chosen on test, report the final epoch)")
