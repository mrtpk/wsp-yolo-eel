"""v2: temporal re-scoring. Frames come in clips of ~13 consecutive video frames; an eel seen confidently in
neighbouring frames is likely present in this one too. Detections are re-scored using support from the same
clip's neighbouring frames, and box AP (own COCO-style evaluator, IoU 0.50:0.95, 101-point) is compared.

support(d) = max over neighbour frames n (|t-n| <= K, same clip; causal: n < t) of max score of n's detections
             overlapping d with IoU >= TAU
  pullup:  s' = s + alpha * max(0, support - s)          (only raises weak detections)
  blend:   s' = (1 - alpha) * s + alpha * support          (also lowers unsupported ones)
  fill:    pullup + insert an interpolated box when frames t-1 and t+1 both have a detection that matches
           each other (IoU >= TAU) but frame t has nothing overlapping it (two-sided only)

Usage: uv run python experiments/temporal_rescore.py <dets.jsonl> <labels_dir>
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

dets_path, labels = sys.argv[1], Path(sys.argv[2])
D = [json.loads(l) for l in open(dets_path)]
num = lambda s: int(re.search(r"(\d+)", s).group(1))
D.sort(key=lambda r: num(r["image"]))
frames = [r["image"] for r in D]
fnum = np.array([num(f) for f in frames])
clip = np.cumsum(np.r_[0, np.diff(fnum) != 1])
boxes = [np.array(r["boxes"], float).reshape(-1, 4) for r in D]
scores = [np.array(r["scores"], float) for r in D]


def gt_for(name):
    out = []
    for line in open(labels / f"{name}.txt"):
        v = line.split()
        if v:
            xy = np.array(v[1:], float).reshape(-1, 2) * [640, 360]
            out.append([*xy.min(0), *xy.max(0)])
    return np.array(out, float).reshape(-1, 4)


GT = [gt_for(f) for f in frames]


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    ar = lambda x: np.prod(x[:, 2:] - x[:, :2], 1)
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def evaluate(B, S):
    """COCO-style box AP, single class. Returns (AP50, AP50-95, recall at best F1, precision at best F1)."""
    ths = np.linspace(0.5, 0.95, 10)
    allsc, tps = [], []
    n_gt = sum(len(g) for g in GT)
    for b, s, g in zip(B, S, GT):
        order = np.argsort(-s)
        b, s = b[order], s[order]
        I = iou(b, g)
        tp = np.zeros((len(b), len(ths)), bool)
        for t, th in enumerate(ths):
            used = np.zeros(len(g), bool)
            for k in range(len(b)):
                if len(g) == 0:
                    break
                cand = np.where(~used & (I[k] >= th))[0]
                if len(cand):
                    j = cand[np.argmax(I[k, cand])]
                    used[j] = True
                    tp[k, t] = True
        allsc.append(s)
        tps.append(tp)
    sc = np.concatenate(allsc)
    tp = np.concatenate(tps)
    o = np.argsort(-sc, kind="stable")
    tp = tp[o]
    ctp = np.cumsum(tp, 0)
    cfp = np.cumsum(~tp, 0)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    aps = []
    for t in range(len(ths)):
        mpre = np.concatenate([[1.0], prec[:, t], [0.0]])
        mrec = np.concatenate([[0.0], rec[:, t], [1.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        aps.append(np.trapezoid(np.interp(x, mrec, mpre), x))
    f1 = 2 * prec[:, 0] * rec[:, 0] / (prec[:, 0] + rec[:, 0] + 1e-9)
    k = int(np.argmax(f1))
    return aps[0], float(np.mean(aps)), rec[k, 0], prec[k, 0]


def support(i, K, tau, causal):
    s_sup = np.zeros(len(boxes[i]))
    for d in range(1, K + 1):
        for n in ([i - d] if causal else [i - d, i + d]):
            if 0 <= n < len(frames) and clip[n] == clip[i] and abs(fnum[n] - fnum[i]) == d and len(boxes[n]):
                I = iou(boxes[i], boxes[n])
                ok = I >= tau
                if ok.any():
                    s_sup = np.maximum(s_sup, np.where(ok, scores[n][None, :], 0).max(1))
    return s_sup


def run(mode, K, tau, alpha, causal):
    B, S = [], []
    for i in range(len(frames)):
        b, s = boxes[i].copy(), scores[i].copy()
        if mode != "none" and len(b):
            sup = support(i, K, tau, causal)
            s = s + alpha * np.clip(sup - s, 0, None) if mode in ("pullup", "fill") else (1 - alpha) * s + alpha * sup
        if mode == "fill" and 0 < i < len(frames) - 1:
            p_, n_ = i - 1, i + 1
            if clip[p_] == clip[i] == clip[n_] and len(boxes[p_]) and len(boxes[n_]):
                I = iou(boxes[p_], boxes[n_])
                for a_, c_ in zip(*np.where(I >= tau)):
                    sp, sn = scores[p_][a_], scores[n_][c_]
                    if min(sp, sn) < 0.25:
                        continue
                    nb = (boxes[p_][a_] + boxes[n_][c_]) / 2
                    if len(b) == 0 or iou(nb[None], b[s >= 0.05]).max(initial=0) < tau:
                        b = np.vstack([b, nb]) if len(b) else nb[None]
                        s = np.append(s, alpha * min(sp, sn))
        B.append(b)
        S.append(s)
    return evaluate(B, S)


print(f"frames {len(frames)}, clips {clip.max() + 1}, GT {sum(len(g) for g in GT)}, dets {sum(len(b) for b in boxes)}")
base = run("none", 0, 0, 0, False)
print(f"{'variant':40} {'AP50':>7} {'AP50-95':>8} {'R@F1':>6} {'P@F1':>6}")
print(f"{'none (own evaluator)':40} {base[0]:7.4f} {base[1]:8.4f} {base[2]:6.3f} {base[3]:6.3f}")
for mode, K, tau, alpha, causal in itertools.product(["pullup", "blend", "fill"], [1, 2, 3], [0.3, 0.5],
                                                     [0.5, 0.8], [True, False]):
    if mode == "fill" and causal:
        continue
    r = run(mode, K, tau, alpha, causal)
    tag = f"{mode} K={K} tau={tau} a={alpha} {'causal' if causal else 'two-sided'}"
    print(f"{tag:40} {r[0]:7.4f} {r[1]:8.4f} {r[2]:6.3f} {r[3]:6.3f}   (dAP50 {r[0]-base[0]:+.4f}, dAP {r[1]-base[1]:+.4f})", flush=True)
