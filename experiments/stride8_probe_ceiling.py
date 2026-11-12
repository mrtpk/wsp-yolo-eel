"""Ceiling test: is ANY linear stride-8 eel row able to recover small eels without flooding false positives?

Fits a class-weighted logistic regression (a probe, not an edit we deploy) on the early-video cache and
traces held-out (late-video) TP-vs-FP curves for the current row and the probe row, both under a bias sweep.
FPs are reported two ways: all (end-to-end inference, no NMS) and background-only (what would survive NMS).

Usage: uv run python experiments/stride8_probe_ceiling.py <fit.pt> <test.pt>
"""
import sys
import torch

CHUNK = 262144


def time_half(c, first):
    m = (c["frame"] < c["n_frames"] // 2) if first else (c["frame"] >= c["n_frames"] // 2)
    return {k: (v[m] if torch.is_tensor(v) and v.dim() >= 1 and len(v) == len(m) else v) for k, v in c.items()}


def logits(F16, w, b):
    return torch.cat([F16[i : i + CHUNK].float() @ w for i in range(0, len(F16), CHUNK)]) + b


fit = time_half(torch.load(sys.argv[1]), True)
test = time_half(torch.load(sys.argv[2]), False)
torch.manual_seed(0)

# ---- training set for the probe: all targets + all non-background-trivial negatives + random background ----
lab = fit["labels"]
z0 = logits(fit["feats"], fit["w"].float(), fit["b"].float())
pos = (lab == 1).nonzero().flatten()
hard = ((lab != 1) & ((lab != 0) | (z0 > -6))).nonzero().flatten()  # duplicates, neighbours, hard background
easy = ((lab == 0) & (z0 <= -6)).nonzero().flatten()
easy = easy[torch.randperm(len(easy))[:200_000]]
idx = torch.cat([pos, hard, easy])
X = fit["feats"][idx].float()
y = (lab[idx] == 1).float()
# weights: targets up-weighted; easy negatives re-weighted to stand for the full background population
wt = torch.ones(len(idx))
wt[: len(pos)] = 50.0
wt[len(pos) + len(hard) :] = ((lab == 0) & (z0 <= -6)).sum() / len(easy)
wt = wt / wt.mean()
print(f"probe training: {len(pos)} targets, {len(hard)} hard negatives, {len(easy)} sampled easy negatives")

w = fit["w"].float().clone().requires_grad_(True)
b = fit["b"].float().clone().requires_grad_(True)
opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")


def closure():
    opt.zero_grad()
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, reduction="none") * wt).mean()
    loss = loss + 1e-4 * (w - fit["w"].float()).pow(2).sum()
    loss.backward()
    return loss


opt.step(closure)
w, b = w.detach(), b.detach()
print(f"probe row: |w|={w.norm():.2f} (current {fit['w'].norm():.2f}), cos(probe, current)={torch.nn.functional.cosine_similarity(w, fit['w'].float(), 0):.3f}")


def curve(name, c, w, b):
    z = logits(c["feats"], w, b)
    lab = c["labels"]
    t = lab == 1
    T = int(t.sum())
    print(f"\n{name}: held-out targets={T}")
    print(f"  {'thr(logit)':>10} {'TP':>5} {'recall':>7} {'FP all':>7} {'FP bg-only':>10}")
    # sweep thresholds equivalent to bias shifts, report at matched recall levels
    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        thr = torch.quantile(z[t], 1 - q).item()
        hit = z >= thr
        print(f"  {thr:10.2f} {int((hit & t).sum()):5} {int((hit & t).sum())/T:7.2f} {int((hit & ~t).sum()):7} {int((hit & (lab == 0)).sum()):10}")


curve("CURRENT row (any bias)", test, test["w"].float(), test["b"].float())
curve("BEST LINEAR probe row (any bias)", test, w, b)
curve("BEST LINEAR probe row on FIT half (optimistic)", fit, w, b)
