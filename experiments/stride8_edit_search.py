"""Search edits of the stride-8 eel classifier row on cached features (exact, no network run).

E1: bias shift        b' = b + beta
E2: rank-1 LDA edit   w' = w + alpha * d,  b' = b + beta,  d = (Sigma_neg + lam I)^-1 (mu_target - mu_neg)

Fit d on the first half of the video (fit cache), report on the second half (test cache): no shared clips.
Usage: uv run python experiments/stride8_edit_search.py <fit.pt> <test.pt> <out_edits.pt>
"""
import sys
import math
import torch

fit, test = torch.load(sys.argv[1]), torch.load(sys.argv[2])
out = sys.argv[3]


def time_half(c, first):
    """Leak-free split: keep only frames in the first (or second) half of the video timeline."""
    m = (c["frame"] < c["n_frames"] // 2) if first else (c["frame"] >= c["n_frames"] // 2)
    return {k: (v[m] if torch.is_tensor(v) and v.dim() >= 1 and len(v) == len(m) else v) for k, v in c.items()}


fit, test = time_half(fit, True), time_half(test, False)  # fit: early video, test: late video
CHUNK = 262144


def logits(F16, w, b):
    """F16 @ w + b in float32 chunks (keeps memory small)."""
    return torch.cat([F16[i : i + CHUNK].float() @ w for i in range(0, len(F16), CHUNK)]) + b


def moments(F16, mask):
    """Mean and covariance of the masked rows, accumulated in float64 chunks."""
    n, s, ss = 0, torch.zeros(64, dtype=torch.float64), torch.zeros(64, 64, dtype=torch.float64)
    for i in range(0, len(F16), CHUNK):
        x = F16[i : i + CHUNK][mask[i : i + CHUNK]].double()
        n += len(x)
        s += x.sum(0)
        ss += x.T @ x
    mu = s / n
    return mu.float(), (ss / n - torch.outer(mu, mu)).float()


THR = math.log(0.25 / 0.75)  # logit of conf 0.25
BINS = [0, 60, 100, 150, 2000]


def evaluate(c, w, b):
    z = logits(c["feats"], w, b)
    lab, ln = c["labels"], c["lengths"]
    hit = z > THR
    t = lab == 1
    res = {"tp8": int((hit & t).sum()), "targets": int(t.sum())}
    res["fp_bg"] = int((hit & (lab == 0)).sum())
    res["fp_dup16"] = int((hit & (lab == 2)).sum())
    res["fp_neigh"] = int((hit & (lab == 3)).sum())
    res["fp"] = res["fp_bg"] + res["fp_dup16"] + res["fp_neigh"]
    res["bins"] = []
    for lo, hi in zip(BINS, BINS[1:]):
        m = t & (ln >= lo) & (ln < hi)
        res["bins"].append(f"{int((hit & m).sum())}/{int(m.sum())}")
    return res


def fmt(name, r, r0):
    return (f"{name:28} fit: tp8 {r[0]['tp8']:4}/{r[0]['targets']} fp {r[0]['fp']:4} (Δtp {r[0]['tp8']-r0[0]['tp8']:+4} Δfp {r[0]['fp']-r0[0]['fp']:+4})"
            f" | test: tp8 {r[1]['tp8']:4}/{r[1]['targets']} fp {r[1]['fp']:4} [bg {r[1]['fp_bg']} dup16 {r[1]['fp_dup16']} neigh {r[1]['fp_neigh']}]"
            f" (Δtp {r[1]['tp8']-r0[1]['tp8']:+4} Δfp {r[1]['fp']-r0[1]['fp']:+4})  bins {r[1]['bins']}")


w0, b0 = fit["w"].float(), fit["b"].float().squeeze()
base = [evaluate(c, w0, b0) for c in (fit, test)]
print(f"stride-8 row: |w|={w0.norm():.2f}  b={b0:.3f}   size bins {BINS}")
print(fmt("baseline", base, base))

# ---- E1: bias shift ----
results = []
for beta in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    r = [evaluate(c, w0, b0 + beta) for c in (fit, test)]
    print(fmt(f"E1 bias +{beta}", r, base))
    results.append(("E1", beta, 0.0, r))

# ---- E2: LDA direction ----
lab = fit["labels"]
mu_p, _ = moments(fit["feats"], lab == 1)
mu_n, cov = moments(fit["feats"], lab != 1)
lam = 1e-2 * cov.diagonal().mean()
d = torch.linalg.solve(cov + lam * torch.eye(64), mu_p - mu_n)
d = d / (d @ (mu_p - mu_n))  # unit separation: d.(mu_p - mu_n) = 1
cos = torch.nn.functional.cosine_similarity(d, w0, dim=0)
print(f"\nLDA direction: cos(d, w0) = {cos:.3f}; current row's mean-logit gap target-vs-neg = {(mu_p - mu_n) @ w0:.2f}")
for alpha in [1, 2, 4, 6, 8]:
    for beta in [-1.0, -0.5, 0.0, 0.5]:
        # keep background logit mean fixed: shift bias by -alpha*d.mu_n, then add beta
        b = b0 - alpha * (d @ mu_n) + beta
        r = [evaluate(c, w0 + alpha * d, b) for c in (fit, test)]
        print(fmt(f"E2 a={alpha} beta={beta:+}", r, base))
        results.append(("E2", beta, alpha, r))

torch.save(dict(d=d, mu_n=mu_n, w0=w0, b0=b0, results=results), out)
