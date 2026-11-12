"""Weight-space interpolation between the base model and a fine-tuned copy (WiSE-FT / task vectors).

    theta(alpha) = theta_base + alpha * (theta_ft - theta_base)          for the selected parameter groups

Groups (by state-dict key prefix inside the head, layer 23):
    all      every parameter and BN buffer
    cls      class branches (cv3, one2one_cv3)
    box      box branches (cv2, one2one_cv2)
    backbone everything outside the head
Parameters outside the selected groups keep the base values.

Usage: uv run python experiments/wise_ft.py <base.pt> <ft.pt> <out.pt> <alpha> [groups, comma-separated; default all]
"""
import sys

import torch

base_p, ft_p, out_p, alpha = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
groups = set((sys.argv[5] if len(sys.argv) > 5 else "all").split(","))
HEAD = "model.23."


def group_of(k):
    if not k.startswith(HEAD):
        return "backbone"
    rest = k[len(HEAD):]
    if rest.startswith(("cv3.", "one2one_cv3.")):
        return "cls"
    if rest.startswith(("cv2.", "one2one_cv2.")):
        return "box"
    return "headother"


base = torch.load(base_p, map_location="cpu", weights_only=False)
ft = torch.load(ft_p, map_location="cpu", weights_only=False)
mb = (base.get("ema") or base["model"]).float()
mf = (ft.get("ema") or ft["model"]).float()
sb, sf = mb.state_dict(), mf.state_dict()
assert sb.keys() == sf.keys(), "architectures differ"
new, moved = {}, 0
for k, v in sb.items():
    g = group_of(k)
    if v.is_floating_point() and ("all" in groups or g in groups):
        new[k] = v + alpha * (sf[k] - v)
        moved += v.numel()
    else:
        new[k] = v
mb.load_state_dict(new)
base["model"], base["ema"] = mb.half(), None
base["wise_ft"] = dict(ft=ft_p, alpha=alpha, groups=sorted(groups))
torch.save(base, out_p)
print(f"saved {out_p}: alpha={alpha} groups={sorted(groups)} interpolated {moved:,} values")
