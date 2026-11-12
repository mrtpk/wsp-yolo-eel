"""Ablation: silence the stride-32 eel classifier (bias -> -100 in cv3[2] and one2one_cv3[2]).

If the stride-32 branch is truly dead, predictions must be unchanged.
Usage: uv run python experiments/ablate_stride32.py <in_1cls.pt> <out.pt>
"""
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
for key in ("model", "ema"):
    m = ckpt.get(key)
    if m is None:
        continue
    head = m.model[-1]
    assert head.nc == 1
    with torch.no_grad():
        for branch in (head.cv3, head.one2one_cv3):
            print(f"{key} stride-32 eel bias {float(branch[2][-1].bias[0]):.3f} -> -100")
            branch[2][-1].bias.fill_(-100.0)
torch.save(ckpt, dst)
print(f"saved {dst}")
