"""Weight-space head blending: interpolate the deployed one-to-one head toward the one-to-many head.

YOLO26 trains two identical-shaped heads: one2one_* (deployed, no NMS) and cv2/cv3/cv4 (one-to-many,
trained with denser targets, discarded at deploy). This writes
    one2one_X <- (1 - t) * one2one_X + t * X        for X in cv2 (box), cv3 (class), cv4 (mask coef)
Usage: uv run python experiments/blend_heads.py <in_1cls.pt> <out.pt> <t> [parts=cv2,cv3,cv4]
"""
import sys
import torch

src, dst, t = sys.argv[1], sys.argv[2], float(sys.argv[3])
parts = (sys.argv[4] if len(sys.argv) > 4 else "cv2,cv3,cv4").split(",")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
for key in ("model", "ema"):
    m = ckpt.get(key)
    if m is None:
        continue
    head = m.model[-1]
    with torch.no_grad():
        for part in parts:
            o2m, o2o = getattr(head, part), getattr(head, f"one2one_{part}")
            sd_m, sd_o = o2m.state_dict(), o2o.state_dict()
            new = {}
            for k, v in sd_o.items():
                if v.is_floating_point() and "num_batches" not in k:
                    new[k] = ((1 - t) * v.float() + t * sd_m[k].float()).to(v.dtype)
                else:
                    new[k] = v
            o2o.load_state_dict(new)
    print(f"{key}: blended {parts} with t={t}")
torch.save(ckpt, dst)
print(f"saved {dst}")
