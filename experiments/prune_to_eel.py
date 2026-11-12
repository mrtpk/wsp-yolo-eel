"""Experiment 1: prune the 22-class head to the single live class (American_Eel, index 21).

Pure weight edit, no training: slice row 21 out of every nc-sized output conv
(cv3, one2one_cv3, proto.semseg) and update the class metadata.

Usage: uv run python experiments/prune_to_eel.py <in.pt> <out.pt>
"""
import sys
import torch
from torch import nn

KEEP = 21

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)  # our own checkpoint


def slice_conv(conv: nn.Conv2d, rows: list[int]) -> nn.Conv2d:
    new = nn.Conv2d(conv.in_channels, len(rows), conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)  # type: ignore[arg-type]
    new = new.to(conv.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(conv.weight[rows])
        if conv.bias is not None:
            new.bias.copy_(conv.bias[rows])
    return new


for key in ("model", "ema"):
    model = ckpt.get(key)
    if model is None:
        continue
    head = model.model[-1]
    for branch in (head.cv3, head.one2one_cv3):
        for seq in branch:
            seq[-1] = slice_conv(seq[-1], [KEEP])
    if getattr(head.proto, "semseg", None) is not None:
        head.proto.semseg[-1] = slice_conv(head.proto.semseg[-1], [KEEP])
    head.nc = 1
    head.no = head.nc + head.reg_max * 4
    model.nc = 1
    model.names = {0: "American_Eel"}
    model.yaml["nc"] = 1
    if hasattr(model, "args") and isinstance(model.args, dict):
        model.args["single_cls"] = False

ckpt["train_args"] = {**ckpt.get("train_args", {}), "data": "eel_1cls"}
torch.save(ckpt, dst)
print(f"saved {dst}")
