"""Exact weight-space head merge by WIDENING (not averaging).

Averaging the weights of the one-to-one and one-to-many class branches destroys the model (v2 B/C/D).
Averaging their OUTPUTS helps. This builds a single 128-channel class branch whose output equals
    w * o2o_branch(x) + (1 - w) * o2m_branch(x)
exactly (up to float rounding), then installs it as the deployed one2one_cv3. Optionally also copies the
one-to-many box and mask-coefficient branches into the deployed slots.

Branch layout (per scale): DWConv3x3(64) -> Conv1x1(64->64) -> DWConv3x3(64) -> Conv1x1(64->64) -> Conv2d1x1(64->nc)
Merged:                     DWConv3x3(64->128, groups 64; interleaved) -> Conv1x1(128->128, un-interleave + block-diag)
                            -> DWConv3x3(128) -> Conv1x1(128->128 block-diag) -> Conv2d1x1(128->nc, [w*W1 | (1-w)*W2])
BatchNorms are folded into the convs first so every layer is a plain conv + SiLU.

Usage: uv run python experiments/merge_heads_wide.py <in_1cls.pt> <out.pt> [w=0.5] [--box-o2m] [--mask-o2m]
"""
import copy
import sys

import torch
from torch import nn
from ultralytics.utils.torch_utils import fuse_conv_and_bn

src, dst = sys.argv[1], sys.argv[2]
w = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 0.5
box_o2m, mask_o2m = "--box-o2m" in sys.argv, "--mask-o2m" in sys.argv


def fused(conv_mod):
    """Return (weight, bias) of an ultralytics Conv/DWConv with BN folded in."""
    c = fuse_conv_and_bn(copy.deepcopy(conv_mod.conv), copy.deepcopy(conv_mod.bn))
    return c.weight.detach().float(), c.bias.detach().float()


class ConvAct(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv, self.act = conv, nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


def merge_branch(A, B, w):
    (a0w, a0b), (b0w, b0b) = fused(A[0][0]), fused(B[0][0])  # DW 3x3, (64,1,3,3)
    (a1w, a1b), (b1w, b1b) = fused(A[0][1]), fused(B[0][1])  # PW (64,64,1,1)
    (a2w, a2b), (b2w, b2b) = fused(A[1][0]), fused(B[1][0])  # DW
    (a3w, a3b), (b3w, b3b) = fused(A[1][1]), fused(B[1][1])  # PW
    a4, b4 = A[2], B[2]  # final Conv2d with bias
    Ci, C = a0w.shape[0], a1w.shape[0]  # input channels (64/128/256 by scale), branch width (64)
    nc = a4.weight.shape[0]
    # layer 0: depthwise 3x3 Ci -> 2Ci, output 2g = A filter g, 2g+1 = B filter g
    l0 = nn.Conv2d(Ci, 2 * Ci, 3, 1, 1, groups=Ci, bias=True)
    W0 = torch.empty(2 * Ci, 1, 3, 3)
    W0[0::2], W0[1::2] = a0w, b0w
    B0 = torch.empty(2 * Ci)
    B0[0::2], B0[1::2] = a0b, b0b
    l0.weight.data, l0.bias.data = W0, B0
    # layer 1: 1x1, interleaved 2Ci -> [A C | B C]
    l1 = nn.Conv2d(2 * Ci, 2 * C, 1, bias=True)
    W1 = torch.zeros(2 * C, 2 * Ci, 1, 1)
    W1[:C, 0::2] = a1w
    W1[C:, 1::2] = b1w
    l1.weight.data, l1.bias.data = W1, torch.cat([a1b, b1b])
    # layer 2: depthwise over 2C
    l2 = nn.Conv2d(2 * C, 2 * C, 3, 1, 1, groups=2 * C, bias=True)
    l2.weight.data, l2.bias.data = torch.cat([a2w, b2w]), torch.cat([a2b, b2b])
    # layer 3: block-diagonal 1x1
    l3 = nn.Conv2d(2 * C, 2 * C, 1, bias=True)
    W3 = torch.zeros(2 * C, 2 * C, 1, 1)
    W3[:C, :C], W3[C:, C:] = a3w, b3w
    l3.weight.data, l3.bias.data = W3, torch.cat([a3b, b3b])
    # layer 4: weighted sum of both heads' logits
    l4 = nn.Conv2d(2 * C, nc, 1, bias=True)
    l4.weight.data = torch.cat([w * a4.weight.detach().float(), (1 - w) * b4.weight.detach().float()], 1)
    l4.bias.data = w * a4.bias.detach().float() + (1 - w) * b4.bias.detach().float()
    return nn.Sequential(nn.Sequential(l0, nn.SiLU()), nn.Sequential(l1, nn.SiLU()), nn.Sequential(l2, nn.SiLU()), nn.Sequential(l3, nn.SiLU()), l4)


ckpt = torch.load(src, map_location="cpu", weights_only=False)
for key in ("model", "ema"):
    m = ckpt.get(key)
    if m is None:
        continue
    m = m.float().eval()
    head = m.model[-1]
    merged = nn.ModuleList(merge_branch(head.one2one_cv3[i], head.cv3[i], w) for i in range(head.nl))
    # exactness check on random features
    with torch.no_grad():
        for i in range(head.nl):
            x = torch.randn(2, merged[i][0][0].in_channels, 20, 32)  # type: ignore[index]
            ref = w * head.one2one_cv3[i](x) + (1 - w) * head.cv3[i](x)
            err = (merged[i](x) - ref).abs().max().item()
            print(f"{key} scale {i}: max |merged - average| = {err:.2e}")
    head.one2one_cv3 = merged
    if box_o2m:
        head.one2one_cv2 = copy.deepcopy(head.cv2)
    if mask_o2m:
        head.one2one_cv4 = copy.deepcopy(head.cv4)
    ckpt[key] = m
ckpt["merge_info"] = dict(w=w, box_o2m=box_o2m, mask_o2m=mask_o2m, note="one2one_cv3 = exact widened merge of o2o and o2m class branches")
torch.save(ckpt, dst)
print(f"saved {dst}")
