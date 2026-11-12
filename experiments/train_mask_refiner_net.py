"""Mask boundary refiner network (see train_mask_refiner.py)."""
import torch
from torch import nn


class Refiner(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        layers, c = [], 3
        for d in (1, 2, 4, 8, 1):
            layers += [nn.Conv2d(c, w, 3, padding=d, dilation=d), nn.SiLU()]
            c = w
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(w, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, gray, logit, inbox):
        x = torch.stack([gray, (logit / 8).clamp(-3, 3), inbox], 1)
        return logit + self.out(self.body(x))[:, 0]
