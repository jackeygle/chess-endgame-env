"""Policy-value network: shared residual trunk, a policy head and a value head.

Policy head: a 1x1 convolution to 320 channels gives, for every from-square (the spatial
position), a logit per (to-square, promotion) pair, i.e. the 64 * 320 = 20480 actions of
`encoding`, in index order from * 320 + to * 5 + promotion. A dense layer to 20480 outputs
would cost ~2.6M parameters; this costs ~21k.

Value head: the expected result for the side to move, in [-1, 1] (tanh), or, with
`wdl=True`, logits over (win, draw, loss) with v = p_win - p_loss for the search.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.encoding import N_ACTIONS, N_PLANES


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, blocks: int = 6, channels: int = 64, wdl: bool = False):
        super().__init__()
        self.wdl = wdl
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 320, 1)
        self.value = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1),
                                   nn.ReLU(), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(),
                                   nn.Linear(64, 3 if wdl else 1))

    def forward(self, x):
        h = self.trunk(self.stem(x))
        # (B, 320, 8, 8) -> (B, 64 from-squares, 320) -> (B, 20480); square = rank * 8 + file
        logits = self.policy(h).flatten(2).transpose(1, 2).reshape(x.shape[0], N_ACTIONS)
        v = self.value(h)
        return logits, (v if self.wdl else torch.tanh(v).squeeze(1))

    def scalar_value(self, v):
        """Value in [-1, 1] for the search, from either head type."""
        if self.wdl:
            p = torch.softmax(v, dim=1)
            return p[:, 0] - p[:, 2]
        return v


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
