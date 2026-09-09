"""
ResMLP-DP: the ~9,857-parameter residual MLP used as the client-side model.

Forward pass matches the manuscript exactly:
    h1 = ReLU(GN(W1 x + b1))                      W1 in R^{64 x d}
    h2 = h1 + ReLU(GN(W2 h1 + b2))                 W2 in R^{64 x 64}   (residual block)
    h3 = ReLU(W3 h2 + b3)                          W3 in R^{32 x 64}
    z  = W4 h3 + b4                                W4 in R^{1 x 32}
    yhat = sigmoid(z)

GroupNorm (not BatchNorm) is used because BatchNorm mixes statistics across a
batch, which is incompatible with Opacus's per-sample gradient computation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import MODEL, DATA


class ResMLP_DP(nn.Module):
    def __init__(
        self,
        input_dim: int = DATA.input_dim,
        hidden_1: int = MODEL.hidden_1,
        hidden_2: int = MODEL.hidden_2,
        hidden_3: int = MODEL.hidden_3,
        output_dim: int = MODEL.output_dim,
        gn_groups: int = MODEL.group_norm_groups,
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_1)
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=hidden_1)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.gn2 = nn.GroupNorm(num_groups=gn_groups, num_channels=hidden_2)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(hidden_2, hidden_3)
        self.relu3 = nn.ReLU()

        self.linear4 = nn.Linear(hidden_3, output_dim)

        self._param_count = sum(p.numel() for p in self.parameters())

    @property
    def param_count(self) -> int:
        return self._param_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GroupNorm expects (N, C, *) -- with C = feature dim here, treat the
        # feature axis as channels on a length-1 spatial axis.
        h1 = self.relu1(self._group_norm_1d(self.linear1(x), self.gn1))
        h2 = h1 + self.relu2(self._group_norm_1d(self.linear2(h1), self.gn2))
        h3 = self.relu3(self.linear3(h2))
        z = self.linear4(h3)
        return z  # logits -- apply torch.sigmoid(z) for probabilities

    @staticmethod
    def _group_norm_1d(x: torch.Tensor, gn: nn.GroupNorm) -> torch.Tensor:
        # x: (N, C) -> (N, C, 1) -> GroupNorm -> (N, C)
        return gn(x.unsqueeze(-1)).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def build_resmlp_dp() -> ResMLP_DP:
    model = ResMLP_DP()
    return model


def gn_groups_that_divide(num_channels: int, preferred: int = 8) -> int:
    """
    GroupNorm requires num_groups to divide num_channels evenly. The manuscript
    does not state the group count numerically; this helper picks the largest
    divisor of num_channels that is <= preferred, so the architecture stays
    valid even if hidden widths are changed in config.py.
    """
    for g in range(min(preferred, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1
