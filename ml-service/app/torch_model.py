"""Small MLP. Defined in its own module so both train/ and app/ import the exact same
architecture - loading a state_dict into a differently shaped class is a classic
production footgun."""

import torch
from torch import nn


class FantasyMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
