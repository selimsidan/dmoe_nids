"""Gate: latent ``z`` -> softmax weights over the experts.

The gate never receives dataset identity as an input. How it is trained is
selected in Stage C: from the task loss (M, B1, A1 to A3) or only from a
dataset-ID cross-entropy (the B3 router).
"""
from __future__ import annotations

import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self, latent_dim: int, num_datasets: int, hidden_dims: list[int] = ()) -> None:
        super().__init__()
        if not hidden_dims:
            self.net = nn.Linear(latent_dim, num_datasets)
        else:
            dims = [latent_dim, *hidden_dims]
            layers: list[nn.Module] = []
            for in_d, out_d in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(in_d, out_d), nn.ReLU()]
            layers.append(nn.Linear(dims[-1], num_datasets))
            self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        return torch.softmax(logits, dim=-1)
