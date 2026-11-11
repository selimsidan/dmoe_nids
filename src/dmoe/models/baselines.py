"""Capacity-matched pooled MLP (condition B2).

One dataset-blind classifier: M's shared encoder followed by a two-hidden-layer
head whose widths are chosen deterministically so the total parameter count
matches the dataset-expert MoE (see ``training.model_utils.build_matched_dense_model``).
"""
from __future__ import annotations

import torch
from torch import nn

from .encoder import SharedEncoder


class MatchedDenseHead(nn.Module):
    """Two-hidden-layer dense comparator resolved against an MoE budget.

    ``match_info`` is intentionally stored on the module (but not in the
    state_dict) so runners and reports can record the exact integer solution
    and residual instead of describing an approximate match as exact.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, int],
        dropout: float,
        match_info: dict,
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(latent_dim, first),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second, num_classes),
        )
        self.hidden_dims = (first, second)
        self.match_info = dict(match_info)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MatchedDenseClassifier(nn.Module):
    """Shared encoder plus a deterministic capacity/compute-matched head."""

    def __init__(self, encoder: SharedEncoder, head: MatchedDenseHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def match_info(self) -> dict:
        return self.head.match_info

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {"z": z, "logits": self.head(z)}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["logits"].argmax(dim=1)
