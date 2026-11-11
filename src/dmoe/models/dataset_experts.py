"""Expert banks for the MoE conditions.

``DatasetExpertBank`` holds one ``Expert`` per active dataset (M, A1, A2, B3)
or, for the generic MoE (B1), the same number of experts with no dataset
assigned (named ``expert_0`` ... ``expert_3``). Every expert predicts the full
22-class vocabulary.

``forward`` is the dense path and runs every expert on every row.
``forward_selected`` is the top-1 path used by B3: rows are grouped by their
selected expert and only that expert runs for them.
"""
from __future__ import annotations

import torch
from torch import nn


class Expert(nn.Module):
    """One expert for one dataset: z -> num_classes logits (full task vocabulary)."""

    def __init__(self, latent_dim: int, num_classes: int, hidden_dims: list[int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        dims = [latent_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DatasetExpertBank(nn.Module):
    """Holds one Expert per active dataset.

    ``forward`` is dense; ``forward_selected`` is true grouped top-1 dispatch.
    Number of experts is always
    `len(dataset_names)`, derived at construction time -- never hardcoded.
    """

    def __init__(
        self,
        dataset_names: list[str],
        latent_dim: int,
        num_classes: int,
        hidden_dims: list[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dataset_names = list(dataset_names)
        self.num_classes = num_classes
        self.experts = nn.ModuleList(
            [Expert(latent_dim, num_classes, hidden_dims, dropout) for _ in dataset_names]
        )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def expert_names(self) -> list[str]:
        """Semantic alias for the historical ``dataset_names`` field."""
        return self.dataset_names

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, num_datasets, num_classes)."""
        batch = z.shape[0]
        assert batch > 0
        outputs = [expert(z) for expert in self.experts]  # every expert runs on the full batch, unconditionally
        stacked = torch.stack(outputs, dim=1)
        assert stacked.shape == (batch, self.num_experts, self.num_classes)
        return stacked

    def forward_selected(self, z: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Run exactly one expert per row and return ``(batch, classes)`` logits."""
        batch = z.shape[0]
        if selected_experts.shape != (batch,):
            raise ValueError("selected_experts must have shape (batch,)")
        if batch == 0:
            raise ValueError("top-1 dispatch requires a non-empty batch")
        if int(selected_experts.min()) < 0 or int(selected_experts.max()) >= self.num_experts:
            raise ValueError("selected_experts contains an index outside the expert bank")
        output = z.new_zeros((batch, self.num_classes))
        for expert_id, expert in enumerate(self.experts):
            rows = torch.nonzero(selected_experts == expert_id, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            logits = expert(z.index_select(0, rows))
            output = output.index_copy(0, rows, logits)
        return output
