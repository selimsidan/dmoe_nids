"""Gate load balancing and optional dataset-identity supervision losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def load_balance_penalty(gate_weights: torch.Tensor) -> torch.Tensor:
    """Coefficient-of-variation penalty on mean per-expert utilization
    within a batch: 0 when every dataset-expert gets equal average gate
    weight, growing as utilization concentrates onto a subset (gate
    collapse).
    """
    mean_utilization = gate_weights.mean(dim=0)
    cv = mean_utilization.std() / (mean_utilization.mean() + 1e-12)
    return cv**2


def dataset_aux_loss(gate_weights: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
    """CE(gate_weights, dataset_id): the router objective of baseline B3.

    Only B3 (``gate_supervision: damex``) uses it; every other condition
    trains its gate from the task loss alone. ``gate_weights`` are already
    softmax probabilities, so this is NLL on their log.
    """
    log_probs = torch.log(gate_weights + 1e-12)
    return F.nll_loss(log_probs, dataset_id)
