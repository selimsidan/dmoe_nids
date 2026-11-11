"""Stage-A objective: weighted cross-entropy plus balanced SupCon.

Stage A trains the shared encoder with a linear probe (cross-entropy with
square-root inverse-frequency class weights) plus ``weight`` times a balanced
supervised contrastive loss on the encoder output ``z``. No projection head
is used. ``objective: ce`` drops the contrastive term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import ProbeHead


def _normalized_similarity(z: torch.Tensor, temperature: float) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("embeddings must have shape (batch, latent_dim)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = F.normalize(z, dim=1)
    return normalized @ normalized.T / temperature


class BalancedSupervisedContrastiveLoss(nn.Module):
    """Class-averaged SupCon denominator for long-tailed batches.

    Every represented class contributes the mean exponentiated similarity of
    its members to the denominator rather than a sum proportional to its batch
    frequency, so frequent classes do not dominate the contrast. Batches
    themselves are not class-balanced: they are plain block-shuffled rows.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarity = _normalized_similarity(z, self.temperature)
        batch = len(labels)
        identity = torch.eye(batch, dtype=torch.bool, device=z.device)
        positives = labels[:, None].eq(labels[None, :]) & ~identity
        valid = positives.any(dim=1)
        if not valid.any():
            return z.sum() * 0.0

        # Subtract a row constant before exponentiation for numerical stability.
        stable = similarity - similarity.masked_fill(identity, -torch.inf).max(dim=1, keepdim=True).values
        exp_similarity = stable.exp().masked_fill(identity, 0.0)
        class_ids = torch.unique(labels)
        class_terms = []
        for class_id in class_ids:
            members = labels.eq(class_id)[None, :].expand(batch, -1) & ~identity
            counts = members.sum(dim=1)
            class_mean = (exp_similarity * members).sum(dim=1) / counts.clamp_min(1)
            class_terms.append(torch.where(counts > 0, class_mean, torch.zeros_like(class_mean)))
        class_terms = torch.stack(class_terms, dim=1)
        denominator = class_terms.sum(dim=1).clamp_min(torch.finfo(z.dtype).tiny)
        own_columns = labels[:, None].eq(class_ids[None, :]).to(class_terms.dtype)
        own_class_mean = (class_terms * own_columns).sum(dim=1).clamp_min(torch.finfo(z.dtype).tiny)
        per_anchor = -(own_class_mean.log() - denominator.log())
        return per_anchor[valid].mean()


def representation_config(config: dict) -> dict:
    """Resolved Stage-A representation settings (stored in the Stage-A checkpoint)."""
    values = dict(config.get("training", {}).get("representation", {}) or {})
    resolved = {
        "objective": values.get("objective", "ce"),
        "sampling": values.get("sampling", "legacy"),
        "class_weighting": values.get("class_weighting", "legacy"),
        "weight": float(values.get("weight", 0.1)),
        "temperature": float(values.get("temperature", 0.1)),
    }
    if resolved["sampling"] != "legacy":
        raise ValueError("training.representation.sampling must be 'legacy' (block-shuffled batches)")
    if resolved["class_weighting"] not in {"legacy", "none"}:
        raise ValueError("training.representation.class_weighting must be legacy or none")
    return resolved


class StageARepresentationObjective(nn.Module):
    """Probe cross-entropy plus the configured representation objective."""

    VALID_OBJECTIVES = {"ce", "balanced_supcon"}

    def __init__(self, latent_dim: int, num_classes: int, config: dict) -> None:
        super().__init__()
        self.config = representation_config(config)
        self.objective = self.config["objective"]
        if self.objective not in self.VALID_OBJECTIVES:
            raise ValueError(
                f"Unknown representation objective {self.objective!r}; "
                f"expected one of {sorted(self.VALID_OBJECTIVES)}"
            )
        self.classifier = ProbeHead(latent_dim, num_classes)
        self.metric: nn.Module | None
        if self.objective == "balanced_supcon":
            self.metric = BalancedSupervisedContrastiveLoss(self.config["temperature"])
        else:
            self.metric = None
        self.num_classes = int(num_classes)

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        logits = self.classifier(z)
        ce = F.cross_entropy(logits, labels, weight=class_weights)
        metric = ce.new_zeros(())
        if self.metric is not None:
            metric = self.metric(z, labels)
        total = ce + self.config["weight"] * metric
        return total, {"ce": ce, "metric": metric, "total": total}, logits
