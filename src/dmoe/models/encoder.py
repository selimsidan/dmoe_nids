"""Shared encoder: 47 standardized features -> representation ``z``.

The encoder takes only the feature vector; it has no dataset input, so the
representation it produces cannot depend on dataset identity. Roles:
``encoder`` (the shared encoder of every shared-encoder condition),
``gate_encoder`` and ``private_encoder`` (the A3 ablation, where the gate and
each expert own separate encoders; both fall back to ``model.encoder``).

Kinds (``model.encoder.kind``):

* ``mlp``: ``SharedEncoder``, an MLP ending in a ``latent_dim`` projection.
* ``identity``: ``IdentityEncoder``, no parameters; ``z`` is the standardized
  feature vector itself, so its width is the input width (A1). A condition
  with an identity encoder has no Stage A.
"""
from __future__ import annotations

import torch
from copy import deepcopy

from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def resolve_encoder_config(model_cfg: dict, role: str = "encoder") -> dict:
    """Resolve a role-specific encoder config with ``model.encoder`` as fallback."""
    if role not in {"encoder", "gate_encoder", "private_encoder"}:
        raise ValueError(f"Unknown encoder role {role!r}")
    base = deepcopy(model_cfg["encoder"])
    if role != "encoder":
        base.update(deepcopy(model_cfg.get(role, {}) or {}))
    return base


class SharedEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128),
        latent_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
        *,
        kind: str = "mlp",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.kind = str(kind)
        if self.kind != "mlp":
            raise ValueError("encoder kind must be 'mlp'")
        act_cls = _ACTIVATIONS[activation]
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), act_cls(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def last_layer_parameters(self):
        """Parameters updated by ``stage_c_unfreeze=last_layer``."""
        for layer in reversed(self.net):
            if isinstance(layer, nn.Linear):
                return layer.parameters()
        raise RuntimeError("encoder contains no linear projection")


class IdentityEncoder(nn.Module):
    """No encoder: returns the standardized features unchanged (zero parameters).

    Exposes the attributes the pipeline reads from ``SharedEncoder``
    (``input_dim``, ``latent_dim`` as the output width, ``kind``).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(input_dim)
        self.kind = "identity"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def last_layer_parameters(self):
        return iter(())


ENCODER_KINDS = {"mlp", "identity"}


def is_identity_encoder(encoder_cfg: dict) -> bool:
    kind = encoder_cfg.get("kind", "mlp")
    if kind not in ENCODER_KINDS:
        raise ValueError(f"encoder kind must be one of {sorted(ENCODER_KINDS)}, got {kind!r}")
    return kind == "identity"


def encoder_output_dim(input_dim: int, latent_dim: int, encoder_cfg: dict) -> int:
    """Width of the encoder output, known without building the encoder."""
    return int(input_dim) if is_identity_encoder(encoder_cfg) else int(latent_dim)


def build_encoder(input_dim: int, latent_dim: int, encoder_cfg: dict) -> SharedEncoder | IdentityEncoder:
    """Construct the encoder of a ``model.encoder``-style config block."""
    if is_identity_encoder(encoder_cfg):
        return IdentityEncoder(input_dim)
    return SharedEncoder(
        input_dim=input_dim,
        hidden_dims=list(encoder_cfg.get("hidden_dims", [])),
        latent_dim=latent_dim,
        activation=encoder_cfg.get("activation", "relu"),
        dropout=float(encoder_cfg.get("dropout", 0.1)),
        kind=encoder_cfg.get("kind", "mlp"),
    )


class ProbeHead(nn.Module):
    """Stage-A-only linear probe on ``z``; discarded after Stage A."""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)
