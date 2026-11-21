"""The one place that maps ``architecture`` to concrete ``nn.Module`` wiring.

==================================================  =========  =====================
architecture                                        condition  model
==================================================  =========  =====================
moe_dataset_class_conditional                       M          class-conditional MoE
matched_dense                                       B2         encoder + matched MLP head
moe_dataset_damex (gate.routing: top1)              B3         top-1 routed baseline
moe_dataset_no_encoder_class_conditional            A1         class-conditional MoE, no encoder
moe_dataset_uniform                                 A2         uniform average of the experts
moe_dataset_private_encoders_class_conditional      A3         class-conditional MoE, private encoders
==================================================  =========  =====================

Gate and expert input widths come from the encoder's output width
(``encoder.latent_dim``): 64 for the MLP encoder, 47 for A1's identity encoder.

B4 (``pooled_xgboost``) is not a torch model; see ``xgboost_baseline``.
"""
from __future__ import annotations

import torch
from torch import nn

from ..models.baselines import MatchedDenseClassifier, MatchedDenseHead
from ..models.dataset_experts import DatasetExpertBank
from ..models.encoder import SharedEncoder, is_identity_encoder, resolve_encoder_config
from ..models.gate import Gate, UniformGate
from ..models.moe import ClassConditionalMoEDatasetNIDS, MoEDatasetNIDS
from ..models.private_encoder_experts import PrivateEncoderExpertBank

CLASS_CONDITIONAL_ARCHITECTURES = {
    "moe_dataset_class_conditional",
    "moe_dataset_no_encoder_class_conditional",
    "moe_dataset_private_encoders_class_conditional",
}
NO_ENCODER_ARCHITECTURES = {"moe_dataset_no_encoder_class_conditional"}
PRIVATE_ENCODER_ARCHITECTURES = {"moe_dataset_private_encoders_class_conditional"}
UNIFORM_GATE_ARCHITECTURES = {"moe_dataset_uniform"}
DATASET_MOE_ARCHITECTURES = {
    "moe_dataset_uniform",
    "moe_dataset_damex",
    "moe_dataset_class_conditional",
    "moe_dataset_no_encoder_class_conditional",
    "moe_dataset_private_encoders_class_conditional",
}


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _linear_macs(module: nn.Module) -> int:
    """Static per-sample MACs for Linear weights; nonlinearities are reported separately."""
    return sum(layer.weight.numel() for layer in module.modules() if isinstance(layer, nn.Linear))


def _resolve_dense_hidden_dims(
    latent_dim: int,
    num_classes: int,
    target: int,
    axis: str,
    reference_hidden_dims: list[int],
    max_width: int,
) -> tuple[tuple[int, int], int]:
    """Find the deterministic closest two-layer integer architecture."""
    if target <= 0 or max_width < 1:
        raise ValueError("dense matching target and max_width must be positive")
    best_key = None
    best = None
    reference_second = reference_hidden_dims[-1] if reference_hidden_dims else latent_dim
    reference_first = reference_hidden_dims[0] if reference_hidden_dims else latent_dim
    for first in range(1, max_width + 1):
        if axis in {"total_params", "active_params"}:
            constant = (latent_dim + 1) * first + num_classes
            slope = first + 1 + num_classes
        else:
            constant = latent_dim * first
            slope = first + num_classes
        ideal = (target - constant) / slope
        candidates = {
            1,
            max_width,
            first,
            max(1, min(max_width, int(ideal))),
            max(1, min(max_width, int(ideal) + 1)),
        }
        for second in sorted(candidates):
            # Preserve the expert family's pyramidal shape; an exact count
            # obtained via a tiny first layer and enormous second layer would
            # be a materially different architecture rather than a fair widen.
            if second > first:
                continue
            params = (latent_dim + 1) * first + (first + 1) * second + (second + 1) * num_classes
            macs = latent_dim * first + first * second + second * num_classes
            value = params if axis in {"total_params", "active_params"} else macs
            key = (
                abs(value - target),
                abs(second - reference_second),
                abs(first - reference_first),
                first,
                second,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (first, second, value)
    assert best is not None
    return (best[0], best[1]), best[2]


def build_matched_dense_model(
    encoder: SharedEncoder,
    dataset_names: list[str],
    class_names: list[str],
    model_cfg: dict,
) -> MatchedDenseClassifier:
    """Build a dense head matched to the configured full-expert MoE reference."""
    dense_cfg = model_cfg.get("dense_match", {})
    axis = dense_cfg.get("axis", "total_params")
    if axis not in {"total_params", "active_params", "forward_macs"}:
        raise ValueError("model.dense_match.axis must be total_params, active_params, or forward_macs")
    reference_routing = dense_cfg.get(
        "reference_routing", "dense" if axis == "total_params" else "top1"
    )
    if reference_routing not in {"dense", "top1"}:
        raise ValueError("model.dense_match.reference_routing must be dense or top1")
    bank = build_expert_bank(
        "full", dataset_names, model_cfg["latent_dim"], len(class_names), model_cfg
    )
    gate = Gate(model_cfg["latent_dim"], len(dataset_names), model_cfg["gate"]["hidden_dims"])
    if not bank.experts:
        raise ValueError("matched_dense requires at least one reference expert")
    gate_params, gate_macs = _parameter_count(gate), _linear_macs(gate)
    expert_params = [_parameter_count(expert) for expert in bank.experts]
    expert_macs = [_linear_macs(expert) for expert in bank.experts]
    if reference_routing == "dense":
        active_params = gate_params + sum(expert_params)
        forward_macs = gate_macs + sum(expert_macs)
    else:
        active_params = gate_params + max(expert_params)
        forward_macs = gate_macs + max(expert_macs)
    targets = {
        "total_params": (
            gate_params + sum(expert_params)
            + int(dense_cfg.get("extra_reference_parameters", 0))
        ),
        "active_params": (
            active_params + int(dense_cfg.get("extra_reference_active_parameters", 0))
        ),
        "forward_macs": (
            forward_macs + int(dense_cfg.get("extra_reference_forward_macs", 0))
        ),
    }
    target = targets[axis]
    hidden_dims, achieved = _resolve_dense_hidden_dims(
        model_cfg["latent_dim"],
        len(class_names),
        target,
        axis,
        list(model_cfg["expert"]["hidden_dims"]),
        int(dense_cfg.get("max_width", 2048)),
    )
    relative_error = abs(achieved - target) / target
    tolerance = float(dense_cfg.get("tolerance", 0.005))
    if relative_error > tolerance:
        raise ValueError(
            f"Could not match {axis}: target={target}, achieved={achieved}, "
            f"relative_error={relative_error:.6f} exceeds tolerance={tolerance:.6f}"
        )
    info = {
        "axis": axis,
        "reference_routing": reference_routing,
        "target": target,
        "achieved": achieved,
        "residual": achieved - target,
        "relative_error": relative_error,
        "hidden_dims": list(hidden_dims),
        "reference_total_downstream_params": targets["total_params"],
        "reference_active_downstream_params": targets["active_params"],
        "reference_forward_downstream_macs": targets["forward_macs"],
    }
    head = MatchedDenseHead(
        model_cfg["latent_dim"],
        len(class_names),
        hidden_dims,
        dropout=float(model_cfg["expert"].get("dropout", 0.1)),
        match_info=info,
    )
    return MatchedDenseClassifier(encoder, head)


def bank_kind_for_architecture(architecture: str) -> str:
    if architecture in PRIVATE_ENCODER_ARCHITECTURES:
        return "private_encoder"
    return "full"


def build_expert_bank(
    bank_kind: str,
    dataset_names: list[str],
    latent_dim: int,
    num_classes: int,
    model_cfg: dict,
    input_dim: int | None = None,
) -> nn.Module:
    if bank_kind == "full":
        expert_cfg = model_cfg["expert"]
        return DatasetExpertBank(
            dataset_names,
            latent_dim,
            num_classes,
            hidden_dims=expert_cfg["hidden_dims"],
            dropout=expert_cfg["dropout"],
        )
    if bank_kind == "private_encoder":
        if input_dim is None:
            raise ValueError("private_encoder banks require input_dim")
        encoder_cfg = resolve_encoder_config(model_cfg, "private_encoder")
        expert_cfg = model_cfg["expert"]
        return PrivateEncoderExpertBank(
            dataset_names=dataset_names,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_classes=num_classes,
            encoder_hidden_dims=encoder_cfg["hidden_dims"],
            expert_hidden_dims=expert_cfg["hidden_dims"],
            activation=encoder_cfg["activation"],
            encoder_dropout=encoder_cfg["dropout"],
            expert_dropout=expert_cfg["dropout"],
            encoder_config=encoder_cfg,
        )
    raise ValueError(
        f"Unknown bank_kind '{bank_kind}'. Expected 'full' or 'private_encoder'."
    )


def initialize_private_expert_encoders(bank: nn.Module, stage_a_encoder: SharedEncoder) -> None:
    """Clone Stage-A weights when the bank owns full private encoders."""
    if isinstance(bank, PrivateEncoderExpertBank):
        bank.initialize_encoders(stage_a_encoder)


def model_encoder_modules(model: MoEDatasetNIDS) -> list[SharedEncoder]:
    """All encoder modules controlled by ``training.stage_c_unfreeze``."""
    encoders = [model.encoder]
    if isinstance(model.expert_bank, PrivateEncoderExpertBank):
        encoders.extend(model.expert_bank.encoders)
    return encoders


def expert_train_params(bank: nn.Module, dataset_idx: int) -> list[torch.nn.Parameter]:
    """Parameters optimized in Stage B for expert ``dataset_idx``.

    For a shared-encoder bank this is the expert head; for the private-encoder
    bank (A3) it is the expert's private encoder plus its head.
    """
    return list(bank.experts[dataset_idx].parameters())


def expert_forward_one(bank: nn.Module, dataset_idx: int, representation: torch.Tensor) -> torch.Tensor:
    """Run only expert ``dataset_idx`` (Stage B trains one expert at a time)."""
    return bank.experts[dataset_idx](representation)


def apply_stage_c_trainability(model: MoEDatasetNIDS, config: dict) -> None:
    """Freeze encoders (``stage_c_unfreeze``) and experts (``stage_c.freeze_experts``)."""
    mode = config["training"].get("stage_c_unfreeze", "all")
    for encoder in model_encoder_modules(model):
        for parameter in encoder.parameters():
            parameter.requires_grad_(mode == "all")
        if mode == "last_layer":
            for parameter in encoder.last_layer_parameters():
                parameter.requires_grad_(True)
        elif mode not in {"all", "none"}:
            raise ValueError(f"Unknown stage_c_unfreeze mode: {mode}")
    if bool(config["training"].get("stage_c", {}).get("freeze_experts", False)):
        for parameter in model.expert_bank.parameters():
            parameter.requires_grad_(False)


def check_encoder_matches_architecture(architecture: str, model_cfg: dict) -> None:
    """The no-encoder architecture needs ``model.encoder.kind: identity`` and nothing else may use it."""
    identity = is_identity_encoder(model_cfg["encoder"])
    if identity != (architecture in NO_ENCODER_ARCHITECTURES):
        raise ValueError(
            f"architecture {architecture!r} with model.encoder.kind={model_cfg['encoder'].get('kind', 'mlp')!r}: "
            f"the identity encoder is used by, and only by, {sorted(NO_ENCODER_ARCHITECTURES)}"
        )


def expert_input_dim(architecture: str, encoder_output_width: int, model_cfg: dict) -> int:
    """Width the experts read: the encoder output, or a private encoder's latent (A3)."""
    if bank_kind_for_architecture(architecture) == "private_encoder":
        return int(model_cfg["latent_dim"])
    return int(encoder_output_width)


def build_model(
    architecture: str,
    encoder: SharedEncoder,
    dataset_names: list[str],
    class_names: list[str],
    model_cfg: dict,
) -> MoEDatasetNIDS:
    check_encoder_matches_architecture(architecture, model_cfg)
    bank_kind = bank_kind_for_architecture(architecture)
    bank = build_expert_bank(
        bank_kind,
        list(dataset_names),
        expert_input_dim(architecture, encoder.latent_dim, model_cfg),
        len(class_names),
        model_cfg,
        input_dim=encoder.input_dim,
    )
    if architecture in UNIFORM_GATE_ARCHITECTURES:
        gate = UniformGate(len(dataset_names))
    else:
        gate = Gate(encoder.latent_dim, len(dataset_names), model_cfg["gate"]["hidden_dims"])
    routing_mode = model_cfg.get("gate", {}).get("routing", "dense")
    model_class = (
        ClassConditionalMoEDatasetNIDS
        if architecture in CLASS_CONDITIONAL_ARCHITECTURES
        else MoEDatasetNIDS
    )
    return model_class(encoder, bank, gate, class_names, routing_mode=routing_mode)


def check_parameter_count(model: nn.Module, config: dict, input_dim: int, num_classes: int) -> int | None:
    """Compare the built model with the condition's ``expected_total_parameters``.

    The documented counts hold for the paper's data shape (``data.expected_features``
    features, ``data.expected_classes`` classes); on other shapes (the synthetic test
    data) the check is skipped and None is returned. ``parameter_tolerance`` is a
    relative tolerance (0 when absent: the count must match exactly).
    """
    header = config.get("condition", {}) or {}
    expected = header.get("expected_total_parameters")
    data_cfg = config.get("data", {})
    if expected is None or data_cfg.get("expected_features") != input_dim or data_cfg.get("expected_classes") != num_classes:
        return None
    actual = _parameter_count(model)
    tolerance = float(header.get("parameter_tolerance", 0.0) or 0.0)
    if abs(actual - int(expected)) > tolerance * int(expected):
        raise ValueError(
            f"{header.get('condition', config.get('architecture'))}: the built model has {actual:,} parameters, "
            f"the config expects {int(expected):,} (parameter_tolerance {tolerance})"
        )
    return actual
