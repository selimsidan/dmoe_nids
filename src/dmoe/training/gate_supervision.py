"""How the gate is trained in Stage C.

``gate_supervision`` selects the router's training signal:

* ``none``: task loss only (M, B1, A1, A2, A3).
* ``damex``: dataset-ID cross-entropy only; the task-loss gradient into the
  gate is cut by detaching the gate weights in the task path (baseline B3).
* ``light_aux`` / ``hard``: task loss plus a dataset-ID auxiliary term
  (kept for completeness; no release condition uses them).
"""
from __future__ import annotations

import torch


def lambda_dataset_aux(config: dict) -> float:
    stage_c_cfg = config["training"]["stage_c"]
    mode = stage_c_cfg.get("gate_supervision", "light_aux")
    if mode == "none":
        return 0.0
    if mode == "light_aux":
        return float(stage_c_cfg.get("lambda_dataset_aux", 0.1))
    if mode == "hard":
        return float(stage_c_cfg.get("lambda_dataset_aux_hard", 5.0))
    if mode == "damex":
        return float(stage_c_cfg.get("lambda_dataset_aux_damex", 1.0))
    raise ValueError(
        f"Unknown training.stage_c.gate_supervision '{mode}'. "
        "Expected none | light_aux | hard | damex."
    )


def gate_weights_for_task(gate_weights: torch.Tensor, gate_supervision: str) -> torch.Tensor:
    """Keep the task objective out of the router graph in ``damex`` mode."""
    return gate_weights.detach() if gate_supervision == "damex" else gate_weights


def set_encoder_trainable(encoder, mode: str) -> None:
    """Apply ``training.stage_c_unfreeze`` (all | last_layer | none) to one encoder."""
    for parameter in encoder.parameters():
        parameter.requires_grad_(mode == "all")
    if mode == "last_layer":
        for parameter in encoder.last_layer_parameters():
            parameter.requires_grad_(True)
    elif mode not in {"all", "none"}:
        raise ValueError(f"Unknown stage_c_unfreeze mode: {mode}")
