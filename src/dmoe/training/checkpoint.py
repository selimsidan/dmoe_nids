"""Checkpoint I/O for the three training stages and the dense baseline.

Layout of one run's ``checkpoints/`` folder::

    run_contract.json  Training_History.csv  stage_c_training_summary.json
    stage_A/  stage_a_encoder.pt  (stage_a_progress.pt while training)
    stage_B/  stage_b_expert_bank.pt or baseline_stage_b.pt  (stage_b_progress.pt)
    stage_C/  stage_c_full.pt or baseline_full.pt  (stage_c_progress.pt)

Every stage writes its final checkpoint atomically (write to ``.tmp``, then
rename) and reopens it before clearing its per-epoch progress file, so an
interrupted Colab session always resumes from a valid state. Stage A and B
checkpoints can be read from another run's directory (for example M's) via
``training.stage_a_checkpoint`` / ``training.stage_b_checkpoint``.
"""
from __future__ import annotations

import os
import time

import torch

from ..models.encoder import resolve_encoder_config
from ..models.representation_losses import representation_config

STAGE_A_FILE = "stage_a_encoder.pt"
STAGE_B_FILE = "stage_b_expert_bank.pt"
STAGE_C_FILE = "stage_c_full.pt"
BASELINE_MODEL_FILE = "baseline_full.pt"
BASELINE_STAGE_B_FILE = "baseline_stage_b.pt"
STAGE_DIRS = {"A": "stage_A", "B": "stage_B", "C": "stage_C"}
# Progress keys of the dense baseline map onto the same stage folders.
_PROGRESS_STAGE = {"A": "A", "B": "B", "C": "C", "DB": "B", "DC": "C"}


def stage_dir(checkpoint_dir: str, stage: str) -> str:
    return os.path.join(checkpoint_dir, STAGE_DIRS[stage])


def stage_file(checkpoint_dir: str, stage: str, filename: str) -> str:
    """Path of ``filename`` inside the folder of ``stage`` (A, B or C)."""
    return os.path.join(stage_dir(checkpoint_dir, stage), filename)


def _atomic_torch_save(value, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


def _verify_torch_checkpoint(path: str) -> None:
    """Reopen a newly committed checkpoint before its progress file is cleared."""
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Checkpoint verification failed for {path}: {exc}") from exc


def stage_a_metadata(
    config: dict,
    data,
    *,
    split_signature: str,
    feature_columns: list[str],
    encoder_role: str = "encoder",
    encoder_config: dict | None = None,
) -> dict:
    """Facts a reused Stage-A encoder must match (shape, data, split)."""
    encoder_cfg = encoder_config or resolve_encoder_config(config["model"], encoder_role)
    encoder_metadata = {
        "input_dim": int(data.train.features.shape[1]),
        "hidden_dims": list(encoder_cfg.get("hidden_dims", [])),
        "latent_dim": int(config["model"]["latent_dim"]),
        "activation": encoder_cfg.get("activation", "relu"),
        "dropout": float(encoder_cfg.get("dropout", 0.1)),
    }
    metadata = {
        "format_version": 2,
        "seed": int(config.get("seed", 0)),
        "encoder": encoder_metadata,
        "class_names": list(data.class_names),
        "active_datasets": list(data.active_datasets),
        "feature_columns": list(feature_columns),
        "split_signature": split_signature,
    }
    return metadata


def save_stage_a(
    checkpoint_dir: str,
    encoder_state: dict,
    class_names: list[str],
    metadata: dict | None = None,
    representation_state: dict | None = None,
    representation_config: dict | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = stage_file(checkpoint_dir, "A", STAGE_A_FILE)
    _atomic_torch_save(
        {
            "encoder_state": encoder_state,
            "class_names": class_names,
            "metadata": metadata,
            "representation_state": representation_state,
            "representation_config": representation_config,
        },
        path,
    )
    _verify_torch_checkpoint(path)


def load_stage_a(checkpoint_dir: str) -> dict:
    path = stage_file(checkpoint_dir, "A", STAGE_A_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage A checkpoint at {path}. Run Stage A first.")
    return torch.load(path, map_location="cpu")


def resolve_stage_a_path(config: dict, encoder_role: str = "encoder") -> str:
    training_cfg = config.get("training", {})
    if encoder_role == "private_encoder":
        source = training_cfg.get("private_stage_a_checkpoint")
    else:
        source = None
    source = source or training_cfg.get("stage_a_checkpoint")
    if source:
        return source if source.endswith(".pt") else stage_file(source, "A", STAGE_A_FILE)
    return stage_file(config["training"]["checkpoint_dir"], "A", STAGE_A_FILE)


def load_validated_stage_a(
    config: dict,
    expected_metadata: dict,
    encoder_role: str = "encoder",
) -> dict:
    path = resolve_stage_a_path(config, encoder_role)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage-A encoder checkpoint at {path}")
    checkpoint = torch.load(path, map_location="cpu")
    actual = checkpoint.get("metadata")
    if actual is None:
        raise ValueError(
            f"Stage-A checkpoint {path} predates the fairness metadata contract; regenerate it"
        )
    mismatches = {
        key: {"expected": expected_metadata.get(key), "actual": actual.get(key)}
        for key in expected_metadata
        if actual.get(key) != expected_metadata.get(key)
    }
    if mismatches:
        raise ValueError(f"Incompatible Stage-A checkpoint {path}: {mismatches}")
    configured_representation = representation_config(config)
    checkpoint_representation = checkpoint.get("representation_config")
    if checkpoint_representation is None:
        raise ValueError(f"Stage-A checkpoint {path} does not record its representation objective")
    mismatched_representation = {
        key: {"expected": value, "actual": checkpoint_representation.get(key)}
        for key, value in configured_representation.items()
        if checkpoint_representation.get(key) != value
    }
    if mismatched_representation:
        raise ValueError(
            f"Incompatible Stage-A representation checkpoint {path}: "
            f"{mismatched_representation}"
        )
    return checkpoint


def save_stage_b(
    checkpoint_dir: str,
    expert_bank_state: dict,
    dataset_names: list[str],
    bank_kind: str,
    training_summary: dict | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = stage_file(checkpoint_dir, "B", STAGE_B_FILE)
    _atomic_torch_save(
        {
            "expert_bank_state": expert_bank_state,
            "dataset_names": dataset_names,
            "bank_kind": bank_kind,
            "training_summary": training_summary,
        },
        path,
    )
    _verify_torch_checkpoint(path)


def load_stage_b(checkpoint_dir: str) -> dict:
    path = checkpoint_dir if checkpoint_dir.endswith(".pt") else stage_file(checkpoint_dir, "B", STAGE_B_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage B checkpoint at {path}. Run Stage B first.")
    return torch.load(path, map_location="cpu")


def resolve_stage_b_path(config: dict) -> str:
    source = config.get("training", {}).get("stage_b_checkpoint")
    if source:
        return source if source.endswith(".pt") else stage_file(source, "B", STAGE_B_FILE)
    return stage_file(config["training"]["checkpoint_dir"], "B", STAGE_B_FILE)


def load_configured_stage_b(
    config: dict,
    *,
    expected_dataset_names: list[str] | None = None,
    expected_bank_kind: str | None = None,
) -> dict:
    """Load the selected Stage-B checkpoint and reject a mismatched expert bank."""
    checkpoint = load_stage_b(resolve_stage_b_path(config))
    mismatches = {}
    if expected_dataset_names is not None:
        actual = list(checkpoint.get("dataset_names", []))
        expected = list(expected_dataset_names)
        if actual != expected:
            mismatches["dataset_names"] = {"expected": expected, "actual": actual}
    if expected_bank_kind is not None:
        actual = checkpoint.get("bank_kind")
        if actual != expected_bank_kind:
            mismatches["bank_kind"] = {
                "expected": expected_bank_kind,
                "actual": actual,
            }
    if mismatches:
        raise ValueError(
            f"Incompatible Stage-B checkpoint {resolve_stage_b_path(config)}: {mismatches}"
        )
    return checkpoint


def save_stage_c(
    checkpoint_dir: str,
    model_state: dict,
    class_names: list[str],
    dataset_names: list[str],
    bank_kind: str,
    training_summary: dict | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = stage_file(checkpoint_dir, "C", STAGE_C_FILE)
    _atomic_torch_save(
        {
            "model_state": model_state,
            "class_names": class_names,
            "dataset_names": dataset_names,
            "bank_kind": bank_kind,
            "training_summary": training_summary,
        },
        path,
    )
    _verify_torch_checkpoint(path)


def load_stage_c(checkpoint_dir: str) -> dict:
    path = stage_file(checkpoint_dir, "C", STAGE_C_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No Stage C checkpoint at {path}. Run Stage C first.")
    return torch.load(path, map_location="cpu")


def stage_complete(checkpoint_dir: str, stage: str) -> bool:
    filename = {"A": STAGE_A_FILE, "B": STAGE_B_FILE, "C": STAGE_C_FILE}[stage]
    path = stage_file(checkpoint_dir, stage, filename)
    if not os.path.isfile(path):
        return False
    try:
        torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[checkpoint] Stage {stage} final checkpoint is unreadable: {path}: {exc}")
        return False
    return True


def _progress_path(checkpoint_dir: str, stage: str) -> str:
    folder = _PROGRESS_STAGE[stage]
    return stage_file(checkpoint_dir, folder, f"stage_{folder.lower()}_progress.pt")


def save_progress(checkpoint_dir: str, stage: str, state: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    _atomic_torch_save(state, _progress_path(checkpoint_dir, stage))


def load_progress(checkpoint_dir: str, stage: str) -> dict | None:
    path = _progress_path(checkpoint_dir, stage)
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        quarantine = f"{path}.corrupt-{time.time_ns()}"
        os.replace(path, quarantine)
        print(
            f"[checkpoint] unreadable Stage {stage} progress was quarantined at "
            f"{quarantine}; restarting the stage from its last valid input: {exc}",
            flush=True,
        )
        return None


def clear_progress(checkpoint_dir: str, stage: str) -> None:
    path = _progress_path(checkpoint_dir, stage)
    if os.path.isfile(path):
        os.remove(path)
