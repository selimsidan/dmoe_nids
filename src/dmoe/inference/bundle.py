"""Self-contained model bundles: export after training, load for prediction.

A bundle directory holds:

* ``model.safetensors``: the final weights (safetensors only, never pickle).
* ``config.json``: architecture and model hyperparameters, expert names,
  class names, routing mode.
* ``preprocessing.json``: the 47 ordered feature columns, the pooled
  scaler's mean and scale, the train medians used for non-finite values, and
  the preprocessing signature.
* ``labels.json``: the 22 ordered classes, the expert names, and the raw
  label map per dataset.
* ``metadata.json``: condition, seed, origin, split information, selected
  epochs, validation and test macro-F1, parameter count, commit, library
  versions, and the SHA-256 of every other file.

M additionally ships ``stage_a_encoder.safetensors`` and
``stage_b_expert_bank.safetensors`` so the baselines and ablations can be
retrained from its checkpoints.

``load_bundle`` verifies the file hashes and asserts that ``model.forward``
takes exactly one argument (the features), so no dataset identity can reach
the model at prediction time.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from ..data.out_of_core import standardized_batch
from ..models.encoder import build_encoder, resolve_encoder_config
from ..training.checkpoint import STAGE_A_FILE, STAGE_B_FILE, resolve_stage_a_path, resolve_stage_b_path
from ..training.model_utils import PRIVATE_ENCODER_ARCHITECTURES, build_matched_dense_model, build_model

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
PREPROCESSING_FILE = "preprocessing.json"
LABELS_FILE = "labels.json"
METADATA_FILE = "metadata.json"
STAGE_A_BUNDLE_FILE = "stage_a_encoder.safetensors"
STAGE_B_BUNDLE_FILE = "stage_b_expert_bank.safetensors"
FORMAT_VERSION = 1


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: dict, path: str) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def _read_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _state_for_safetensors(state: dict) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def assert_dataset_blind_signature(model: torch.nn.Module) -> None:
    """``forward`` must accept the feature tensor and nothing else."""
    parameters = list(inspect.signature(model.forward).parameters)
    if parameters != ["x"]:
        raise AssertionError(
            f"{type(model).__name__}.forward takes {parameters}; a dataset-blind model takes only ['x']"
        )


def build_model_for_bundle(config: dict) -> torch.nn.Module:
    """Rebuild the architecture from ``config.json`` (no data needed)."""
    model_cfg = config["model"]
    architecture = config["architecture"]
    input_dim = int(config["input_dim"])
    dataset_names = list(config["dataset_names"])
    class_names = list(config["class_names"])
    role = "gate_encoder" if architecture in PRIVATE_ENCODER_ARCHITECTURES else "encoder"
    encoder = build_encoder(input_dim, model_cfg["latent_dim"], resolve_encoder_config(model_cfg, role))
    if architecture == "matched_dense":
        return build_matched_dense_model(encoder, dataset_names, class_names, model_cfg)
    return build_model(architecture, encoder, dataset_names, class_names, model_cfg)


def export_bundle(
    model: torch.nn.Module,
    config: dict,
    context,
    bundle_dir: str,
    *,
    metadata: dict,
    include_stages: bool = False,
) -> dict[str, str]:
    """Write a bundle directory; return ``{filename: sha256}``."""
    os.makedirs(bundle_dir, exist_ok=True)
    data = context.data
    model_config = {
        "format_version": FORMAT_VERSION,
        "architecture": config["architecture"],
        "model": config["model"],
        "input_dim": int(data.train.features.shape[1]),
        "dataset_names": list(data.active_datasets),
        "expert_names": list(getattr(model, "expert_names", [])),
        "class_names": list(data.class_names),
        "routing_mode": getattr(model, "routing_mode", None),
        "match_info": getattr(model, "match_info", None),
    }
    stats = context.preprocessing_stats
    preprocessing = {
        "format_version": FORMAT_VERSION,
        "name": "pooled-train-standard-scaler-v1",
        "feature_columns": list(context.feature_columns),
        "mean": np.asarray(stats["mean"], dtype=np.float64).tolist(),
        "scale": np.asarray(stats["scale"], dtype=np.float64).tolist(),
        "finite_medians": np.asarray(stats["finite_medians"], dtype=np.float64).tolist(),
        "signature": context.preprocessing_signature,
        "split_signatures": dict(context.split_signatures),
    }
    labels = {
        "format_version": FORMAT_VERSION,
        "class_names": list(data.class_names),
        "expert_names": list(getattr(model, "expert_names", [])),
        "dataset_names": list(data.active_datasets),
        "label_column": "Attack",
        "benign_label": "Benign",
        "raw_label_map": {name: config["data"]["label_mapping"][name] for name in data.active_datasets},
    }
    save_file(_state_for_safetensors(model.state_dict()), os.path.join(bundle_dir, MODEL_FILE))
    _write_json(model_config, os.path.join(bundle_dir, CONFIG_FILE))
    _write_json(preprocessing, os.path.join(bundle_dir, PREPROCESSING_FILE))
    _write_json(labels, os.path.join(bundle_dir, LABELS_FILE))
    if include_stages:
        stage_a = torch.load(resolve_stage_a_path(config), map_location="cpu")
        save_file(_state_for_safetensors(stage_a["encoder_state"]), os.path.join(bundle_dir, STAGE_A_BUNDLE_FILE))
        stage_b = torch.load(resolve_stage_b_path(config), map_location="cpu")
        save_file(_state_for_safetensors(stage_b["expert_bank_state"]), os.path.join(bundle_dir, STAGE_B_BUNDLE_FILE))
    hashes = {
        name: file_sha256(os.path.join(bundle_dir, name))
        for name in sorted(os.listdir(bundle_dir))
        if name != METADATA_FILE and not name.endswith(".tmp")
    }
    _write_json({"format_version": FORMAT_VERSION, **metadata, "file_sha256": hashes}, os.path.join(bundle_dir, METADATA_FILE))
    return hashes


@dataclass
class Bundle:
    path: str
    model: torch.nn.Module
    config: dict
    preprocessing: dict
    labels: dict
    metadata: dict
    device: torch.device

    @property
    def class_names(self) -> list[str]:
        return list(self.labels["class_names"])

    @property
    def expert_names(self) -> list[str]:
        return list(self.labels.get("expert_names", []))

    @property
    def feature_columns(self) -> list[str]:
        return list(self.preprocessing["feature_columns"])

    def select_features(self, frame: pd.DataFrame) -> np.ndarray:
        """Pick the 47 features by name and return raw float32 rows (NaN for invalid)."""
        columns = {column.strip(): column for column in frame.columns}
        missing = [name for name in self.feature_columns if name not in columns]
        if missing:
            raise KeyError(
                f"input is missing {len(missing)} of the {len(self.feature_columns)} required NF-v3 features: "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}"
            )
        matrix = np.empty((len(frame), len(self.feature_columns)), dtype=np.float32)
        float32_max = np.finfo(np.float32).max
        for index, name in enumerate(self.feature_columns):
            values = pd.to_numeric(frame[columns[name]], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(values) & (np.abs(values) <= float32_max)
            matrix[:, index] = np.where(valid, values, np.nan).astype(np.float32, copy=False)
        return matrix

    def standardize(self, raw: np.ndarray) -> np.ndarray:
        return standardized_batch(
            raw,
            np.asarray(self.preprocessing["finite_medians"], dtype=np.float64),
            np.asarray(self.preprocessing["mean"], dtype=np.float64),
            np.asarray(self.preprocessing["scale"], dtype=np.float64),
        )

    @torch.no_grad()
    def predict_standardized(self, x: np.ndarray) -> dict[str, np.ndarray]:
        from ..evaluation.predictions import model_outputs

        tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        out = model_outputs(self.model, tensor)
        result = {"probabilities": out["probabilities"].cpu().numpy()}
        if "gate" in out:
            result["gate"] = out["gate"].cpu().numpy()
        return result

    def predict(self, frame: pd.DataFrame, *, with_probs: bool = False, with_gate: bool = False) -> pd.DataFrame:
        """Predictions for a DataFrame of raw NF-v3 rows (any extra columns are ignored)."""
        out = self.predict_standardized(self.standardize(self.select_features(frame)))
        probabilities = out["probabilities"]
        result = pd.DataFrame({
            "predicted_label": [self.class_names[i] for i in probabilities.argmax(axis=1)],
            "confidence": probabilities.max(axis=1),
        }, index=frame.index)
        if with_probs:
            for index, name in enumerate(self.class_names):
                result[f"prob__{name}"] = probabilities[:, index]
        if with_gate and "gate" in out:
            for index, name in enumerate(self.expert_names):
                result[f"gate__{name}"] = out["gate"][:, index]
            result["top_expert"] = [self.expert_names[i] for i in out["gate"].argmax(axis=1)]
        return result


def load_bundle(path: str, device: str | torch.device = "cpu", *, verify: bool = True) -> Bundle:
    """Load a bundle directory (or ``hf:`` reference, see ``dmoe.hub``)."""
    if isinstance(path, str) and path.startswith("hf:"):
        from ..hub import resolve_hub_bundle

        path = resolve_hub_bundle(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"bundle directory not found: {path}")
    metadata = _read_json(os.path.join(path, METADATA_FILE))
    if verify:
        for name, expected in metadata.get("file_sha256", {}).items():
            actual = file_sha256(os.path.join(path, name))
            if actual != expected:
                raise ValueError(f"bundle file {name} has SHA-256 {actual}, expected {expected}")
    config = _read_json(os.path.join(path, CONFIG_FILE))
    preprocessing = _read_json(os.path.join(path, PREPROCESSING_FILE))
    labels = _read_json(os.path.join(path, LABELS_FILE))
    if list(preprocessing["feature_columns"]) != list(preprocessing["feature_columns"]) or len(preprocessing["feature_columns"]) != int(config["input_dim"]):
        raise ValueError("bundle preprocessing feature count does not match the model input")
    if labels["class_names"] != config["class_names"]:
        raise ValueError("bundle labels.json and config.json disagree on the class order")
    model = build_model_for_bundle(config)
    assert_dataset_blind_signature(model)
    state = load_file(os.path.join(path, MODEL_FILE))
    model.load_state_dict(state, strict=True)
    device = torch.device(device)
    model.to(device).eval()
    return Bundle(path=path, model=model, config=config, preprocessing=preprocessing, labels=labels, metadata=metadata, device=device)
