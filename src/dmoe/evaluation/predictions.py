"""Per-row predictions of one model on one split, saved as a single ``.npz``.

``PredictionRecord`` is the only input the report writer needs, so every
table and figure can be regenerated from the saved file without the data or
a GPU. It holds, per row: the true class, the predicted class, the dataset of
origin, the confidence, and (MoE models) the gate weights. It also holds the
bounded-size score histograms that give ROC-AUC and PR-AUC, and the
confusion matrix of every expert evaluated alone on every dataset.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import confusion_from_predictions, update_score_histograms

FORMAT_VERSION = 1
PREDICTION_FILES = {"val": "Val_Predictions.npz", "test": "Test_Predictions.npz"}


@dataclass
class PredictionRecord:
    class_names: list[str]
    dataset_names: list[str]
    expert_names: list[str]
    y_true: np.ndarray          # int16
    y_pred: np.ndarray          # int16
    dataset_idx: np.ndarray     # uint8
    confidence: np.ndarray      # float16
    gate: np.ndarray | None     # float16, (rows, experts) for MoE models
    roc_positive: np.ndarray    # int64, (datasets, classes, bins)
    roc_negative: np.ndarray    # int64, (datasets, classes, bins)
    expert_confusion: np.ndarray | None   # int64, (datasets, experts, classes, classes)
    reliability: np.ndarray | None        # float32, (experts, classes) logits
    native_classes: dict[str, list[str]]  # classes present in each dataset's split
    meta: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def rows(self) -> int:
        return int(len(self.y_true))

    @property
    def bins(self) -> int:
        return int(self.roc_positive.shape[-1])

    def confusion(self, dataset: str | None = None) -> np.ndarray:
        mask = slice(None) if dataset is None else self.dataset_idx == self.dataset_names.index(dataset)
        return confusion_from_predictions(self.y_true[mask], self.y_pred[mask], self.num_classes)

    def histograms(self, dataset: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        if dataset is None:
            return self.roc_positive.sum(axis=0), self.roc_negative.sum(axis=0)
        index = self.dataset_names.index(dataset)
        return self.roc_positive[index], self.roc_negative[index]

    def native_indices(self, dataset: str | None = None) -> set[int]:
        names = (
            set().union(*self.native_classes.values()) if dataset is None else set(self.native_classes[dataset])
        )
        return {self.class_names.index(name) for name in names}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = {
            "y_true": self.y_true.astype(np.int16),
            "y_pred": self.y_pred.astype(np.int16),
            "dataset_idx": self.dataset_idx.astype(np.uint8),
            "confidence": self.confidence.astype(np.float16),
            "roc_positive": self.roc_positive,
            "roc_negative": self.roc_negative,
        }
        if self.gate is not None:
            arrays["gate"] = self.gate.astype(np.float16)
        if self.expert_confusion is not None:
            arrays["expert_confusion"] = self.expert_confusion
        if self.reliability is not None:
            arrays["reliability"] = self.reliability.astype(np.float32)
        header = {
            "format_version": FORMAT_VERSION,
            "class_names": self.class_names,
            "dataset_names": self.dataset_names,
            "expert_names": self.expert_names,
            "native_classes": self.native_classes,
            "meta": self.meta,
        }
        arrays["header"] = np.array(json.dumps(header, sort_keys=True, default=str))
        temporary = path + ".tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, path: str) -> "PredictionRecord":
        with np.load(path, allow_pickle=False) as saved:
            header = json.loads(str(saved["header"]))
            if header.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported prediction format {header.get('format_version')}")
            return cls(
                class_names=list(header["class_names"]),
                dataset_names=list(header["dataset_names"]),
                expert_names=list(header["expert_names"]),
                y_true=saved["y_true"],
                y_pred=saved["y_pred"],
                dataset_idx=saved["dataset_idx"],
                confidence=saved["confidence"],
                gate=saved["gate"] if "gate" in saved.files else None,
                roc_positive=saved["roc_positive"],
                roc_negative=saved["roc_negative"],
                expert_confusion=saved["expert_confusion"] if "expert_confusion" in saved.files else None,
                reliability=saved["reliability"] if "reliability" in saved.files else None,
                native_classes={k: list(v) for k, v in header["native_classes"].items()},
                meta=dict(header.get("meta", {})),
            )


def _peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if os.uname().sysname == "Linux" else 1))
    except (ImportError, AttributeError, OSError):
        return None


def model_outputs(model: torch.nn.Module, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward one batch and normalize the output to probabilities, gate and expert logits."""
    output = model(tensor)
    if "combined_probs" in output:
        probabilities = output["combined_probs"]
    elif "logits" in output:
        probabilities = F.softmax(output["logits"], dim=1)
    else:
        raise KeyError("model output must contain combined_probs or logits")
    result = {"probabilities": probabilities}
    if "gate_weights" in output:
        result["gate"] = output["gate_weights"]
    if hasattr(model, "expert_bank"):
        expert_logits = output.get("expert_logits")
        if expert_logits is None:
            expert_logits = model.all_expert_logits(tensor, output.get("z"))
        result["expert_probabilities"] = F.softmax(expert_logits, dim=2)
    return result


def predict_rows(
    model: torch.nn.Module,
    features,
    y_true,
    dataset_idx,
    *,
    class_names: list[str],
    dataset_names: list[str],
    chunk_rows: int,
    roc_bins: int,
    device: torch.device,
    progress_label: str = "predict",
) -> PredictionRecord:
    """Predict ``features`` (an array-like of standardized rows) chunk by chunk."""
    rows = len(y_true)
    num_classes = len(class_names)
    num_datasets = len(dataset_names)
    expert_names = list(getattr(model, "expert_names", []))
    num_experts = len(expert_names)
    y_pred = np.zeros(rows, dtype=np.int16)
    confidence = np.zeros(rows, dtype=np.float16)
    gate = np.zeros((rows, num_experts), dtype=np.float16) if num_experts else None
    roc_positive = np.zeros((num_datasets, num_classes, roc_bins), dtype=np.int64)
    roc_negative = np.zeros((num_datasets, num_classes, roc_bins), dtype=np.int64)
    expert_confusion = (
        np.zeros((num_datasets, num_experts, num_classes, num_classes), dtype=np.int64) if num_experts else None
    )
    truth_all = np.asarray(y_true, dtype=np.int64)
    origin_all = np.asarray(dataset_idx, dtype=np.int64)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            batch = np.asarray(features[start:stop], dtype=np.float32)
            truth = truth_all[start:stop]
            origin = origin_all[start:stop]
            tensor = torch.from_numpy(batch).to(device, non_blocking=True)
            out = model_outputs(model, tensor)
            probabilities = out["probabilities"].cpu().numpy()
            y_pred[start:stop] = probabilities.argmax(axis=1)
            confidence[start:stop] = probabilities.max(axis=1)
            if gate is not None:
                gate[start:stop] = out["gate"].cpu().numpy()
            expert_probabilities = out.get("expert_probabilities")
            expert_predictions = expert_probabilities.argmax(dim=2).cpu().numpy() if expert_probabilities is not None else None
            for d in np.unique(origin):
                mask = origin == d
                update_score_histograms(roc_positive[d], roc_negative[d], truth[mask], probabilities[mask])
                if expert_predictions is not None:
                    for e in range(num_experts):
                        expert_confusion[d, e] += confusion_from_predictions(
                            truth[mask], expert_predictions[mask, e], num_classes
                        )
            if stop == rows or (start // chunk_rows) % 10 == 0:
                print(f"[{progress_label}] {stop:,}/{rows:,} rows", flush=True)
    seconds = time.monotonic() - started
    reliability = None
    if hasattr(model, "class_reliability"):
        reliability = model.class_reliability.detach().cpu().numpy().astype(np.float32)
    native = {}
    for d, name in enumerate(dataset_names):
        present = np.unique(truth_all[origin_all == d])
        native[name] = [class_names[int(i)] for i in present]
    return PredictionRecord(
        class_names=list(class_names),
        dataset_names=list(dataset_names),
        expert_names=expert_names,
        y_true=truth_all.astype(np.int16),
        y_pred=y_pred,
        dataset_idx=origin_all.astype(np.uint8),
        confidence=confidence,
        gate=gate,
        roc_positive=roc_positive,
        roc_negative=roc_negative,
        expert_confusion=expert_confusion,
        reliability=reliability,
        native_classes=native,
        meta={
            "roc_auc_bins": int(roc_bins),
            "auc_method": f"histogram_approximation_{roc_bins}_bins",
            "inference_wall_seconds": seconds,
            "inference_rows_per_second": rows / seconds if seconds else None,
            "inference_microseconds_per_row": 1e6 * seconds / rows if rows else None,
            "inference_device": device.type,
            "peak_memory_bytes": _peak_memory_bytes(device),
        },
    )


def predict_split(model: torch.nn.Module, context, split_name: str, config: dict) -> PredictionRecord:
    """Predict one split (``val`` or ``test``) of the prepared out-of-core data."""
    split = getattr(context.data, split_name)
    device = next(model.parameters()).device
    record = predict_rows(
        model, split.features, split.class_idx, split.dataset_idx,
        class_names=context.data.class_names,
        dataset_names=context.data.active_datasets,
        chunk_rows=int(config["evaluation"].get("prediction_chunk_rows", 262_144)),
        roc_bins=int(config["evaluation"].get("roc_auc_bins", 4096)),
        device=device,
        progress_label=f"predict:{split_name}",
    )
    record.meta.update({
        "split": split_name,
        "split_mode": config.get("data", {}).get("split_mode"),
        "split_seed": config.get("data", {}).get("split_seed"),
        "split_signatures": dict(context.split_signatures),
        "preprocessing_signature": context.preprocessing_signature,
    })
    return record
