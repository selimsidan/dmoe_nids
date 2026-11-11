"""The one evaluation entry point used by the scripts, the notebooks and training.

``evaluate`` accepts one of three inputs and writes the same set of tables
and figures for each:

* ``predictions``: a saved ``*_Predictions.npz`` (no data or GPU needed);
* ``bundle`` plus ``context``: re-run a bundle on a prepared split;
* ``bundle`` plus ``csv`` plus ``labels_from``: score a labeled NF-v3 CSV.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from .predictions import PREDICTION_FILES, PredictionRecord, predict_rows, predict_split
from .reports import write_reports


def evaluate(
    output_dir: str,
    *,
    predictions: str | PredictionRecord | None = None,
    bundle=None,
    context=None,
    split: str = "test",
    csv: str | None = None,
    labels_from: str | None = None,
    condition: str | None = None,
    seed: int | None = None,
    display_names: dict[str, str] | None = None,
    chunk_rows: int = 262_144,
    roc_bins: int = 4096,
    figures: bool = True,
    save_predictions: bool = True,
) -> dict[str, pd.DataFrame]:
    if predictions is not None:
        record = predictions if isinstance(predictions, PredictionRecord) else PredictionRecord.load(predictions)
    elif bundle is not None and context is not None:
        config = {"evaluation": {"prediction_chunk_rows": chunk_rows, "roc_auc_bins": roc_bins},
                  "data": {"split_mode": None, "split_seed": None}}
        record = predict_split(bundle.model, context, split, config)
    elif bundle is not None and csv is not None:
        record = predict_labeled_csv(bundle, csv, labels_from, chunk_rows=chunk_rows, roc_bins=roc_bins)
    else:
        raise ValueError("evaluate needs predictions, or a bundle with a context, or a bundle with a CSV")
    if save_predictions and predictions is None:
        record.save(os.path.join(output_dir, PREDICTION_FILES.get(split, f"{split.title()}_Predictions.npz")))
    return write_reports(
        record, output_dir, condition=condition, seed=seed, split=split or record.meta.get("split"),
        display_names=display_names, figures=figures,
    )


def map_raw_labels(raw: pd.Series, bundle, labels_from: str) -> np.ndarray:
    """Map an ``Attack`` column to class indices with the bundle's label map for ``labels_from``."""
    label_map = bundle.labels["raw_label_map"].get(labels_from)
    if label_map is None:
        raise KeyError(f"the bundle has no raw label map for {labels_from!r}; known: {sorted(bundle.labels['raw_label_map'])}")
    cleaned = raw.astype("string").fillna("").str.strip()
    benign = bundle.labels.get("benign_label", "Benign")
    index = {name: i for i, name in enumerate(bundle.class_names)}
    mapped = np.full(len(cleaned), -1, dtype=np.int64)
    is_benign = cleaned.str.lower().eq(benign.lower()).to_numpy(dtype=bool, na_value=False)
    mapped[is_benign] = index[benign]
    for raw_value, canonical in label_map.items():
        if canonical is None:
            continue
        mask = cleaned.eq(raw_value).to_numpy(dtype=bool, na_value=False)
        if not mask.any():
            continue
        if canonical not in index:
            raise ValueError(f"raw label {raw_value!r} maps to {canonical!r}, which is not one of the model's classes")
        mapped[mask] = index[canonical]
    unmapped = sorted(set(cleaned[(mapped < 0)].tolist()))
    if unmapped:
        raise ValueError(f"unmapped Attack values for {labels_from}: {unmapped[:10]}")
    return mapped


def predict_labeled_csv(bundle, csv: str, labels_from: str | None, *, chunk_rows: int, roc_bins: int) -> PredictionRecord:
    if not labels_from:
        raise ValueError("scoring a CSV needs --labels-from <dataset> to map its Attack column")
    label_column = bundle.labels.get("label_column", "Attack")
    features, truth = [], []
    usecols = [*bundle.feature_columns, label_column]
    for chunk in pd.read_csv(csv, usecols=lambda c: c.strip() in usecols, chunksize=chunk_rows, low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]
        features.append(bundle.standardize(bundle.select_features(chunk)))
        truth.append(map_raw_labels(chunk[label_column], bundle, labels_from))
    x = np.concatenate(features) if features else np.zeros((0, len(bundle.feature_columns)), dtype=np.float32)
    y = np.concatenate(truth) if truth else np.zeros(0, dtype=np.int64)
    record = predict_rows(
        bundle.model, x, y, np.zeros(len(y), dtype=np.int64),
        class_names=bundle.class_names, dataset_names=[labels_from],
        chunk_rows=chunk_rows, roc_bins=roc_bins, device=bundle.device, progress_label="predict:csv",
    )
    record.meta.update({"split": "csv", "source_csv": os.path.basename(csv), "labels_from": labels_from})
    return record
