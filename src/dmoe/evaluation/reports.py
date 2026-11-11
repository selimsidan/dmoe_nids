"""All per-run, per-split tables and figures, written from a PredictionRecord.

Files in ``runs/<condition>/seed<k>/eval/<split>/``:

* ``Overall_Metrics.csv``: one row for ``ALL`` and one per origin dataset,
  macro-F1 first (see ``metrics.OVERALL_METRIC_ORDER``).
* ``Per_Dataset_Metrics.csv``: the per-origin rows only.
* ``Per_Class_Metrics.csv``: every class, overall and per origin.
* ``Confusion_Matrix_ALL_*`` and ``Confusion_Matrix_<dataset>_*``: long,
  counts and row-normalized CSVs plus PNG/PDF figures (``figures/``).
* ``Per_Class_Table.{png,pdf}``: figure version of the overall per-class table.
* Routing files (MoE conditions): see ``routing.py``.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .confusion import write_confusion_outputs
from .figures import save_figure, table_figure
from .metrics import auc_metrics_from_histograms, overall_from_confusion, per_class_rows
from .predictions import PredictionRecord
from .routing import write_routing_outputs

OVERALL_FILE = "Overall_Metrics.csv"
PER_DATASET_FILE = "Per_Dataset_Metrics.csv"
PER_CLASS_FILE = "Per_Class_Metrics.csv"


def _atomic_csv(frame: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def _short_dataset(name: str) -> str:
    """Compact label for figures: NF-ToN-IoT-v3 -> ToN-IoT."""
    value = name[3:] if name.startswith("NF-") else name
    return value[:-3] if value.endswith("-v3") else value


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def origin_datasets_by_class(record: PredictionRecord) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {name: [] for name in record.class_names}
    for dataset, classes in record.native_classes.items():
        for class_name in classes:
            result[class_name].append(dataset)
    return result


def overall_table(record: PredictionRecord) -> pd.DataFrame:
    rows = []
    for origin in ["ALL", *record.dataset_names]:
        dataset = None if origin == "ALL" else origin
        auc = auc_metrics_from_histograms(*record.histograms(dataset))
        rows.append({
            "origin": origin,
            **overall_from_confusion(record.confusion(dataset), record.native_indices(dataset), auc),
            "auc_method": record.meta.get("auc_method"),
        })
    return pd.DataFrame(rows)


def per_class_table(record: PredictionRecord) -> pd.DataFrame:
    origins = origin_datasets_by_class(record)
    rows = []
    for origin in ["ALL", *record.dataset_names]:
        dataset = None if origin == "ALL" else origin
        auc = auc_metrics_from_histograms(*record.histograms(dataset))
        rows.extend(per_class_rows(
            record.confusion(dataset), record.class_names, origin,
            auc["per_class"], auc["per_class_pr"], origins,
        ))
    frame = pd.DataFrame(rows)
    frame["auc_method"] = record.meta.get("auc_method")
    return frame


def write_reports(
    record: PredictionRecord,
    output_dir: str,
    *,
    condition: str | None = None,
    seed: int | None = None,
    split: str | None = None,
    display_names: dict[str, str] | None = None,
    figures: bool = True,
) -> dict[str, pd.DataFrame]:
    """Write every table and figure for one run and split; return the tables."""
    os.makedirs(output_dir, exist_ok=True)
    display = dict(display_names or {})
    label = " ".join(str(part) for part in (condition, f"seed {seed}" if seed is not None else None, split) if part)
    prefix = {"condition": condition, "seed": seed, "split": split or record.meta.get("split")}

    overall = overall_table(record)
    for key, value in reversed(prefix.items()):
        overall.insert(0, key, value)
    _atomic_csv(overall, os.path.join(output_dir, OVERALL_FILE))
    per_dataset = overall[overall["origin"] != "ALL"].copy()
    per_dataset.insert(per_dataset.columns.get_loc("origin") + 1, "dataset", per_dataset["origin"].map(lambda v: display.get(v, v)))
    _atomic_csv(per_dataset, os.path.join(output_dir, PER_DATASET_FILE))
    per_class = per_class_table(record)
    for key, value in reversed(prefix.items()):
        per_class.insert(0, key, value)
    _atomic_csv(per_class, os.path.join(output_dir, PER_CLASS_FILE))

    figure_dir = os.path.join(output_dir, "figures")
    write_confusion_outputs(
        record.confusion(), record.class_names, output_dir, "Confusion_Matrix_ALL",
        f"Confusion matrix, all datasets ({label})".strip(), figures=False,
    )
    if figures:
        write_confusion_outputs(
            record.confusion(), record.class_names, figure_dir, "Confusion_Matrix_ALL",
            f"Confusion matrix, all datasets ({label})", figures=True,
        )
        for path in ("Confusion_Matrix_ALL_long.csv", "Confusion_Matrix_ALL_counts.csv", "Confusion_Matrix_ALL_row_normalized.csv"):
            os.remove(os.path.join(figure_dir, path))
    for dataset in record.dataset_names:
        stem = f"Confusion_Matrix_{_safe_stem(dataset)}"
        title = f"Confusion matrix, {display.get(dataset, dataset)} ({label})"
        write_confusion_outputs(record.confusion(dataset), record.class_names, output_dir, stem, title, only_present=True, figures=False)
        if figures:
            write_confusion_outputs(record.confusion(dataset), record.class_names, figure_dir, stem, title, only_present=True, figures=True)
            for suffix in ("_long.csv", "_counts.csv", "_row_normalized.csv"):
                os.remove(os.path.join(figure_dir, stem + suffix))
    if figures:
        short = {name: _short_dataset(display.get(name, name)) for name in record.dataset_names}
        table = per_class[per_class["origin"] == "ALL"][
            ["class", "origin_datasets", "support", "precision", "recall", "f1", "roc_auc_ovr", "pr_auc_ovr"]
        ].rename(columns={"origin_datasets": "datasets", "roc_auc_ovr": "ROC-AUC", "pr_auc_ovr": "PR-AUC"})
        table["datasets"] = table["datasets"].map(
            lambda value: ", ".join(short.get(name, name) for name in str(value).split("|") if name)
        )
        fig = table_figure(table.reset_index(drop=True), f"Per-class metrics ({label})".strip())
        save_figure(fig, figure_dir, "Per_Class_Table")
    frames = {"overall": overall, "per_dataset": per_dataset, "per_class": per_class}
    frames.update(write_routing_outputs(record, output_dir, figures=figures))
    return frames


def summary_line(record: PredictionRecord) -> str:
    overall = overall_from_confusion(record.confusion())
    parts = [f"macro_f1={overall['macro_f1']:.6f}", f"accuracy={overall['accuracy']:.6f}", f"rows={record.rows:,}"]
    for dataset in record.dataset_names:
        value = overall_from_confusion(record.confusion(dataset))["macro_f1"]
        parts.append(f"{dataset}={value:.4f}")
    return " ".join(parts)


def macro_f1(record: PredictionRecord) -> float:
    return float(overall_from_confusion(record.confusion())["macro_f1"])


def macro_f1_by_dataset(record: PredictionRecord) -> dict[str, float]:
    return {d: float(overall_from_confusion(record.confusion(d))["macro_f1"]) for d in record.dataset_names}


def frame_from_npz(path: str) -> tuple[PredictionRecord, np.ndarray]:
    record = PredictionRecord.load(path)
    return record, record.confusion()
