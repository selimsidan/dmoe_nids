"""Routing diagnostics for the MoE conditions, computed from a PredictionRecord.

Outputs (all in the split's evaluation directory):

* ``Gate_By_Dataset.csv``: mean gate weight per expert for the rows of each
  origin dataset, mean gate entropy, mean top gate weight, and the fraction
  of rows whose top expert is the dataset's own expert (dataset-named experts only).
* ``Expert_Utilization.csv``: how many rows each expert wins by argmax.
* ``Route_Confusion.csv``: origin dataset (rows) against top expert (columns).
* ``Expert_Performance_By_Dataset.csv``: every expert evaluated alone on every origin.
* ``Class_Conditional_Reliability.csv``: the reliability matrix R (when present).
* Heatmaps of the gate mass, the route confusion and R.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .figures import heatmap_figure, save_figure
from .metrics import overall_from_confusion
from .predictions import PredictionRecord


def _atomic_csv(frame: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def gate_by_dataset(record: PredictionRecord) -> pd.DataFrame:
    gate = record.gate.astype(np.float64)
    rows = []
    top = gate.argmax(axis=1)
    entropy = -(gate * np.log(np.clip(gate, 1e-12, 1.0))).sum(axis=1)
    experts_are_datasets = record.expert_names == record.dataset_names
    for d, name in enumerate(record.dataset_names):
        mask = record.dataset_idx == d
        count = int(mask.sum())
        row = {"origin": name, "rows": count}
        row["own_expert_top_fraction"] = (
            float((top[mask] == d).mean()) if experts_are_datasets and count else float("nan")
        )
        row["mean_gate_entropy"] = float(entropy[mask].mean()) if count else float("nan")
        row["mean_max_gate"] = float(gate[mask].max(axis=1).mean()) if count else float("nan")
        for e, expert in enumerate(record.expert_names):
            row[f"mean_gate__{expert}"] = float(gate[mask, e].mean()) if count else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def expert_utilization(record: PredictionRecord) -> pd.DataFrame:
    top = record.gate.argmax(axis=1)
    counts = np.bincount(top, minlength=len(record.expert_names))
    rows = []
    for e, expert in enumerate(record.expert_names):
        row = {
            "expert": expert,
            "argmax_rows": int(counts[e]),
            "argmax_fraction": float(counts[e] / counts.sum()) if counts.sum() else 0.0,
        }
        for d, name in enumerate(record.dataset_names):
            mask = record.dataset_idx == d
            row[f"argmax_fraction__{name}"] = float((top[mask] == e).mean()) if mask.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def route_confusion(record: PredictionRecord) -> np.ndarray:
    """Counts of (origin dataset, top expert)."""
    top = record.gate.argmax(axis=1).astype(np.int64)
    n_d, n_e = len(record.dataset_names), len(record.expert_names)
    return np.bincount(record.dataset_idx.astype(np.int64) * n_e + top, minlength=n_d * n_e).reshape(n_d, n_e)


def expert_performance(record: PredictionRecord) -> pd.DataFrame:
    rows = []
    for d, origin in enumerate(record.dataset_names):
        native = record.native_indices(origin)
        for e, expert in enumerate(record.expert_names):
            rows.append({
                "origin": origin,
                "expert": expert,
                "is_assigned_expert": expert == origin,
                **overall_from_confusion(record.expert_confusion[d, e], native),
            })
    frame = pd.DataFrame(rows)
    return frame.drop(columns=[c for c in frame.columns if c.startswith(("roc_auc", "pr_auc"))])


def reliability_frame(record: PredictionRecord) -> pd.DataFrame:
    values = record.reliability.astype(np.float64)
    preference = np.exp(values - values.max(axis=0, keepdims=True))
    preference /= preference.sum(axis=0, keepdims=True)
    return pd.DataFrame([
        {
            "expert": expert,
            "class": class_name,
            "reliability_logit": float(values[e, c]),
            "reliability_only_expert_preference": float(preference[e, c]),
        }
        for e, expert in enumerate(record.expert_names)
        for c, class_name in enumerate(record.class_names)
    ])


def write_routing_outputs(record: PredictionRecord, output_dir: str, *, figures: bool = True) -> dict[str, pd.DataFrame]:
    if record.gate is None or not record.expert_names:
        return {}
    frames: dict[str, pd.DataFrame] = {}
    frames["gate_by_dataset"] = gate_by_dataset(record)
    _atomic_csv(frames["gate_by_dataset"], os.path.join(output_dir, "Gate_By_Dataset.csv"))
    frames["expert_utilization"] = expert_utilization(record)
    _atomic_csv(frames["expert_utilization"], os.path.join(output_dir, "Expert_Utilization.csv"))
    routes = route_confusion(record)
    route_frame = pd.DataFrame(routes, index=record.dataset_names, columns=record.expert_names)
    route_frame.index.name = "origin"
    totals = routes.sum(axis=1, keepdims=True)
    route_ratio = np.divide(routes, totals, out=np.zeros(routes.shape), where=totals != 0)
    long_rows = [
        {"origin": origin, "top_expert": expert, "count": int(routes[d, e]), "row_ratio": float(route_ratio[d, e])}
        for d, origin in enumerate(record.dataset_names)
        for e, expert in enumerate(record.expert_names)
    ]
    frames["route_confusion"] = pd.DataFrame(long_rows)
    _atomic_csv(frames["route_confusion"], os.path.join(output_dir, "Route_Confusion.csv"))
    _atomic_csv(route_frame, os.path.join(output_dir, "Route_Confusion_counts.csv"), index=True)
    if record.expert_confusion is not None:
        frames["expert_performance"] = expert_performance(record)
        _atomic_csv(frames["expert_performance"], os.path.join(output_dir, "Expert_Performance_By_Dataset.csv"))
    if record.reliability is not None:
        frames["reliability"] = reliability_frame(record)
        _atomic_csv(frames["reliability"], os.path.join(output_dir, "Class_Conditional_Reliability.csv"))
    if figures:
        figure_dir = os.path.join(output_dir, "figures")
        gate_columns = [f"mean_gate__{expert}" for expert in record.expert_names]
        fig = heatmap_figure(
            frames["gate_by_dataset"][gate_columns].to_numpy(), record.dataset_names, record.expert_names,
            "Mean gate weight by origin dataset", xlabel="Expert", ylabel="Origin dataset",
            colorbar_label="mean gate weight", vmin=0.0, vmax=1.0,
        )
        save_figure(fig, figure_dir, "Gate_Mass_Heatmap")
        fig = heatmap_figure(
            100.0 * route_ratio, record.dataset_names, record.expert_names,
            "Top expert by origin dataset", xlabel="Top expert", ylabel="Origin dataset",
            colorbar_label="row %", fmt="{:.1f}%", cmap="Blues", vmin=0.0, vmax=100.0,
        )
        save_figure(fig, figure_dir, "Route_Confusion")
        if record.reliability is not None:
            values = record.reliability.astype(np.float64)
            limit = float(np.abs(values).max()) or 1.0
            fig = heatmap_figure(
                values, record.expert_names, record.class_names,
                "Class reliability matrix R (logits)", xlabel="Class", ylabel="Expert",
                colorbar_label="logit", fmt="{:.2f}", cmap="coolwarm", vmin=-limit, vmax=limit,
            )
            save_figure(fig, figure_dir, "Class_Reliability_Heatmap")
    return frames
