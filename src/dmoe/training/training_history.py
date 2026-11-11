"""Per-epoch training history (``Training_History.csv``) and learning-curve figures."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd


TRAINING_HISTORY_FILE = "Training_History.csv"
HISTORY_KEYS = ["seed", "stage", "encoder_role", "dataset", "epoch"]
HISTORY_COLUMNS = [
    *HISTORY_KEYS,
    "objective",
    "learning_rate",
    "rows",
    "optimizer_steps",
    "examples_seen",
    "epoch_seconds",
    "train_total_loss",
    "train_ce_loss",
    "train_representation_loss",
    "train_weighted_representation_loss",
    "train_representation_fraction",
    "train_balance_penalty",
    "train_dataset_aux_loss",
    "train_anchor_penalty",
    "train_reliability_penalty",
    "val_macro_f1",
    "best_val_macro_f1",
    "improved",
    "patience_left",
]


def training_history_enabled(config: dict) -> bool:
    return bool(config.get("training", {}).get("save_epoch_history", False))


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in HISTORY_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    if frame.empty:
        return frame[HISTORY_COLUMNS]
    for column in ("stage", "encoder_role", "dataset", "objective"):
        frame[column] = frame[column].fillna("").astype(str)
    for column in ("seed", "epoch"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame = frame[HISTORY_COLUMNS]
    return frame.sort_values(HISTORY_KEYS, kind="stable").reset_index(drop=True)


def history_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, TRAINING_HISTORY_FILE)


def read_training_history(checkpoint_dir: str) -> pd.DataFrame:
    path = history_path(checkpoint_dir)
    if not os.path.isfile(path):
        return _empty_history()
    return _normalize_history(pd.read_csv(path))


def append_training_history(config: dict, row: dict[str, Any]) -> None:
    """Atomically upsert one completed epoch when history is opted in."""
    if not training_history_enabled(config):
        return
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    current = read_training_history(checkpoint_dir)
    value = {column: row.get(column, np.nan) for column in HISTORY_COLUMNS}
    value["seed"] = int(row.get("seed", config.get("seed", 0)))
    updated = pd.concat([current, pd.DataFrame([value])], ignore_index=True)
    updated = updated.drop_duplicates(HISTORY_KEYS, keep="last")
    updated = _normalize_history(updated)
    path = history_path(checkpoint_dir)
    temporary = path + ".tmp"
    updated.to_csv(temporary, index=False)
    os.replace(temporary, path)


def summarize_training_history(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage", "encoder_role", "dataset", "epoch", "objective"]
    numeric = [
        column for column in history.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "epoch"}
    ]
    grouped = history.groupby(keys, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        value if isinstance(value, str) else "__".join(part for part in value if part)
        for value in grouped.columns
    ]
    return grouped


def _save_figure(fig, output_dir: str, filename: str) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seed_training_history(
    history: pd.DataFrame,
    output_dir: str,
    *,
    selected_epoch: int | None = None,
) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    stage_a = history[history["stage"] == "A"]
    if not stage_a.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for role, group in stage_a.groupby("encoder_role"):
            group = group.sort_values("epoch")
            axes[0].plot(group["epoch"], group["train_ce_loss"], marker="o", label=role)
            axes[1].plot(group["epoch"], group["train_total_loss"], marker="o", label=role)
            if group["train_representation_loss"].notna().any():
                axes[1].plot(
                    group["epoch"], group["train_representation_loss"], linestyle="--",
                    label=f"{role} representation",
                )
        axes[0].set_title("Stage A classification loss"); axes[0].set_ylabel("CE")
        axes[1].set_title("Stage A total and representation losses")
        for axis in axes:
            axis.set_xlabel("Epoch"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, "Stage_A_Learning_Curves.png"))

    stage_b = history[history["stage"] == "B"]
    if not stage_b.empty:
        fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for dataset, group in stage_b.groupby("dataset"):
            group = group.sort_values("epoch")
            axis.plot(group["epoch"], group["train_ce_loss"], marker="o", label=dataset)
        axis.set_title("Stage B expert training loss"); axis.set_xlabel("Epoch")
        axis.set_ylabel("Weighted CE"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, "Stage_B_Learning_Curves.png"))

    stage_c = history[history["stage"] == "C"].sort_values("epoch")
    if not stage_c.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        for column, label in (
            ("train_total_loss", "total"), ("train_ce_loss", "CE"),
            ("train_balance_penalty", "balance"),
            ("train_dataset_aux_loss", "dataset auxiliary"),
            ("train_anchor_penalty", "anchor"),
            ("train_reliability_penalty", "class reliability"),
        ):
            if stage_c[column].notna().any():
                axes[0].plot(stage_c["epoch"], stage_c[column], marker="o", label=label)
        axes[0].set_title("Stage C training objectives"); axes[0].set_ylabel("Loss")
        axes[1].plot(
            stage_c["epoch"], stage_c["val_macro_f1"], marker="o", label="validation macro-F1"
        )
        axes[1].plot(
            stage_c["epoch"], stage_c["best_val_macro_f1"], linestyle="--", label="best so far"
        )
        axes[1].set_title("Stage C validation performance"); axes[1].set_ylim(0, 1)
        if selected_epoch is not None:
            for axis in axes:
                axis.axvline(selected_epoch, color="black", linestyle=":", label="selected epoch")
        for axis in axes:
            axis.set_xlabel("Epoch"); axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, "Stage_C_Learning_Curves.png"))
    return written


def plot_cross_seed_training_history(summary: pd.DataFrame, output_dir: str) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    specifications = (
        ("A", "encoder_role", "train_total_loss", "Stage A total loss", "Stage_A_Across_Seeds.png"),
        ("B", "dataset", "train_ce_loss", "Stage B expert CE", "Stage_B_Across_Seeds.png"),
        ("C", "dataset", "val_macro_f1", "Stage C validation macro-F1", "Stage_C_Across_Seeds.png"),
    )
    for stage, series_column, metric, title, filename in specifications:
        subset = summary[summary["stage"] == stage]
        mean_column = f"{metric}__mean"; std_column = f"{metric}__std"
        if subset.empty or mean_column not in subset or subset[mean_column].isna().all():
            continue
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for label, group in subset.groupby(series_column):
            group = group.sort_values("epoch")
            mean = group[mean_column].to_numpy(dtype=float)
            std = group[std_column].fillna(0).to_numpy(dtype=float)
            epochs = group["epoch"].to_numpy(dtype=int)
            axis.plot(epochs, mean, marker="o", label=str(label))
            axis.fill_between(epochs, mean - std, mean + std, alpha=0.18)
        axis.set_title(f"{title}, mean ± sample SD")
        axis.set_xlabel("Epoch"); axis.set_ylabel(metric.replace("_", " "))
        if metric == "val_macro_f1":
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.25); axis.legend(fontsize=8)
        written.append(_save_figure(fig, output_dir, filename))
    return written
