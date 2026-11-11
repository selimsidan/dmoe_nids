"""Confusion matrices as CSV (long, wide counts, row-normalized) and figures.

The figure shows true classes as rows and predicted classes as columns. Each
non-empty cell prints the count (thousands separators) over the row
percentage with one decimal, for example ``12,431`` over ``(87.3%)``. Cell
colour encodes the row percentage; the colour bar is labelled "row %".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .metrics import confusion_long_rows


def _atomic_csv(frame: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def present_classes(confusion: np.ndarray, class_names: list[str]) -> list[int]:
    """Indices of classes that appear as a true or predicted label."""
    confusion = np.asarray(confusion)
    keep = (confusion.sum(axis=1) > 0) | (confusion.sum(axis=0) > 0)
    return [index for index in range(len(class_names)) if keep[index]]


def confusion_figure(
    confusion: np.ndarray,
    class_names: list[str],
    title: str,
    *,
    cmap: str = "Blues",
):
    import matplotlib
    import matplotlib.pyplot as plt

    confusion = np.asarray(confusion, dtype=np.int64)
    n = len(class_names)
    row_totals = confusion.sum(axis=1, keepdims=True)
    row_percent = np.divide(
        100.0 * confusion, row_totals, out=np.zeros_like(confusion, dtype=np.float64), where=row_totals != 0
    )
    cell = 0.62 if n > 12 else 0.78
    size = max(5.5, cell * n + 2.4)
    fig, axis = plt.subplots(figsize=(size, size * 0.92), constrained_layout=True)
    image = axis.imshow(row_percent, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    colorbar.set_label("row %")
    font = 8 if n > 12 else 9
    norm = matplotlib.colors.Normalize(0, 100)
    colormap = plt.get_cmap(cmap)
    for i in range(n):
        for j in range(n):
            count = int(confusion[i, j])
            if count == 0:
                continue
            rgba = colormap(norm(row_percent[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            colour = "white" if luminance < 0.5 else "black"
            axis.text(
                j, i, f"{count:,}\n({row_percent[i, j]:.1f}%)",
                ha="center", va="center", fontsize=font - 1, color=colour, linespacing=1.1,
            )
    axis.set_xticks(range(n))
    axis.set_yticks(range(n))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=font)
    axis.set_yticklabels(class_names, fontsize=font)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title, fontsize=font + 3)
    axis.set_xticks(np.arange(-0.5, n, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.6)
    axis.tick_params(which="minor", length=0)
    return fig


def write_confusion_outputs(
    confusion: np.ndarray,
    class_names: list[str],
    output_dir: str,
    stem: str,
    title: str,
    *,
    only_present: bool = False,
    figures: bool = True,
) -> dict[str, str]:
    """Write ``<stem>_long.csv``, ``<stem>_counts.csv``, ``<stem>_row_normalized.csv`` and figures."""
    import matplotlib.pyplot as plt

    confusion = np.asarray(confusion, dtype=np.int64)
    names = list(class_names)
    if only_present:
        keep = present_classes(confusion, names)
        confusion = confusion[np.ix_(keep, keep)]
        names = [names[index] for index in keep]
    os.makedirs(output_dir, exist_ok=True)
    written = {}
    long_frame = pd.DataFrame(confusion_long_rows(confusion, names))
    written["long"] = os.path.join(output_dir, f"{stem}_long.csv")
    _atomic_csv(long_frame, written["long"])
    counts = pd.DataFrame(confusion, index=names, columns=names)
    counts.index.name = "true_class"
    written["counts"] = os.path.join(output_dir, f"{stem}_counts.csv")
    _atomic_csv(counts, written["counts"], index=True)
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_totals, out=np.zeros(confusion.shape), where=row_totals != 0)
    normalized_frame = pd.DataFrame(normalized, index=names, columns=names)
    normalized_frame.index.name = "true_class"
    written["row_normalized"] = os.path.join(output_dir, f"{stem}_row_normalized.csv")
    _atomic_csv(normalized_frame, written["row_normalized"], index=True)
    if figures and len(names):
        fig = confusion_figure(confusion, names, title)
        for extension in ("png", "pdf"):
            path = os.path.join(output_dir, f"{stem}.{extension}")
            fig.savefig(path, dpi=200 if extension == "png" else None, bbox_inches="tight")
            written[extension] = path
        plt.close(fig)
    return written
