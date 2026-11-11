"""Small shared figure helpers: annotated heatmaps and table figures."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def save_figure(fig, output_dir: str, stem: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    written = {}
    for extension in ("png", "pdf"):
        path = os.path.join(output_dir, f"{stem}.{extension}")
        fig.savefig(path, dpi=200 if extension == "png" else None, bbox_inches="tight")
        written[extension] = path
    plt.close(fig)
    return written


def heatmap_figure(
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    *,
    xlabel: str = "",
    ylabel: str = "",
    colorbar_label: str = "",
    fmt: str = "{:.2f}",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float64)
    rows, columns = values.shape
    fig, axis = plt.subplots(
        figsize=(max(4.5, 0.75 * columns + 2.5), max(3.2, 0.5 * rows + 1.8)), constrained_layout=True
    )
    image = axis.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    low, high = image.get_clim()
    for i in range(rows):
        for j in range(columns):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            share = (value - low) / (high - low) if high > low else 0.5
            axis.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8,
                      color="white" if share < 0.55 else "black")
    axis.set_xticks(range(columns))
    axis.set_yticks(range(rows))
    axis.set_xticklabels(column_labels, rotation=30, ha="right", fontsize=8)
    axis.set_yticklabels(row_labels, fontsize=8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    return fig


def table_figure(frame: pd.DataFrame, title: str, *, float_format: str = "{:.4f}", font_size: int = 8):
    """Render a small table; float columns always use ``float_format``, integer columns thousands separators."""
    import matplotlib.pyplot as plt

    def fmt(value, dtype):
        if pd.isna(value):
            return "--"
        if pd.api.types.is_integer_dtype(dtype):
            return f"{int(value):,}"
        if pd.api.types.is_float_dtype(dtype):
            return float_format.format(float(value))
        return str(value)

    cells = [[fmt(value, frame.dtypes.iloc[i]) for i, value in enumerate(row)] for row in frame.itertuples(index=False)]
    widths = [max(len(str(c)), *(len(r[i]) for r in cells)) if cells else len(str(c)) for i, c in enumerate(frame.columns)]
    total = sum(widths) or 1
    height = 0.28 * (len(frame) + 2) + 0.8
    width = max(5.0, 0.11 * total + 1.0)
    fig, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=list(frame.columns), loc="center", cellLoc="center",
                       colWidths=[w / total for w in widths])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.2)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
    axis.set_title(title, fontsize=font_size + 3, pad=10)
    return fig
