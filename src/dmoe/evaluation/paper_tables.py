"""Paper exhibits generated from ``OUTPUT_ROOT/summary`` into ``OUTPUT_ROOT/paper``.

Every number comes from the aggregated result files; nothing is typed by
hand. Cells read ``mean \\pm SD``; the number of seeds is printed in each
table (per cell when conditions differ). Only the best mean macro-F1 of each
column is bolded; weighted-F1 is never bolded.

Files (each table as ``.tex`` and ``.csv``, each figure as ``.pdf`` and ``.png``):

* ``Table_Main``: M against the baselines B1 to B3, overall and per-dataset macro-F1, parameters.
* ``Table_Ablations``: M against A1 to A3 with the paired difference from M.
* ``Table_Per_Dataset``: every condition and dataset: macro-F1, balanced accuracy, MCC, weighted-F1.
* ``Table_Per_Class``: per-class F1 of M and the baselines.
* ``Figure_Routing_<condition>``: mean gate weight by origin dataset, averaged over seeds.
* ``Figure_Confusion_<condition>``: test confusion matrix summed over seeds.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

from .. import paths
from .figures import heatmap_figure, save_figure

MAIN_CONDITIONS = ["M", "B1", "B2", "B3"]
ABLATION_CONDITIONS = ["M", "A1", "A2", "A3"]


def _read(summary_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(summary_dir, f"{name}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing; run scripts/aggregate.py first")
    return pd.read_csv(path)


def _cell(mean: float, sd: float, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def _bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def _latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def latex_table(frame: pd.DataFrame, caption: str, label: str, bold: set[tuple[int, str]] | None = None) -> str:
    """A booktabs tabular; ``bold`` holds (row position, column) pairs to bold."""
    bold = bold or set()
    columns = list(frame.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * (len(columns) - 1)}}}",
        "\\toprule",
        " & ".join(_latex_escape(c) for c in columns) + " \\\\",
        "\\midrule",
    ]
    for position, (_, row) in enumerate(frame.iterrows()):
        cells = []
        for column in columns:
            value = row[column]
            text = value if isinstance(value, str) and "$" in value else _latex_escape(value)
            cells.append(_bold(text) if (position, column) in bold else text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def _write_table(frame: pd.DataFrame, numeric: pd.DataFrame, out_dir: str, stem: str, caption: str,
                 bold: set[tuple[int, str]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    numeric.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
    with open(os.path.join(out_dir, f"{stem}.tex"), "w") as handle:
        handle.write(latex_table(frame, caption, f"tab:{stem.lower()}", bold))


def _best_rows(values: list[float]) -> set[int]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return set()
    best = max(finite)
    return {i for i, v in enumerate(values) if np.isfinite(v) and np.isclose(v, best, rtol=0, atol=1e-12)}


def _n_note(ns: set[int]) -> str:
    return f"n = {ns.pop()} seeds" if len(ns) == 1 else "n differs between conditions; see the n column"


class Summary:
    def __init__(self, summary_dir: str):
        self.dir = summary_dir
        self.overall = _read(summary_dir, "Summary_Overall")
        self.per_dataset = _read(summary_dir, "Summary_Per_Dataset")
        self.per_class = _read(summary_dir, "Summary_Per_Class")
        self.paired = _read(summary_dir, "Paired_vs_M")
        with open(os.path.join(summary_dir, "Summary_Info.json")) as handle:
            self.info = json.load(handle)
        self.display = self.info.get("display_names", {})

    def stat(self, condition: str, metric: str, origin: str | None = None) -> tuple[float, float, int]:
        frame = self.overall if origin is None else self.per_dataset[self.per_dataset["origin"] == origin]
        row = frame[(frame["condition"] == condition) & (frame["metric"] == metric)]
        if row.empty:
            return float("nan"), float("nan"), 0
        row = row.iloc[0]
        return float(row["mean"]), float(row["sd"]), int(row["n"])

    def params(self, condition: str):
        row = self.overall[self.overall["condition"] == condition]
        return int(row["total_parameters"].iloc[0]) if not row.empty and pd.notna(row["total_parameters"].iloc[0]) else None

    def datasets(self) -> list[str]:
        return list(dict.fromkeys(self.per_dataset["origin"]))

    def available(self, conditions: list[str]) -> list[str]:
        present = set(self.overall["condition"])
        return [c for c in conditions if c in present]


def _dataset_label(name: str, display: dict[str, str]) -> str:
    return display.get(name, name)


def main_table(summary: Summary, out_dir: str, conditions: list[str], stem: str, caption: str,
               dataset_display: dict[str, str]) -> None:
    conditions = summary.available(conditions)
    if not conditions:
        return
    datasets = summary.datasets()
    columns = ["Condition", "Params", "n", "Macro-F1", *[_dataset_label(d, dataset_display) for d in datasets]]
    rows, numeric, ns = [], [], set()
    means: dict[str, list[float]] = {c: [] for c in columns[3:]}
    for condition in conditions:
        mean, sd, n = summary.stat(condition, "macro_f1")
        ns.add(n)
        row = {"Condition": summary.display.get(condition, condition), "Params": f"{summary.params(condition):,}"
               if summary.params(condition) else "--", "n": str(n), "Macro-F1": _cell(mean, sd)}
        num = {"condition": condition, "params": summary.params(condition), "n": n, "macro_f1_mean": mean, "macro_f1_sd": sd}
        means["Macro-F1"].append(mean)
        for dataset, label in zip(datasets, columns[4:]):
            m, s, _ = summary.stat(condition, "macro_f1", dataset)
            row[label] = _cell(m, s)
            num[f"{dataset}__macro_f1_mean"], num[f"{dataset}__macro_f1_sd"] = m, s
            means[label].append(m)
        rows.append(row)
        numeric.append(num)
    bold = {(i, column) for column, values in means.items() for i in _best_rows(values)}
    caption = f"{caption} Test macro-F1, mean $\\pm$ SD over seeds ({_n_note(ns)}); best per column in bold."
    _write_table(pd.DataFrame(rows, columns=columns), pd.DataFrame(numeric), out_dir, stem, caption, bold)


def ablation_table(summary: Summary, out_dir: str) -> None:
    conditions = summary.available(ABLATION_CONDITIONS)
    if not conditions:
        return
    rows, numeric, ns, means = [], [], set(), []
    for condition in conditions:
        mean, sd, n = summary.stat(condition, "macro_f1")
        ns.add(n)
        means.append(mean)
        paired = summary.paired[(summary.paired["condition"] == condition) & (summary.paired["origin"] == "ALL")
                                & (summary.paired["metric"] == "macro_f1")]
        if condition == "M" or paired.empty:
            diff_text, favoring, n_common = "--", "--", None
            diff_mean = diff_sd = float("nan")
        else:
            p = paired.iloc[0]
            diff_mean, diff_sd, n_common = float(p["difference_mean"]), float(p["difference_sd"]), int(p["n_common_seeds"])
            diff_text = _cell(diff_mean, diff_sd)
            favoring = f"{int(p['seeds_favoring_reference'])}/{n_common}"
        params = summary.params(condition)
        rows.append({"Condition": summary.display.get(condition, condition), "Params": f"{params:,}" if params else "--",
                     "n": str(n), "Macro-F1": _cell(mean, sd), "M minus condition": diff_text, "Seeds favouring M": favoring})
        numeric.append({"condition": condition, "params": params, "n": n, "macro_f1_mean": mean, "macro_f1_sd": sd,
                        "difference_vs_m_mean": diff_mean, "difference_vs_m_sd": diff_sd, "n_common_seeds": n_common})
    bold = {(i, "Macro-F1") for i in _best_rows(means)}
    caption = (f"Ablations. Test macro-F1, mean $\\pm$ SD over seeds ({_n_note(ns)}); the difference uses the seeds "
               "both conditions completed. A3 is not capacity matched (see Params).")
    _write_table(pd.DataFrame(rows), pd.DataFrame(numeric), out_dir, "Table_Ablations", caption, bold)


def per_dataset_table(summary: Summary, out_dir: str, dataset_display: dict[str, str]) -> None:
    conditions = summary.available(["M", "B1", "B2", "B3", "A1", "A2", "A3"])
    metrics = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Bal. acc."), ("matthews_corrcoef", "MCC"),
               ("weighted_f1", "Weighted-F1 (ref.)")]
    rows, numeric, bold = [], [], set()
    for dataset in summary.datasets():
        block_start = len(rows)
        macro = []
        for condition in conditions:
            row = {"Dataset": _dataset_label(dataset, dataset_display), "Condition": summary.display.get(condition, condition)}
            num = {"dataset": dataset, "condition": condition}
            for metric, label in metrics:
                m, s, n = summary.stat(condition, metric, dataset)
                row[label] = _cell(m, s)
                num[f"{metric}_mean"], num[f"{metric}_sd"], num["n"] = m, s, n
                if metric == "macro_f1":
                    macro.append(m)
            rows.append(row)
            numeric.append(num)
        bold |= {(block_start + i, "Macro-F1") for i in _best_rows(macro)}
    caption = "Per-dataset test results, mean $\\pm$ SD over seeds. Best macro-F1 per dataset in bold; weighted-F1 is reference only."
    _write_table(pd.DataFrame(rows), pd.DataFrame(numeric), out_dir, "Table_Per_Dataset", caption, bold)


def per_class_table(summary: Summary, out_dir: str) -> None:
    conditions = summary.available(MAIN_CONDITIONS)
    frame = summary.per_class[summary.per_class["metric"] == "f1"]
    classes = list(dict.fromkeys(frame["class"]))
    rows, numeric, bold = [], [], set()
    for position, class_name in enumerate(classes):
        row, num, values = {"Class": class_name}, {"class": class_name}, []
        for condition in conditions:
            match = frame[(frame["class"] == class_name) & (frame["condition"] == condition)]
            m, s = (float(match["mean"].iloc[0]), float(match["sd"].iloc[0])) if not match.empty else (float("nan"), float("nan"))
            label = summary.display.get(condition, condition)
            row[label] = _cell(m, s, 3)
            num[f"{condition}_f1_mean"], num[f"{condition}_f1_sd"] = m, s
            values.append(m)
        bold |= {(position, summary.display.get(conditions[i], conditions[i])) for i in _best_rows(values)}
        rows.append(row)
        numeric.append(num)
    caption = "Per-class test F1 (all datasets pooled), mean $\\pm$ SD over seeds; best per class in bold."
    _write_table(pd.DataFrame(rows), pd.DataFrame(numeric), out_dir, "Table_Per_Class", caption, bold)


def routing_figures(output_root: str, out_dir: str, display: dict[str, str], dataset_display: dict[str, str]) -> None:
    from .aggregate import collect_runs

    runs = collect_runs(output_root)
    for condition in sorted({r.condition for r in runs}):
        frames = [
            pd.read_csv(os.path.join(r.run_dir, "eval", "test", "Gate_By_Dataset.csv"))
            for r in runs if r.condition == condition
            and os.path.isfile(os.path.join(r.run_dir, "eval", "test", "Gate_By_Dataset.csv"))
        ]
        if not frames:
            continue
        columns = [c for c in frames[0].columns if c.startswith("mean_gate__")]
        mean = sum(f[columns].to_numpy() for f in frames) / len(frames)
        experts = [c.replace("mean_gate__", "") for c in columns]
        fig = heatmap_figure(
            mean, [_dataset_label(o, dataset_display) for o in frames[0]["origin"]],
            [_dataset_label(e, dataset_display) for e in experts],
            f"{display.get(condition, condition)}: mean gate weight (n = {len(frames)} seeds)",
            xlabel="Expert", ylabel="Origin dataset", colorbar_label="mean gate weight", vmin=0.0, vmax=1.0,
        )
        save_figure(fig, out_dir, f"Figure_Routing_{condition}")


def confusion_figures(summary_dir: str, out_dir: str) -> None:
    base = os.path.join(summary_dir, "confusion")
    if not os.path.isdir(base):
        return
    for condition in sorted(os.listdir(base)):
        for extension in ("pdf", "png"):
            source = os.path.join(base, condition, f"Confusion_Matrix_{condition}_summed.{extension}")
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(out_dir, f"Figure_Confusion_{condition}.{extension}"))


def make_paper_tables(output_root: str | None = None, dataset_display: dict[str, str] | None = None) -> str:
    root = output_root or paths.output_root()
    summary_dir = paths.summary_dir(root)
    out_dir = paths.paper_dir(root)
    os.makedirs(out_dir, exist_ok=True)
    summary = Summary(summary_dir)
    if dataset_display is None:
        from ..training.config import BASE_CONFIG, load_yaml

        dataset_display = load_yaml(BASE_CONFIG).get("data", {}).get("display_names", {})
    main_table(summary, out_dir, MAIN_CONDITIONS, "Table_Main", "Dataset-expert MoE against the baselines.", dataset_display)
    ablation_table(summary, out_dir)
    per_dataset_table(summary, out_dir, dataset_display)
    per_class_table(summary, out_dir)
    routing_figures(root, out_dir, summary.display, dataset_display)
    confusion_figures(summary_dir, out_dir)
    return out_dir
