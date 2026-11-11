"""Cross-seed aggregation of completed runs into ``OUTPUT_ROOT/summary/``.

Reads every ``runs/<condition>/seed<k>/manifest.json`` with status
``complete`` and writes:

* ``Summary_Overall.csv`` / ``Summary_Per_Dataset.csv`` / ``Summary_Per_Class.csv``:
  per condition (and dataset or class) and metric: mean, SD (ddof 1), min,
  max, n, the seeds used, and a t-based 95% interval of the mean.
* ``Paired_vs_M.csv`` and ``Paired_vs_M_Per_Seed.csv``: differences from the
  reference condition (M) on the seeds both have completed.
* ``Significance.csv``: per common seed, a paired bootstrap of the test
  macro-F1 difference and McNemar's test on per-row correctness.
* ``Recommended_Seeds.csv``: the seed with the highest **validation**
  macro-F1 per condition (ties go to the lower seed). Test results never
  enter this choice.
* ``confusion/<condition>/``: the confusion matrix summed over seeds.
* ``training/<condition>/``: learning curves across seeds (mean and SD bands).
* ``Summary_Info.json``: seeds per condition, warnings, split signature.

Runs trained on different splits are never combined: aggregation stops with
an error naming them.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .. import paths
from .confusion import write_confusion_outputs
from .metrics import overall_from_confusion
from .predictions import PREDICTION_FILES, PredictionRecord

REFERENCE = "M"
SUMMARY_METRICS = [
    "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "accuracy",
    "balanced_accuracy", "matthews_corrcoef", "cohen_kappa",
    "roc_auc_ovr_macro", "pr_auc_ovr_macro",
]
PER_CLASS_METRICS = ["f1", "precision", "recall", "roc_auc_ovr", "pr_auc_ovr"]


@dataclass
class RunInfo:
    condition: str
    seed: int
    run_dir: str
    manifest: dict

    @property
    def split_key(self) -> str:
        split = self.manifest.get("split", {})
        return json.dumps(
            {"signatures": split.get("signatures"), "preprocessing": split.get("preprocessing_signature")},
            sort_keys=True,
        )

    def eval_csv(self, split: str, name: str) -> str:
        return os.path.join(self.run_dir, "eval", split, name)

    def predictions(self, split: str = "test") -> str:
        return os.path.join(self.run_dir, "eval", split, PREDICTION_FILES[split])


def collect_runs(output_root: str, conditions: list[str] | None = None, seeds: list[int] | None = None) -> list[RunInfo]:
    base = os.path.join(output_root, "runs")
    runs = []
    if not os.path.isdir(base):
        return runs
    for condition in sorted(os.listdir(base)):
        if conditions and condition not in conditions:
            continue
        for name in sorted(os.listdir(os.path.join(base, condition))):
            if not name.startswith("seed") or not name[4:].isdigit():
                continue
            seed = int(name[4:])
            if seeds is not None and seed not in seeds:
                continue
            run_dir = os.path.join(base, condition, name)
            manifest_path = os.path.join(run_dir, "manifest.json")
            if not os.path.isfile(manifest_path):
                continue
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            if manifest.get("status") == "complete":
                runs.append(RunInfo(condition, seed, run_dir, manifest))
    return runs


def check_single_split(runs: list[RunInfo]) -> str:
    groups: dict[str, list[str]] = {}
    for run in runs:
        groups.setdefault(run.split_key, []).append(f"{run.condition} seed {run.seed}")
    if len(groups) > 1:
        detail = "; ".join(f"split {i + 1}: {', '.join(v)}" for i, v in enumerate(groups.values()))
        raise ValueError(f"runs were trained on different splits and cannot be compared: {detail}")
    return next(iter(groups)) if groups else ""


def t_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    mean = values.mean()
    half = stats.t.ppf(0.5 + level / 2, n - 1) * values.std(ddof=1) / math.sqrt(n)
    return float(mean - half), float(mean + half)


def describe(values, seeds) -> dict:
    values = np.asarray(values, dtype=np.float64)
    low, high = t_interval(values)
    return {
        "mean": float(values.mean()) if len(values) else float("nan"),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(values.min()) if len(values) else float("nan"),
        "max": float(values.max()) if len(values) else float("nan"),
        "n": int(len(values)),
        "seeds": " ".join(str(int(s)) for s in seeds),
        "ci95_low": low,
        "ci95_high": high,
    }


def _per_run_table(runs: list[RunInfo], split: str, filename: str) -> pd.DataFrame:
    frames = []
    for run in runs:
        path = run.eval_csv(split, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{run.condition} seed {run.seed} is complete but has no {split}/{filename}")
        frame = pd.read_csv(path)
        frame["condition"] = run.condition
        frame["seed"] = run.seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(table: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for group, frame in table.groupby(keys, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        frame = frame.sort_values("seed")
        for metric in metrics:
            if metric not in frame:
                continue
            rows.append({**dict(zip(keys, group)), "metric": metric, **describe(frame[metric], frame["seed"])})
    return pd.DataFrame(rows)


def paired_differences(table: pd.DataFrame, keys: list[str], metrics: list[str], reference: str = REFERENCE):
    """Differences (reference minus condition) on the seeds both conditions completed."""
    per_seed, rows = [], []
    if reference not in set(table["condition"]):
        return pd.DataFrame(), pd.DataFrame()
    group_keys = [k for k in keys if k != "condition"]
    for condition in [c for c in table["condition"].unique() if c != reference]:
        for metric in metrics:
            if metric not in table:
                continue
            ref = table[table["condition"] == reference].set_index([*group_keys, "seed"])[metric]
            other = table[table["condition"] == condition].set_index([*group_keys, "seed"])[metric]
            joined = pd.concat([ref.rename("reference"), other.rename("condition")], axis=1, join="inner").reset_index()
            for group, frame in joined.groupby(group_keys, sort=False) if group_keys else [((), joined)]:
                group = group if isinstance(group, tuple) else (group,)
                frame = frame.sort_values("seed")
                diff = (frame["reference"] - frame["condition"]).to_numpy()
                label = dict(zip(group_keys, group))
                for seed, value in zip(frame["seed"], diff):
                    per_seed.append({"reference": reference, "condition": condition, **label, "metric": metric,
                                     "seed": int(seed), "difference": float(value)})
                rows.append({
                    "reference": reference, "condition": condition, **label, "metric": metric,
                    **{f"difference_{k}": v for k, v in describe(diff, frame["seed"]).items() if k not in ("seeds", "n")},
                    "n_common_seeds": int(len(diff)),
                    "common_seeds": " ".join(str(int(s)) for s in frame["seed"]),
                    "seeds_favoring_reference": int((diff > 0).sum()),
                })
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    """Macro-F1 for a stack of confusion matrices (B x C x C), classes with support only."""
    support = confusions.sum(axis=2)
    predicted = confusions.sum(axis=1)
    tp = np.diagonal(confusions, axis1=1, axis2=2)
    precision = np.divide(tp, predicted, out=np.zeros(tp.shape), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(tp.shape), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(tp.shape), where=(precision + recall) != 0)
    present = support > 0
    return (f1 * present).sum(axis=1) / np.maximum(present.sum(axis=1), 1)


def paired_bootstrap_macro_f1(y_true, pred_a, pred_b, num_classes: int, resamples: int, seed: int) -> dict:
    """Row-level paired bootstrap of macro-F1(a) minus macro-F1(b).

    Resampling rows with replacement is equivalent to drawing multinomial
    counts over the (true, a, b) cells, which makes the bootstrap exact and
    fast even for tens of millions of rows.
    """
    c = num_classes
    cells = (np.asarray(y_true, np.int64) * c + np.asarray(pred_a, np.int64)) * c + np.asarray(pred_b, np.int64)
    counts = np.bincount(cells, minlength=c ** 3)
    nonzero = np.flatnonzero(counts)
    total = int(counts.sum())
    probabilities = counts[nonzero] / total
    t_idx, a_idx, b_idx = nonzero // (c * c), (nonzero // c) % c, nonzero % c
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(total, probabilities, size=resamples)  # resamples x cells
    conf_a = np.zeros((resamples, c, c))
    conf_b = np.zeros((resamples, c, c))
    np.add.at(conf_a, (slice(None), t_idx, a_idx), draws)
    np.add.at(conf_b, (slice(None), t_idx, b_idx), draws)
    differences = _macro_f1_from_confusions(conf_a) - _macro_f1_from_confusions(conf_b)
    base_a = overall_from_confusion(np.bincount(t_idx * c + a_idx, weights=counts[nonzero], minlength=c * c).reshape(c, c))["macro_f1"]
    base_b = overall_from_confusion(np.bincount(t_idx * c + b_idx, weights=counts[nonzero], minlength=c * c).reshape(c, c))["macro_f1"]
    observed = base_a - base_b
    p_two_sided = float(min(1.0, 2 * min((differences <= 0).mean(), (differences >= 0).mean())))
    return {
        "observed_difference": float(observed),
        "bootstrap_ci95_low": float(np.percentile(differences, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(differences, 97.5)),
        "bootstrap_p_two_sided": p_two_sided,
        "bootstrap_resamples": int(resamples),
    }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b
    if discordant == 0:
        p, method, statistic = 1.0, "none (no discordant rows)", 0.0
    elif discordant < 25:
        p = float(stats.binomtest(only_a, discordant, 0.5).pvalue)
        method, statistic = "exact binomial", float(only_a)
    else:
        statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
        p = float(stats.chi2.sf(statistic, 1))
        method = "chi-square with continuity correction"
    return {"mcnemar_only_reference_correct": only_a, "mcnemar_only_condition_correct": only_b,
            "mcnemar_statistic": float(statistic), "mcnemar_p": p, "mcnemar_method": method}


def significance(runs: list[RunInfo], resamples: int, seed: int, reference: str = REFERENCE) -> pd.DataFrame:
    by_key = {(run.condition, run.seed): run for run in runs}
    rows = []
    reference_seeds = sorted(s for (c, s) in by_key if c == reference)
    for condition in sorted({c for c, _ in by_key if c != reference}):
        for s in reference_seeds:
            if (condition, s) not in by_key:
                continue
            a = PredictionRecord.load(by_key[(reference, s)].predictions("test"))
            b = PredictionRecord.load(by_key[(condition, s)].predictions("test"))
            if not np.array_equal(a.y_true, b.y_true) or not np.array_equal(a.dataset_idx, b.dataset_idx):
                raise ValueError(f"{reference} and {condition} seed {s} were evaluated on different test rows")
            rows.append({
                "reference": reference, "condition": condition, "seed": s, "rows": a.rows,
                **paired_bootstrap_macro_f1(a.y_true, a.y_pred, b.y_pred, a.num_classes, resamples, seed + s),
                **mcnemar(a.y_true, a.y_pred, b.y_pred),
            })
    return pd.DataFrame(rows)


def recommended_seeds(runs: list[RunInfo], reference: str = REFERENCE) -> pd.DataFrame:
    """Highest validation macro-F1 per condition; ties go to the lower seed. Test is never read."""
    rows = []
    for condition in sorted({run.condition for run in runs}):
        candidates = []
        for run in sorted((r for r in runs if r.condition == condition), key=lambda r: r.seed):
            frame = pd.read_csv(run.eval_csv("val", "Overall_Metrics.csv"))
            value = float(frame.loc[frame["origin"] == "ALL", "macro_f1"].iloc[0])
            candidates.append((value, run.seed))
        best_value = max(v for v, _ in candidates)
        best_seed = min(s for v, s in candidates if v == best_value)
        rows.append({
            "condition": condition, "recommended_seed": best_seed, "val_macro_f1": best_value,
            "rule": "max_val_macro_f1, ties to lower seed", "seeds_considered": " ".join(str(s) for _, s in candidates),
            "advertised": condition == reference,
        })
    return pd.DataFrame(rows)


def _atomic_csv(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def summed_confusions(runs: list[RunInfo], output_dir: str, display: dict[str, str]) -> None:
    for condition in sorted({run.condition for run in runs}):
        selected = [r for r in runs if r.condition == condition]
        total = None
        for run in selected:
            record = PredictionRecord.load(run.predictions("test"))
            matrix = record.confusion()
            total = matrix if total is None else total + matrix
            class_names = record.class_names
        seeds = ", ".join(str(r.seed) for r in sorted(selected, key=lambda r: r.seed))
        write_confusion_outputs(
            total, class_names, os.path.join(output_dir, "confusion", condition), f"Confusion_Matrix_{condition}_summed",
            f"{display.get(condition, condition)}: test confusion summed over seeds {seeds}",
        )


def cross_seed_curves(runs: list[RunInfo], output_dir: str) -> None:
    from ..training.training_history import plot_cross_seed_training_history, summarize_training_history

    for condition in sorted({run.condition for run in runs}):
        frames = []
        for run in runs:
            path = os.path.join(run.run_dir, "Training_History.csv")
            if run.condition == condition and os.path.isfile(path):
                frames.append(pd.read_csv(path))
        if not frames:
            continue
        history = pd.concat(frames, ignore_index=True)
        summary = summarize_training_history(history)
        target = os.path.join(output_dir, "training", condition)
        _atomic_csv(summary, os.path.join(target, "Training_History_Across_Seeds.csv"))
        plot_cross_seed_training_history(summary, target)


def aggregate(
    output_root: str | None = None,
    *,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    resamples: int = 1000,
    aggregation_seed: int = 0,
    figures: bool = True,
    display_names: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    root = output_root or paths.output_root()
    out = paths.summary_dir(root)
    runs = collect_runs(root, conditions, seeds)
    if not runs:
        raise FileNotFoundError(f"no completed runs under {os.path.join(root, 'runs')}")
    split_key = check_single_split(runs)
    seeds_by_condition = {c: sorted(r.seed for r in runs if r.condition == c) for c in sorted({r.condition for r in runs})}
    warnings = []
    if len({tuple(v) for v in seeds_by_condition.values()}) > 1:
        warnings.append("conditions have different completed seed sets: " + "; ".join(
            f"{c}: {' '.join(map(str, v))}" for c, v in seeds_by_condition.items()))
    for condition, seed_list in seeds_by_condition.items():
        print(f"[aggregate] {condition}: n={len(seed_list)} seeds={seed_list}", flush=True)
    for warning in warnings:
        print(f"[aggregate] WARNING: {warning}", flush=True)

    overall = _per_run_table(runs, "test", "Overall_Metrics.csv")
    per_class = _per_run_table(runs, "test", "Per_Class_Metrics.csv")
    tables = {
        "Summary_Overall": summarize(overall[overall["origin"] == "ALL"], ["condition"], SUMMARY_METRICS),
        "Summary_Per_Dataset": summarize(overall[overall["origin"] != "ALL"], ["condition", "origin"], SUMMARY_METRICS),
        "Summary_Per_Class": summarize(per_class[per_class["origin"] == "ALL"], ["condition", "class"], PER_CLASS_METRICS),
    }
    paired, paired_seed = paired_differences(overall, ["condition", "origin"], ["macro_f1", "accuracy", "weighted_f1"])
    tables["Paired_vs_M"], tables["Paired_vs_M_Per_Seed"] = paired, paired_seed
    tables["Significance"] = significance(runs, resamples, aggregation_seed)
    tables["Recommended_Seeds"] = recommended_seeds(runs)
    params = {r.condition: r.manifest.get("total_parameters") for r in runs}
    tables["Summary_Overall"].insert(1, "total_parameters", tables["Summary_Overall"]["condition"].map(params))
    for name, frame in tables.items():
        _atomic_csv(frame, os.path.join(out, f"{name}.csv"))
    display = {r.condition: r.manifest.get("display_name") or r.condition for r in runs}
    display.update(display_names or {})
    if figures:
        summed_confusions(runs, out, display)
        cross_seed_curves(runs, out)
    info = {
        "seeds_by_condition": seeds_by_condition,
        "warnings": warnings,
        "split": json.loads(split_key) if split_key else None,
        "reference": REFERENCE,
        "bootstrap_resamples": resamples,
        "aggregation_seed": aggregation_seed,
        "display_names": display,
    }
    with open(os.path.join(out, "Summary_Info.json"), "w") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
    return tables
