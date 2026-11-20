"""Supplementary analysis: per-class score offsets tuned on validation for macro-F1.

No model optimizes macro-F1 directly: every condition minimizes a
class-weighted cross-entropy, and macro-F1 enters only through the class
weights and validation selection. This analysis asks how much each trained
model gains from one post-hoc step that targets macro-F1, applied identically
to every condition and seed:

* scores are the model's log class probabilities;
* one offset per class (class 0, Benign, fixed at 0) is added before the
  argmax; the offsets are chosen by coordinate ascent over a fixed grid to
  maximize validation macro-F1;
* the fit uses a class-capped subset of the validation rows (every row of
  classes up to ``fit_cap_per_class``, ``fit_cap_per_class`` rows of larger
  classes), with each kept row weighted by n_c / k_c so the weighted
  confusion matrix estimates the full validation one;
* the offsets are then applied unchanged to the test split.

Test data is never used for fitting. The main results stay untuned; this is
reported separately (``summary/Bias_Tuning_*.csv``).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .metrics import confusion_from_predictions, overall_from_confusion

BIAS_FILE = "Bias_Tuning.json"
DEFAULTS = {
    "fit_cap_per_class": 200_000,
    "fit_seed": 0,
    "grid": [round(float(v), 4) for v in np.arange(-8.0, 8.0001, 0.25)],
    "passes": 3,
    "probability_floor": 1e-12,
}
REPORT_METRICS = ("macro_f1", "macro_precision", "macro_recall", "accuracy", "weighted_f1")


def fit_subset(labels: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted row indices of a class-capped subset and each kept row's weight n_c / k_c."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(int(seed))
    chosen, weights = [], []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        kept = rows if len(rows) <= cap else np.sort(rng.choice(rows, size=int(cap), replace=False))
        chosen.append(kept)
        weights.append(np.full(len(kept), len(rows) / len(kept)))
    order = np.argsort(np.concatenate(chosen), kind="stable")
    return np.concatenate(chosen)[order], np.concatenate(weights)[order]


def weighted_confusion(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray, num_classes: int) -> np.ndarray:
    flat = np.asarray(truth, dtype=np.int64) * num_classes + np.asarray(prediction, dtype=np.int64)
    return np.bincount(flat, weights=weights, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def macro_f1(confusion: np.ndarray) -> float:
    """Macro-F1 over classes with support (the evaluation definition)."""
    confusion = np.asarray(confusion, dtype=np.float64)
    support, predicted, tp = confusion.sum(axis=1), confusion.sum(axis=0), np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0


def fit_offsets(scores: np.ndarray, truth: np.ndarray, weights: np.ndarray, grid, passes: int) -> tuple[np.ndarray, float, float]:
    """Coordinate ascent on per-class offsets (class 0 fixed); returns offsets, start and final weighted macro-F1."""
    scores = np.asarray(scores, dtype=np.float32)
    rows, num_classes = scores.shape
    offsets = np.zeros(num_classes, dtype=np.float32)
    start = best = macro_f1(weighted_confusion(truth, scores.argmax(axis=1), weights, num_classes))
    index = np.arange(rows)
    grid = np.asarray(grid, dtype=np.float32)
    for _ in range(int(passes)):
        changed = False
        for label in range(1, num_classes):
            others = scores + offsets
            others[:, label] = -np.inf
            runner_up = others.argmax(axis=1)
            margin = others[index, runner_up] - scores[:, label]
            del others
            current, chosen = best, offsets[label]
            for value in grid:
                prediction = np.where(value > margin, label, runner_up)
                score = macro_f1(weighted_confusion(truth, prediction, weights, num_classes))
                if score > current + 1e-12:
                    current, chosen = score, value
            if chosen != offsets[label]:
                offsets[label], best, changed = chosen, current, True
        if not changed:
            break
    return offsets, start, best


def _log_scores(bundle, features: np.ndarray, floor: float) -> np.ndarray:
    probabilities = bundle.predict_standardized(features)["probabilities"]
    return np.log(np.maximum(np.asarray(probabilities, dtype=np.float32), floor))


def _metrics(confusion: np.ndarray) -> dict:
    row = overall_from_confusion(confusion)
    return {key: float(row[key]) for key in REPORT_METRICS}


def tune_run(bundle, context, out_path: str, *, chunk_rows: int = 262_144, settings: dict | None = None) -> dict:
    """Fit offsets on validation for one trained bundle, apply them to test, write ``out_path``."""
    settings = {**DEFAULTS, **(settings or {})}
    if os.path.isfile(out_path):
        with open(out_path) as handle:
            existing = json.load(handle)
        if existing.get("settings") == settings:
            return existing
    data = context.data
    num_classes = len(data.class_names)
    floor = float(settings["probability_floor"])

    val = data.val
    total = len(val.class_idx)
    truth = np.empty(total, dtype=np.int64)
    scores = np.empty((total, num_classes), dtype=np.float32)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        truth[start:stop] = np.asarray(val.class_idx[start:stop])
        scores[start:stop] = _log_scores(bundle, val.features[start:stop], floor)
    fit_rows, fit_weights = fit_subset(truth, int(settings["fit_cap_per_class"]), int(settings["fit_seed"]))
    offsets, fit_start, fit_best = fit_offsets(scores[fit_rows], truth[fit_rows], fit_weights,
                                               settings["grid"], int(settings["passes"]))
    val_raw = confusion_from_predictions(truth, scores.argmax(axis=1), num_classes)
    val_tuned = confusion_from_predictions(truth, (scores + offsets).argmax(axis=1), num_classes)
    del scores

    test = data.test
    names = list(data.active_datasets)
    raw = np.zeros((len(names), num_classes, num_classes), dtype=np.int64)
    tuned = np.zeros_like(raw)
    total = len(test.class_idx)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        y = np.asarray(test.class_idx[start:stop], dtype=np.int64)
        origin = np.asarray(test.dataset_idx[start:stop], dtype=np.int64)
        block = _log_scores(bundle, test.features[start:stop], floor)
        plain, adjusted = block.argmax(axis=1), (block + offsets).argmax(axis=1)
        for index in np.unique(origin):
            mask = origin == index
            raw[index] += confusion_from_predictions(y[mask], plain[mask], num_classes)
            tuned[index] += confusion_from_predictions(y[mask], adjusted[mask], num_classes)
    result = {
        "settings": settings,
        "class_names": list(data.class_names),
        "offsets": [float(v) for v in offsets],
        "fit_rows": int(len(fit_rows)),
        "fit_weighted_val_macro_f1": {"raw": fit_start, "tuned": fit_best},
        "val": {"raw": _metrics(val_raw), "tuned": _metrics(val_tuned)},
        "test": {"raw": _metrics(raw.sum(axis=0)), "tuned": _metrics(tuned.sum(axis=0))},
        "test_per_dataset": {
            name: {"raw": _metrics(raw[i]), "tuned": _metrics(tuned[i])} for i, name in enumerate(names)
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w") as handle:
        json.dump(result, handle, indent=2)
    os.replace(out_path + ".tmp", out_path)
    return result


def summarize(output_root: str, conditions: list[str], seeds: list[int], reference: str = "M") -> dict[str, pd.DataFrame]:
    """Per-seed and per-condition tables from every run's ``eval/Bias_Tuning.json``."""
    rows = []
    for condition in conditions:
        for seed in seeds:
            path = os.path.join(output_root, "runs", condition, f"seed{seed}", "eval", BIAS_FILE)
            if not os.path.isfile(path):
                continue
            with open(path) as handle:
                result = json.load(handle)
            blocks = [("ALL", result["test"])] + list(result["test_per_dataset"].items())
            for origin, block in blocks:
                for metric in REPORT_METRICS:
                    rows.append({"condition": condition, "seed": seed, "origin": origin, "metric": metric,
                                 "raw": block["raw"][metric], "tuned": block["tuned"][metric],
                                 "gain": block["tuned"][metric] - block["raw"][metric]})
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return {"per_seed": per_seed, "summary": per_seed, "paired": per_seed}
    keys = ["condition", "origin", "metric"]
    summary = per_seed.groupby(keys, sort=False).agg(
        n=("seed", "count"), raw_mean=("raw", "mean"), raw_sd=("raw", "std"),
        tuned_mean=("tuned", "mean"), tuned_sd=("tuned", "std"), gain_mean=("gain", "mean"),
    ).reset_index()
    ref = per_seed[per_seed.condition == reference].set_index(["seed", "origin", "metric"])
    paired = []
    for (condition, origin, metric), group in per_seed[per_seed.condition != reference].groupby(keys, sort=False):
        common = [s for s in group.seed if (s, origin, metric) in ref.index]
        if not common:
            continue
        mine = group.set_index("seed").loc[common]
        theirs = ref.loc[[(s, origin, metric) for s in common]]
        for kind in ("raw", "tuned"):
            difference = mine[kind].to_numpy() - theirs[kind].to_numpy()
            paired.append({"reference": reference, "condition": condition, "origin": origin, "metric": metric,
                           "scores": kind, "difference_mean": float(difference.mean()),
                           "difference_sd": float(difference.std(ddof=1)) if len(difference) > 1 else float("nan"),
                           "n_common_seeds": len(common), "seeds_favoring_reference": int((difference < 0).sum())})
    return {"per_seed": per_seed, "summary": summary, "paired": pd.DataFrame(paired)}


def write_summary(output_root: str, conditions: list[str], seeds: list[int], reference: str = "M") -> dict[str, str]:
    tables = summarize(output_root, conditions, seeds, reference)
    folder = os.path.join(output_root, "summary")
    os.makedirs(folder, exist_ok=True)
    names = {"per_seed": "Bias_Tuning_Per_Seed.csv", "summary": "Bias_Tuning_Summary.csv",
             "paired": "Bias_Tuning_Paired_vs_M.csv"}
    written = {}
    for key, name in names.items():
        path = os.path.join(folder, name)
        tables[key].to_csv(path, index=False)
        written[key] = path
    return written
