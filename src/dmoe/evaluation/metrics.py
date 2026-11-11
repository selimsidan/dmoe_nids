"""Classification metrics computed from confusion matrices and score histograms.

Everything here is a pure function of counts, so the same code serves the
training-time evaluation (tens of millions of rows, streamed) and the offline
re-evaluation from saved per-row predictions.

Macro-F1 is the headline and selection metric. Micro precision, recall and
F1 equal accuracy for single-label multiclass prediction and are reported for
completeness. Weighted-F1 is reference only. ROC-AUC and PR-AUC (one-vs-rest)
are computed from fixed-bin score histograms; the number of bins is recorded
in every output so the approximation is visible.
"""
from __future__ import annotations

import numpy as np

OVERALL_METRIC_ORDER = [
    "macro_f1", "macro_precision", "macro_recall",
    "micro_f1", "micro_precision", "micro_recall",
    "weighted_f1", "weighted_precision", "weighted_recall",
    "accuracy", "balanced_accuracy", "matthews_corrcoef", "cohen_kappa",
    "roc_auc_ovr_macro", "roc_auc_ovr_weighted", "roc_auc_ovr_micro",
    "pr_auc_ovr_macro", "pr_auc_ovr_weighted", "pr_auc_ovr_micro",
    "foreign_prediction_rate", "classes_with_support", "rows",
]


def confusion_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def update_score_histograms(
    positive: np.ndarray,
    negative: np.ndarray,
    truth: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Accumulate one-vs-rest score histograms (``positive``/``negative``: C x bins)."""
    bins = positive.shape[1]
    score_bins = np.clip((probabilities * (bins - 1)).astype(np.int64), 0, bins - 1)
    for class_index in range(positive.shape[0]):
        is_positive = truth == class_index
        positive[class_index] += np.bincount(score_bins[is_positive, class_index], minlength=bins)
        negative[class_index] += np.bincount(score_bins[~is_positive, class_index], minlength=bins)


def auc_from_counts(positive: np.ndarray, negative: np.ndarray) -> float:
    positives = int(positive.sum())
    negatives = int(negative.sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    true_positive_rate = np.r_[0.0, np.cumsum(positive[::-1]) / positives]
    false_positive_rate = np.r_[0.0, np.cumsum(negative[::-1]) / negatives]
    widths = np.diff(false_positive_rate)
    heights = (true_positive_rate[:-1] + true_positive_rate[1:]) * 0.5
    return float(np.sum(widths * heights))


def average_precision_from_counts(positive: np.ndarray, negative: np.ndarray) -> float:
    positives = int(positive.sum())
    negatives = int(negative.sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    true_positive = np.cumsum(positive[::-1], dtype=np.float64)
    false_positive = np.cumsum(negative[::-1], dtype=np.float64)
    precision = np.divide(
        true_positive, true_positive + false_positive,
        out=np.ones_like(true_positive), where=(true_positive + false_positive) != 0,
    )
    recall = true_positive / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def auc_metrics_from_histograms(positive: np.ndarray, negative: np.ndarray) -> dict:
    """Per-class and macro/weighted/micro ROC-AUC and PR-AUC from C x bins histograms."""
    per_class = np.asarray([auc_from_counts(positive[i], negative[i]) for i in range(len(positive))])
    per_class_pr = np.asarray(
        [average_precision_from_counts(positive[i], negative[i]) for i in range(len(positive))]
    )
    support = positive.sum(axis=1)
    valid = np.isfinite(per_class)
    valid_pr = np.isfinite(per_class_pr)
    return {
        "per_class": per_class,
        "per_class_pr": per_class_pr,
        "roc_auc_ovr_macro": float(per_class[valid].mean()) if valid.any() else float("nan"),
        "roc_auc_ovr_weighted": (
            float(np.average(per_class[valid], weights=support[valid]))
            if valid.any() and support[valid].sum() else float("nan")
        ),
        "roc_auc_ovr_micro": auc_from_counts(positive.sum(axis=0), negative.sum(axis=0)),
        "pr_auc_ovr_macro": float(per_class_pr[valid_pr].mean()) if valid_pr.any() else float("nan"),
        "pr_auc_ovr_weighted": (
            float(np.average(per_class_pr[valid_pr], weights=support[valid_pr]))
            if valid_pr.any() and support[valid_pr].sum() else float("nan")
        ),
        "pr_auc_ovr_micro": average_precision_from_counts(positive.sum(axis=0), negative.sum(axis=0)),
    }


def overall_from_confusion(
    confusion: np.ndarray,
    native_class_indices: set[int] | None = None,
    auc: dict | None = None,
) -> dict[str, float | int]:
    """All overall metrics of one confusion matrix (rows: true, columns: predicted)."""
    confusion = np.asarray(confusion, dtype=np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    present = support > 0
    total = confusion.sum()
    accuracy = tp.sum() / total if total else 0.0
    numerator = total * tp.sum() - float(np.dot(support, predicted))
    denominator = np.sqrt(
        (total**2 - float(np.dot(predicted, predicted))) * (total**2 - float(np.dot(support, support)))
    )
    expected = float(np.dot(support, predicted)) / total**2 if total else 0.0
    foreign_prediction_rate = 0.0
    if native_class_indices is not None and total:
        foreign = [index for index in range(len(support)) if index not in native_class_indices]
        foreign_prediction_rate = float(confusion[:, foreign].sum() / total) if foreign else 0.0
    auc = auc or {}
    nan = float("nan")
    row = {
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "micro_f1": float(accuracy),
        "micro_precision": float(accuracy),
        "micro_recall": float(accuracy),
        "weighted_f1": float((f1 * support).sum() / total) if total else 0.0,
        "weighted_precision": float((precision * support).sum() / total) if total else 0.0,
        "weighted_recall": float((recall * support).sum() / total) if total else 0.0,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "matthews_corrcoef": float(numerator / denominator) if denominator else 0.0,
        "cohen_kappa": float((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else 0.0,
        "roc_auc_ovr_macro": auc.get("roc_auc_ovr_macro", nan),
        "roc_auc_ovr_weighted": auc.get("roc_auc_ovr_weighted", nan),
        "roc_auc_ovr_micro": auc.get("roc_auc_ovr_micro", nan),
        "pr_auc_ovr_macro": auc.get("pr_auc_ovr_macro", nan),
        "pr_auc_ovr_weighted": auc.get("pr_auc_ovr_weighted", nan),
        "pr_auc_ovr_micro": auc.get("pr_auc_ovr_micro", nan),
        "foreign_prediction_rate": foreign_prediction_rate,
        "classes_with_support": int(present.sum()),
        "rows": int(total),
    }
    return {key: row[key] for key in OVERALL_METRIC_ORDER}


def per_class_rows(
    confusion: np.ndarray,
    class_names: list[str],
    origin: str,
    per_class_roc_auc: np.ndarray | None = None,
    per_class_pr_auc: np.ndarray | None = None,
    origin_datasets: dict[str, list[str]] | None = None,
) -> list[dict]:
    """One row per class with counts, rates and (when given) AUCs."""
    total = int(confusion.sum())
    rows = []
    for index, label in enumerate(class_names):
        tp = int(confusion[index, index])
        fn = int(confusion[index, :].sum() - tp)
        fp = int(confusion[:, index].sum() - tp)
        tn = total - tp - fn - fp
        support = tp + fn
        predicted_support = tp + fp
        precision = tp / predicted_support if predicted_support else 0.0
        recall = tp / support if support else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "origin": origin,
            "class": label,
            "origin_datasets": "|".join((origin_datasets or {}).get(label, [])),
            "support": support,
            "predicted_support": predicted_support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "false_positive_rate": 1.0 - specificity,
            "false_negative_rate": 1.0 - recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "prevalence": support / total if total else 0.0,
            "roc_auc_ovr": float(per_class_roc_auc[index]) if per_class_roc_auc is not None else float("nan"),
            "pr_auc_ovr": float(per_class_pr_auc[index]) if per_class_pr_auc is not None else float("nan"),
        })
    return rows


def confusion_long_rows(confusion: np.ndarray, class_names: list[str]) -> list[dict]:
    """Long format: one row per (true, predicted) cell with count, row ratio and column ratio."""
    confusion = np.asarray(confusion, dtype=np.int64)
    row_totals = confusion.sum(axis=1)
    column_totals = confusion.sum(axis=0)
    rows = []
    for i, true_name in enumerate(class_names):
        for j, predicted_name in enumerate(class_names):
            count = int(confusion[i, j])
            rows.append({
                "true_class": true_name,
                "predicted_class": predicted_name,
                "count": count,
                "row_ratio": count / row_totals[i] if row_totals[i] else 0.0,
                "column_ratio": count / column_totals[j] if column_totals[j] else 0.0,
            })
    return rows
