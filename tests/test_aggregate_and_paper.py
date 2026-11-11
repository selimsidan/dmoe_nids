"""Cross-seed aggregation, significance, recommended seeds and paper tables."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from dmoe.evaluation.aggregate import (
    aggregate, collect_runs, mcnemar, paired_bootstrap_macro_f1, recommended_seeds,
)
from dmoe.evaluation.metrics import confusion_from_predictions, overall_from_confusion
from dmoe.evaluation.paper_tables import make_paper_tables


@pytest.fixture(scope="module")
def summary(trained):
    tables = aggregate(trained["output"], resamples=200, aggregation_seed=0)
    return trained["output"], tables


def test_n_and_mixed_seed_warning(summary):
    root, tables = summary
    overall = tables["Summary_Overall"].query("metric == 'macro_f1'").set_index("condition")
    assert overall.loc["M", "n"] == 2 and overall.loc["M", "seeds"] == "0 1"
    assert overall.loc["B1", "n"] == 1 and np.isnan(overall.loc["B1", "sd"])
    info = json.load(open(os.path.join(root, "summary", "Summary_Info.json")))
    assert any("different completed seed sets" in w for w in info["warnings"])


def test_pairs_only_common_seeds(summary):
    _, tables = summary
    paired = tables["Paired_vs_M"].query("metric == 'macro_f1' and origin == 'ALL'").set_index("condition")
    assert set(paired.index) == {"B1", "B2", "B3", "A1", "A2", "A3"}
    assert (paired["n_common_seeds"] == 1).all() and (paired["common_seeds"] == "0").all()
    per_seed = tables["Paired_vs_M_Per_Seed"]
    assert set(per_seed["seed"]) == {0}


def test_summary_matches_per_run_files(summary, trained):
    _, tables = summary
    values = []
    for seed in (0, 1):
        frame = pd.read_csv(os.path.join(trained["output"], "runs", "M", f"seed{seed}", "eval", "test", "Overall_Metrics.csv"))
        values.append(float(frame.loc[frame["origin"] == "ALL", "macro_f1"].iloc[0]))
    row = tables["Summary_Overall"].query("condition == 'M' and metric == 'macro_f1'").iloc[0]
    assert row["mean"] == pytest.approx(np.mean(values)) and row["sd"] == pytest.approx(np.std(values, ddof=1))


def test_recommended_seed_reads_validation_only(trained, tmp_path):
    copy = tmp_path / "out"
    shutil.copytree(os.path.join(trained["output"], "runs"), copy / "runs")
    before = recommended_seeds(collect_runs(str(copy))).set_index("condition").loc["M", "recommended_seed"]
    for seed in (0, 1):  # make the test split favour the other seed strongly
        path = copy / "runs" / "M" / f"seed{seed}" / "eval" / "test" / "Overall_Metrics.csv"
        frame = pd.read_csv(path)
        frame["macro_f1"] = 1.0 if seed != before else 0.0
        frame.to_csv(path, index=False)
    after = recommended_seeds(collect_runs(str(copy))).set_index("condition").loc["M", "recommended_seed"]
    assert after == before
    val = {s: pd.read_csv(copy / "runs" / "M" / f"seed{s}" / "eval" / "val" / "Overall_Metrics.csv")
           .query("origin == 'ALL'")["macro_f1"].iloc[0] for s in (0, 1)}
    best = max(val.values())
    assert before == min(s for s, v in val.items() if v == best)


def test_different_splits_are_refused(trained, tmp_path):
    copy = tmp_path / "out"
    shutil.copytree(os.path.join(trained["output"], "runs"), copy / "runs")
    path = copy / "runs" / "M" / "seed1" / "manifest.json"
    manifest = json.load(open(path))
    manifest["split"]["preprocessing_signature"] = "something-else"
    json.dump(manifest, open(path, "w"))
    with pytest.raises(ValueError, match="different splits"):
        aggregate(str(copy), figures=False, resamples=10)


def test_cell_bootstrap_is_consistent_and_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 3000)
    a = np.where(rng.random(3000) < 0.8, y, rng.integers(0, 4, 3000))
    b = np.where(rng.random(3000) < 0.7, y, rng.integers(0, 4, 3000))
    first = paired_bootstrap_macro_f1(y, a, b, 4, 300, seed=1)
    second = paired_bootstrap_macro_f1(y, a, b, 4, 300, seed=1)
    assert first == second
    observed = overall_from_confusion(confusion_from_predictions(y, a, 4))["macro_f1"] - \
        overall_from_confusion(confusion_from_predictions(y, b, 4))["macro_f1"]
    assert first["observed_difference"] == pytest.approx(observed)
    assert first["bootstrap_ci95_low"] < observed < first["bootstrap_ci95_high"]
    assert first["bootstrap_p_two_sided"] < 0.05  # a is clearly better than b


def test_mcnemar_exact_and_chi_square():
    y = np.zeros(40, dtype=int)
    exact = mcnemar(y, np.r_[np.zeros(10), np.ones(30)], np.r_[np.ones(5), np.zeros(5), np.ones(30)])
    assert exact["mcnemar_method"] == "exact binomial"
    large = mcnemar(np.zeros(200), np.r_[np.zeros(100), np.ones(100)], np.r_[np.ones(100), np.zeros(100)])
    assert large["mcnemar_method"].startswith("chi-square") and large["mcnemar_p"] == pytest.approx(1.0, abs=0.1)


def test_paper_tables(summary):
    root, _ = summary
    out = make_paper_tables(root)
    files = set(os.listdir(out))
    assert {"Table_Main.tex", "Table_Main.csv", "Table_Ablations.tex", "Table_Per_Dataset.tex",
            "Table_Per_Class.tex", "Figure_Routing_M.pdf", "Figure_Confusion_M.pdf"} <= files
    main = open(os.path.join(out, "Table_Main.tex")).read()
    assert "Dataset-expert MoE" in main and "\\pm" in main and "\\textbf{" in main
    per_dataset = open(os.path.join(out, "Table_Per_Dataset.tex")).read()
    weighted_column = [line.split("&")[-1] for line in per_dataset.splitlines() if line.count("&") >= 5]
    assert not any("\\textbf" in cell for cell in weighted_column)  # weighted-F1 is never bolded
    numbers = pd.read_csv(os.path.join(out, "Table_Main.csv"))
    summary_value = pd.read_csv(os.path.join(root, "summary", "Summary_Overall.csv")) \
        .query("condition == 'M' and metric == 'macro_f1'")["mean"].iloc[0]
    assert numbers.set_index("condition").loc["M", "macro_f1_mean"] == pytest.approx(summary_value)
