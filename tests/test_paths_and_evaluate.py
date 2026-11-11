"""Path resolution order and the offline evaluation path."""
import os

import pandas as pd

from dmoe import paths
from dmoe.evaluation.evaluate import evaluate


def test_explicit_over_env_over_default(monkeypatch, tmp_path):
    monkeypatch.delenv("NIDS_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_root("data") == paths.ResolvedRoot(str(tmp_path / "data"), "default")
    monkeypatch.setenv("NIDS_DATA_DIR", str(tmp_path / "env"))
    assert paths.resolve_root("data").source == "env"
    assert paths.resolve_root("data", str(tmp_path / "flag")) == paths.ResolvedRoot(str(tmp_path / "flag"), "explicit")


def test_dataset_layout():
    assert paths.dataset_csv_path("NF-ToN-IoT-v3", "/r") == os.path.join("/r", "NF-ToN-IoT-v3", "data", "NF-ToN-IoT-v3.csv")


def test_offline_evaluation_reproduces_training_reports(trained, tmp_path):
    run = os.path.join(trained["output"], "runs", "M", "seed0", "eval", "test")
    evaluate(str(tmp_path), predictions=os.path.join(run, "Test_Predictions.npz"), condition="M", seed=0, split="test", figures=False)
    original = pd.read_csv(os.path.join(run, "Overall_Metrics.csv"))
    again = pd.read_csv(tmp_path / "Overall_Metrics.csv")
    pd.testing.assert_frame_equal(original, again)


def test_labeled_csv_scoring(trained, tmp_path):
    from dmoe.inference.bundle import load_bundle

    bundle = load_bundle(os.path.join(trained["output"], "runs", "M", "seed0", "bundle"))
    csv = paths.dataset_csv_path("NF-UNSW-NB15-v3", trained["data"])
    frames = evaluate(str(tmp_path), bundle=bundle, csv=csv, labels_from="NF-UNSW-NB15-v3", split="csv", figures=False)
    overall = frames["overall"]
    assert overall.loc[overall["origin"] == "ALL", "rows"].item() == len(pd.read_csv(csv))
