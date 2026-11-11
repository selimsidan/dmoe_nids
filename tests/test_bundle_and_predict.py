"""Bundles round-trip, predictions need no dataset identity, and the scaler sees train rows only."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from dmoe import paths
from dmoe.evaluation.predictions import PredictionRecord
from dmoe.inference.bundle import load_bundle
from dmoe.training.out_of_core_data import assert_scaler_fit_on_train_only


def _bundle_dir(root, condition, seed=0):
    return os.path.join(root, "runs", condition, f"seed{seed}", "bundle")


@pytest.mark.parametrize("condition", ["M", "B1", "B2", "B3", "A1", "A2", "A3"])
def test_bundle_reproduces_saved_test_predictions(trained, condition):
    bundle = load_bundle(_bundle_dir(trained["output"], condition))
    record = PredictionRecord.load(os.path.join(trained["output"], "runs", condition, "seed0", "eval", "test", "Test_Predictions.npz"))
    split_dirs = sorted(os.listdir(os.path.join(trained["output"], "cache", "splits")))
    # Rebuild test features in the same order as training (active dataset order).
    names = bundle.labels["dataset_names"]
    raw = []
    for name in names:
        directory = next(
            os.path.join(trained["output"], "cache", "splits", name, d)
            for d in os.listdir(os.path.join(trained["output"], "cache", "splits", name)) if d.startswith("split-")
        )
        raw.append(np.load(os.path.join(directory, "test_X.npy")))
    assert split_dirs == sorted(names)
    probabilities = bundle.predict_standardized(bundle.standardize(np.concatenate(raw)))["probabilities"]
    np.testing.assert_array_equal(probabilities.argmax(axis=1), record.y_pred)


def test_tampered_bundle_is_rejected(trained, tmp_path):
    import shutil

    target = tmp_path / "bundle"
    shutil.copytree(_bundle_dir(trained["output"], "M"), target)
    with open(target / "labels.json", "a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(str(target))


def test_predict_raw_csv_without_dataset_column(trained):
    bundle = load_bundle(_bundle_dir(trained["output"], "M"))
    csv = paths.dataset_csv_path("NF-BoT-IoT-v3", trained["data"])
    frame = pd.read_csv(csv, nrows=50).drop(columns=["Attack", "Label"])
    out = bundle.predict(frame, with_probs=True, with_gate=True)
    assert len(out) == 50 and set(out["predicted_label"]) <= set(bundle.class_names)
    assert out["top_expert"].isin(bundle.expert_names).all()
    probs = out[[c for c in out if c.startswith("prob__")]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_missing_feature_named(trained):
    bundle = load_bundle(_bundle_dir(trained["output"], "B2"))
    frame = pd.read_csv(paths.dataset_csv_path("NF-BoT-IoT-v3", trained["data"]), nrows=5).drop(columns=["IN_BYTES"])
    with pytest.raises(KeyError, match="IN_BYTES"):
        bundle.predict(frame)


def test_scaler_fit_on_train_rows_only(trained):
    metadata = json.load(open(os.path.join(_bundle_dir(trained["output"], "M"), "metadata.json")))
    manifest = json.load(open(os.path.join(trained["output"], "runs", "M", "seed0", "manifest.json")))
    assert metadata["split"]["preprocessing_signature"] == manifest["split"]["preprocessing_signature"]
    stats = np.load(next(
        os.path.join(trained["output"], "cache", "preprocessors", f)
        for f in os.listdir(os.path.join(trained["output"], "cache", "preprocessors"))
    ))
    train_rows = 0
    for name in manifest["dataset_names"]:
        base = os.path.join(trained["output"], "cache", "splits", name)
        directory = os.path.join(base, next(d for d in os.listdir(base) if d.startswith("split-")))
        train_rows += json.load(open(os.path.join(directory, "metadata.json")))["split_rows"]["train"]
    assert int(np.asarray(stats["samples_seen"]).max()) == train_rows
    assert_scaler_fit_on_train_only({"samples_seen": stats["samples_seen"]}, train_rows)
    with pytest.raises(AssertionError):
        assert_scaler_fit_on_train_only({"samples_seen": stats["samples_seen"]}, train_rows + 1)


def test_m_bundle_ships_stage_weights(trained):
    files = set(os.listdir(_bundle_dir(trained["output"], "M")))
    assert {"stage_a_encoder.safetensors", "stage_b_expert_bank.safetensors"} <= files
    assert "stage_a_encoder.safetensors" not in os.listdir(_bundle_dir(trained["output"], "B1"))
