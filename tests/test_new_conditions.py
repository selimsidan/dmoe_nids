"""A1 (no contrastive loss), A2 (uniform expert average) and B4 (pooled LightGBM)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from dmoe.evaluation.predictions import PredictionRecord
from dmoe.inference.bundle import load_bundle


def _run(root, condition, seed=0):
    return os.path.join(root, "runs", condition, f"seed{seed}")


def _test_features(root, bundle):
    raw = []
    for name in bundle.labels["dataset_names"]:
        base = os.path.join(root, "cache", "splits", name)
        directory = os.path.join(base, next(d for d in os.listdir(base) if d.startswith("split-")))
        raw.append(np.load(os.path.join(directory, "test_X.npy")))
    return bundle.standardize(np.concatenate(raw))


# ---------------------------------------------------------------- A2


def test_a2_is_the_mean_of_m_expert_probabilities(trained):
    m = load_bundle(os.path.join(_run(trained["output"], "M"), "bundle"))
    a2 = load_bundle(os.path.join(_run(trained["output"], "A2"), "bundle"))
    x = torch.from_numpy(_test_features(trained["output"], m)[:300])
    with torch.no_grad():
        expected = m.model(x)["expert_probs"].mean(dim=1)
        out = a2.model(x)
    torch.testing.assert_close(out["combined_probs"], expected, rtol=0, atol=1e-6)
    assert torch.all(out["gate_weights"] == 0.25)
    assert sum(p.numel() for p in a2.model.gate.parameters()) == 0
    assert not hasattr(a2.model, "class_reliability")


def test_a2_expert_confusions_equal_m(trained):
    m = PredictionRecord.load(os.path.join(_run(trained["output"], "M"), "eval", "test", "Test_Predictions.npz"))
    a2 = PredictionRecord.load(os.path.join(_run(trained["output"], "A2"), "eval", "test", "Test_Predictions.npz"))
    np.testing.assert_array_equal(m.expert_confusion, a2.expert_confusion)
    assert a2.reliability is None and np.all(a2.gate == np.float16(0.25))


def test_a2_reports_and_accounting(trained):
    run = _run(trained["output"], "A2")
    test_dir = os.path.join(run, "eval", "test")
    assert not os.path.exists(os.path.join(test_dir, "Class_Conditional_Reliability.csv"))
    assert not os.path.exists(os.path.join(test_dir, "Route_Confusion.csv"))  # no top expert under equal weights
    gate = pd.read_csv(os.path.join(test_dir, "Gate_By_Dataset.csv"))
    assert gate["own_expert_top_fraction"].isna().all()
    assert os.path.isfile(os.path.join(test_dir, "Expert_Performance_By_Dataset.csv"))
    resources = pd.read_csv(os.path.join(run, "eval", "Resource_Accounting.csv")).iloc[0]
    assert resources["gate_router_parameters"] == 0 and resources["trainable_parameters"] == 0
    manifest = json.load(open(os.path.join(run, "manifest.json")))
    assert manifest["stages_trained"] == ["C"] and manifest["selected_epochs"]["stage_c"] is None
    summary = json.load(open(os.path.join(run, "checkpoints", "stage_c_training_summary.json")))
    assert summary["selection_mode"] == "none" and summary["training_summary"]["optimizer_steps"] == 0


# ---------------------------------------------------------------- A1


def test_a1_trains_its_own_ce_only_encoder(trained):
    a1 = torch.load(os.path.join(_run(trained["output"], "A1"), "checkpoints", "stage_A", "stage_a_encoder.pt"))
    m = torch.load(os.path.join(_run(trained["output"], "M"), "checkpoints", "stage_A", "stage_a_encoder.pt"))
    assert a1["representation_config"]["objective"] == "ce"
    assert m["representation_config"]["objective"] == "balanced_supcon"
    assert any(not torch.equal(a1["encoder_state"][k], v) for k, v in m["encoder_state"].items())
    manifest = json.load(open(os.path.join(_run(trained["output"], "A1"), "manifest.json")))
    assert manifest["stages_trained"] == ["A", "B", "C"] and manifest["depends_on"] == {}


def test_a1_and_m_start_from_the_same_initialization():
    from dmoe.models.encoder import build_encoder, resolve_encoder_config
    from dmoe.models.representation_losses import StageARepresentationObjective
    from dmoe.training.campaign import load_campaign
    from dmoe.training.config import load_config

    campaign = load_campaign()
    states = []
    for condition in ("M", "A1"):
        config = load_config(campaign.entry(condition).config_path)
        torch.manual_seed(3)
        encoder = build_encoder(47, config["model"]["latent_dim"], resolve_encoder_config(config["model"], "encoder"))
        objective = StageARepresentationObjective(config["model"]["latent_dim"], 22, config)
        states.append({**encoder.state_dict(), **{f"objective.{k}": v for k, v in objective.state_dict().items()}})
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])  # SupCon adds no parameters


def test_m_encoder_is_rejected_for_a1(trained):
    from dmoe.training.campaign import load_campaign, resolve_unit_config
    from dmoe.training.checkpoint import load_validated_stage_a
    from synthetic import FAST_OVERRIDES

    config = resolve_unit_config(load_campaign(), "A1", 0, output_root=trained["output"], overrides=FAST_OVERRIDES)
    config["training"]["stage_a_checkpoint"] = os.path.join(
        _run(trained["output"], "M"), "checkpoints", "stage_A", "stage_a_encoder.pt")
    metadata = torch.load(config["training"]["stage_a_checkpoint"])["metadata"]
    expected = {key: metadata[key] for key in ("encoder", "class_names", "feature_columns", "split_signature")}
    with pytest.raises(ValueError, match="representation"):
        load_validated_stage_a(config, expected)


# ---------------------------------------------------------------- B4


def test_b4_tuning_log_and_choice(trained):
    tuning = os.path.join(trained["output"], "runs", "B4", "tuning")
    log = pd.read_csv(os.path.join(tuning, "Tuning_Log.csv"))
    assert len(log) == 4 and set(log["data_mode"]) == {"full"} and log["val_macro_f1"].notna().all()
    chosen = json.load(open(os.path.join(tuning, "chosen.json")))
    best = log.sort_values("val_macro_f1", ascending=False, kind="stable").iloc[0]
    assert chosen["val_macro_f1"] == pytest.approx(best["val_macro_f1"])
    assert chosen["tuning_seed"] == 0 and chosen["seed_max_rounds"] in (40, 60)


def test_b4_run_outputs(trained):
    run = _run(trained["output"], "B4")
    manifest = json.load(open(os.path.join(run, "manifest.json")))
    assert manifest["total_parameters"] is None and manifest["model_size"]["lightgbm_trees"] > 0
    assert manifest["stages_trained"] == ["lightgbm"] and manifest["environment"]["lightgbm"]
    assert manifest["selected_epochs"]["lightgbm_best_round"] % 10 == 0
    record = PredictionRecord.load(os.path.join(run, "eval", "test", "Test_Predictions.npz"))
    assert record.gate is None and record.expert_confusion is None and record.reliability is None
    assert record.roc_positive.sum() > 0
    resources = pd.read_csv(os.path.join(run, "eval", "Resource_Accounting.csv"), keep_default_na=False).iloc[0]
    assert resources["total_parameters"] == "n/a" and resources["stored_linear_macs"] == "n/a"
    assert int(resources["lightgbm_trees"]) == manifest["model_size"]["lightgbm_trees"]
    history = pd.read_csv(os.path.join(run, "Training_History.csv"))
    assert set(history["encoder_role"]) == {"lightgbm"} and history["val_macro_f1"].notna().all()


def test_b4_sample_weights_equal_m_class_weights():
    from dmoe.training.lightgbm_baseline import class_weights
    from dmoe.training.out_of_core_train import _loss_weights

    labels = np.repeat(np.arange(5), [1000, 300, 50, 7, 1])
    expected = _loss_weights(labels, 6, "cpu").numpy()
    np.testing.assert_allclose(class_weights(labels, 6), expected)
    assert class_weights(labels, 6)[5] == 0 and np.mean(class_weights(labels, 6)[:5]) == pytest.approx(1.0)


def test_b4_tuning_sample_is_uniform_and_fixed():
    from dmoe.training.lightgbm_baseline import sample_indices

    labels = np.repeat(np.arange(4), [70_000, 20_000, 9_000, 1_000])
    first = sample_indices(len(labels), 0.1, seed=0)
    np.testing.assert_array_equal(first, sample_indices(len(labels), 0.1, seed=0))
    assert len(first) == 10_000 and np.all(np.diff(first) > 0)
    share = np.bincount(labels[first], minlength=4) / len(first)
    np.testing.assert_allclose(share, [0.7, 0.2, 0.09, 0.01], atol=0.01)


def test_b4_default_data_mode_is_the_declared_rule():
    from dmoe.training.campaign import load_campaign
    from dmoe.training.config import load_config
    from dmoe.training.lightgbm_baseline import grid_configs, lightgbm_settings

    settings = lightgbm_settings(load_config(load_campaign().entry("B4").config_path))
    assert settings["tuning"]["data_mode"] == "auto" and settings["tuning"]["max_projected_hours"] == 24
    assert grid_configs(settings) == [
        {"learning_rate": 0.05, "num_leaves": 63}, {"learning_rate": 0.05, "num_leaves": 255},
        {"learning_rate": 0.1, "num_leaves": 63}, {"learning_rate": 0.1, "num_leaves": 255},
    ]
    assert settings["params"]["bagging_fraction"] == settings["params"]["feature_fraction"] == 0.8


def test_b4_forbidden_settings_are_rejected():
    from dmoe.training.lightgbm_baseline import lightgbm_settings

    base = {"params": {"objective": "multiclass"}, "grid": {}, "max_rounds": 10, "eval_every_rounds": 10,
            "patience_rounds": 10, "chunk_rounds": 10, "tuning": {}}
    for key in ("is_unbalance", "class_weight", "boosting"):
        with pytest.raises(ValueError, match="forbids"):
            lightgbm_settings({"lightgbm": {**base, "params": {key: "x"}}})


class _Arrays:
    """Small in-memory stand-in for ``PreparedArrays``."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(4000, 5)).astype(np.float32)
        self.y = ((self.x[:, 0] > 0).astype(int) + (self.x[:, 1] > 1)).astype(np.int32)
        self.x_val = rng.normal(size=(1500, 5)).astype(np.float32)
        self.y_val = ((self.x_val[:, 0] > 0).astype(int) + (self.x_val[:, 1] > 1)).astype(np.int32)
        self.weights = np.ones(len(self.y))
        self.rows = len(self.y)
        self.feature_columns = [f"f{i}" for i in range(5)]
        self.dataset_params = {"max_bin": 63, "verbose": -1, "num_threads": 2, "feature_pre_filter": False}

    def datasets(self):
        from dmoe.training.lightgbm_baseline import PreparedArrays

        return PreparedArrays.datasets(self)


def test_b4_interrupted_run_resumes_identically(tmp_path, monkeypatch):
    from dmoe.training import lightgbm_baseline as lb

    params = dict(objective="multiclass", num_class=3, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  deterministic=True, force_row_wise=True, metric="None", verbose=-1, num_threads=2,
                  bagging_seed=5, feature_fraction_seed=6, num_leaves=7, learning_rate=0.1)
    settings = dict(max_rounds=60, eval_every=10, patience=1000, chunk_rounds=20)
    whole = lb.train_booster(params, _Arrays(), state_dir=str(tmp_path / "whole"), label="whole", **settings)

    original = lb._write_json

    class Disconnect(Exception):
        pass

    def write_then_disconnect(value, path):
        original(value, path)
        raise Disconnect

    monkeypatch.setattr(lb, "_write_json", write_then_disconnect)
    with pytest.raises(Disconnect):
        lb.train_booster(params, _Arrays(), state_dir=str(tmp_path / "split"), label="split", **settings)
    monkeypatch.setattr(lb, "_write_json", original)
    resumed = lb.train_booster(params, _Arrays(), state_dir=str(tmp_path / "split"), label="split", **settings)

    def trees(name):
        return (tmp_path / name / "model_progress.txt").read_text().split("parameters:")[0]

    assert trees("whole") == trees("split")
    assert whole["checks"] == resumed["checks"] and whole["best_round"] == resumed["best_round"]


def test_b4_early_stopping_keeps_the_best_round(tmp_path):
    from dmoe.training import lightgbm_baseline as lb

    params = dict(objective="multiclass", num_class=3, deterministic=True, force_row_wise=True, metric="None",
                  verbose=-1, num_threads=2, bagging_seed=1, feature_fraction_seed=2, num_leaves=7, learning_rate=0.3)
    state = lb.train_booster(params, _Arrays(), max_rounds=400, eval_every=10, patience=30, chunk_rounds=20,
                             state_dir=str(tmp_path / "run"), label="es")
    assert state["finished"] and state["stopped_early"]
    assert state["rounds_done"] - state["best_round"] >= 30
    assert state["best_val_macro_f1"] == max(c["val_macro_f1"] for c in state["checks"])
    text = lb.final_model_text(str(tmp_path / "run"), state["best_round"])
    booster = lb._lgb().Booster(model_str=text)
    assert booster.current_iteration() == state["best_round"]
