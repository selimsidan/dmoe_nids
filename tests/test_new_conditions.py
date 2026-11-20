"""A1 (no contrastive loss), A2 (uniform expert average) and B4 (pooled XGBoost)."""
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
    assert len(log) == 4 and log["val_macro_f1"].notna().all() and set(log["device"]) == {"cpu"}
    assert log["train_rows"].nunique() == 1  # all training rows, the same for every configuration
    assert not os.path.exists(os.path.join(tuning, "pilot.json"))  # the tests disable the pilot
    chosen = json.load(open(os.path.join(tuning, "chosen.json")))
    best = log.sort_values("val_macro_f1", ascending=False, kind="stable").iloc[0]
    assert chosen["val_macro_f1"] == pytest.approx(best["val_macro_f1"])
    assert chosen["tuning_seed"] == 0 and chosen["seed_max_rounds"] in (40, 60)
    assert set(chosen["chosen"]) == {"learning_rate", "max_leaves"}


def test_b4_trains_on_all_training_rows(trained):
    from dmoe.training.campaign import load_campaign, resolve_unit_config
    from dmoe.training.out_of_core_data import prepare_out_of_core_data
    from synthetic import FAST_OVERRIDES

    config = resolve_unit_config(load_campaign(), "B4", 0, output_root=trained["output"], overrides=FAST_OVERRIDES)
    rows = len(prepare_out_of_core_data(config).data.train.class_idx)
    chosen = json.load(open(os.path.join(trained["output"], "runs", "B4", "tuning", "chosen.json")))
    assert chosen["train_rows"] == rows


def test_b4_run_outputs(trained):
    run = _run(trained["output"], "B4")
    manifest = json.load(open(os.path.join(run, "manifest.json")))
    assert manifest["total_parameters"] is None and manifest["model_size"]["xgboost_trees"] > 0
    assert manifest["stages_trained"] == ["xgboost"] and manifest["environment"]["xgboost"]
    assert manifest["selected_epochs"]["xgboost_best_round"] % 10 == 0
    record = PredictionRecord.load(os.path.join(run, "eval", "test", "Test_Predictions.npz"))
    assert record.gate is None and record.expert_confusion is None and record.reliability is None
    assert record.roc_positive.sum() > 0
    resources = pd.read_csv(os.path.join(run, "eval", "Resource_Accounting.csv"), keep_default_na=False).iloc[0]
    assert resources["total_parameters"] == "n/a" and resources["stored_linear_macs"] == "n/a"
    assert int(resources["xgboost_trees"]) == manifest["model_size"]["xgboost_trees"]
    assert int(resources["xgboost_trees"]) == record.num_classes * int(resources["xgboost_best_round"])  # one per class
    history = pd.read_csv(os.path.join(run, "Training_History.csv"))
    assert set(history["encoder_role"]) == {"xgboost"} and history["val_macro_f1"].notna().all()


def test_b4_sample_weights_equal_m_class_weights():
    from dmoe.training.out_of_core_train import _loss_weights
    from dmoe.training.xgboost_baseline import class_weights

    labels = np.repeat(np.arange(5), [1000, 300, 50, 7, 1])
    expected = _loss_weights(labels, 6, "cpu").numpy()
    np.testing.assert_allclose(class_weights(labels, 6), expected)
    assert class_weights(labels, 6)[5] == 0 and np.mean(class_weights(labels, 6)[:5]) == pytest.approx(1.0)


def test_b4_default_settings_are_declared():
    from dmoe.training.campaign import load_campaign
    from dmoe.training.config import load_config
    from dmoe.training.xgboost_baseline import grid_configs, xgboost_settings

    settings = xgboost_settings(load_config(load_campaign().entry("B4").config_path))
    assert grid_configs(settings) == [
        {"learning_rate": 0.05, "max_leaves": 63}, {"learning_rate": 0.05, "max_leaves": 255},
        {"learning_rate": 0.1, "max_leaves": 63}, {"learning_rate": 0.1, "max_leaves": 255},
    ]
    params = settings["params"]
    assert params["subsample"] == params["colsample_bytree"] == 0.8 and params["reg_lambda"] == 1.0
    assert params["grow_policy"] == "lossguide" and params["max_depth"] == 0
    assert settings["patience_rounds"] == 200 and settings["max_rounds"] == 2000 and settings["device"] == "auto"
    assert settings["tuning"] == {"seed": 0, "pilot_rounds": 20}


def test_b4_forbidden_settings_are_rejected():
    from dmoe.training.xgboost_baseline import xgboost_settings

    base = {"params": {}, "grid": {}, "max_rounds": 10, "eval_every_rounds": 10, "patience_rounds": 10,
            "checkpoint_rounds": 10, "tuning": {}, "device": "cpu"}
    for key in ("scale_pos_weight", "objective", "booster", "max_delta_step", "seed"):
        with pytest.raises(ValueError, match="forbids"):
            xgboost_settings({"xgboost": {**base, "params": {key: "x"}}})


class _Matrices:
    def __init__(self):
        import xgboost as xgb

        rng = np.random.default_rng(0)
        x = rng.normal(size=(4000, 5)).astype(np.float32)
        y = ((x[:, 0] > 0).astype(int) + (x[:, 1] > 1)).astype(np.int32)
        x_val = rng.normal(size=(1500, 5)).astype(np.float32)
        self.y_val = ((x_val[:, 0] > 0).astype(int) + (x_val[:, 1] > 1)).astype(np.int32)
        names = [f"f{i}" for i in range(5)]
        self.train = xgb.QuantileDMatrix(x, label=y, feature_names=names, max_bin=63)
        self.val = xgb.QuantileDMatrix(x_val, label=self.y_val, feature_names=names, ref=self.train, max_bin=63)
        self.rows, self.num_classes = len(y), 3


_PARAMS = dict(objective="multi:softprob", num_class=3, tree_method="hist", device="cpu", grow_policy="lossguide",
               max_depth=0, max_leaves=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, seed=5,
               nthread=2, verbosity=0, max_bin=63)


def test_b4_uninterrupted_runs_are_identical_and_interrupted_runs_resume(tmp_path, monkeypatch):
    from dmoe.training import xgboost_baseline as xb

    settings = dict(max_rounds=60, eval_every=10, patience=1000, checkpoint_rounds=20)
    whole = xb.train_booster(_PARAMS, _Matrices(), state_dir=str(tmp_path / "whole"), label="whole", **settings)
    again = xb.train_booster(_PARAMS, _Matrices(), state_dir=str(tmp_path / "again"), label="again", **settings)

    def model(name):
        return (tmp_path / name / "model_progress.ubj").read_bytes()

    assert model("whole") == model("again") and whole["checks"] == again["checks"]
    assert whole["finished"] and whole["rounds_done"] == 60 and whole["resumed_after_rounds"] == []

    original = xb._write_json

    class Disconnect(Exception):
        pass

    def write_then_disconnect(value, path):
        original(value, path)
        raise Disconnect

    monkeypatch.setattr(xb, "_write_json", write_then_disconnect)
    with pytest.raises(Disconnect):
        xb.train_booster(_PARAMS, _Matrices(), state_dir=str(tmp_path / "split"), label="split", **settings)
    monkeypatch.setattr(xb, "_write_json", original)
    saved = json.loads((tmp_path / "split" / "progress.json").read_text())
    assert saved["rounds_done"] == 20 and not saved["finished"]
    resumed = xb.train_booster(_PARAMS, _Matrices(), state_dir=str(tmp_path / "split"), label="split", **settings)
    assert resumed["finished"] and resumed["rounds_done"] == 60 and resumed["resumed_after_rounds"] == [20]
    assert [c["round"] for c in resumed["checks"]] == [10, 20, 30, 40, 50, 60]
    assert [c["val_macro_f1"] for c in resumed["checks"][:2]] == [c["val_macro_f1"] for c in whole["checks"][:2]]
    booster = xb._xgb().Booster(model_file=str(tmp_path / "split" / "model_progress.ubj"))
    assert booster.num_boosted_rounds() == 60
    assert xb.train_booster(_PARAMS, None, state_dir=str(tmp_path / "split"), label="split", **settings) == resumed


def test_b4_early_stopping_keeps_the_best_round(tmp_path):
    from dmoe.training import xgboost_baseline as xb

    params = {**_PARAMS, "learning_rate": 0.3, "subsample": 1.0, "colsample_bytree": 1.0}
    state = xb.train_booster(params, _Matrices(), max_rounds=400, eval_every=10, patience=30, checkpoint_rounds=20,
                             state_dir=str(tmp_path / "run"), label="es")
    assert state["finished"] and state["stopped_early"]
    assert state["rounds_done"] - state["best_round"] >= 30
    assert state["best_val_macro_f1"] == max(c["val_macro_f1"] for c in state["checks"])
    model = xb.XGBoostModel(xb.final_model_bytes(str(tmp_path / "run"), state["best_round"]),
                            ["a", "b", "c"], [f"f{i}" for i in range(5)], device="cpu")
    assert model.best_round() == state["best_round"] and model.num_trees() == 3 * state["best_round"]
    assert model.total_leaves() > 0
    assert model.predict_proba(np.zeros((4, 5), dtype=np.float32)).shape == (4, 3)


# ---------------------------------------------------------------- post-hoc analyses


def _cli(name, trained, *args):
    import subprocess
    import sys

    from synthetic import FAST_OVERRIDES

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = ["--data-root", trained["data"], "--output-root", trained["output"], "--scratch-root", trained["scratch"]]
    sets = [arg for override in FAST_OVERRIDES for arg in ("--set", override)]
    return subprocess.run([sys.executable, os.path.join(repo, "scripts", name), *roots, *sets, *args],
                          capture_output=True, text=True, cwd=repo)


def test_bias_offsets_fit_on_weighted_subset_and_improve_macro_f1():
    from dmoe.evaluation.bias_tuning import fit_offsets, fit_subset, macro_f1, weighted_confusion

    rng = np.random.default_rng(0)
    truth = rng.choice(3, size=30_000, p=[0.9, 0.08, 0.02])
    scores = rng.normal(size=(len(truth), 3)).astype(np.float32)
    scores[np.arange(len(truth)), truth] += 1.5
    scores[:, 0] += 1.0  # the model over-predicts the majority class
    rows, weights = fit_subset(truth, cap=2_000, seed=0)
    assert np.all(np.diff(rows) > 0)
    np.testing.assert_allclose(np.bincount(truth[rows], weights=weights), np.bincount(truth))
    offsets, start, best = fit_offsets(scores[rows], truth[rows], weights, np.arange(-3, 3.01, 0.25), passes=3)
    assert offsets[0] == 0 and best > start
    full_raw = macro_f1(weighted_confusion(truth, scores.argmax(1), np.ones(len(truth)), 3))
    full_tuned = macro_f1(weighted_confusion(truth, (scores + offsets).argmax(1), np.ones(len(truth)), 3))
    assert full_tuned > full_raw and start == pytest.approx(full_raw, abs=0.02)
    zero, _, _ = fit_offsets(scores[rows], truth[rows], weights, [0.0], passes=1)
    assert not zero.any()


def test_bias_tuning_cli_writes_per_run_results_and_summary(trained):
    result = _cli("bias_tuning.py", trained, "--only", "M", "B4", "A2", "--seeds", "0", "--device", "cpu")
    assert result.returncode == 0, result.stderr
    for condition in ("M", "B4", "A2"):
        run = json.load(open(os.path.join(_run(trained["output"], condition), "eval", "Bias_Tuning.json")))
        assert run["offsets"][0] == 0 and len(run["offsets"]) == len(run["class_names"])
        assert run["val"]["tuned"]["macro_f1"] >= run["val"]["raw"]["macro_f1"] - 1e-9
        raw = pd.read_csv(os.path.join(_run(trained["output"], condition), "eval", "test", "Overall_Metrics.csv"))
        overall = raw[raw["origin"] == "ALL"] if "origin" in raw else raw
        assert run["test"]["raw"]["macro_f1"] == pytest.approx(float(overall["macro_f1"].iloc[0]), abs=1e-6)
    summary = os.path.join(trained["output"], "summary")
    paired = pd.read_csv(os.path.join(summary, "Bias_Tuning_Paired_vs_M.csv"))
    assert set(paired["condition"]) == {"B4", "A2"} and set(paired["scores"]) == {"raw", "tuned"}
    assert set(pd.read_csv(os.path.join(summary, "Bias_Tuning_Summary.csv"))["origin"]) >= {"ALL"}
