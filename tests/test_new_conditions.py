"""A1 (no shared encoder), A2 (uniform expert average) and B4 (pooled XGBoost)."""
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


def _a1_model():
    from dmoe.data.registry import DATASET_NAMES
    from dmoe.inference.bundle import build_model_for_bundle
    from dmoe.training.campaign import load_campaign
    from dmoe.training.config import load_config

    config = load_config(load_campaign().entry("A1").config_path)
    model = build_model_for_bundle({"architecture": config["architecture"], "model": config["model"], "input_dim": 47,
                                    "dataset_names": list(DATASET_NAMES), "class_names": [f"c{i}" for i in range(22)]})
    return config, model


def test_a1_identity_encoder():
    from dmoe.models.encoder import IdentityEncoder

    _, model = _a1_model()
    assert isinstance(model.encoder, IdentityEncoder) and model.encoder.latent_dim == 47
    assert sum(p.numel() for p in model.encoder.parameters()) == 0
    x = torch.randn(5, 47)
    assert torch.equal(model.encoder(x), x)
    assert model.gate.net[0].in_features == 47 and model.expert_bank.experts[0].net[0].in_features == 47


def test_a1_trains_b_and_c_without_any_stage_a(trained):
    run = _run(trained["output"], "A1")
    assert not os.path.exists(os.path.join(run, "checkpoints", "stage_A"))
    manifest = json.load(open(os.path.join(run, "manifest.json")))
    assert manifest["stages_trained"] == ["B", "C"] and manifest["depends_on"] == {}
    assert "stage_a" not in manifest.get("checkpoints", {})
    bundle_files = os.listdir(os.path.join(run, "bundle"))
    assert "stage_a_encoder.safetensors" not in bundle_files and "stage_b_expert_bank.safetensors" not in bundle_files
    stage_c = torch.load(os.path.join(run, "checkpoints", "stage_C", "stage_c_full.pt"))["model_state"]
    assert not any(key.startswith("encoder.") for key in stage_c)


def test_a1_stage_b_gradients_reach_only_the_trained_expert():
    from dmoe.training.model_utils import expert_forward_one, expert_train_params

    _, model = _a1_model()
    bank = model.expert_bank
    x, y = torch.randn(16, 47), torch.randint(0, 22, (16,))
    torch.nn.functional.cross_entropy(expert_forward_one(bank, 2, model.encoder(x)), y).backward()
    assert {id(p) for p in expert_train_params(bank, 2)} == {id(p) for p in bank.experts[2].parameters()}
    for index, expert in enumerate(bank.experts):
        has_grad = [p.grad is not None and bool(p.grad.abs().sum() > 0) for p in expert.parameters()]
        assert all(has_grad) if index == 2 else not any(has_grad), index
    assert all(p.grad is None for p in model.gate.parameters()) and model.class_reliability.grad is None


def test_a1_stage_c_trains_only_gate_and_reliability(trained):
    from dmoe.training.model_utils import apply_stage_c_trainability

    config, model = _a1_model()
    apply_stage_c_trainability(model, config)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {name for name, _ in model.named_parameters() if name.startswith("gate.")} | {"class_reliability"}
    run = _run(trained["output"], "A1")
    stage_b = torch.load(os.path.join(run, "checkpoints", "stage_B", "stage_b_expert_bank.pt"))["expert_bank_state"]
    stage_c = torch.load(os.path.join(run, "checkpoints", "stage_C", "stage_c_full.pt"))["model_state"]
    assert all(torch.equal(stage_c[f"expert_bank.{k}"], v) for k, v in stage_b.items())  # experts frozen in Stage C


def test_parameter_count_is_checked_at_runtime():
    from dmoe.training.model_utils import check_parameter_count

    config, model = _a1_model()
    assert check_parameter_count(model, config, 47, 22) == 22284
    assert check_parameter_count(model, config, 47, 11) is None  # other data shapes (synthetic tests): skipped
    wrong = {**config, "condition": {**config["condition"], "expected_total_parameters": 22420}}
    with pytest.raises(ValueError, match="22,284 parameters, the config expects 22,420"):
        check_parameter_count(model, wrong, 47, 22)
    tolerant = {**config, "condition": {**config["condition"], "expected_total_parameters": 22420,
                                        "parameter_tolerance": 0.01}}
    assert check_parameter_count(model, tolerant, 47, 22) == 22284


def _old_a1_contract(trained):
    """The run contract an A1 run of the earlier condition set (M with a CE-only Stage A) wrote."""
    contract = json.load(open(os.path.join(_run(trained["output"], "M"), "checkpoints", "run_contract.json")))
    contract["training"]["representation"]["objective"] = "ce"
    return contract


def test_old_a1_checkpoints_are_rejected(trained, tmp_path):
    from dmoe.training.campaign import load_campaign, resolve_unit_config
    from dmoe.training.out_of_core_data import prepare_out_of_core_data
    from dmoe.training.out_of_core_train import ensure_run_contract
    from synthetic import FAST_OVERRIDES

    config = resolve_unit_config(load_campaign(), "A1", 0, output_root=trained["output"], overrides=FAST_OVERRIDES)
    checkpoints = config["training"]["checkpoint_dir"] = str(tmp_path / "A1" / "seed0" / "checkpoints")
    os.makedirs(checkpoints)
    json.dump(_old_a1_contract(trained), open(os.path.join(checkpoints, "run_contract.json"), "w"))
    with pytest.raises(ValueError, match="Move that run directory"):
        ensure_run_contract(config, prepare_out_of_core_data(config))


def test_old_a1_manifest_is_not_complete(trained, tmp_path):
    from dmoe.training.campaign import load_campaign, plan_units, resolve_unit_config, unit_complete
    from dmoe.training.config import config_hash
    from synthetic import FAST_OVERRIDES

    campaign = load_campaign()
    old = resolve_unit_config(campaign, "M", 0, output_root=str(tmp_path), overrides=FAST_OVERRIDES)
    old["training"]["representation"]["objective"] = "ce"  # the earlier A1
    run = os.path.join(str(tmp_path), "runs", "A1", "seed0")
    os.makedirs(run)
    json.dump({"status": "complete", "config_hash": config_hash(old)}, open(os.path.join(run, "manifest.json"), "w"))
    assert not unit_complete(campaign, "A1", 0, str(tmp_path), FAST_OVERRIDES)
    unit = plan_units(campaign, ["A1"], [0], output_root=str(tmp_path), overrides=FAST_OVERRIDES)[0]
    assert unit.status != "complete"


def test_stage_a_with_another_representation_objective_is_rejected(trained):
    """A Stage-A encoder is reused only under the representation settings it was trained with."""
    from dmoe.training.campaign import load_campaign, resolve_unit_config
    from dmoe.training.checkpoint import load_validated_stage_a
    from synthetic import FAST_OVERRIDES

    config = resolve_unit_config(load_campaign(), "B2", 0, output_root=trained["output"],
                                 overrides=[*FAST_OVERRIDES, "training.representation.objective=ce"])
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
