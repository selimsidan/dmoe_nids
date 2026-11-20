"""Same seed, same data: identical weights and metrics. Different seed: different weights."""
import json
import os

import torch

from conftest import set_roots, train_unit


def _stage_c(root, condition, seed):
    return torch.load(os.path.join(root, "runs", condition, f"seed{seed}", "checkpoints", "stage_C", "stage_c_full.pt"))["model_state"]


def _test_f1(root, condition, seed):
    manifest = json.load(open(os.path.join(root, "runs", condition, f"seed{seed}", "manifest.json")))
    return manifest["results"]["test"]["macro_f1"], manifest["results"]["val"]["macro_f1"]


def test_same_seed_identical(trained, tmp_path, monkeypatch):
    roots = set_roots(monkeypatch, str(tmp_path))
    roots["data"] = trained["data"]
    train_unit("M", 0, roots)
    first, second = _stage_c(trained["output"], "M", 0), _stage_c(roots["output"], "M", 0)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert _test_f1(trained["output"], "M", 0) == _test_f1(roots["output"], "M", 0)


def test_different_seed_differs(trained):
    first, second = _stage_c(trained["output"], "M", 0), _stage_c(trained["output"], "M", 1)
    assert any(not torch.equal(first[k], second[k]) for k in first)


def test_dependent_conditions_reuse_m_weights(trained):
    """B3 and A2 use M's Stage A+B of the same seed unchanged; A1 trains its own experts."""
    root = trained["output"]
    m_bank = torch.load(os.path.join(root, "runs", "M", "seed0", "checkpoints", "stage_B", "stage_b_expert_bank.pt"))["expert_bank_state"]
    m_encoder = torch.load(os.path.join(root, "runs", "M", "seed0", "checkpoints", "stage_A", "stage_a_encoder.pt"))["encoder_state"]
    for condition, own_weights in (("B3", False), ("A2", False), ("A1", True)):
        state = _stage_c(root, condition, 0)
        same_experts = all(torch.equal(state[f"expert_bank.{k}"], v) for k, v in m_bank.items())
        same_encoder = all(torch.equal(state[f"encoder.{k}"], v) for k, v in m_encoder.items())
        assert same_experts is (not own_weights) and same_encoder is (not own_weights), condition


def test_b4_same_seed_identical_different_seed_differs(trained, tmp_path, monkeypatch):
    """B4 seed 0 retrained from scratch (not reused from tuning) is identical; seed 1 differs."""
    import shutil

    roots = set_roots(monkeypatch, str(tmp_path))
    roots["data"] = trained["data"]
    tuning = os.path.join(roots["output"], "runs", "B4", "tuning")
    shutil.copytree(os.path.join(trained["output"], "runs", "B4", "tuning"), tuning)
    chosen = json.load(open(os.path.join(tuning, "chosen.json")))
    shutil.rmtree(os.path.join(tuning, chosen["config_id"]))  # force a fresh training run
    train_unit("B4", 0, roots)
    train_unit("B4", 1, roots)

    def model_text(root, seed):
        path = os.path.join(root, "runs", "B4", f"seed{seed}", "checkpoints", "stage_C", "xgboost_model.ubj")
        return open(path, "rb").read()

    assert model_text(trained["output"], 0) == model_text(roots["output"], 0)
    assert model_text(roots["output"], 0) != model_text(roots["output"], 1)
    assert _test_f1(trained["output"], "B4", 0) == _test_f1(roots["output"], "B4", 0)
