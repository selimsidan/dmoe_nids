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
    """A1/A2/B3 start Stage C from M's Stage A+B of the same seed; A1 then changes the experts."""
    root = trained["output"]
    m_bank = torch.load(os.path.join(root, "runs", "M", "seed0", "checkpoints", "stage_B", "stage_b_expert_bank.pt"))["expert_bank_state"]
    for condition, experts_trained in (("B3", False), ("A2", False), ("A1", True)):
        state = _stage_c(root, condition, 0)
        same = all(torch.equal(state[f"expert_bank.{k}"], v) for k, v in m_bank.items())
        assert same is (not experts_trained), condition
