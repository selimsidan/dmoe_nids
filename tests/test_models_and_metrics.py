"""Model-level guarantees (routing, supervision, ownership) and metric arithmetic."""
import numpy as np
import pytest
import torch

from dmoe.evaluation.confusion import confusion_figure
from dmoe.evaluation.metrics import confusion_long_rows, overall_from_confusion
from dmoe.models.dataset_experts import DatasetExpertBank
from dmoe.models.moe import ClassConditionalMoEDatasetNIDS, MoEDatasetNIDS
from dmoe.training.gate_supervision import gate_weights_for_task, lambda_dataset_aux

from test_config_param_counts import _model


def test_top1_runs_only_selected_experts():
    bank = DatasetExpertBank(["a", "b", "c"], 8, 5, hidden_dims=[], dropout=0.0)
    calls = []
    for index, expert in enumerate(bank.experts):
        expert.register_forward_hook(lambda module, inputs, output, i=index: calls.append((i, inputs[0].shape[0])))
    z = torch.randn(6, 8)
    bank.forward_selected(z, torch.tensor([0, 0, 2, 2, 2, 0]))
    assert sorted(calls) == [(0, 3), (2, 3)]


def test_class_reliability_zero_equals_soft_mixture():
    _, model = _model("M")
    assert isinstance(model, ClassConditionalMoEDatasetNIDS)
    assert torch.count_nonzero(model.class_reliability) == 0
    model.eval()
    x = torch.randn(16, 47)
    out = model(x)
    soft = torch.einsum("bd,bdc->bc", out["gate_weights"], out["expert_probs"])
    torch.testing.assert_close(out["combined_probs"], soft)


def test_m_gate_gets_no_dataset_supervision():
    m_config, _ = _model("M")
    assert lambda_dataset_aux(m_config) == 0.0
    b3_config, _ = _model("B3")
    assert lambda_dataset_aux(b3_config) == 1.0
    weights = torch.rand(4, 3, requires_grad=True)
    assert not gate_weights_for_task(weights, "damex").requires_grad
    assert gate_weights_for_task(weights, "none").requires_grad


def test_assigned_only_masks_foreign_expert_gradients():
    gate = torch.full((2, 2), 0.5)
    probs = torch.rand(2, 2, 3, requires_grad=True)
    out = MoEDatasetNIDS.combine_probs_for_training(gate, probs, torch.tensor([0, 1]), "assigned_only")
    out.sum().backward()
    assert torch.all(probs.grad[0, 1] == 0) and torch.all(probs.grad[1, 0] == 0)
    assert torch.all(probs.grad[0, 0] != 0) and torch.all(probs.grad[1, 1] != 0)


def test_metrics_on_known_confusion():
    confusion = np.array([[5, 1, 0], [2, 2, 0], [0, 0, 0]])
    row = overall_from_confusion(confusion)
    f1 = [2 * (5 / 7) * (5 / 6) / (5 / 7 + 5 / 6), 2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4)]
    assert row["macro_f1"] == pytest.approx(np.mean(f1))
    assert row["accuracy"] == row["micro_f1"] == pytest.approx(0.7)
    assert row["classes_with_support"] == 2 and row["rows"] == 10
    assert list(row)[0] == "macro_f1"


def test_confusion_counts_and_ratios():
    confusion = np.array([[12431, 1808], [0, 0]])
    rows = confusion_long_rows(confusion, ["Benign", "DoS"])
    first = [r for r in rows if r["true_class"] == "Benign"]
    assert sum(r["row_ratio"] for r in first) == pytest.approx(1.0)
    assert all(r["row_ratio"] == 0 for r in rows if r["true_class"] == "DoS")
    fig = confusion_figure(confusion, ["Benign", "DoS"], "test")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "12,431\n(87.3%)" in texts and len(texts) == 2  # zero cells stay blank
