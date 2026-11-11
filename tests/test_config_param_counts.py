"""Every condition config builds a model with exactly the documented parameter count."""
import pytest

from dmoe.data.registry import DATASET_NAMES
from dmoe.inference.bundle import assert_dataset_blind_signature, build_model_for_bundle
from dmoe.training.campaign import load_campaign
from dmoe.training.config import load_config

CLASSES = ["Benign", *[f"class_{i}" for i in range(21)]]
EXPECTED = {"M": 22420, "B1": 22420, "B2": 22419, "B3": 22332, "A1": 22420, "A2": 22332, "A3": 80020}


def _model(condition):
    config = load_config(load_campaign().entry(condition).config_path)
    model = build_model_for_bundle({
        "architecture": config["architecture"], "model": config["model"], "input_dim": 47,
        "dataset_names": list(DATASET_NAMES), "class_names": CLASSES,
    })
    return config, model


@pytest.mark.parametrize("condition", sorted(EXPECTED))
def test_parameter_count(condition):
    config, model = _model(condition)
    total = sum(p.numel() for p in model.parameters())
    assert total == EXPECTED[condition] == config["condition"]["expected_total_parameters"]


def test_b2_head_widths():
    _, model = _model("B2")
    assert model.match_info["hidden_dims"] == [61, 48]
    assert abs(model.match_info["relative_error"]) < 0.005


def test_campaign_lists_all_conditions_in_dependency_order():
    campaign = load_campaign()
    assert campaign.condition_ids == ["M", "B1", "B2", "B3", "A1", "A2", "A3"]
    assert campaign.seeds == [0, 1, 2, 3, 4]
    assert campaign.split_mode == "fixed" and campaign.split_seed == 0


@pytest.mark.parametrize("condition", sorted(EXPECTED))
def test_forward_is_dataset_blind(condition):
    _, model = _model(condition)
    assert_dataset_blind_signature(model)


def test_conditions_differ_from_m_only_where_documented():
    base = load_config(load_campaign().entry("M").config_path)
    a1 = load_config(load_campaign().entry("A1").config_path)
    assert a1["training"]["stage_c"]["freeze_experts"] is False
    a1["training"]["stage_c"]["freeze_experts"] = True
    for key in ("model", "training", "load_balance", "data"):
        assert a1[key] == base[key]
    b3 = load_config(load_campaign().entry("B3").config_path)
    assert b3["training"]["stage_c"]["gate_supervision"] == "damex"
    assert b3["model"]["gate"]["routing"] == "top1"
    assert b3["load_balance"]["lambda_balance"] == 0.0
