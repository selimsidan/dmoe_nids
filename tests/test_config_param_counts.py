"""Every condition config builds a model with exactly the documented parameter count."""
import pytest

from dmoe.data.registry import DATASET_NAMES
from dmoe.inference.bundle import assert_dataset_blind_signature, build_model_for_bundle
from dmoe.training.campaign import load_campaign
from dmoe.training.config import load_config

CLASSES = ["Benign", *[f"class_{i}" for i in range(21)]]
EXPECTED = {"M": 22420, "B2": 22419, "B3": 22332, "A1": 22420, "A2": 20120, "A3": 80020}  # B4: trees, no count


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


def test_b4_has_no_parameter_count():
    config = load_config(load_campaign().entry("B4").config_path)
    assert config["architecture"] == "pooled_xgboost"
    assert config["condition"]["expected_total_parameters"] is None


def test_campaign_lists_all_conditions_in_dependency_order():
    campaign = load_campaign()
    assert campaign.condition_ids == ["M", "B2", "B3", "B4", "A1", "A2", "A3"]
    assert campaign.seeds == [0, 1, 2, 3, 4]
    assert campaign.split_mode == "fixed" and campaign.split_seed == 0


@pytest.mark.parametrize("condition", sorted(EXPECTED))
def test_forward_is_dataset_blind(condition):
    _, model = _model(condition)
    assert_dataset_blind_signature(model)


def test_conditions_differ_from_m_only_where_documented():
    base = load_config(load_campaign().entry("M").config_path)
    a1 = load_config(load_campaign().entry("A1").config_path)
    assert base["training"]["representation"]["objective"] == "balanced_supcon"
    assert a1["training"]["representation"]["objective"] == "ce"
    a1["training"]["representation"]["objective"] = "balanced_supcon"
    for key in ("model", "training", "load_balance", "data", "architecture"):
        assert a1[key] == base[key]
    a2 = load_config(load_campaign().entry("A2").config_path)
    assert a2["architecture"] == "moe_dataset_uniform"
    for key in ("model", "training", "load_balance", "data"):
        assert a2[key] == base[key]
    b3 = load_config(load_campaign().entry("B3").config_path)
    assert b3["training"]["stage_c"]["gate_supervision"] == "damex"
    assert b3["model"]["gate"]["routing"] == "top1"
    assert b3["load_balance"]["lambda_balance"] == 0.0


# Config hashes recorded in the manifests of the completed paper runs of the
# conditions that were kept when the condition set changed. If any of these
# changes, the existing runs would count as outdated and be retrained.
KEPT_RUN_HASHES = {
    ("M", 0): "691c5f123aeccb44", ("M", 1): "8408baff1c50c413", ("M", 2): "637d54e9d6450d77",
    ("M", 3): "2df5dcb3badd14d9", ("M", 4): "1c5324acd846429c",
    ("B2", 0): "64b44a64fae009c4", ("B2", 1): "61e44e6d3e6fa052", ("B2", 2): "c985173ac4ec3b86",
    ("B2", 3): "29d7d66ebbe2599d", ("B2", 4): "d8584831dd4f1a7e",
    ("B3", 0): "12a7184a137fe31a", ("B3", 1): "caec788310aecbb8", ("B3", 2): "e098c26530ac9240",
    ("B3", 3): "f85b043b7666b988", ("B3", 4): "6cc331de16479f9a",
    ("A3", 0): "67cff7afe694aeaf", ("A3", 1): "33352f068049f4f0", ("A3", 2): "d3e0ed335bf5df76",
    ("A3", 3): "fc0b09ac1f6c8fd4", ("A3", 4): "1d1c12075bd69be5",
}


@pytest.mark.parametrize("condition,seed", sorted(KEPT_RUN_HASHES))
def test_kept_conditions_keep_their_config_hash(condition, seed, tmp_path):
    from dmoe.training.campaign import resolve_unit_config
    from dmoe.training.config import config_hash

    config = resolve_unit_config(load_campaign(), condition, seed, output_root=str(tmp_path))
    assert config_hash(config) == KEPT_RUN_HASHES[(condition, seed)]
