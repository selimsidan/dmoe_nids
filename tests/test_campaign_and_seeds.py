"""Campaign order, dependencies, seed subsets, the split policy and completion detection."""
import json
import os

import pytest

from dmoe.training.campaign import (
    load_campaign, missing_dependencies, plan_units, resolve_unit_config, stages_for,
)
from dmoe.training.config import CAMPAIGN_CONFIG, config_hash, load_yaml


def test_units_are_seed_major_with_m_first(tmp_path):
    campaign = load_campaign()
    units = plan_units(campaign, seeds=[3, 4], output_root=str(tmp_path))
    assert [(u.seed, u.condition) for u in units][:7] == [(3, c) for c in campaign.condition_ids]
    assert units[7].seed == 4 and units[7].condition == "M"
    b3 = next(u for u in units if u.condition == "B3" and u.seed == 3)
    assert b3.waits_for == ["M seed 3 Stage A", "M seed 3 Stage B"] and not b3.missing


def test_missing_parent_is_named(tmp_path):
    campaign = load_campaign()
    units = plan_units(campaign, ["A1"], [3], output_root=str(tmp_path))
    assert units[0].status == "blocked"
    assert "M seed 3 Stage B" in units[0].describe()
    assert missing_dependencies(campaign, "A1", 3, str(tmp_path)) == ["M seed 3 Stage A", "M seed 3 Stage B"]


def test_stage_reuse_per_condition():
    campaign = load_campaign()
    expected = {"M": ["A", "B", "C"], "B1": ["B", "C"], "B2": ["B", "C"], "B3": ["C"],
                "A1": ["C"], "A2": ["C"], "A3": ["B", "C"]}
    assert {c: stages_for(campaign.entry(c)) for c in expected} == expected


def test_any_seed_list_resolves(tmp_path):
    campaign = load_campaign()
    for seed in (0, 3, 5, 17):
        config = resolve_unit_config(campaign, "A3", seed, output_root=str(tmp_path))
        assert config["seed"] == seed
        assert config["training"]["stage_a_checkpoint"].endswith(os.path.join("M", f"seed{seed}", "checkpoints", "stage_A", "stage_a_encoder.pt"))
        assert config["training"]["private_stage_a_checkpoint"] == config["training"]["stage_a_checkpoint"]


def test_fixed_split_never_follows_training_seed(tmp_path):
    campaign = load_campaign()
    split_seeds = {resolve_unit_config(campaign, "M", s, output_root=str(tmp_path))["data"]["split_seed"] for s in range(5)}
    assert split_seeds == {0}


def test_per_seed_split_mode(tmp_path):
    raw = load_yaml(CAMPAIGN_CONFIG)
    raw["split"]["mode"] = "per_seed"
    path = tmp_path / "campaign.yaml"
    import yaml

    path.write_text(yaml.safe_dump(raw))
    campaign = load_campaign(str(path))
    assert [resolve_unit_config(campaign, "M", s, output_root=str(tmp_path))["data"]["split_seed"] for s in (0, 2, 4)] == [0, 2, 4]


def test_config_hash_ignores_paths(tmp_path):
    campaign = load_campaign()
    a = resolve_unit_config(campaign, "M", 0, output_root=str(tmp_path / "a"))
    b = resolve_unit_config(campaign, "M", 0, output_root=str(tmp_path / "b"))
    c = resolve_unit_config(campaign, "M", 1, output_root=str(tmp_path / "a"))
    assert config_hash(a) == config_hash(b) != config_hash(c)


def test_bad_seed_lists_rejected(tmp_path):
    import yaml

    raw = load_yaml(CAMPAIGN_CONFIG)
    for seeds in ([], [1, 1], [-1]):
        raw["seeds"] = seeds
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError):
            load_campaign(str(path))


def test_completed_units_are_skipped(trained):
    """With seed 0 of everything and seed 1 of M done, only the pending units remain."""
    from synthetic import FAST_OVERRIDES

    campaign = load_campaign()
    units = plan_units(campaign, seeds=[0, 1], output_root=trained["output"], overrides=FAST_OVERRIDES)
    status = {(u.condition, u.seed): u.status for u in units}
    assert all(status[(c, 0)] == "complete" for c in campaign.condition_ids)
    assert status[("M", 1)] == "complete"
    assert status[("B1", 1)] == "pending"
    # A different setting (more epochs) is a different config, so nothing counts as done.
    changed = plan_units(campaign, ["M"], [0], output_root=trained["output"], overrides=[*FAST_OVERRIDES, "training.epochs_c=3"])
    assert changed[0].status != "complete"


def test_manifest_records_split_and_environment(trained):
    manifest = json.load(open(os.path.join(trained["output"], "runs", "B3", "seed0", "manifest.json")))
    assert manifest["status"] == "complete"
    assert manifest["split"]["mode"] == "fixed" and manifest["split"]["seed"] == 0
    assert manifest["stages_trained"] == ["C"]
    assert manifest["checkpoints"]["stage_b"]["shared_from"] == "M"
    for key in ("torch", "numpy", "python", "commit"):
        assert key in manifest["environment"]
    text = json.dumps(manifest["checkpoints"]) + json.dumps(manifest["files"]) + json.dumps(manifest["bundle"])
    assert trained["output"] not in text  # committed manifests use paths relative to OUTPUT_ROOT


def test_all_runs_share_one_split(trained):
    signatures = set()
    for condition in ("M", "B1", "B2", "B3", "A1", "A2", "A3"):
        manifest = json.load(open(os.path.join(trained["output"], "runs", condition, "seed0", "manifest.json")))
        signatures.add(json.dumps(manifest["split"]["signatures"], sort_keys=True))
    seed1 = json.load(open(os.path.join(trained["output"], "runs", "M", "seed1", "manifest.json")))
    signatures.add(json.dumps(seed1["split"]["signatures"], sort_keys=True))
    assert len(signatures) == 1


def test_checkpoint_folder_layout(trained):
    """One checkpoints/ folder per run, with a subfolder per stage; reused stages stay with M."""
    def listing(condition):
        base = os.path.join(trained["output"], "runs", condition, "seed0", "checkpoints")
        return {os.path.relpath(os.path.join(d, f), base) for d, _, files in os.walk(base) for f in files}

    m = listing("M")
    assert {"run_contract.json", "stage_A/stage_a_encoder.pt", "stage_B/stage_b_expert_bank.pt",
            "stage_C/stage_c_full.pt", "stage_A/Representation_Batch_Coverage.csv"} <= m
    assert not any(name.endswith("_progress.pt") for name in m)  # cleared after each stage
    b3 = listing("B3")
    assert "stage_C/stage_c_full.pt" in b3 and not any(n.startswith(("stage_A/", "stage_B/")) for n in b3)
    b2 = listing("B2")
    assert {"stage_B/baseline_stage_b.pt", "stage_C/baseline_full.pt"} <= b2
