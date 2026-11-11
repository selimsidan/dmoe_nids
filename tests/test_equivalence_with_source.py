"""Pruning did not change the math: release and development code agree exactly.

Needs a checkout of the development repository; set ``DMOE_SOURCE_REPO`` to
its path. Skipped otherwise (strangers do not need it).
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

SOURCE = os.environ.get("DMOE_SOURCE_REPO")
pytestmark = pytest.mark.skipif(
    not SOURCE or not os.path.isdir(os.path.join(SOURCE or "", "models")),
    reason="set DMOE_SOURCE_REPO to the development repository to run this check",
)

DATASETS = ["NF-UNSW-NB15-v3", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CICIDS2018-v3"]
CLASSES = ["Benign", *[f"class_{i}" for i in range(21)]]


def _source_outputs(architecture, model_cfg, tmp_path):
    """Build the model with the source code in a clean subprocess and return state + outputs."""
    script = textwrap.dedent(f"""
        import sys, json, torch
        sys.path.insert(0, {SOURCE!r})
        from models.encoder import build_encoder, resolve_encoder_config
        from training.model_utils import build_model, build_matched_dense_model
        cfg = json.loads({__import__('json').dumps(model_cfg)!r})
        torch.manual_seed(7)
        role = "gate_encoder" if "{architecture}".startswith("moe_dataset_private") else "encoder"
        encoder = build_encoder(47, cfg["latent_dim"], resolve_encoder_config(cfg, role))
        if "{architecture}" == "matched_dense":
            model = build_matched_dense_model(encoder, {DATASETS!r}, {CLASSES!r}, cfg)
        else:
            model = build_model("{architecture}", encoder, {DATASETS!r}, {CLASSES!r}, cfg)
        model.eval()
        torch.manual_seed(11)
        x = torch.randn(64, 47)
        with torch.no_grad():
            out = model(x)
        probs = out["combined_probs"] if "combined_probs" in out else torch.softmax(out["logits"], 1)
        torch.save({{"state": model.state_dict(), "probs": probs, "x": x}}, {str(tmp_path / "source.pt")!r})
    """)
    subprocess.run([sys.executable, "-c", script], check=True, cwd=str(tmp_path))
    return torch.load(tmp_path / "source.pt")


def _release_outputs(architecture, model_cfg, x):
    from dmoe.inference.bundle import build_model_for_bundle

    torch.manual_seed(7)
    model = build_model_for_bundle({
        "architecture": architecture, "model": model_cfg, "input_dim": 47,
        "dataset_names": DATASETS, "class_names": CLASSES,
    })
    model.eval()
    with torch.no_grad():
        out = model(x)
    probs = out["combined_probs"] if "combined_probs" in out else torch.softmax(out["logits"], 1)
    return model.state_dict(), probs


SOURCE_ARCHITECTURE = {  # A3 is new in the release; its source counterpart is the plain private-encoder MoE.
    "moe_dataset_private_encoders_class_conditional": None,
}


@pytest.mark.parametrize("condition", ["M", "B1", "B2", "B3", "A1", "A2"])
def test_build_model_matches_source(condition, tmp_path):
    from dmoe.training.campaign import load_campaign
    from dmoe.training.config import load_config

    config = load_config(load_campaign().entry(condition).config_path)
    source = _source_outputs(config["architecture"], config["model"], tmp_path)
    state, probs = _release_outputs(config["architecture"], config["model"], source["x"])
    assert list(state) == list(source["state"])
    for key in state:
        assert torch.equal(state[key], source["state"][key]), key
    assert torch.equal(probs, source["probs"])


def test_split_assignment_matches_source(tmp_path):
    """Same CSV, same split seed: the release writes exactly the source's split arrays."""
    sys.path.insert(0, os.path.dirname(__file__))
    from synthetic import write_dataset

    from dmoe.data import out_of_core as release
    from dmoe.data.registry import DatasetSpec
    from dmoe.training.config import load_yaml, BASE_CONFIG

    name = "NF-ToN-IoT-v3"
    csv = tmp_path / "data" / name / "data" / f"{name}.csv"
    write_dataset(str(csv), name, 1500, seed=3)
    base = load_yaml(BASE_CONFIG)
    mapping = base["data"]["label_mapping"][name]
    from dmoe.data.registry import FEATURE_COLUMNS

    os.environ["NIDS_DATA_DIR"] = str(tmp_path / "data")
    prepared = release.prepare_dataset(DatasetSpec(name), mapping, FEATURE_COLUMNS, base["data"]["split"], 0,
                                       str(tmp_path / "release"), 300)
    script = textwrap.dedent(f"""
        import sys, os
        os.environ["NIDS_DRIVE_BASE"] = {str(tmp_path / "data")!r}
        sys.path.insert(0, {SOURCE!r})
        from data.out_of_core import prepare_dataset
        from data.registry import get_spec, NETFLOW_V3_ALIAS
        import yaml
        mapping = yaml.safe_load(open({os.path.join(SOURCE, "config", "default.yaml")!r}))["data"]["label_mapping"]["{name}"]
        prepare_dataset(get_spec("{name}"), mapping, list(NETFLOW_V3_ALIAS.values()),
                        {{"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "min_class_count": 3}},
                        0, {str(tmp_path / "source")!r}, 300)
    """)
    subprocess.run([sys.executable, "-c", script], check=True, cwd=str(tmp_path))
    source_dir = next((tmp_path / "source" / name).glob("split-*"))
    for split in ("train", "val", "test"):
        np.testing.assert_array_equal(prepared.features(split), np.load(source_dir / f"{split}_X.npy"))
        np.testing.assert_array_equal(prepared.labels(split), np.load(source_dir / f"{split}_y.npy"))
