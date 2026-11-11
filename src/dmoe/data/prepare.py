"""Build the train/validation/test split once and describe it in ``split.json``.

Training builds the split automatically on first use; running this ahead of
time (for example once on a high-memory runtime) is optional. The written
``OUTPUT_ROOT/cache/split.json`` records the split policy, seed, signatures
and per-class row counts of every split; ``scripts/sync_results.py`` copies it
to ``results/splits/split.json``.
"""
from __future__ import annotations

import json
import os

from .. import paths
from ..training.campaign import load_campaign, resolve_unit_config
from ..training.config import CAMPAIGN_CONFIG


def describe_split(context, split_mode: str) -> dict:
    datasets = {}
    for name, prepared in context.prepared_by_dataset.items():
        datasets[name] = {
            "signature": prepared.signature,
            "rows": prepared.split_rows,
            "class_rows": prepared.class_split_counts,
        }
    return {
        "split_mode": split_mode,
        "split_seed": context.split_seed,
        "fractions": "per class within each dataset: train 0.70, validation 0.15, test 0.15 (classes under 3 rows go to train)",
        "assignment_algorithm": "streamed-classwise-multivariate-hypergeometric-v1",
        "preprocessing": "pooled-train-standard-scaler-v1",
        "preprocessing_signature": context.preprocessing_signature,
        "class_names": list(context.data.class_names),
        "datasets": datasets,
        "totals": {
            split: int(sum(d["rows"][split] for d in datasets.values())) for split in ("train", "val", "test")
        },
    }


def prepare_split(campaign_path: str = CAMPAIGN_CONFIG, overrides: list[str] | None = None, seed: int = 0) -> dict:
    from ..training.out_of_core_data import prepare_out_of_core_data

    campaign = load_campaign(campaign_path)
    config = resolve_unit_config(campaign, "M", seed, overrides=overrides)
    context = prepare_out_of_core_data(config)
    description = describe_split(context, campaign.split_mode)
    target = os.path.join(paths.output_root(), "cache", "split.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as handle:
        json.dump(description, handle, indent=2, sort_keys=True)
    print(f"[prepare] split written; totals {description['totals']}; description at {target}")
    return description
