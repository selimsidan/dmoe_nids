"""Tiny synthetic NF-v3-like datasets for CPU tests.

Each dataset gets a CSV with all 47 features, the identity columns that the
loader must ignore, and ``Label``/``Attack`` columns whose raw values follow
``configs/base.yaml``'s label map. Classes are made separable by shifting a
few features per class, so short training runs learn something.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from dmoe.data.registry import FEATURE_COLUMNS, NETFLOW_V3_EXCLUDED

# Raw labels per dataset (subset of the real ones, enough to exercise mapping).
RAW_LABELS = {
    "NF-UNSW-NB15-v3": ["Benign", "Exploits", "Fuzzers", "Generic"],
    "NF-ToN-IoT-v3": ["Benign", "scanning", "ddos", "xss"],
    "NF-BoT-IoT-v3": ["Benign", "DDoS", "Reconnaissance", "Theft"],
    "NF-CICIDS2018-v3": ["BENIGN", "DDOS_attack-HOIC", "SSH-Bruteforce", "Bot"],
}

# Tests override training sizes with these settings.
FAST_OVERRIDES = [
    "training.epochs_a=2",
    "training.epochs_b=1",
    "training.epochs_c=2",
    "training.batch_size=64",
    "training.shuffle_block_rows=256",
    "training.validation_chunk_rows=500",
    "training.progress_every_rows=1000000",
    "training.device=cpu",
    "evaluation.prediction_chunk_rows=500",
    "evaluation.roc_auc_bins=64",
    "data.chunk_rows=300",
    "data.expected_classes=null",
]


def write_dataset(path: str, name: str, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = RAW_LABELS[name]
    weights = np.array([0.55, 0.2, 0.15, 0.10])
    y = rng.choice(len(labels), size=rows, p=weights)
    features = rng.normal(size=(rows, len(FEATURE_COLUMNS))) * 10 + 100
    dataset_shift = (seed % 7) * 3.0
    for k in range(len(labels)):
        features[y == k, k * 3 : k * 3 + 3] += 40.0 * (k + 1)
    features[:, 5] += dataset_shift
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame.iloc[::97, 7] = np.nan  # some missing values
    frame.iloc[::131, 9] = np.inf
    for column in NETFLOW_V3_EXCLUDED:
        frame[column] = rng.integers(0, 1000, size=rows)
    frame["Label"] = (y > 0).astype(int)
    frame["Attack"] = [labels[k] for k in y]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.sample(frac=1.0, random_state=seed).to_csv(path, index=False)


def make_data_root(root: str, rows: int = 900) -> str:
    from dmoe import paths

    for index, name in enumerate(RAW_LABELS):
        write_dataset(paths.dataset_csv_path(name, root), name, rows + 100 * index, seed=index + 1)
    return root
