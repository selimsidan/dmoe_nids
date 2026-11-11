"""End of a training unit: export the bundle, evaluate val and test, write the manifest.

Layout of ``runs/<condition>/seed<k>/`` afterwards::

    checkpoints/                 stage checkpoints, progress files, run contract, train history
    bundle/                      self-contained model (see dmoe.inference.bundle)
    eval/selection/              validation reports written during Stage C selection
    eval/val/, eval/test/        all tables and figures of that split, plus <Split>_Predictions.npz
    eval/Resource_Accounting.csv parameters, FLOPs, training time, inference time, peak memory
    Training_History.csv         per-epoch losses and validation macro-F1 of every stage
    figures/                     learning curves
    manifest.json                status, config hash, split info, environment, hashes, metrics
    log.txt                      streamed training output (written by the campaign runner)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone

import pandas as pd

from .. import paths
from ..inference.bundle import export_bundle
from ..training.checkpoint import (
    BASELINE_MODEL_FILE, BASELINE_STAGE_B_FILE, STAGE_C_FILE,
    resolve_stage_a_path, resolve_stage_b_path, stage_file,
)
from ..training.config import config_hash
from ..training.reproducibility import environment_info
from ..training.training_history import TRAINING_HISTORY_FILE, read_training_history, plot_seed_training_history
from .predictions import PREDICTION_FILES, predict_split
from .reports import macro_f1, macro_f1_by_dataset, summary_line, write_reports
from .resource_accounting import resource_profile, write_resource_accounting

MANIFEST_FILE = "manifest.json"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_records(config: dict, output_root: str) -> dict:
    checkpoint_dir = config["training"]["checkpoint_dir"]
    candidates = {
        "stage_a": resolve_stage_a_path(config),
        "stage_b": resolve_stage_b_path(config),
        "stage_c": stage_file(checkpoint_dir, "C", STAGE_C_FILE),
        "baseline_stage_b": stage_file(checkpoint_dir, "B", BASELINE_STAGE_B_FILE),
        "baseline": stage_file(checkpoint_dir, "C", BASELINE_MODEL_FILE),
    }
    records = {}
    for key, path in candidates.items():
        if os.path.isfile(path):
            records[key] = {
                "path": paths.relative_to_root(path, output_root),
                "sha256": _sha256(path),
                "size_bytes": os.path.getsize(path),
                "shared_from": None if path.startswith(checkpoint_dir) else config["run"]["depends_on"].get(key),
            }
    return records


def _selected_epochs(config: dict) -> dict:
    summary_path = os.path.join(config["training"]["checkpoint_dir"], "stage_c_training_summary.json")
    selected = {"stage_a": config["training"].get("epochs_a"), "stage_b": config["training"].get("epochs_b")}
    if os.path.isfile(summary_path):
        with open(summary_path) as handle:
            value = json.load(handle)
        selected["stage_c"] = value.get("best_epoch")
        selected["stage_c_best_validation_macro_f1"] = value.get("best_validation_macro_f1")
    return selected


def write_manifest(path: str, manifest: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def finalize_unit(model, context, config: dict, contract: dict, *, training_seconds: float) -> dict:
    run = config["run"]
    run_dir, output_root = run["run_dir"], run["output_root"]
    eval_dir = config["evaluation"]["output_dir"]
    condition, seed = run["condition"], int(run["seed"])
    display_names = config.get("data", {}).get("display_names", {})
    started = time.monotonic()
    environment = environment_info()
    manifest_path = os.path.join(run_dir, MANIFEST_FILE)
    write_manifest(manifest_path, {
        "status": "evaluating", "condition": condition, "seed": seed, "config_hash": config_hash(config),
    })

    # 1. Per-split predictions and reports (validation first, test last).
    results = {}
    for split in ("val", "test"):
        record = predict_split(model, context, split, config)
        split_dir = os.path.join(eval_dir, split)
        record.save(os.path.join(split_dir, PREDICTION_FILES[split]))
        write_reports(record, split_dir, condition=condition, seed=seed, split=split, display_names=display_names)
        print(f"[evaluate:{split}] {summary_line(record)}", flush=True)
        results[split] = {
            "macro_f1": macro_f1(record),
            "macro_f1_by_dataset": macro_f1_by_dataset(record),
            "rows": record.rows,
            "inference": {k: v for k, v in record.meta.items() if k.startswith(("inference", "peak"))},
        }

    # 2. Resource accounting (parameters, FLOPs, training and inference cost).
    training_summary = dict(getattr(model, "training_summary", {}))
    stage_a_path = resolve_stage_a_path(config)
    if os.path.isfile(stage_a_path):
        import torch

        stage_metadata = torch.load(stage_a_path, map_location="cpu").get("metadata") or {}
        if stage_metadata.get("training_summary"):
            training_summary["A"] = stage_metadata["training_summary"]
    row = resource_profile(
        model, config, trial_id=f"{condition}-seed{seed}",
        split_signature=contract.get("preprocessing_signature"),
        checkpoint_hash=_sha256(stage_a_path) if os.path.isfile(stage_a_path) else None,
        training_summary=training_summary,
    )
    row.update({f"test_{k}": v for k, v in results["test"]["inference"].items()})
    row["training_wall_seconds_this_unit"] = training_seconds
    write_resource_accounting(eval_dir, [row])

    # 3. Training history and learning curves.
    history = read_training_history(config["training"]["checkpoint_dir"])
    stage_dirs = {os.path.dirname(resolve_stage_a_path(config)), os.path.dirname(resolve_stage_b_path(config))}
    shared = [read_training_history(d) for d in stage_dirs if d != config["training"]["checkpoint_dir"]]
    for frame in shared:
        if not frame.empty:
            frame = frame[frame["stage"].isin([s for s in ("A", "B") if s not in config["training"]["stages"]])]
            history = pd.concat([history, frame], ignore_index=True)
    history_path = os.path.join(run_dir, TRAINING_HISTORY_FILE)
    history.to_csv(history_path + ".tmp", index=False)
    os.replace(history_path + ".tmp", history_path)
    figure_paths = plot_seed_training_history(
        history, os.path.join(run_dir, "figures"), selected_epoch=_selected_epochs(config).get("stage_c"),
    ) if not history.empty else []

    # 4. Bundle export.
    bundle_dir = os.path.join(run_dir, "bundle")
    if os.path.isdir(bundle_dir):
        shutil.rmtree(bundle_dir)
    bundle_metadata = {
        "condition": condition,
        "display_name": config.get("condition", {}).get("display_name"),
        "seed": seed,
        "origin": "release_trained",
        "architecture": config["architecture"],
        "split": {
            "mode": config["data"].get("split_mode"), "seed": config["data"].get("split_seed"),
            "signatures": dict(context.split_signatures), "preprocessing_signature": context.preprocessing_signature,
        },
        "selected_epochs": _selected_epochs(config),
        "val_macro_f1": results["val"]["macro_f1"],
        "test_macro_f1": results["test"]["macro_f1"],
        "total_parameters": int(row["total_parameters"]),
        "config_hash": config_hash(config),
        "environment": environment,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
    }
    bundle_hashes = export_bundle(
        model, config, context, bundle_dir, metadata=bundle_metadata,
        include_stages=config["architecture"] != "matched_dense" and "A" in config["training"]["stages"],
    )

    # 5. Split description (shared by every run; written once).
    split_path = os.path.join(output_root, "cache", "split.json")
    if not os.path.isfile(split_path):
        from ..data.prepare import describe_split

        with open(split_path + ".tmp", "w") as handle:
            json.dump(describe_split(context, config["data"].get("split_mode")), handle, indent=2, sort_keys=True)
        os.replace(split_path + ".tmp", split_path)

    # 6. Manifest.
    manifest = {
        "format_version": 1,
        "status": "complete",
        "condition": condition,
        "display_name": config.get("condition", {}).get("display_name"),
        "seed": seed,
        "origin": "release_trained",
        "architecture": config["architecture"],
        "config_hash": config_hash(config),
        "contract_signature": contract.get("signature"),
        "expected_total_parameters": config.get("condition", {}).get("expected_total_parameters"),
        "total_parameters": int(row["total_parameters"]),
        "trainable_parameters_stage_c": int(row["trainable_parameters"]),
        "stages_trained": list(config["training"]["stages"]),
        "depends_on": dict(run["depends_on"]),
        "split": bundle_metadata["split"],
        "class_names": list(context.data.class_names),
        "dataset_names": list(context.data.active_datasets),
        "selected_epochs": _selected_epochs(config),
        "results": results,
        "checkpoints": _checkpoint_records(config, output_root),
        "bundle": {"path": paths.relative_to_root(bundle_dir, output_root), "file_sha256": bundle_hashes},
        "files": {
            "eval_dir": paths.relative_to_root(eval_dir, output_root),
            "training_history": paths.relative_to_root(history_path, output_root),
            "figures": [paths.relative_to_root(p, output_root) for p in figure_paths],
        },
        "environment": environment,
        "times": {
            "training_wall_seconds_this_unit": training_seconds,
            "evaluation_wall_seconds": time.monotonic() - started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_manifest(manifest_path, manifest)
    return manifest
