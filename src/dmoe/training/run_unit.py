"""Train, export and evaluate one (condition, seed) unit in its own process.

Normally launched by ``dmoe.training.campaign.run_campaign`` (the notebooks and
``scripts/train.py`` / ``scripts/run_campaign.py``), which streams this
process's output into ``runs/<condition>/seed<k>/log.txt``. Each stage is
skipped when its final checkpoint already exists, so rerunning the same unit
resumes where it stopped.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import torch

from .. import paths
from .campaign import load_campaign, resolve_unit_config
from .checkpoint import (
    BASELINE_MODEL_FILE,
    STAGE_DIRS,
    clear_progress,
    load_configured_stage_b,
    load_stage_c,
    resolve_stage_a_path,
    resolve_stage_b_path,
    stage_complete,
    stage_dir,
    stage_file,
)
from .config import CAMPAIGN_CONFIG
from .dense_ooc import load_dense_ooc, run_dense_stage_b_ooc, run_dense_stage_c_ooc
from .out_of_core_data import prepare_out_of_core_data
from .out_of_core_train import (
    CONTRACT_FILE,
    STAGE_C_SUMMARY_FILE,
    build_ooc_model,
    ensure_run_contract,
    run_stage_a_ooc,
    run_stage_b_ooc,
    run_stage_c_ooc,
)
from .reproducibility import configure_reproducibility, environment_info
from .training_history import TRAINING_HISTORY_FILE


def _resolve_device(config: dict) -> None:
    requested = str(config["training"].get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select a GPU runtime.")
    config["training"]["device"] = requested


def _restart(checkpoint_dir: str) -> None:
    for stage in ("A", "B", "C", "DB", "DC"):
        clear_progress(checkpoint_dir, stage)
    for stage in STAGE_DIRS:
        shutil.rmtree(stage_dir(checkpoint_dir, stage), ignore_errors=True)
    for filename in (CONTRACT_FILE, STAGE_C_SUMMARY_FILE, TRAINING_HISTORY_FILE):
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            os.remove(path)


def train_unit(config: dict, context) -> torch.nn.Module:
    """Run the requested stages and return the selected final model."""
    checkpoint_dir = config["training"]["checkpoint_dir"]
    stages = list(config["training"]["stages"])
    device = torch.device(config["training"]["device"])

    if config["architecture"] == "matched_dense":
        if "A" in stages and not stage_complete(checkpoint_dir, "A"):
            run_stage_a_ooc(config, context)
        if os.path.isfile(stage_file(checkpoint_dir, "C", BASELINE_MODEL_FILE)):
            return load_dense_ooc(config, context)
        model = run_dense_stage_b_ooc(config, context)
        return run_dense_stage_c_ooc(config, context, model)

    for stage, function in (("A", run_stage_a_ooc), ("B", run_stage_b_ooc), ("C", run_stage_c_ooc)):
        if stage not in stages:
            continue
        if stage_complete(checkpoint_dir, stage):
            print(f"[run] Stage {stage} already complete; skipping", flush=True)
            continue
        function(config, context)
    for required in (resolve_stage_a_path(config), resolve_stage_b_path(config)):
        if not os.path.isfile(required):
            raise FileNotFoundError(f"missing checkpoint {required}")
    model = build_ooc_model(config, context, device)
    stage_c = load_stage_c(checkpoint_dir)
    model.load_state_dict(stage_c["model_state"])
    model.training_summary = {}
    stage_b_summary = load_configured_stage_b(config).get("training_summary")
    if stage_b_summary:
        model.training_summary["B"] = stage_b_summary
    if stage_c.get("training_summary"):
        model.training_summary["C"] = stage_c["training_summary"]
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--scratch-root")
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    roots = paths.apply_cli_roots(args.data_root, args.output_root, args.scratch_root)
    campaign = load_campaign(args.campaign)
    config = resolve_unit_config(
        campaign, args.condition, args.seed,
        output_root=roots["output"].path, overrides=args.overrides,
    )
    configure_reproducibility(config["seed"], deterministic=bool(config["training"].get("deterministic", True)))
    _resolve_device(config)
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    started = time.monotonic()
    info = environment_info()
    print("=" * 80, flush=True)
    print(f"[run] {args.condition} seed {args.seed} started {time.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    print(f"[run] architecture={config['architecture']} stages={config['training']['stages']} "
          f"split_mode={config['data']['split_mode']} split_seed={config['data']['split_seed']}", flush=True)
    print(f"[run] device={config['training']['device']} gpu={info.get('gpu')} torch={info.get('torch')}", flush=True)
    for label, path in (("output", roots["output"].path), ("scratch", roots["scratch"].path)):
        os.makedirs(path, exist_ok=True)
        usage = shutil.disk_usage(path)
        print(f"[run] {label} free={usage.free / 2**30:.1f}GiB", flush=True)
    if args.force_restart:
        _restart(checkpoint_dir)

    context = prepare_out_of_core_data(config)
    contract = ensure_run_contract(config, context)
    print(f"[run] contract={contract['signature']} split_signatures={context.split_signatures}", flush=True)
    print(
        f"[run] rows train={len(context.data.train.class_idx):,} "
        f"val={len(context.data.val.class_idx):,} test={len(context.data.test.class_idx):,}",
        flush=True,
    )
    model = train_unit(config, context)
    if args.skip_evaluation:
        print("[run] training complete; evaluation skipped by request", flush=True)
        return
    from ..evaluation.finalize import finalize_unit

    finalize_unit(model, context, config, contract, training_seconds=time.monotonic() - started)
    print(f"[run] {args.condition} seed {args.seed} complete in {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
