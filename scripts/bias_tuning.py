"""Supplementary macro-F1 analysis: per-class offsets tuned on validation, for every run.

For each completed run (``runs/<condition>/seed<k>/bundle``), fits one score
offset per class on validation to maximize macro-F1, applies the offsets
unchanged to test, and writes ``eval/Bias_Tuning.json``; then writes
``summary/Bias_Tuning_Per_Seed.csv``, ``Bias_Tuning_Summary.csv`` and
``Bias_Tuning_Paired_vs_M.csv``. The same procedure is applied to every
condition; the main results stay untuned. See ``dmoe.evaluation.bias_tuning``.
Resumable: runs with an up-to-date ``Bias_Tuning.json`` are skipped.

Example:
  python scripts/bias_tuning.py                      # all conditions, seeds 0 to 4
  python scripts/bias_tuning.py --only M B2 --seeds 0
"""
from __future__ import annotations

import argparse
import os

from _common import add_root_arguments, apply_roots

from dmoe.evaluation.bias_tuning import BIAS_FILE, tune_run, write_summary
from dmoe.inference.bundle import load_bundle
from dmoe.training.campaign import load_campaign, resolve_unit_config
from dmoe.training.config import CAMPAIGN_CONFIG
from dmoe.training.out_of_core_data import prepare_out_of_core_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--only", nargs="+", metavar="CONDITION", help="default: every campaign condition")
    parser.add_argument("--seeds", nargs="+", type=int, help="default: the campaign seeds")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_root_arguments(parser)
    args = parser.parse_args(argv)
    roots = apply_roots(args)
    output = roots["output"].path
    campaign = load_campaign(args.campaign)
    conditions = args.only or campaign.condition_ids
    seeds = args.seeds if args.seeds is not None else list(campaign.seeds)
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = resolve_unit_config(campaign, "M", 0, output_root=output, overrides=args.overrides)
    context = prepare_out_of_core_data(config)
    chunk_rows = int(config["evaluation"].get("prediction_chunk_rows", 262_144))
    for condition in conditions:
        for seed in seeds:
            run = os.path.join(output, "runs", condition, f"seed{seed}")
            bundle_dir = os.path.join(run, "bundle")
            if not os.path.isdir(bundle_dir):
                print(f"[bias] {condition} seed {seed}: no bundle, skipped", flush=True)
                continue
            bundle = load_bundle(bundle_dir, device)
            if bundle.preprocessing["signature"] != context.preprocessing_signature:
                raise SystemExit(f"{bundle_dir} was trained on a different prepared split")
            result = tune_run(bundle, context, os.path.join(run, "eval", BIAS_FILE), chunk_rows=chunk_rows)
            print(f"[bias] {condition} seed {seed}: test macro-F1 {result['test']['raw']['macro_f1']:.4f} -> "
                  f"{result['test']['tuned']['macro_f1']:.4f} (validation "
                  f"{result['val']['raw']['macro_f1']:.4f} -> {result['val']['tuned']['macro_f1']:.4f})", flush=True)
    for name, path in write_summary(output, conditions, seeds).items():
        print(f"[bias] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
