"""Evaluate saved predictions, a bundle on the prepared split, or a bundle on a labeled CSV.

Examples:
  python scripts/evaluate.py --predictions outputs/runs/M/seed0/eval/test/Test_Predictions.npz --out eval_again
  python scripts/evaluate.py --bundle outputs/runs/M/seed0/bundle --split test --out eval_full
  python scripts/evaluate.py --bundle hf:recommended --csv my_flows.csv --labels-from NF-ToN-IoT-v3 --out eval_csv
"""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.evaluation.evaluate import evaluate
from dmoe.inference.bundle import load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", help="a saved *_Predictions.npz (no data or GPU needed)")
    source.add_argument("--bundle", help="bundle directory or hf:<condition>/seed<k> | hf:recommended")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="with --bundle and no --csv")
    parser.add_argument("--csv", help="labeled NF-v3 CSV to score with --bundle")
    parser.add_argument("--labels-from", help="dataset whose label map applies to the CSV's Attack column")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-figures", action="store_true")
    add_root_arguments(parser)
    args = parser.parse_args()
    apply_roots(args)
    if args.predictions:
        evaluate(args.out, predictions=args.predictions, figures=not args.no_figures)
        return
    bundle = load_bundle(args.bundle, args.device)
    if args.csv:
        evaluate(args.out, bundle=bundle, csv=args.csv, labels_from=args.labels_from, split="csv", figures=not args.no_figures)
        return
    from dmoe.training.campaign import load_campaign, resolve_unit_config
    from dmoe.training.out_of_core_data import prepare_out_of_core_data

    config = resolve_unit_config(load_campaign(), "M", 0)
    context = prepare_out_of_core_data(config)
    if context.preprocessing_signature != bundle.preprocessing["signature"]:
        raise SystemExit("the prepared split differs from the one this bundle was trained on; refusing to evaluate")
    evaluate(args.out, bundle=bundle, context=context, split=args.split, figures=not args.no_figures)


if __name__ == "__main__":
    main()
