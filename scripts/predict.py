"""Predict any NF-v3 CSV with a bundle. No dataset identity is used for prediction.

Example:
  python scripts/predict.py --bundle outputs/runs/M/seed0/bundle --input flows.csv --output predictions.csv --probs --gate
  add --labels-from NF-ToN-IoT-v3 to also score the CSV's Attack column (written next to the output).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from dmoe.evaluation.evaluate import evaluate
from dmoe.inference.bundle import load_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="bundle directory or hf:<condition>/seed<k> | hf:recommended")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probs", action="store_true", help="add one probability column per class")
    parser.add_argument("--gate", action="store_true", help="add gate weights and the top expert (MoE models)")
    parser.add_argument("--labels-from", help="also evaluate against the Attack column, mapped with this dataset's labels")
    parser.add_argument("--chunk-rows", type=int, default=262_144)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    bundle = load_bundle(args.bundle, args.device)
    first = True
    rows = 0
    for chunk in pd.read_csv(args.input, chunksize=args.chunk_rows, low_memory=False):
        out = bundle.predict(chunk, with_probs=args.probs, with_gate=args.gate)
        out.to_csv(args.output, mode="w" if first else "a", header=first, index=False)
        first = False
        rows += len(out)
    print(f"[predict] wrote {rows:,} prediction(s) to {args.output}")
    if args.labels_from:
        target = os.path.splitext(args.output)[0] + "_evaluation"
        evaluate(target, bundle=bundle, csv=args.input, labels_from=args.labels_from, split="csv",
                 chunk_rows=args.chunk_rows)
        print(f"[predict] evaluation written to {target}")


if __name__ == "__main__":
    main()
