"""B4 timing pilot: a few rounds of the largest grid configuration on all training rows.

Loads all training and validation rows, builds the XGBoost matrices, trains
`--rounds` rounds of the largest configuration on the GPU, and prints seconds
per round, the time of one validation check, GPU memory in use and the
worst-case projections (no early stopping) for one run and for the tuning
grid. Writes nothing to the results.

Example (Colab A100 runtime, data on Drive):
  python scripts/xgboost_pilot.py --rounds 20
"""
from __future__ import annotations

import argparse
import json

from dmoe.openmp import prepare_for_trees

prepare_for_trees()  # before anything imports torch (see dmoe.openmp)

from _common import add_root_arguments, apply_roots  # noqa: E402

from dmoe.training.campaign import load_campaign, resolve_unit_config  # noqa: E402
from dmoe.training.config import CAMPAIGN_CONFIG  # noqa: E402
from dmoe.training.out_of_core_data import prepare_out_of_core_data  # noqa: E402
from dmoe.training.xgboost_baseline import (  # noqa: E402
    PreparedMatrices, grid_configs, measure_cost, xgboost_settings,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--condition", default="B4")
    parser.add_argument("--rounds", type=int, default=None, help="default: xgboost.tuning.pilot_rounds")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_root_arguments(parser)
    args = parser.parse_args(argv)
    roots = apply_roots(args)
    campaign = load_campaign(args.campaign)
    config = resolve_unit_config(campaign, args.condition, 0, output_root=roots["output"].path,
                                 overrides=args.overrides)
    settings = xgboost_settings(config)
    rounds = args.rounds or int(settings["tuning"].get("pilot_rounds", 20))
    context = prepare_out_of_core_data(config)
    cost = measure_cost(config, context, PreparedMatrices(config, context), rounds=rounds)
    seeds = len(campaign.seeds)
    print(json.dumps(cost, indent=2, default=str))
    print(f"[pilot] {cost['seconds_per_round']:.2f} s per round on {cost['rows']:,} rows ({cost['device']}); "
          f"one validation check {cost['validation_check_seconds']:.1f} s")
    print(f"[pilot] one run at {settings['max_rounds']} rounds (no early stop): "
          f"{cost['projected_hours_per_run_at_max_rounds']:.1f} h")
    print(f"[pilot] tuning grid ({len(grid_configs(settings))} configurations): "
          f"{cost['projected_hours_grid']:.1f} h (worst case, no early stopping)")
    print(f"[pilot] {seeds} seed runs: up to {seeds * cost['projected_hours_per_run_at_max_rounds']:.1f} h "
          "(early stopping usually ends runs sooner; seed 0 reuses the tuning run)")
    if cost.get("gpu_memory_used_bytes"):
        print(f"[pilot] GPU memory in use {cost['gpu_memory_used_bytes'] / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
