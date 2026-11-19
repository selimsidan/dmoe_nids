"""B4 timing pilot: train a few rounds of the largest grid configuration and project the cost.

Loads B4's class-capped training sample and all validation rows into memory
(as B4 does), trains
`--rounds` rounds of the largest configuration with validation macro-F1
checked every round, and prints seconds per round, the cost of continuing
chunks from a saved model, peak memory, and projections for the tuning grid
and for the seed runs. Writes nothing to the results.

Example (Colab CPU high-RAM runtime, data on Drive):
  python scripts/lightgbm_pilot.py --rounds 20
"""
from __future__ import annotations

import argparse
import json

from dmoe.openmp import prepare_for_lightgbm

prepare_for_lightgbm()  # before anything imports torch (see dmoe.openmp)

from _common import add_root_arguments, apply_roots  # noqa: E402

from dmoe.training.campaign import load_campaign, resolve_unit_config  # noqa: E402
from dmoe.training.config import CAMPAIGN_CONFIG  # noqa: E402
from dmoe.training.lightgbm_baseline import (  # noqa: E402
    PreparedArrays, grid_configs, lightgbm_settings, measure_cost,
)
from dmoe.training.out_of_core_data import prepare_out_of_core_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--condition", default="B4")
    parser.add_argument("--rounds", type=int, default=None, help="default: lightgbm.tuning.pilot_rounds")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    campaign = load_campaign(args.campaign)
    config = resolve_unit_config(campaign, args.condition, 0, output_root=roots["output"].path, overrides=args.overrides)
    settings = lightgbm_settings(config)
    rounds = args.rounds or int(settings["tuning"].get("pilot_rounds", 20))
    context = prepare_out_of_core_data(config)
    cost = measure_cost(config, context, PreparedArrays(config, context), rounds=rounds)
    seeds = len(campaign.seeds)
    print(json.dumps(cost, indent=2, default=str))
    print(f"[pilot] {cost['seconds_per_round']:.2f} s per round on {cost['rows']:,} rows with {cost['threads']} threads")
    print(f"[pilot] one run at {settings['max_rounds']} rounds (no early stop): "
          f"{cost['projected_hours_per_run_at_max_rounds']:.1f} h")
    print(f"[pilot] tuning grid ({len(grid_configs(settings))} configurations, training sample): "
          f"{cost['projected_hours_grid']:.1f} h (limit {settings['tuning']['max_projected_hours']} h)")
    print(f"[pilot] {seeds} seed runs: up to {seeds * cost['projected_hours_per_run_at_max_rounds']:.1f} h "
          "(early stopping usually ends runs sooner)")
    if cost.get("peak_memory_bytes"):
        print(f"[pilot] peak memory {cost['peak_memory_bytes'] / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
