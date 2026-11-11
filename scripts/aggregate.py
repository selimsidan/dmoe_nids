"""Aggregate completed runs across seeds into OUTPUT_ROOT/summary/."""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.evaluation.aggregate import aggregate
from dmoe.training.campaign import load_campaign
from dmoe.training.config import CAMPAIGN_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--only", nargs="+", metavar="CONDITION")
    parser.add_argument("--seeds", nargs="+", type=int, help="default: every completed seed")
    parser.add_argument("--no-figures", action="store_true")
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    campaign = load_campaign(args.campaign)
    aggregate(roots["output"].path, conditions=args.only, seeds=args.seeds,
              resamples=campaign.bootstrap_resamples, aggregation_seed=campaign.aggregation_seed,
              figures=not args.no_figures)


if __name__ == "__main__":
    main()
