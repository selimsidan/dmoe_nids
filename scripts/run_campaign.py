"""Plan and train the campaign (configs/campaign.yaml). Dry run unless --execute.

Examples:
  python scripts/run_campaign.py                         # plan for all conditions and seeds
  python scripts/run_campaign.py --only M --seeds 0 1 --execute
  python scripts/run_campaign.py --only B1 B2 B3 --execute --runs-per-session 2
"""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.training.campaign import load_campaign, run_campaign
from dmoe.training.config import CAMPAIGN_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--only", nargs="+", metavar="CONDITION", help="conditions to include (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, help="seeds (default: the campaign's list)")
    parser.add_argument("--execute", action="store_true", help="train; without it only the plan is printed")
    parser.add_argument("--runs-per-session", type=int, help="stop after this many (condition, seed) units")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="config override a.b=value")
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    run_campaign(
        load_campaign(args.campaign), args.only, args.seeds, execute=args.execute,
        max_units=args.runs_per_session, output_root=roots["output"].path, overrides=args.overrides,
    )


if __name__ == "__main__":
    main()
