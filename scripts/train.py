"""Train one condition for one or more seeds (dependencies must already exist).

Example: python scripts/train.py --condition M --seeds 0 1 2 --execute
"""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.training.campaign import load_campaign, run_campaign
from dmoe.training.config import CAMPAIGN_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, help="default: the campaign's seed list")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    run_campaign(load_campaign(args.campaign), [args.condition], args.seeds, execute=args.execute,
                 output_root=roots["output"].path, overrides=args.overrides)


if __name__ == "__main__":
    main()
