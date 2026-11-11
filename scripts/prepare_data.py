"""Build the fixed train/validation/test split once and write its description (split.json)."""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.data.prepare import prepare_split
from dmoe.training.config import CAMPAIGN_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_root_arguments(parser)
    args = parser.parse_args()
    apply_roots(args)
    prepare_split(args.campaign, args.overrides)


if __name__ == "__main__":
    main()
