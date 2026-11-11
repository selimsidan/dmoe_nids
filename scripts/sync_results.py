"""Copy small, shareable result files from OUTPUT_ROOT into results/ and paper/ (paths made relative).

Never copies data, caches, checkpoints, bundles, per-row predictions or logs.
"""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.release import sync_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true", help="skip PNG/PDF figures of the summary")
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    sync_results(roots["output"].path, figures=not args.no_figures)


if __name__ == "__main__":
    main()
