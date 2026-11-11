"""Check that the NF-v3 CSVs exist, have the expected columns and row counts, and match their hashes.

Example: python scripts/verify_data.py --data-root /path/to/NIDS_datasets
Exit code 0 when everything checks out, 1 otherwise. See docs/DATA.md.
"""
from __future__ import annotations

import argparse
import os
import sys

from _common import add_root_arguments, apply_roots

from dmoe.data.verify import verify_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="DATASET")
    parser.add_argument("--quick", action="store_true", help="only existence and header (skip counting and hashing)")
    parser.add_argument("--no-hash", action="store_true", help="count rows but skip the content hash")
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    results = verify_data(
        roots["data"].path, args.only, count=not args.quick, hash_files=not (args.quick or args.no_hash),
        hash_cache_dir=os.path.join(roots["output"].path, "cache"),
    )
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
