"""Maintainer tool: count rows and hash the CSVs, then write configs/checksums.json.

Reads every byte of every CSV (tens of GB), so run it once, for example on
Colab with the data on Drive. Prints a comparison with the recorded values
before writing; --write updates the file.
"""
from __future__ import annotations

import argparse
import json
import os

from _common import add_root_arguments, apply_roots

from dmoe import paths
from dmoe.data.registry import DATASET_NAMES
from dmoe.data.source_hash import cached_digests
from dmoe.data.verify import CHECKSUMS_FILE, count_rows, load_checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="DATASET")
    parser.add_argument("--write", action="store_true", help="update configs/checksums.json")
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    checksums = load_checksums()
    for dataset in args.only or list(DATASET_NAMES):
        path = paths.dataset_csv_path(dataset, roots["data"].path)
        rows = count_rows(path)
        digests = cached_digests(path, os.path.join(roots["output"].path, "cache"))
        entry = checksums.setdefault(dataset, {})
        for key, value in (("rows", rows), ("size_bytes", os.path.getsize(path)), ("sha256", digests["sha256"]), ("sha1", digests["sha1"])):
            recorded = entry.get(key)
            flag = "" if recorded in (None, value) else "   <-- differs from recorded value"
            print(f"{dataset} {key}: {value} (recorded: {recorded}){flag}")
            if key != "sha1" or recorded is None:
                entry[key] = value
    if args.write:
        with open(CHECKSUMS_FILE, "w") as handle:
            json.dump(checksums, handle, indent=2)
            handle.write("\n")
        print(f"updated {CHECKSUMS_FILE}")


if __name__ == "__main__":
    main()
