"""Download the datasets that have a direct URL in configs/sources.yaml; print manual steps for the rest."""
from __future__ import annotations

import argparse
import os

from _common import add_root_arguments, apply_roots

from dmoe.data.download import download_all
from dmoe.data.verify import verify_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="DATASET")
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    status = download_all(roots["data"].path, args.only)
    downloaded = [name for name, value in status.items() if value == "downloaded"]
    if downloaded:
        verify_data(roots["data"].path, downloaded, hash_cache_dir=os.path.join(roots["output"].path, "cache"))


if __name__ == "__main__":
    main()
