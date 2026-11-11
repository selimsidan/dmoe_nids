"""Upload every run's bundle and the recommended-seed pointer to the Hugging Face Hub (dry run by default)."""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.release import publish_hub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="default: hub.repo_id in configs/base.yaml")
    parser.add_argument("--execute", action="store_true", help="upload (needs HF_TOKEN)")
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    publish_hub(roots["output"].path, repo_id=args.repo_id, execute=args.execute)


if __name__ == "__main__":
    main()
