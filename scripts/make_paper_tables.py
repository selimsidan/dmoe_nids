"""Write the paper tables and figures (LaTeX, CSV, PDF) into OUTPUT_ROOT/paper/."""
from __future__ import annotations

import argparse

from _common import add_root_arguments, apply_roots

from dmoe.evaluation.paper_tables import make_paper_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    print(f"[paper] written to {make_paper_tables(roots['output'].path)}")


if __name__ == "__main__":
    main()
