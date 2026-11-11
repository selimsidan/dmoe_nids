"""Shared argument handling for the command-line scripts (no logic of its own)."""
from __future__ import annotations

import argparse

from dmoe import paths


def add_root_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("storage roots (flag, then environment variable, then default)")
    group.add_argument("--data-root", help="DATA_ROOT (env NIDS_DATA_DIR, default ./data)")
    group.add_argument("--output-root", help="OUTPUT_ROOT (env NIDS_OUTPUT_DIR, default ./outputs)")
    group.add_argument("--scratch-root", help="SCRATCH_ROOT (env NIDS_SCRATCH_DIR, default ./scratch)")


def apply_roots(args: argparse.Namespace) -> dict:
    roots = paths.apply_cli_roots(args.data_root, args.output_root, args.scratch_root)
    for kind, root in roots.items():
        print(f"[roots] {kind}={root.path} ({root.source})")
    return roots
