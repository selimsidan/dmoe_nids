"""Resolution of the three storage roots used by every entry point.

Each root is resolved, in order, from an explicit value (a CLI flag or a
notebook setting), then an environment variable, then a generic default
relative to the current working directory:

==============  ====================  ==================  ============
Root            CLI flag              Environment         Default
==============  ====================  ==================  ============
DATA_ROOT       ``--data-root``       ``NIDS_DATA_DIR``   ``./data``
OUTPUT_ROOT     ``--output-root``     ``NIDS_OUTPUT_DIR`` ``./outputs``
SCRATCH_ROOT    ``--scratch-root``    ``NIDS_SCRATCH_DIR`` ``./scratch``
==============  ====================  ==================  ============

Nothing is resolved at import time, so tests and notebooks can change the
environment before the first call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

ENV_VARS = {
    "data": "NIDS_DATA_DIR",
    "output": "NIDS_OUTPUT_DIR",
    "scratch": "NIDS_SCRATCH_DIR",
}
DEFAULTS = {
    "data": "data",
    "output": "outputs",
    "scratch": "scratch",
}


@dataclass(frozen=True)
class ResolvedRoot:
    path: str
    source: str  # "explicit" | "env" | "default"


def resolve_root(kind: str, explicit: str | None = None) -> ResolvedRoot:
    """Resolve one root: explicit value, then environment variable, then default."""
    if kind not in ENV_VARS:
        raise ValueError(f"unknown root kind {kind!r}; expected one of {sorted(ENV_VARS)}")
    if explicit:
        return ResolvedRoot(os.path.abspath(os.path.expanduser(explicit)), "explicit")
    from_env = os.environ.get(ENV_VARS[kind])
    if from_env:
        return ResolvedRoot(os.path.abspath(os.path.expanduser(from_env)), "env")
    return ResolvedRoot(os.path.abspath(DEFAULTS[kind]), "default")


def data_root(explicit: str | None = None) -> str:
    return resolve_root("data", explicit).path


def output_root(explicit: str | None = None) -> str:
    return resolve_root("output", explicit).path


def scratch_root(explicit: str | None = None) -> str:
    return resolve_root("scratch", explicit).path


def apply_cli_roots(data: str | None = None, output: str | None = None, scratch: str | None = None) -> dict[str, ResolvedRoot]:
    """Resolve the roots and export them so child processes see the same values."""
    resolved = {
        "data": resolve_root("data", data),
        "output": resolve_root("output", output),
        "scratch": resolve_root("scratch", scratch),
    }
    for kind, root in resolved.items():
        os.environ[ENV_VARS[kind]] = root.path
    return resolved


def dataset_csv_path(dataset: str, root: str | None = None) -> str:
    """Expected location of one dataset: ``DATA_ROOT/<dataset>/data/<dataset>.csv``."""
    return os.path.join(root or data_root(), dataset, "data", f"{dataset}.csv")


# Layout below OUTPUT_ROOT. Every generated artifact lives here, never in the repo.
def split_cache_dir(root: str | None = None) -> str:
    return os.path.join(root or output_root(), "cache", "splits")


def preprocessor_cache_dir(root: str | None = None) -> str:
    return os.path.join(root or output_root(), "cache", "preprocessors")


def run_dir(condition: str, seed: int, root: str | None = None) -> str:
    return os.path.join(root or output_root(), "runs", condition, f"seed{int(seed)}")


def summary_dir(root: str | None = None) -> str:
    return os.path.join(root or output_root(), "summary")


def paper_dir(root: str | None = None) -> str:
    return os.path.join(root or output_root(), "paper")


def relative_to_root(path: str, root: str) -> str:
    """Store paths relative to a root in manifests so they never leak a machine layout."""
    absolute = os.path.abspath(path)
    base = os.path.abspath(root)
    if os.path.commonpath([absolute, base]) != base:
        raise ValueError(f"{path!r} is not inside root {root!r}")
    return os.path.relpath(absolute, base).replace(os.sep, "/")
