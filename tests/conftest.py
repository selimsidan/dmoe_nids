"""Shared fixtures: synthetic data roots and one tiny trained campaign per session."""
from __future__ import annotations

import os
import sys

from dmoe.openmp import prepare_for_lightgbm  # noqa: E402

# The tests train every condition, B4 included, in this one process; the campaign uses one
# process per unit. LightGBM is loaded before torch and torch keeps one thread (dmoe.openmp).
prepare_for_lightgbm()

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

from synthetic import FAST_OVERRIDES, make_data_root  # noqa: E402

from dmoe import paths  # noqa: E402

CONDITIONS = ("M", "B2", "B3", "B4", "A1", "A2", "A3")


def set_roots(monkeypatch, base: str) -> dict[str, str]:
    roots = {
        "data": os.path.join(base, "data"),
        "output": os.path.join(base, "outputs"),
        "scratch": os.path.join(base, "scratch"),
    }
    for kind, path in roots.items():
        monkeypatch.setenv(paths.ENV_VARS[kind], path)
    return roots


def train_unit(condition: str, seed: int, roots: dict[str, str], extra: list[str] | None = None,
               tune: bool = False) -> None:
    from dmoe.training import run_unit

    argv = ["--condition", condition, "--seed", str(seed), *(["--tune"] if tune else []),
            "--data-root", roots["data"], "--output-root", roots["output"], "--scratch-root", roots["scratch"]]
    for override in [*FAST_OVERRIDES, *(extra or [])]:
        argv += ["--set", override]
    run_unit.main(argv)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Seed 0 of all seven conditions (B4 after its tuning) plus seed 1 of M, trained once on synthetic data."""
    base = str(tmp_path_factory.mktemp("campaign"))
    mp = pytest.MonkeyPatch()
    roots = set_roots(mp, base)
    make_data_root(roots["data"])
    for condition in CONDITIONS:
        if condition == "B4":
            train_unit("B4", 0, roots, tune=True)
        train_unit(condition, 0, roots)
    train_unit("M", 1, roots)
    yield roots
    mp.undo()
