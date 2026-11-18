"""Entry point of B4 (LightGBM) units: ``dmoe.training.run_unit`` with LightGBM loaded first.

The campaign launches B4's tuning and seed units through this module; see
``dmoe.openmp`` for why the load order matters.
"""
from dmoe.openmp import prepare_for_lightgbm

prepare_for_lightgbm()

from dmoe.training.run_unit import main  # noqa: E402

if __name__ == "__main__":
    main()
