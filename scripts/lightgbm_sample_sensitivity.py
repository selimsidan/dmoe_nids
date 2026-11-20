"""B4 training-sample sensitivity: does B4 improve with more training rows?

Retrains B4's chosen configuration (tuning seed) on the sample fractions in
``lightgbm.sample_sensitivity.fractions`` (same class-capped, reweighted
sampling as the main 10% sample) and writes
``runs/B4/sample_sensitivity/Sample_Sensitivity.csv`` with the best validation
macro-F1, its round and the test macro-F1 per fraction. The main fraction
reuses the chosen tuning run. Run it after B4's tuning; resumable.

Example (Colab CPU high-RAM runtime, data on Drive):
  python scripts/lightgbm_sample_sensitivity.py
"""
from __future__ import annotations

import argparse

from dmoe.openmp import prepare_for_lightgbm

prepare_for_lightgbm()  # before anything imports torch (see dmoe.openmp)

from _common import add_root_arguments, apply_roots  # noqa: E402

from dmoe.training.campaign import load_campaign, resolve_unit_config  # noqa: E402
from dmoe.training.config import CAMPAIGN_CONFIG  # noqa: E402
from dmoe.training.lightgbm_baseline import run_sample_sensitivity  # noqa: E402
from dmoe.training.out_of_core_data import prepare_out_of_core_data  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--condition", default="B4")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_root_arguments(parser)
    args = parser.parse_args(argv)
    roots = apply_roots(args)
    campaign = load_campaign(args.campaign)
    config = resolve_unit_config(campaign, args.condition, 0, output_root=roots["output"].path,
                                 overrides=args.overrides)
    table = run_sample_sensitivity(config, prepare_out_of_core_data(config), args.condition, roots["output"].path)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
