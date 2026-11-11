"""Re-export the bundle (and re-run evaluation) of a completed run from its checkpoints."""
from __future__ import annotations

import argparse
import time

from _common import add_root_arguments, apply_roots

from dmoe.evaluation.finalize import finalize_unit
from dmoe.training.campaign import load_campaign, resolve_unit_config
from dmoe.training.config import CAMPAIGN_CONFIG
from dmoe.training.out_of_core_data import prepare_out_of_core_data
from dmoe.training.out_of_core_train import ensure_run_contract
from dmoe.training.run_unit import _resolve_device, train_unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN_CONFIG)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    add_root_arguments(parser)
    args = parser.parse_args()
    roots = apply_roots(args)
    config = resolve_unit_config(load_campaign(args.campaign), args.condition, args.seed,
                                 output_root=roots["output"].path, overrides=args.overrides)
    config["training"]["stages"] = []  # load only; never train here
    _resolve_device(config)
    context = prepare_out_of_core_data(config)
    contract = ensure_run_contract(config, context)
    started = time.monotonic()
    model = train_unit(config, context)
    finalize_unit(model, context, config, contract, training_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
