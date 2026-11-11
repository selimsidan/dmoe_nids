"""Experiment configuration: ``configs/base.yaml`` + one condition file + overrides.

A condition file (for example ``configs/method/M_dataset_expert_moe.yaml``)
holds a small header (``condition``, ``display_name``,
``expected_total_parameters``) and an ``overrides`` block that is deep-merged
into the base settings. ``--set a.b=value`` strings are applied last. The
resolved config of one training unit additionally carries the training seed,
the split policy, and the absolute checkpoint and output paths (which are
excluded from every signature).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")
BASE_CONFIG = os.path.join(CONFIG_ROOT, "base.yaml")
CAMPAIGN_CONFIG = os.path.join(CONFIG_ROOT, "campaign.yaml")

MOE_ARCHITECTURES = {
    "moe_dataset_soft",
    "moe_dataset_damex",
    "moe_basic_class_conditional",
    "moe_dataset_class_conditional",
    "moe_dataset_private_encoders_class_conditional",
}
DENSE_ARCHITECTURES = {"matched_dense"}
ARCHITECTURES = MOE_ARCHITECTURES | DENSE_ARCHITECTURES

# Presets applied unless the condition sets the key explicitly.
_ARCHITECTURE_DEFAULTS = {
    "moe_dataset_damex": {
        "training.stage_c.gate_supervision": "damex",
        "training.stage_c.expert_update_policy": "assigned_only",
    },
    "moe_basic_class_conditional": {
        "training.stage_b.warmstart_mode": "generic_pooled",
        "training.stage_c.gate_supervision": "none",
        "training.stage_c.expert_update_policy": "all",
        "training.stage_c.lambda_expert_anchor": 0.0,
    },
}

# Keys that never change what is computed: paths, scheduling and logging.
NON_SCIENTIFIC_TRAINING_KEYS = {
    "checkpoint_dir",
    "device",
    "force_restart",
    "progress_every_rows",
    "run_final_evaluation",
    "save_epoch_history",
    "stages",
    "stage_a_checkpoint",
    "stage_b_checkpoint",
    "private_stage_a_checkpoint",
}
CONDITION_HEADER_KEYS = {
    "condition", "display_name", "role", "description", "expected_total_parameters",
    "parameter_tolerance", "overrides",
}


def load_yaml(path: str) -> dict:
    with open(path) as handle:
        value = yaml.safe_load(handle)
    return value or {}


def deep_merge(base: dict, update: dict) -> dict:
    """Return ``base`` updated recursively with ``update`` (inputs untouched)."""
    result = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        return [_coerce(v.strip()) for v in value[1:-1].split(",") if v.strip()]
    return value


def set_dotted(cfg: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def get_dotted(cfg: dict, dotted_key: str, default=None):
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override {override!r} must look like a.b.c=value")
        key, value = override.split("=", 1)
        set_dotted(cfg, key.strip(), _coerce(value.strip()))
    return cfg


def apply_architecture_defaults(config: dict, explicit_keys: set[str]) -> None:
    for dotted_key, value in _ARCHITECTURE_DEFAULTS.get(config.get("architecture"), {}).items():
        if dotted_key not in explicit_keys:
            set_dotted(config, dotted_key, value)


def _flatten_keys(value: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, item in (value or {}).items():
        dotted = f"{prefix}{key}"
        keys.add(dotted)
        if isinstance(item, dict):
            keys |= _flatten_keys(item, dotted + ".")
    return keys


def load_condition_file(path: str) -> dict:
    condition = load_yaml(path)
    unknown = set(condition) - CONDITION_HEADER_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}; put settings under 'overrides'")
    if "condition" not in condition:
        raise ValueError(f"{path}: missing 'condition'")
    return condition


def load_config(
    condition_path: str,
    overrides: list[str] | None = None,
    base_path: str = BASE_CONFIG,
) -> dict:
    """Resolve base + condition + overrides into one experiment config."""
    condition = load_condition_file(condition_path)
    config = deep_merge(load_yaml(base_path), condition.get("overrides", {}))
    apply_overrides(config, overrides)
    explicit = _flatten_keys(condition.get("overrides", {})) | {
        override.split("=", 1)[0] for override in overrides or []
    }
    apply_architecture_defaults(config, explicit)
    if config.get("architecture") not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture {config.get('architecture')!r}; expected one of {sorted(ARCHITECTURES)}")
    config["condition"] = {
        key: condition[key] for key in condition if key != "overrides"
    }
    return config


def scientific_view(config: dict) -> dict:
    """The part of a config that determines results (paths and scheduling removed)."""
    view = copy.deepcopy(config)
    for key in ("condition", "paths", "run"):
        view.pop(key, None)
    training = view.get("training", {})
    for key in NON_SCIENTIFIC_TRAINING_KEYS:
        training.pop(key, None)
    view.get("evaluation", {}).pop("output_dir", None)
    view.get("data", {}).pop("display_names", None)
    return view


def config_hash(config: dict) -> str:
    payload = json.dumps(scientific_view(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
