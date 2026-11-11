"""The experiment campaign: which conditions and seeds to train, in what order.

``configs/campaign.yaml`` lists the conditions, the default seeds, the split
policy and the stage dependencies (``depends_on``). One *unit* is one
(condition, seed) pair. Units are ordered seed-major and, within a seed, in
the campaign's condition order, so M of a seed always precedes the baselines
and ablations that reuse its Stage A or Stage B.

A unit is complete when ``runs/<condition>/seed<k>/manifest.json`` says so and
its recorded config hash matches the current config. Completed units are
skipped; interrupted units resume from their last epoch checkpoint.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

from .. import paths
from .checkpoint import BASELINE_MODEL_FILE, STAGE_A_FILE, STAGE_B_FILE, STAGE_C_FILE, stage_file
from .config import CAMPAIGN_CONFIG, CONFIG_ROOT, config_hash, load_config, load_yaml

MANIFEST_FILE = "manifest.json"
LOG_FILE = "log.txt"
STAGE_FILES = {"A": STAGE_A_FILE, "B": STAGE_B_FILE}
SPLIT_MODES = {"fixed", "per_seed"}


@dataclass(frozen=True)
class ConditionEntry:
    id: str
    config: str
    depends_on: dict[str, str] = field(default_factory=dict)  # {"stage_a": "M", "stage_b": "M"}

    @property
    def config_path(self) -> str:
        return self.config if os.path.isabs(self.config) else os.path.join(CONFIG_ROOT, self.config)


@dataclass
class Campaign:
    path: str
    seeds: list[int]
    split_mode: str
    split_seed: int
    conditions: list[ConditionEntry]
    recommended_seed_rule: str
    aggregation_seed: int
    bootstrap_resamples: int

    @property
    def condition_ids(self) -> list[str]:
        return [entry.id for entry in self.conditions]

    def entry(self, condition: str) -> ConditionEntry:
        for entry in self.conditions:
            if entry.id == condition:
                return entry
        raise KeyError(f"unknown condition {condition!r}; campaign has {self.condition_ids}")

    def split_seed_for(self, seed: int) -> int:
        return int(self.split_seed) if self.split_mode == "fixed" else int(seed)


def load_campaign(path: str = CAMPAIGN_CONFIG) -> Campaign:
    raw = load_yaml(path)
    split = raw.get("split", {}) or {}
    mode = split.get("mode", "fixed")
    if mode not in SPLIT_MODES:
        raise ValueError(f"campaign split.mode must be one of {sorted(SPLIT_MODES)}")
    seeds = [int(seed) for seed in raw.get("seeds", [])]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("campaign seeds must be a non-empty list of distinct non-negative integers")
    conditions = []
    seen: set[str] = set()
    for item in raw.get("conditions", []):
        entry = ConditionEntry(
            id=str(item["id"]),
            config=str(item["config"]),
            depends_on={str(k): str(v) for k, v in (item.get("depends_on") or {}).items()},
        )
        unknown = set(entry.depends_on) - {"stage_a", "stage_b"}
        if unknown:
            raise ValueError(f"condition {entry.id}: unknown depends_on keys {sorted(unknown)}")
        for parent in entry.depends_on.values():
            if parent not in seen:
                raise ValueError(f"condition {entry.id} depends on {parent!r}, which must be listed before it")
        seen.add(entry.id)
        conditions.append(entry)
    return Campaign(
        path=os.path.abspath(path),
        seeds=seeds,
        split_mode=mode,
        split_seed=int(split.get("split_seed", 0)),
        conditions=conditions,
        recommended_seed_rule=str(raw.get("recommended_seed_rule", "max_val_macro_f1")),
        aggregation_seed=int(raw.get("aggregation_seed", 0)),
        bootstrap_resamples=int(raw.get("bootstrap_resamples", 1000)),
    )


def run_dir(condition: str, seed: int, output_root: str | None = None) -> str:
    return paths.run_dir(condition, seed, output_root)


def checkpoint_dir(condition: str, seed: int, output_root: str | None = None) -> str:
    return os.path.join(run_dir(condition, seed, output_root), "checkpoints")


def stages_for(entry: ConditionEntry) -> list[str]:
    stages = ["A", "B", "C"]
    if "stage_a" in entry.depends_on:
        stages.remove("A")
    if "stage_b" in entry.depends_on:
        stages.remove("B")
    return stages


def resolve_unit_config(
    campaign: Campaign,
    condition: str,
    seed: int,
    *,
    output_root: str | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Full config of one (condition, seed) unit, including its paths."""
    entry = campaign.entry(condition)
    config = load_config(entry.config_path, overrides)
    root = output_root or paths.output_root()
    config["seed"] = int(seed)
    config.setdefault("data", {})["split_seed"] = campaign.split_seed_for(seed)
    config["data"]["split_mode"] = campaign.split_mode
    training = config.setdefault("training", {})
    training["checkpoint_dir"] = checkpoint_dir(condition, seed, root)
    training["stages"] = stages_for(entry)
    training["save_epoch_history"] = True
    for key, parent in entry.depends_on.items():
        stage = "A" if key == "stage_a" else "B"
        training[f"{key}_checkpoint"] = stage_file(checkpoint_dir(parent, seed, root), stage, STAGE_FILES[stage])
    if "stage_a" in entry.depends_on and config["architecture"] == "moe_dataset_private_encoders_class_conditional":
        # A3: M's Stage-A encoder initializes both the gate encoder and every private encoder.
        training["private_stage_a_checkpoint"] = training["stage_a_checkpoint"]
    config.setdefault("evaluation", {})["output_dir"] = os.path.join(run_dir(condition, seed, root), "eval")
    config["run"] = {
        "condition": condition,
        "seed": int(seed),
        "run_dir": run_dir(condition, seed, root),
        "output_root": root,
        "depends_on": dict(entry.depends_on),
    }
    return config


def read_manifest(condition: str, seed: int, output_root: str | None = None) -> dict | None:
    path = os.path.join(run_dir(condition, seed, output_root), MANIFEST_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def unit_complete(
    campaign: Campaign, condition: str, seed: int, output_root: str | None = None,
    overrides: list[str] | None = None,
) -> bool:
    manifest = read_manifest(condition, seed, output_root)
    if not manifest or manifest.get("status") != "complete":
        return False
    expected = config_hash(resolve_unit_config(campaign, condition, seed, output_root=output_root, overrides=overrides))
    return manifest.get("config_hash") == expected


def _final_checkpoint(config: dict) -> str:
    name = BASELINE_MODEL_FILE if config["architecture"] == "matched_dense" else STAGE_C_FILE
    return stage_file(config["training"]["checkpoint_dir"], "C", name)


def missing_dependencies(campaign: Campaign, condition: str, seed: int, output_root: str | None = None) -> list[str]:
    """Human-readable list of parent stages that do not exist yet."""
    entry = campaign.entry(condition)
    missing = []
    for key, parent in entry.depends_on.items():
        stage = "A" if key == "stage_a" else "B"
        path = stage_file(checkpoint_dir(parent, seed, output_root), stage, STAGE_FILES[stage])
        if not os.path.isfile(path):
            missing.append(f"{parent} seed {seed} Stage {stage}")
    return missing


@dataclass
class Unit:
    condition: str
    seed: int
    status: str  # complete | in_progress | pending | blocked
    stages: list[str]
    waits_for: list[str]  # parents scheduled earlier in this plan
    missing: list[str]  # parents neither present nor scheduled

    def describe(self) -> str:
        text = f"{self.condition} seed {self.seed}: {self.status}"
        if self.status != "complete":
            text += f" (trains stages {','.join(self.stages)})"
        if self.waits_for:
            text += f"; after {', '.join(self.waits_for)} in this plan"
        if self.missing:
            text += f"; needs {', '.join(self.missing)}: run that first"
        return text


def plan_units(
    campaign: Campaign,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    output_root: str | None = None,
    overrides: list[str] | None = None,
) -> list[Unit]:
    selected = list(conditions or campaign.condition_ids)
    for condition in selected:
        campaign.entry(condition)
    seed_list = [int(seed) for seed in (seeds if seeds is not None else campaign.seeds)]
    order = [entry.id for entry in campaign.conditions if entry.id in selected]
    units: list[Unit] = []
    scheduled: set[tuple[str, int]] = set()
    for seed in seed_list:
        for condition in order:
            entry = campaign.entry(condition)
            if unit_complete(campaign, condition, seed, output_root, overrides):
                units.append(Unit(condition, seed, "complete", stages_for(entry), [], []))
                continue
            waits, missing = [], []
            for text in missing_dependencies(campaign, condition, seed, output_root):
                parent = text.split(" seed ")[0]
                (waits if (parent, seed) in scheduled else missing).append(text)
            started = os.path.isdir(checkpoint_dir(condition, seed, output_root))
            status = "blocked" if missing else ("in_progress" if started else "pending")
            units.append(Unit(condition, seed, status, stages_for(entry), waits, missing))
            if status != "blocked":
                scheduled.add((condition, seed))
    return units


def print_plan(units: list[Unit]) -> None:
    done = sum(unit.status == "complete" for unit in units)
    print(f"[plan] {len(units)} unit(s): {done} complete, {len(units) - done} to do")
    for unit in units:
        print(f"[plan]   {unit.describe()}")


def run_campaign(
    campaign: Campaign,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    *,
    execute: bool = False,
    max_units: int | None = None,
    output_root: str | None = None,
    overrides: list[str] | None = None,
    heartbeat_seconds: float = 60.0,
) -> list[Unit]:
    """Print the plan; with ``execute`` train pending units in order, each in its own process."""
    from .logging_utils import run_streaming_logged

    units = plan_units(campaign, conditions, seeds, output_root, overrides)
    print_plan(units)
    if not execute:
        print("[plan] dry run only; pass --execute (or EXECUTE=True) to train")
        return units
    ran = 0
    for unit in units:
        if unit.status == "complete":
            continue
        if max_units is not None and ran >= max_units:
            print(f"[campaign] reached the per-session limit of {max_units} unit(s); rerun to continue")
            break
        still_missing = missing_dependencies(campaign, unit.condition, unit.seed, output_root)
        if still_missing:
            print(f"[campaign] skipping {unit.condition} seed {unit.seed}: needs {', '.join(still_missing)}")
            continue
        command = [
            sys.executable, "-u", "-m", "dmoe.training.run_unit",
            "--campaign", campaign.path,
            "--condition", unit.condition,
            "--seed", str(unit.seed),
        ]
        if output_root:
            command += ["--output-root", output_root]
        for override in overrides or []:
            command += ["--set", override]
        environment = {
            "PYTHONHASHSEED": str(unit.seed),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        }
        log_path = os.path.join(run_dir(unit.condition, unit.seed, output_root), LOG_FILE)
        run_streaming_logged(
            command, log_path, env=environment,
            heartbeat_seconds=heartbeat_seconds,
            label=f"{unit.condition} seed {unit.seed}",
        )
        ran += 1
    return plan_units(campaign, conditions, seeds, output_root, overrides)
