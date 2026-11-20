"""Baseline B4: one pooled gradient-boosted tree model (XGBoost, GPU), dataset-blind.

Fairness contract with M (see docs/METHOD.md): all pooled training rows, the
same 47 prepared (median-imputed, standardized) features, no dataset
identity, the same square-root inverse-frequency class weights (as per-row
sample weights, computed by the same function), no resampling, and model
selection by validation macro-F1 (the best boosting round, like M's best
Stage-C epoch).

Two kinds of unit:

* **Tuning** (``runs/B4/tuning/``), once, before the first B4 seed: every
  configuration of the grid trains with the tuning seed; the one with the
  highest validation macro-F1 is written to ``chosen.json`` and used
  unchanged for every seed. A short timing pilot first records seconds per
  round and GPU memory (``pilot.json``).
* **Seed runs** (``runs/B4/seed<k>/``): the chosen configuration.

Every run boosts round by round in one pass and saves its model and progress
every ``checkpoint_rounds`` rounds. A run interrupted by a disconnect resumes
from its last checkpoint (recorded in ``resumed_after_rounds``); row and
column subsampling restarts there, so only a resumed run differs from an
uninterrupted one. Validation macro-F1 is checked every ``eval_every_rounds``
rounds; training stops after ``patience_rounds`` rounds without improvement
and keeps the best round.

Training uses the GPU (``device: cuda``) when one is present; ``device: auto``
falls back to the CPU (used by the tests).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from .. import paths
from ..evaluation.metrics import confusion_from_predictions, overall_from_confusion
from .checkpoint import stage_file
from .out_of_core_data import OutOfCoreContext
from .training_history import append_training_history

ARCHITECTURE = "pooled_xgboost"
STAGES = ["xgboost"]
MODEL_FILE = "xgboost_model.ubj"
STATE_FILE = "progress.json"
PROGRESS_MODEL_FILE = "model_progress.ubj"
TUNING_LOG_FILE = "Tuning_Log.csv"
CHOSEN_FILE = "chosen.json"
PILOT_FILE = "pilot.json"
CHECKS_FILE = "Validation_Checks.csv"
# Set by the code (objective, num_class) or not allowed by the fairness contract.
FORBIDDEN_PARAMS = {
    "objective", "num_class", "scale_pos_weight", "booster", "eval_metric", "max_delta_step",
    "sampling_method", "multi_strategy", "seed", "random_state", "device", "tree_method",
}


def _xgb():
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError("B4 needs XGBoost: pip install xgboost") from exc
    return xgboost


def xgboost_settings(config: dict) -> dict:
    values = copy.deepcopy(config.get("xgboost") or {})
    required = ("params", "grid", "max_rounds", "eval_every_rounds", "patience_rounds", "checkpoint_rounds",
                "tuning", "device")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"xgboost config is missing {missing}")
    for key in ("checkpoint_rounds", "patience_rounds"):
        if int(values[key]) % int(values["eval_every_rounds"]):
            raise ValueError(f"xgboost.{key} must be a multiple of eval_every_rounds")
    forbidden = FORBIDDEN_PARAMS & set(values["params"])
    if forbidden:
        raise ValueError(f"B4 forbids {sorted(forbidden)} (fairness contract; see docs/METHOD.md)")
    if values["device"] not in {"auto", "cuda", "cpu"}:
        raise ValueError("xgboost.device must be auto, cuda or cpu")
    return values


def grid_configs(settings: dict) -> list[dict]:
    """The tuning grid in a fixed order (keys sorted: learning_rate outer, max_leaves inner)."""
    grid = settings["grid"]
    keys = sorted(grid)
    configs = [{}]
    for key in keys:
        configs = [{**config, key: value} for config in configs for value in grid[key]]
    return configs


def _config_id(values: dict) -> str:
    return "_".join(f"{key}-{values[key]}" for key in sorted(values))


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def tuning_dir(condition: str, output_root: str | None = None) -> str:
    return os.path.join(output_root or paths.output_root(), "runs", condition, "tuning")


def tuning_signature(config: dict) -> str:
    """What the tuning result depends on: the B4 settings (not the device) and the data."""
    settings = xgboost_settings(config)
    settings.pop("device", None)
    settings.get("tuning", {}).pop("pilot_rounds", None)
    payload = {"xgboost": settings,
               "data": {k: v for k, v in config.get("data", {}).items() if k not in {"display_names"}}}
    return _hash(payload)


def read_chosen(condition: str, output_root: str | None = None) -> dict | None:
    path = os.path.join(tuning_dir(condition, output_root), CHOSEN_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def tuning_complete(config: dict, condition: str, output_root: str | None = None) -> bool:
    chosen = read_chosen(condition, output_root)
    return bool(chosen) and chosen.get("tuning_config_signature") == tuning_signature(config)


def resolve_device(settings: dict) -> str:
    if settings["device"] != "auto":
        return settings["device"]
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return "cuda" if result.returncode == 0 and "GPU" in result.stdout else "cpu"
    except (OSError, subprocess.SubprocessError):
        return "cpu"


def gpu_memory_used_bytes() -> int | None:
    """Memory in use on GPU 0 according to nvidia-smi (None without a GPU)."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20)
        return int(result.stdout.strip().splitlines()[0]) * 2**20 if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def num_threads() -> int:
    return int(os.cpu_count() or 1)


# --------------------------------------------------------------------------------------
# Data


def _labels(split, chunk_rows: int = 4_000_000) -> np.ndarray:
    labels = np.empty(len(split.class_idx), dtype=np.int32)
    for start in range(0, len(labels), chunk_rows):
        stop = min(start + chunk_rows, len(labels))
        labels[start:stop] = np.asarray(split.class_idx[start:stop], dtype=np.int32)
    return labels


def class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Square-root inverse-frequency class weights, normalized to mean 1 (M's ``_loss_weights``)."""
    from .out_of_core_train import _loss_weights

    return _loss_weights(labels, num_classes, "cpu").numpy().astype(np.float64)


def _materialize(rows, chunk_rows: int = 1_000_000) -> np.ndarray:
    """Standardized float32 rows of a split as one in-memory array, read in fixed windows."""
    out = np.empty(rows.shape, dtype=np.float32)
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        out[start:stop] = rows[start:stop]
    return out


class PreparedMatrices:
    """XGBoost matrices of all training rows and all validation rows.

    The rows are read once into memory, quantized into ``QuantileDMatrix``
    objects (bin boundaries from the training rows; validation uses the same
    ones) and the raw arrays are released.
    """

    def __init__(self, config: dict, context: OutOfCoreContext):
        xgb = _xgb()
        settings = xgboost_settings(config)
        data = context.data
        started = time.monotonic()
        self.feature_columns = list(context.feature_columns)
        self.num_classes = len(data.class_names)
        y = _labels(data.train)
        weights = class_weights(y, self.num_classes)[y]
        self.rows = int(len(y))
        print(f"[B4] loading all {self.rows:,} training rows into memory", flush=True)
        x = _materialize(data.train.features)
        max_bin = int(settings["params"].get("max_bin", 256))
        self.train = xgb.QuantileDMatrix(x, label=y, weight=weights, feature_names=self.feature_columns,
                                         max_bin=max_bin, nthread=num_threads())
        del x, weights
        self.y_val = _labels(data.val)
        x_val = _materialize(data.val.features)
        self.val = xgb.QuantileDMatrix(x_val, label=self.y_val, feature_names=self.feature_columns,
                                       ref=self.train, max_bin=max_bin, nthread=num_threads())
        del x_val
        print(f"[B4] matrices ready in {(time.monotonic() - started) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------------------
# Training


def macro_f1_from_probabilities(probabilities: np.ndarray, truth: np.ndarray, num_classes: int) -> float:
    """Macro-F1 over classes with support (the definition used by every evaluation)."""
    prediction = np.asarray(probabilities).argmax(axis=1)
    confusion = confusion_from_predictions(truth, prediction, num_classes)
    return float(overall_from_confusion(confusion)["macro_f1"])


def training_params(settings: dict, chosen: dict, seed: int, num_classes: int, device: str) -> dict:
    params = dict(settings["params"])
    params.update(chosen)
    params.update({
        "objective": "multi:softprob",
        "num_class": int(num_classes),
        "tree_method": "hist",
        "device": device,
        "seed": 1000 + int(seed),
        "nthread": num_threads(),
        "verbosity": 1,
    })
    return params


def _read_json(path: str):
    with open(path) as handle:
        return json.load(handle)


def _write_json(value, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
    os.replace(path + ".tmp", path)


def _comparable(params: dict) -> dict:
    """Parameters that define a run (not where or with how many threads it runs)."""
    return {k: v for k, v in json.loads(json.dumps(params, default=str)).items() if k not in {"device", "nthread"}}


def train_booster(
    params: dict,
    matrices: "PreparedMatrices",
    *,
    max_rounds: int,
    eval_every: int,
    patience: int,
    checkpoint_rounds: int,
    state_dir: str,
    label: str,
) -> dict:
    """Train (or resume) one early-stopped run in one pass, checkpointing as it goes; return its final state.

    The state directory holds ``model_progress.ubj`` (every round trained so
    far) and ``progress.json`` (rounds done, validation checks, best round).
    """
    xgb = _xgb()
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, STATE_FILE)
    model_path = os.path.join(state_dir, PROGRESS_MODEL_FILE)
    state = _read_json(state_path) if os.path.isfile(state_path) else {
        "rounds_done": 0, "checks": [], "best_round": None, "best_val_macro_f1": None,
        "stopped_early": False, "finished": False, "wall_seconds": 0.0, "params": _comparable(params),
        "resumed_after_rounds": [],
    }
    if state.get("params") != _comparable(params):
        raise ValueError(f"{state_dir} was started with different XGBoost parameters; move it away to retrain")
    if state["finished"]:
        return state
    resume_from = int(state["rounds_done"])
    if resume_from:
        state.setdefault("resumed_after_rounds", []).append(resume_from)
        print(f"[B4 {label}] resuming after round {resume_from} (checkpoint)", flush=True)
    truth = np.asarray(matrices.y_val, dtype=np.int64)
    num_classes = int(params["num_class"])
    started = time.monotonic()
    wall_before = float(state["wall_seconds"])
    booster = xgb.Booster(params, [matrices.train, matrices.val], model_file=model_path if resume_from else None)
    booster.set_param(params)

    def checkpoint(rounds_done: int) -> None:
        state["rounds_done"] = int(rounds_done)
        state["wall_seconds"] = wall_before + time.monotonic() - started
        booster.save_model(model_path + ".tmp.ubj")
        os.replace(model_path + ".tmp.ubj", model_path)
        _write_json(state, state_path)
        pd.DataFrame(state["checks"]).to_csv(os.path.join(state_dir, CHECKS_FILE), index=False)
        print(f"[B4 {label}] checkpoint after round {rounds_done} "
              f"({state['wall_seconds'] / 60:.1f} min so far)", flush=True)

    for iteration in range(resume_from, max_rounds):
        booster.update(matrices.train, iteration)
        round_number = iteration + 1
        if round_number % eval_every:
            continue
        probabilities = np.asarray(booster.predict(matrices.val)).reshape(len(truth), num_classes)
        f1 = macro_f1_from_probabilities(probabilities, truth, num_classes)
        picked = np.clip(probabilities[np.arange(len(truth)), truth], 1e-15, 1.0)
        logloss = float(-np.log(picked).mean())
        del probabilities, picked
        state["checks"].append({"round": round_number, "val_macro_f1": f1, "val_multi_logloss": logloss})
        if state["best_val_macro_f1"] is None or f1 > state["best_val_macro_f1"] + 1e-12:
            state["best_round"], state["best_val_macro_f1"] = round_number, f1
        print(f"[B4 {label}] round {round_number}: val_macro_f1={f1:.6f} best={state['best_val_macro_f1']:.6f} "
              f"(round {state['best_round']}) val_logloss={logloss:.4f}", flush=True)
        if round_number - state["best_round"] >= patience:
            state["stopped_early"] = True
            break
        if round_number % checkpoint_rounds == 0 and round_number < max_rounds:
            checkpoint(round_number)
    state["finished"] = True
    checkpoint(booster.num_boosted_rounds())
    return state


def final_model_bytes(state_dir: str, best_round: int) -> bytes:
    """The model truncated to its best round, in XGBoost's UBJSON format."""
    xgb = _xgb()
    booster = xgb.Booster(model_file=os.path.join(state_dir, PROGRESS_MODEL_FILE))
    return bytes(booster[: int(best_round)].save_raw(raw_format="ubj"))


# --------------------------------------------------------------------------------------
# Model wrapper used by evaluation, resource accounting and bundles


class XGBoostModel:
    """A trained B4 booster behind the interface the evaluation code uses.

    ``forward(x)`` takes only the feature rows; the booster's feature names
    must equal the 47 contract columns, so no dataset identity can enter.
    """

    architecture = ARCHITECTURE
    routing_mode = None
    expert_names: list[str] = []

    def __init__(self, model_bytes: bytes, class_names: list[str], feature_columns: list[str],
                 device: str | None = None):
        xgb = _xgb()
        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(model_bytes))
        self.model_bytes = bytes(model_bytes)
        self.class_names = list(class_names)
        self.feature_columns = list(feature_columns)
        self.device = device or resolve_device({"device": "auto"})
        self.booster.set_param({"device": self.device, "nthread": num_threads(), "verbosity": 0})
        self.threads = num_threads()
        self.training_summary: dict = {}
        self.xgboost_info: dict = {}
        assert_dataset_blind_booster(self.booster, self.feature_columns)

    def eval(self):
        return self

    def to(self, _device):
        return self

    def parameters(self):
        return iter(())

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = self.booster.inplace_predict(np.ascontiguousarray(x, dtype=np.float32))
        return np.asarray(values, dtype=np.float64).reshape(len(x), len(self.class_names))

    def forward(self, x):
        return {"combined_probs": self.predict_proba(x)}

    def best_round(self) -> int:
        return int(self.booster.num_boosted_rounds())

    def num_trees(self) -> int:
        return self.best_round() * len(self.class_names)

    def total_leaves(self) -> int:
        frame = self.booster.trees_to_dataframe()
        return int((frame["Feature"] == "Leaf").sum())


def assert_dataset_blind_booster(booster, feature_columns: list[str]) -> None:
    names = list(booster.feature_names or [])
    if names != list(feature_columns):
        raise AssertionError(
            f"the XGBoost model uses features {names[:5]}... ({len(names)}); a dataset-blind B4 model uses "
            f"exactly the {len(feature_columns)} contract feature columns"
        )


# --------------------------------------------------------------------------------------
# Timing pilot, tuning and seed runs


def measure_cost(config: dict, context: OutOfCoreContext, matrices: PreparedMatrices, *, rounds: int) -> dict:
    """Seconds per round of the largest grid configuration on all rows, projected to max_rounds."""
    xgb = _xgb()
    settings = xgboost_settings(config)
    device = resolve_device(settings)
    configs = grid_configs(settings)
    largest = max(configs, key=lambda c: (c.get("max_leaves", 0), c.get("learning_rate", 0)))
    params = training_params(settings, largest, int(settings["tuning"]["seed"]), matrices.num_classes, device)
    booster = xgb.Booster(params, [matrices.train, matrices.val])
    booster.set_param(params)
    booster.update(matrices.train, 0)  # first round includes one-time GPU setup; timed separately
    started = time.monotonic()
    for iteration in range(1, int(rounds) + 1):
        booster.update(matrices.train, iteration)
    seconds_per_round = (time.monotonic() - started) / int(rounds)
    started = time.monotonic()
    booster.predict(matrices.val)
    validation_check_seconds = time.monotonic() - started
    max_rounds = int(settings["max_rounds"])
    checks = max_rounds // int(settings["eval_every_rounds"])
    run_hours = (max_rounds * seconds_per_round + checks * validation_check_seconds) / 3600
    return {
        "pilot_rounds": int(rounds), "pilot_config": largest, "rows": int(matrices.rows), "device": device,
        "seconds_per_round": seconds_per_round, "validation_check_seconds": validation_check_seconds,
        "projected_hours_per_run_at_max_rounds": run_hours,
        "projected_hours_grid": run_hours * len(configs),
        "gpu_memory_used_bytes": gpu_memory_used_bytes(),
        "note": "worst case: no early stopping",
    }


def run_pilot(config: dict, context: OutOfCoreContext, directory: str, matrices: PreparedMatrices) -> dict | None:
    tuning = xgboost_settings(config)["tuning"]
    if not int(tuning.get("pilot_rounds", 0)):
        return None
    path = os.path.join(directory, PILOT_FILE)
    if os.path.isfile(path):
        pilot = _read_json(path)
    else:
        pilot = measure_cost(config, context, matrices, rounds=int(tuning["pilot_rounds"]))
        _write_json(pilot, path)
    memory = pilot.get("gpu_memory_used_bytes")
    print(f"[B4 tuning] pilot ({pilot['device']}): {pilot['seconds_per_round']:.2f} s per round on "
          f"{pilot['rows']:,} rows; worst case (no early stopping) "
          f"{pilot['projected_hours_per_run_at_max_rounds']:.1f} h per run, {pilot['projected_hours_grid']:.1f} h "
          f"for the grid" + (f"; GPU memory in use {memory / 2**30:.1f} GiB" if memory else ""), flush=True)
    return pilot


def run_tuning(config: dict, context: OutOfCoreContext, condition: str, output_root: str | None = None) -> dict:
    settings = xgboost_settings(config)
    device = resolve_device(settings)
    directory = tuning_dir(condition, output_root)
    os.makedirs(directory, exist_ok=True)
    signature = tuning_signature(config)
    chosen = read_chosen(condition, output_root)
    if chosen and chosen.get("tuning_config_signature") == signature:
        print("[B4 tuning] already complete", flush=True)
        return chosen
    marker = os.path.join(directory, "tuning_signature.json")
    if os.path.isfile(marker) and _read_json(marker).get("signature") != signature:
        raise ValueError(f"{directory} holds tuning for different B4 settings; move it away to retune")
    _write_json({"signature": signature}, marker)
    matrices = PreparedMatrices(config, context)
    run_pilot(config, context, directory, matrices)
    seed = int(settings["tuning"]["seed"])
    log_path = os.path.join(directory, TUNING_LOG_FILE)
    log = pd.read_csv(log_path) if os.path.isfile(log_path) else pd.DataFrame()
    done = set(log["config_id"]) if not log.empty else set()
    for values in grid_configs(settings):
        config_id = _config_id(values)
        if config_id in done:
            continue
        params = training_params(settings, values, seed, matrices.num_classes, device)
        state = train_booster(
            params, matrices, max_rounds=int(settings["max_rounds"]),
            eval_every=int(settings["eval_every_rounds"]), patience=int(settings["patience_rounds"]),
            checkpoint_rounds=int(settings["checkpoint_rounds"]), state_dir=os.path.join(directory, config_id),
            label=f"tuning {config_id}",
        )
        row = {"config_id": config_id, **values, "seed": seed, "train_rows": matrices.rows, "device": device,
               "best_round": state["best_round"], "val_macro_f1": state["best_val_macro_f1"],
               "rounds_trained": state["rounds_done"], "stopped_early": state["stopped_early"],
               "wall_seconds": state["wall_seconds"]}
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        log.to_csv(log_path + ".tmp", index=False)
        os.replace(log_path + ".tmp", log_path)
    order = [_config_id(values) for values in grid_configs(settings)]
    log["grid_order"] = log["config_id"].map(order.index)
    best = log.sort_values(["val_macro_f1", "grid_order"], ascending=[False, True], kind="stable").iloc[0]
    values = {key: (int(best[key]) if key == "max_leaves" else float(best[key])) for key in settings["grid"]}
    max_rounds = int(settings["max_rounds"])
    near_cap = int(best["best_round"]) > max_rounds - int(settings["patience_rounds"])
    chosen = {
        "condition": condition,
        "chosen": values,
        "config_id": str(best["config_id"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "best_round": int(best["best_round"]),
        "tuning_seed": seed,
        "train_rows": int(matrices.rows),
        "seed_max_rounds": int(settings["extended_max_rounds"]) if near_cap else max_rounds,
        "cap_extended": bool(near_cap),
        "rule": "highest validation macro-F1; ties to the earlier grid configuration",
        "tuning_config_signature": signature,
    }
    _write_json(chosen, os.path.join(directory, CHOSEN_FILE))
    print(f"[B4 tuning] chosen {chosen['config_id']} val_macro_f1={chosen['val_macro_f1']:.6f} "
          f"best_round={chosen['best_round']} seed_max_rounds={chosen['seed_max_rounds']}", flush=True)
    return chosen


def train_xgboost_unit(config: dict, context: OutOfCoreContext) -> XGBoostModel:
    """One B4 seed: the chosen configuration on all training rows."""
    settings = xgboost_settings(config)
    device = resolve_device(settings)
    run = config["run"]
    condition, seed = run["condition"], int(run["seed"])
    chosen = read_chosen(condition, run["output_root"])
    if not chosen or chosen.get("tuning_config_signature") != tuning_signature(config):
        raise RuntimeError(f"{condition} tuning is missing or outdated; run the campaign (it tunes {condition} first)")
    checkpoint_dir = config["training"]["checkpoint_dir"]
    final_path = stage_file(checkpoint_dir, "C", MODEL_FILE)
    num_classes = len(context.data.class_names)
    params = training_params(settings, chosen["chosen"], seed, num_classes, device)
    state_dir = os.path.join(checkpoint_dir, "xgboost")
    max_rounds = int(chosen["seed_max_rounds"])
    reusable = os.path.join(tuning_dir(condition, run["output_root"]), chosen["config_id"])
    if (not os.path.isdir(state_dir) and seed == int(chosen["tuning_seed"])
            and max_rounds == int(settings["max_rounds"]) and os.path.isfile(os.path.join(reusable, STATE_FILE))
            and _read_json(os.path.join(reusable, STATE_FILE)).get("finished")):
        # Same parameters, data and seed as the chosen tuning run: identical result, so reuse it.
        shutil.copytree(reusable, state_dir)
        print(f"[B4 seed {seed}] reusing the identical tuning run {chosen['config_id']}", flush=True)
    if not os.path.isfile(final_path):
        state_file = os.path.join(state_dir, STATE_FILE)
        finished = os.path.isfile(state_file) and _read_json(state_file).get("finished")
        if finished:
            state = _read_json(state_file)
        else:
            state = train_booster(
                params, PreparedMatrices(config, context), max_rounds=max_rounds,
                eval_every=int(settings["eval_every_rounds"]), patience=int(settings["patience_rounds"]),
                checkpoint_rounds=int(settings["checkpoint_rounds"]), state_dir=state_dir, label=f"seed {seed}",
            )
        # Training history from the run's validation checks (also when the tuning run was reused).
        best_so_far, best_round = None, None
        for check in state["checks"]:
            if best_so_far is None or check["val_macro_f1"] > best_so_far + 1e-12:
                best_so_far, best_round = check["val_macro_f1"], check["round"]
            append_training_history(config, {
                "stage": "C", "encoder_role": "xgboost", "dataset": "ALL", "epoch": check["round"],
                "objective": "boosting", "rows": int(chosen["train_rows"]),
                "val_macro_f1": check["val_macro_f1"], "best_val_macro_f1": best_so_far,
                "improved": check["round"] == best_round,
                "patience_left": int(settings["patience_rounds"]) - (check["round"] - best_round),
            })
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path + ".tmp", "wb") as handle:
            handle.write(final_model_bytes(state_dir, state["best_round"]))
        os.replace(final_path + ".tmp", final_path)
        summary = {
            "optimizer_steps": 0, "examples_seen": 0, "epochs_completed": int(state["rounds_done"]),
            "wall_seconds": float(state["wall_seconds"]), "selected_epoch": int(state["best_round"]),
        }
        _write_json({"best_epoch": int(state["best_round"]), "best_validation_macro_f1": state["best_val_macro_f1"],
                     "selection_mode": "best_val_round", "training_summary": summary,
                     "xgboost": {"chosen": chosen, "rounds_trained": state["rounds_done"],
                                 "stopped_early": state["stopped_early"],
                                 "resumed_after_rounds": state.get("resumed_after_rounds", []),
                                 "params": params}},
                    os.path.join(checkpoint_dir, "stage_c_training_summary.json"))
    with open(final_path, "rb") as handle:
        model = XGBoostModel(handle.read(), context.data.class_names, context.feature_columns, device=device)
    summary = _read_json(os.path.join(checkpoint_dir, "stage_c_training_summary.json"))
    model.training_summary = {"C": summary["training_summary"]}
    model.xgboost_info = summary.get("xgboost", {})
    return model


def xgboost_version() -> str | None:
    try:
        import xgboost

        return xgboost.__version__
    except ImportError:
        return None
