"""Baseline B4: one pooled LightGBM model, trained on CPU, dataset-blind.

Fairness contract with M (see docs/METHOD.md): the same pooled training rows,
the same 47 prepared (median-imputed, standardized) features, no dataset
identity, the same square-root inverse-frequency class weights (as per-row
sample weights, computed by the same function), no resampling, and model
selection by validation macro-F1 (the best boosting round, like M's best
Stage-C epoch).

Two kinds of unit:

* **Tuning** (``runs/B4/tuning/``), once, before the first B4 seed: every
  configuration of the grid trains with the tuning seed; the one with the
  highest validation macro-F1 is written to ``chosen.json`` and used
  unchanged for every seed. With ``tuning.data_mode: auto`` a short timing
  pilot decides whether the grid runs on all training rows or, when the
  projected grid time exceeds ``tuning.max_projected_hours``, on a fixed
  uniform sample. The decision is recorded.
* **Seed runs** (``runs/B4/seed<k>/``): the chosen configuration on all
  training rows.

Every run trains in fixed chunks of ``chunk_rounds`` rounds, each continued
from the saved model, so a run interrupted by a disconnect resumes at the
same chunk boundary and follows exactly the path of an uninterrupted run.
Validation macro-F1 is checked every ``eval_every_rounds`` rounds; training
stops after ``patience_rounds`` rounds without improvement and keeps the
best round.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from .. import paths
from ..evaluation.metrics import confusion_from_predictions
from .checkpoint import stage_file
from .out_of_core_data import OutOfCoreContext
from .training_history import append_training_history

MODEL_FILE = "lightgbm_model.txt"
STATE_FILE = "progress.json"
PROGRESS_MODEL_FILE = "model_progress.txt"
TUNING_LOG_FILE = "Tuning_Log.csv"
CHOSEN_FILE = "chosen.json"
DATA_MODE_FILE = "data_mode.json"
CHECKS_FILE = "Validation_Checks.csv"


def _lgb():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError("B4 needs LightGBM: pip install lightgbm") from exc
    return lightgbm


def lightgbm_settings(config: dict) -> dict:
    values = copy.deepcopy(config.get("lightgbm") or {})
    required = ("params", "grid", "max_rounds", "eval_every_rounds", "patience_rounds", "chunk_rounds", "tuning")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"lightgbm config is missing {missing}")
    if int(values["chunk_rounds"]) % int(values["eval_every_rounds"]):
        raise ValueError("lightgbm.chunk_rounds must be a multiple of eval_every_rounds")
    if int(values["patience_rounds"]) % int(values["eval_every_rounds"]):
        raise ValueError("lightgbm.patience_rounds must be a multiple of eval_every_rounds")
    forbidden = {"is_unbalance", "class_weight", "scale_pos_weight", "boosting", "boosting_type"} & set(values["params"])
    if forbidden:
        raise ValueError(f"B4 forbids {sorted(forbidden)} (fairness contract; see docs/METHOD.md)")
    return values


def grid_configs(settings: dict) -> list[dict]:
    """The tuning grid in a fixed order (keys sorted: learning_rate outer, num_leaves inner)."""
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


def tuning_signature(config: dict, context_signatures: dict | None = None) -> str:
    """What the tuning result depends on: the B4 settings, the data and the split."""
    settings = lightgbm_settings(config)
    payload = {
        "lightgbm": settings,
        "data": {k: v for k, v in config.get("data", {}).items() if k not in {"display_names"}},
    }
    if context_signatures is not None:
        payload["split_signatures"] = context_signatures
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


# --------------------------------------------------------------------------------------
# Data


def _labels(split, indices: np.ndarray | None = None, chunk_rows: int = 4_000_000) -> np.ndarray:
    labels = np.empty(len(split.class_idx), dtype=np.int32)
    for start in range(0, len(labels), chunk_rows):
        stop = min(start + chunk_rows, len(labels))
        labels[start:stop] = np.asarray(split.class_idx[start:stop], dtype=np.int32)
    return labels if indices is None else labels[indices]


def class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Square-root inverse-frequency class weights, normalized to mean 1 (M's ``_loss_weights``)."""
    from .out_of_core_train import _loss_weights

    return _loss_weights(labels, num_classes, "cpu").numpy().astype(np.float64)


def sample_indices(total_rows: int, fraction: float, seed: int) -> np.ndarray:
    """A fixed uniform sample of row indices (natural class distribution), sorted."""
    size = int(round(total_rows * float(fraction)))
    if not 0 < size <= total_rows:
        raise ValueError(f"sample fraction {fraction} gives {size} of {total_rows} rows")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=size, replace=False))


def _dataset_params(settings: dict, num_threads: int) -> dict:
    params = settings["params"]
    return {
        "max_bin": int(params.get("max_bin", 255)),
        "feature_pre_filter": bool(params.get("feature_pre_filter", False)),
        "verbose": -1,
        "num_threads": int(num_threads),
    }


def num_threads() -> int:
    return int(os.cpu_count() or 1)


def _materialize(rows, indices: np.ndarray | None = None, chunk_rows: int = 1_000_000) -> np.ndarray:
    """Standardized float32 rows (all, or the sorted ``indices``) as one in-memory array."""
    count = len(rows) if indices is None else len(indices)
    out = np.empty((count, rows.shape[1]), dtype=np.float32)
    for start in range(0, count, chunk_rows):
        stop = min(start + chunk_rows, count)
        if indices is None:
            out[start:stop] = rows[start:stop]
        else:
            wanted = indices[start:stop]
            block = rows[int(wanted[0]): int(wanted[-1]) + 1]
            out[start:stop] = block[wanted - wanted[0]]
    return out


class PreparedArrays:
    """In-memory training and validation arrays; ``datasets()`` gives fresh ``lgb.Dataset`` objects.

    LightGBM continues a chunk from a saved model by predicting the kept raw
    rows, so the rows stay in memory (``free_raw_data=False``). Each run gets
    its own Dataset objects; bin boundaries are deterministic.
    """

    def __init__(self, config: dict, context: OutOfCoreContext, mode: str):
        settings = lightgbm_settings(config)
        data = context.data
        indices = None
        if mode == "sample":
            indices = sample_indices(len(data.train.class_idx), settings["tuning"]["sample_fraction"],
                                     int(context.split_seed))
        elif mode != "full":
            raise ValueError(f"unknown B4 data mode {mode!r}")
        started = time.monotonic()
        self.mode = mode
        self.feature_columns = list(context.feature_columns)
        self.dataset_params = _dataset_params(settings, num_threads())
        self.y = _labels(data.train, indices)
        self.weights = class_weights(self.y, len(data.class_names))[self.y]
        print(f"[B4] loading {len(self.y):,} training rows ({mode}) into memory", flush=True)
        self.x = _materialize(data.train.features, indices)
        self.y_val = _labels(data.val)
        self.x_val = _materialize(data.val.features)
        self.rows = len(self.y)
        print(f"[B4] arrays ready in {(time.monotonic() - started) / 60:.1f} min "
              f"(train {self.x.nbytes / 2**30:.1f} GiB, val {self.x_val.nbytes / 2**30:.1f} GiB)", flush=True)

    def datasets(self):
        lgb = _lgb()
        train = lgb.Dataset(self.x, label=self.y, weight=self.weights, feature_name=self.feature_columns,
                            params=dict(self.dataset_params), free_raw_data=False)
        val = lgb.Dataset(self.x_val, label=self.y_val, reference=train, feature_name=self.feature_columns,
                          params=dict(self.dataset_params), free_raw_data=False)
        return train, val


# --------------------------------------------------------------------------------------
# Training


def macro_f1_from_probabilities(probabilities: np.ndarray, truth: np.ndarray, num_classes: int) -> float:
    """Macro-F1 over classes with support (the definition used by every evaluation)."""
    prediction = np.asarray(probabilities).argmax(axis=1)
    confusion = confusion_from_predictions(truth, prediction, num_classes).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes), where=(precision + recall) != 0)
    return float(f1[support > 0].mean())


def training_params(settings: dict, chosen: dict, seed: int, num_classes: int) -> dict:
    params = dict(settings["params"])
    params.update(chosen)
    params.update({
        "num_class": int(num_classes),
        "metric": "None",
        "verbose": -1,
        "num_threads": num_threads(),
        "bagging_seed": 1000 + int(seed),
        "feature_fraction_seed": 2000 + int(seed),
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


def train_booster(
    params: dict,
    arrays: "PreparedArrays",
    *,
    max_rounds: int,
    eval_every: int,
    patience: int,
    chunk_rounds: int,
    state_dir: str,
    label: str,
    on_check=None,
) -> dict:
    """Train (or resume) one chunked, early-stopped run; return its final state.

    The state directory holds ``model_progress.txt`` (every round trained so
    far) and ``progress.json`` (rounds done, validation checks, best round).
    """
    lgb = _lgb()
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, STATE_FILE)
    model_path = os.path.join(state_dir, PROGRESS_MODEL_FILE)
    state = _read_json(state_path) if os.path.isfile(state_path) else {
        "rounds_done": 0, "checks": [], "best_round": None, "best_val_macro_f1": None,
        "stopped_early": False, "finished": False, "wall_seconds": 0.0, "params": params,
    }
    if state.get("params") != json.loads(json.dumps(params, default=str)):
        raise ValueError(f"{state_dir} was started with different LightGBM parameters; move it away to retrain")
    if state["rounds_done"]:
        print(f"[B4 {label}] resuming after round {state['rounds_done']}", flush=True)
    truth = np.asarray(arrays.y_val, dtype=np.int64)
    num_classes = int(params["num_class"])
    current = {"round": 0}
    train = val = None

    def feval(preds, _data):
        round_number = current["round"]
        if round_number % eval_every:
            return [("macro_f1", float("nan"), True)]
        probabilities = np.asarray(preds).reshape(len(truth), num_classes)
        f1 = macro_f1_from_probabilities(probabilities, truth, num_classes)
        picked = np.clip(probabilities[np.arange(len(truth)), truth], 1e-15, 1.0)
        current["logloss"] = float(-np.log(picked).mean())
        return [("macro_f1", f1, True)]

    def before(env):
        current["round"] = env.iteration + 1

    before.before_iteration = True

    def after(env):
        round_number = env.iteration + 1
        if round_number % eval_every:
            return
        f1 = next(value for _, name, value, _ in env.evaluation_result_list if name == "macro_f1")
        check = {"round": round_number, "val_macro_f1": f1, "val_multi_logloss": current.get("logloss")}
        state["checks"].append(check)
        improved = state["best_val_macro_f1"] is None or f1 > state["best_val_macro_f1"] + 1e-12
        if improved:
            state["best_round"], state["best_val_macro_f1"] = round_number, f1
        print(f"[B4 {label}] round {round_number}: val_macro_f1={f1:.6f} best={state['best_val_macro_f1']:.6f} "
              f"(round {state['best_round']})", flush=True)
        if on_check is not None:
            on_check(check, state)
        if round_number - state["best_round"] >= patience:
            state["stopped_early"] = True
            raise lgb.callback.EarlyStopException(state["best_round"] - 1, [])

    while not state["finished"]:
        rounds = min(chunk_rounds, max_rounds - state["rounds_done"])
        if train is None:
            train, val = arrays.datasets()
        started = time.monotonic()
        checks_before = len(state["checks"])
        try:
            booster = lgb.train(
                params, train, num_boost_round=rounds,
                init_model=model_path if state["rounds_done"] else None,
                valid_sets=[val], valid_names=["val"], feval=feval,
                callbacks=[before, after], keep_training_booster=True,
            )
        except Exception:
            del state["checks"][checks_before:]
            raise
        state["rounds_done"] = int(booster.current_iteration())
        state["wall_seconds"] += time.monotonic() - started
        if state["stopped_early"] or state["rounds_done"] >= max_rounds:
            state["finished"] = True
        booster.save_model(model_path + ".tmp", num_iteration=-1)
        os.replace(model_path + ".tmp", model_path)
        _write_json(state, state_path)
        pd.DataFrame(state["checks"]).to_csv(os.path.join(state_dir, CHECKS_FILE), index=False)
        print(f"[B4 {label}] saved after round {state['rounds_done']} "
              f"({state['wall_seconds'] / 60:.1f} min so far)", flush=True)
    return state


def final_model_text(state_dir: str, best_round: int) -> str:
    lgb = _lgb()
    booster = lgb.Booster(model_file=os.path.join(state_dir, PROGRESS_MODEL_FILE))
    return booster.model_to_string(num_iteration=int(best_round))


# --------------------------------------------------------------------------------------
# Model wrapper used by evaluation, resource accounting and bundles


class LightGBMModel:
    """A trained B4 booster behind the interface the evaluation code uses.

    ``forward(x)`` takes only the feature rows; the booster's feature names
    must equal the 47 contract columns, so no dataset identity can enter.
    """

    architecture = "pooled_lightgbm"
    routing_mode = None
    expert_names: list[str] = []

    def __init__(self, model_text: str, class_names: list[str], feature_columns: list[str], threads: int | None = None):
        lgb = _lgb()
        self.booster = lgb.Booster(model_str=model_text)
        self.model_text = model_text
        self.class_names = list(class_names)
        self.feature_columns = list(feature_columns)
        self.threads = int(threads or num_threads())
        self.training_summary: dict = {}
        assert_dataset_blind_booster(self.booster, self.feature_columns)

    def eval(self):
        return self

    def to(self, _device):
        return self

    def parameters(self):
        return iter(())

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = self.booster.predict(np.asarray(x, dtype=np.float32), num_threads=self.threads)
        return np.asarray(values, dtype=np.float64).reshape(len(x), len(self.class_names))

    def forward(self, x):
        return {"combined_probs": self.predict_proba(x)}

    def num_trees(self) -> int:
        return int(self.booster.num_trees())

    def best_round(self) -> int:
        return int(self.booster.current_iteration())

    def total_leaves(self) -> int:
        return sum(int(line.split("=", 1)[1]) for line in self.model_text.splitlines() if line.startswith("num_leaves="))


def assert_dataset_blind_booster(booster, feature_columns: list[str]) -> None:
    names = list(booster.feature_name())
    if names != list(feature_columns):
        raise AssertionError(
            f"the LightGBM model uses features {names[:5]}... ({len(names)}); a dataset-blind B4 model uses "
            f"exactly the {len(feature_columns)} contract feature columns"
        )


# --------------------------------------------------------------------------------------
# Cost measurement, data mode decision, tuning and seed runs


def measure_cost(config: dict, context: OutOfCoreContext, arrays: PreparedArrays, *, rounds: int) -> dict:
    """Seconds per round (with validation) of the largest grid configuration, and the
    projected cost of continuing chunks from a saved model."""
    lgb = _lgb()
    settings = lightgbm_settings(config)
    configs = grid_configs(settings)
    largest = max(configs, key=lambda c: (c.get("num_leaves", 0), c.get("learning_rate", 0)))
    params = training_params(settings, largest, int(settings["tuning"]["seed"]), len(context.data.class_names))
    truth = np.asarray(arrays.y_val, dtype=np.int64)
    train, val = arrays.datasets()

    def feval(preds, _data):
        probabilities = np.asarray(preds).reshape(len(truth), -1)
        return [("macro_f1", macro_f1_from_probabilities(probabilities, truth, probabilities.shape[1]), True)]

    started = time.monotonic()
    booster = lgb.train(params, train, num_boost_round=int(rounds), valid_sets=[val], feval=feval,
                        keep_training_booster=True)
    seconds_per_round = (time.monotonic() - started) / int(rounds)
    # One real chunk continuation (as every B4 run does), so the peak memory below includes
    # the re-scoring of all training rows from the saved model.
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        saved = os.path.join(folder, "pilot_model.txt")
        booster.save_model(saved, num_iteration=-1)
        started = time.monotonic()
        lgb.train(params, train, num_boost_round=1, init_model=saved, valid_sets=[val], feval=feval,
                  keep_training_booster=True)
        measured_continuation_seconds = time.monotonic() - started - seconds_per_round
    sample_rows = min(1_000_000, arrays.rows)
    sample = arrays.x[:sample_rows]
    started = time.monotonic()
    booster.predict(sample, num_threads=num_threads(), raw_score=True)
    predict_seconds_per_tree_row = (time.monotonic() - started) / (booster.num_trees() * sample_rows)
    max_rounds = int(settings["max_rounds"]); chunk = int(settings["chunk_rounds"])
    num_classes = len(context.data.class_names)
    rows = arrays.rows
    continuation_trees = sum(start * num_classes for start in range(chunk, max_rounds, chunk))
    continuation_seconds = continuation_trees * rows * predict_seconds_per_tree_row
    run_hours = (max_rounds * seconds_per_round + continuation_seconds) / 3600
    peak = None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(value * (1024 if os.uname().sysname == "Linux" else 1))
    except (ImportError, AttributeError, OSError):
        pass
    return {
        "pilot_rounds": int(rounds), "pilot_config": largest, "rows": int(rows), "threads": num_threads(),
        "seconds_per_round": seconds_per_round,
        "continuation_seconds_per_run_at_max_rounds": continuation_seconds,
        "measured_continuation_seconds_after_pilot": measured_continuation_seconds,
        "projected_hours_per_run_at_max_rounds": run_hours,
        "projected_hours_grid": run_hours * len(configs),
        "peak_memory_bytes": peak,
        "peak_memory_note": "process peak including one continuation from a saved model",
    }


def decide_data_mode(config: dict, context: OutOfCoreContext, directory: str) -> dict:
    """Apply the declared rule once and record it: full rows unless the grid projection exceeds the limit."""
    path = os.path.join(directory, DATA_MODE_FILE)
    if os.path.isfile(path):
        return _read_json(path)
    tuning = lightgbm_settings(config)["tuning"]
    requested = tuning.get("data_mode", "auto")
    decision = {"requested": requested, "max_projected_hours": float(tuning["max_projected_hours"])}
    if requested in {"full", "sample"}:
        decision["mode"] = requested
    elif requested == "auto":
        arrays = PreparedArrays(config, context, "full")
        cost = measure_cost(config, context, arrays, rounds=int(tuning.get("pilot_rounds", 20)))
        del arrays
        decision.update(cost)
        decision["mode"] = "full" if cost["projected_hours_grid"] <= decision["max_projected_hours"] else "sample"
        print(f"[B4 tuning] projected grid time {cost['projected_hours_grid']:.1f} h on all rows "
              f"(limit {decision['max_projected_hours']:.0f} h): tuning on {decision['mode']} rows", flush=True)
    else:
        raise ValueError("lightgbm.tuning.data_mode must be auto, full or sample")
    if decision["mode"] == "sample":
        decision["sample_fraction"] = float(tuning["sample_fraction"])
    _write_json(decision, path)
    return decision


def run_tuning(config: dict, context: OutOfCoreContext, condition: str, output_root: str | None = None) -> dict:
    settings = lightgbm_settings(config)
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
    decision = decide_data_mode(config, context, directory)
    arrays = None
    seed = int(settings["tuning"]["seed"])
    num_classes = len(context.data.class_names)
    log_path = os.path.join(directory, TUNING_LOG_FILE)
    log = pd.read_csv(log_path) if os.path.isfile(log_path) else pd.DataFrame()
    done = set(log["config_id"]) if not log.empty else set()
    for values in grid_configs(settings):
        config_id = _config_id(values)
        if config_id in done:
            continue
        params = training_params(settings, values, seed, num_classes)
        if arrays is None:
            arrays = PreparedArrays(config, context, decision["mode"])
        state = train_booster(
            params, arrays, max_rounds=int(settings["max_rounds"]),
            eval_every=int(settings["eval_every_rounds"]), patience=int(settings["patience_rounds"]),
            chunk_rounds=int(settings["chunk_rounds"]), state_dir=os.path.join(directory, config_id),
            label=f"tuning {config_id}",
        )
        row = {"config_id": config_id, **values, "seed": seed, "data_mode": decision["mode"],
               "train_rows": arrays.rows, "best_round": state["best_round"],
               "val_macro_f1": state["best_val_macro_f1"], "rounds_trained": state["rounds_done"],
               "stopped_early": state["stopped_early"], "wall_seconds": state["wall_seconds"]}
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        log.to_csv(log_path + ".tmp", index=False)
        os.replace(log_path + ".tmp", log_path)
    order = [_config_id(values) for values in grid_configs(settings)]
    log["grid_order"] = log["config_id"].map(order.index)
    best = log.sort_values(["val_macro_f1", "grid_order"], ascending=[False, True], kind="stable").iloc[0]
    values = {key: (int(best[key]) if key == "num_leaves" else float(best[key])) for key in settings["grid"]}
    max_rounds = int(settings["max_rounds"])
    near_cap = int(best["best_round"]) > max_rounds - int(settings["patience_rounds"])
    chosen = {
        "condition": condition,
        "chosen": values,
        "config_id": str(best["config_id"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "best_round": int(best["best_round"]),
        "tuning_seed": seed,
        "data_mode": decision["mode"],
        "seed_max_rounds": int(settings["extended_max_rounds"]) if near_cap else max_rounds,
        "cap_extended": bool(near_cap),
        "rule": "highest validation macro-F1; ties to the earlier grid configuration",
        "tuning_config_signature": signature,
    }
    _write_json(chosen, os.path.join(directory, CHOSEN_FILE))
    print(f"[B4 tuning] chosen {chosen['config_id']} val_macro_f1={chosen['val_macro_f1']:.6f} "
          f"best_round={chosen['best_round']} seed_max_rounds={chosen['seed_max_rounds']}", flush=True)
    return chosen


def train_lightgbm_unit(config: dict, context: OutOfCoreContext) -> LightGBMModel:
    """One B4 seed: the chosen configuration on all training rows."""
    settings = lightgbm_settings(config)
    run = config["run"]
    condition, seed = run["condition"], int(run["seed"])
    chosen = read_chosen(condition, run["output_root"])
    if not chosen or chosen.get("tuning_config_signature") != tuning_signature(config):
        raise RuntimeError(f"{condition} tuning is missing or outdated; run the campaign (it tunes {condition} first)")
    checkpoint_dir = config["training"]["checkpoint_dir"]
    final_path = stage_file(checkpoint_dir, "C", MODEL_FILE)
    num_classes = len(context.data.class_names)
    params = training_params(settings, chosen["chosen"], seed, num_classes)
    state_dir = os.path.join(checkpoint_dir, "lightgbm")
    max_rounds = int(chosen["seed_max_rounds"])
    reusable = os.path.join(tuning_dir(condition, run["output_root"]), chosen["config_id"])
    if (not os.path.isdir(state_dir) and seed == int(chosen["tuning_seed"]) and chosen["data_mode"] == "full"
            and max_rounds == int(settings["max_rounds"]) and os.path.isfile(os.path.join(reusable, STATE_FILE))
            and _read_json(os.path.join(reusable, STATE_FILE)).get("finished")):
        # Same parameters, data and seed as the chosen tuning run: identical result, so reuse it.
        shutil.copytree(reusable, state_dir)
        print(f"[B4 seed {seed}] reusing the identical tuning run {chosen['config_id']}", flush=True)
    if not os.path.isfile(final_path):
        state_file = os.path.join(state_dir, STATE_FILE)
        finished = os.path.isfile(state_file) and _read_json(state_file).get("finished")
        arrays = None if finished else PreparedArrays(config, context, "full")

        def on_check(check, state):
            append_training_history(config, {
                "stage": "C", "encoder_role": "lightgbm", "dataset": "ALL", "epoch": check["round"],
                "objective": "boosting", "rows": len(context.data.train.class_idx),
                "val_macro_f1": check["val_macro_f1"], "best_val_macro_f1": state["best_val_macro_f1"],
                "improved": check["round"] == state["best_round"],
                "patience_left": int(settings["patience_rounds"]) - (check["round"] - state["best_round"]),
            })

        state = train_booster(
            params, arrays, max_rounds=max_rounds,
            eval_every=int(settings["eval_every_rounds"]), patience=int(settings["patience_rounds"]),
            chunk_rounds=int(settings["chunk_rounds"]), state_dir=state_dir, label=f"seed {seed}",
            on_check=on_check,
        ) if not finished else _read_json(state_file)
        text = final_model_text(state_dir, state["best_round"])
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path + ".tmp", "w") as handle:
            handle.write(text)
        os.replace(final_path + ".tmp", final_path)
        summary = {
            "optimizer_steps": 0, "examples_seen": 0, "epochs_completed": int(state["rounds_done"]),
            "wall_seconds": float(state["wall_seconds"]), "selected_epoch": int(state["best_round"]),
        }
        _write_json({"best_epoch": int(state["best_round"]), "best_validation_macro_f1": state["best_val_macro_f1"],
                     "selection_mode": "best_val_round", "training_summary": summary,
                     "lightgbm": {"chosen": chosen, "rounds_trained": state["rounds_done"],
                                  "stopped_early": state["stopped_early"], "params": params}},
                    os.path.join(checkpoint_dir, "stage_c_training_summary.json"))
    with open(final_path) as handle:
        model = LightGBMModel(handle.read(), context.data.class_names, context.feature_columns)
    summary = _read_json(os.path.join(checkpoint_dir, "stage_c_training_summary.json"))
    model.training_summary = {"C": summary["training_summary"]}
    model.lightgbm_info = summary.get("lightgbm", {})
    return model


def lightgbm_version() -> str | None:
    try:
        import lightgbm

        return lightgbm.__version__
    except ImportError:
        return None
