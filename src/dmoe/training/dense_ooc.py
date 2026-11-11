"""Training of the capacity-matched pooled MLP (baseline B2).

B2 starts from M's frozen Stage-A encoder. Its head is trained in Stage B on
four pooled random row streams whose sizes equal the four datasets' training
sizes (the same exposure M's experts get, without dataset identity), then in
Stage C on all pooled rows with best-validation-epoch selection.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..evaluation.validation_report import write_validation_report_ooc
from ..models.encoder import build_encoder, resolve_encoder_config
from .checkpoint import (
    BASELINE_MODEL_FILE,
    BASELINE_STAGE_B_FILE,
    _atomic_torch_save,
    clear_progress,
    load_progress,
    load_validated_stage_a,
    save_progress,
    stage_a_metadata,
    stage_file,
)
from .gate_supervision import set_encoder_trainable
from .model_utils import build_matched_dense_model
from .out_of_core_data import OutOfCoreContext
from .out_of_core_train import (
    _batch,
    _hash,
    _loss_weights,
    _progress_reporter,
    _settings,
    _validation_macro_f1,
    pooled_budget_row_batches,
    shuffled_row_batches,
)
from .optim import optimizer_hparams
from .training_history import append_training_history


def _baseline_config(config: dict) -> dict:
    values = config.get("training", {}).get("baseline", {})
    encoder_init = values.get("encoder_init", "stage_a")
    warmstart = values.get("stage_b_warmstart", "none")
    if encoder_init not in {"random", "stage_a"}:
        raise ValueError("training.baseline.encoder_init must be random or stage_a")
    if warmstart not in {"none", "matched_exposure", "pooled_matched_exposure"}:
        raise ValueError(
            "training.baseline.stage_b_warmstart must be none, matched_exposure, "
            "or pooled_matched_exposure"
        )
    if warmstart != "none" and encoder_init != "stage_a":
        raise ValueError("matched exposure requires training.baseline.encoder_init=stage_a")
    return {"encoder_init": encoder_init, "stage_b_warmstart": warmstart}


def _new_encoder(config: dict, data, device: torch.device):
    model_cfg = config["model"]
    return build_encoder(
        data.train.features.shape[1], model_cfg["latent_dim"],
        resolve_encoder_config(model_cfg),
    ).to(device)


def _expected_stage_a_metadata(config: dict, context: OutOfCoreContext) -> dict:
    return stage_a_metadata(
        config,
        context.data,
        split_signature=_hash(context.split_signatures),
        feature_columns=context.feature_columns,
    )


def build_dense_ooc(config: dict, context: OutOfCoreContext):
    device = torch.device(config["training"]["device"])
    curriculum = _baseline_config(config)
    encoder = _new_encoder(config, context.data, device)
    if curriculum["encoder_init"] == "stage_a":
        encoder.load_state_dict(
            load_validated_stage_a(config, _expected_stage_a_metadata(config, context))["encoder_state"]
        )
    if config["architecture"] == "matched_dense":
        return build_matched_dense_model(
            encoder, context.data.active_datasets, context.data.class_names, config["model"]
        ).to(device)
    raise ValueError("the dense builder supports matched_dense only")


def resolve_dense_stage_b_path(config: dict) -> str:
    source = config.get("training", {}).get("dense_stage_b_checkpoint")
    if source:
        return source if source.endswith(".pt") else stage_file(source, "B", BASELINE_STAGE_B_FILE)
    return stage_file(config["training"]["checkpoint_dir"], "B", BASELINE_STAGE_B_FILE)


def run_dense_stage_b_ooc(config: dict, context: OutOfCoreContext, model=None):
    stage_b_path = resolve_dense_stage_b_path(config)
    if model is None and os.path.isfile(stage_b_path):
        model = build_dense_ooc(config, context)
        checkpoint = torch.load(stage_b_path, map_location="cpu")
        expected = {
            "architecture": config["architecture"],
            "dataset_names": list(context.data.active_datasets),
        }
        actual = checkpoint.get("metadata")
        if actual is not None and actual != expected:
            raise ValueError(
                f"Incompatible dense Stage-B checkpoint {stage_b_path}: "
                f"expected {expected}, actual {actual}"
            )
        model.load_state_dict(checkpoint["model_state"])
        model.training_summary = {"B": checkpoint["training_summary"]}
        return model
    model = model or build_dense_ooc(config, context)
    curriculum = _baseline_config(config)
    if curriculum["stage_b_warmstart"] == "none":
        summary = {
            "optimizer_steps": 0, "examples_seen": 0, "epochs_completed": 0,
            "wall_seconds": 0.0, "selected_epoch": 0,
        }
        model.training_summary = {"B": summary}
        _atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "training_summary": summary,
                "metadata": {
                    "architecture": config["architecture"],
                    "dataset_names": list(context.data.active_datasets),
                },
            },
            stage_file(config["training"]["checkpoint_dir"], "B", BASELINE_STAGE_B_FILE),
        )
        return model
    device = next(model.parameters()).device
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    lr, weight_decay = optimizer_hparams(config, "b")
    optimizer = torch.optim.Adam(
        model.head.parameters(), lr=lr, weight_decay=weight_decay,
    )
    progress = load_progress(config["training"]["checkpoint_dir"], "DB")
    resume_dataset = resume_epoch = 0
    steps = examples = 0
    if progress:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        resume_dataset = int(progress["dataset_i"])
        resume_epoch = int(progress["epoch"])
        steps = int(progress.get("optimizer_steps", 0))
        examples = int(progress.get("examples_seen", 0))
    batch_size, block_rows, buffer_blocks = _settings(config)
    started = time.monotonic()
    pooled = curriculum["stage_b_warmstart"] == "pooled_matched_exposure"
    budgets = sorted(
        (
            bounds.stop - bounds.start
            for bounds in context.data.train.dataset_slices.values()
        ),
        reverse=True,
    )
    pooled_weights = (
        _loss_weights(
            context.data.train.class_idx, len(context.data.class_names), device
        )
        if pooled else None
    )
    for dataset_index, name in enumerate(context.data.active_datasets):
        if dataset_index < resume_dataset:
            continue
        bounds = context.data.train.dataset_slices[name]
        weights = (
            pooled_weights if pooled else _loss_weights(
                context.data.train.class_idx[bounds], len(context.data.class_names), device
            )
        )
        first_epoch = resume_epoch if dataset_index == resume_dataset else 0
        for epoch in range(first_epoch, int(config["training"]["epochs_b"])):
            rng = np.random.default_rng(config.get("seed", 0) + dataset_index * 10_000 + epoch)
            target_rows = budgets[dataset_index] if pooled else bounds.stop - bounds.start
            label = f"pooled-budget-{dataset_index}" if pooled else name
            report = _progress_reporter(
                f"{config['architecture']} Stage B/ooc:{label} epoch {epoch + 1}",
                target_rows,
                int(config["training"].get("progress_every_rows", 1_000_000)),
            )
            batches = (
                pooled_budget_row_batches(
                    len(context.data.train.class_idx), budgets[dataset_index], rng,
                    batch_size, block_rows, buffer_blocks,
                )
                if pooled else shuffled_row_batches(
                    bounds.start, bounds.stop, rng,
                    batch_size, block_rows, buffer_blocks,
                )
            )
            loss_sum = 0.0
            rows_seen = 0
            for row_ids in batches:
                features, labels, _ = _batch(context.data.train, row_ids, device)
                with torch.no_grad():
                    latent = model.encoder(features)
                loss = F.cross_entropy(model.head(latent), labels, weight=weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward(); optimizer.step()
                steps += 1; examples += len(row_ids)
                loss_sum += float(loss.item()) * len(row_ids)
                rows_seen += len(row_ids)
                report(rows_seen)
            report(rows_seen, force=True)
            print(
                f"[{config['architecture']} Stage B/ooc:{label}] epoch {epoch + 1}: "
                f"CE={loss_sum / max(1, rows_seen):.6f} rows={rows_seen:,}",
                flush=True,
            )
            append_training_history(config, {
                "stage": "B", "encoder_role": "dense_head", "dataset": label,
                "epoch": epoch + 1, "objective": "ce",
                "learning_rate": optimizer.param_groups[0]["lr"], "rows": rows_seen,
                "optimizer_steps": steps, "examples_seen": examples,
                "train_ce_loss": loss_sum / max(1, rows_seen),
                "train_total_loss": loss_sum / max(1, rows_seen),
            })
            save_progress(config["training"]["checkpoint_dir"], "DB", {
                "dataset_i": dataset_index, "epoch": epoch + 1,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "optimizer_steps": steps, "examples_seen": examples,
            })
    clear_progress(config["training"]["checkpoint_dir"], "DB")
    summary = {
        "optimizer_steps": steps, "examples_seen": examples,
        "epochs_completed": int(config["training"]["epochs_b"]),
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": int(config["training"]["epochs_b"]),
        "pooled_examples_seen": examples if pooled else 0,
    }
    model.training_summary = {"B": summary}
    _atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "training_summary": summary,
            "metadata": {
                "architecture": config["architecture"],
                "dataset_names": list(context.data.active_datasets),
            },
        },
        stage_file(config["training"]["checkpoint_dir"], "B", BASELINE_STAGE_B_FILE),
    )
    return model


def run_dense_stage_c_ooc(config: dict, context: OutOfCoreContext, model=None):
    model = model or run_dense_stage_b_ooc(config, context)
    device = next(model.parameters()).device
    set_encoder_trainable(model.encoder, config["training"]["stage_c_unfreeze"])
    lr, weight_decay = optimizer_hparams(config, "c")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    weights = _loss_weights(context.data.train.class_idx, len(context.data.class_names), device)
    selection = config["training"].get("selection_mode", "fixed_epochs")
    if selection not in {"fixed_epochs", "best_val"}:
        raise ValueError("training.selection_mode must be fixed_epochs or best_val")
    progress = load_progress(config["training"]["checkpoint_dir"], "DC")
    start_epoch = steps = examples = 0
    best_score = -float("inf"); best_state = None; best_epoch = None
    patience = int(config["training"].get("early_stopping_patience", 5))
    if progress:
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        start_epoch = int(progress["epoch"])
        steps = int(progress.get("optimizer_steps", 0)); examples = int(progress.get("examples_seen", 0))
        best_score = float(progress.get("best_score", best_score)); best_state = progress.get("best_state")
        best_epoch = progress.get("best_epoch"); patience = int(progress.get("patience", patience))
    batch_size, block_rows, buffer_blocks = _settings(config)
    started = time.monotonic(); completed = start_epoch
    for epoch in range(start_epoch, int(config["training"]["epochs_c"])):
        completed = epoch + 1
        rng = np.random.default_rng(config.get("seed", 0) + 100_000 + epoch)
        model.train(); loss_sum = 0.0; rows_seen = 0
        report = _progress_reporter(
            f"{config['architecture']} Stage C/ooc epoch {completed}",
            len(context.data.train.class_idx),
            int(config["training"].get("progress_every_rows", 1_000_000)),
        )
        for row_ids in shuffled_row_batches(0, len(context.data.train.class_idx), rng, batch_size, block_rows, buffer_blocks):
            features, labels, _ = _batch(context.data.train, row_ids, device)
            loss = F.cross_entropy(model(features)["logits"], labels, weight=weights)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            steps += 1; examples += len(row_ids); rows_seen += len(row_ids)
            loss_sum += float(loss.item()) * len(row_ids); report(rows_seen)
        report(rows_seen, force=True)
        print(f"[{config['architecture']} Stage C/ooc] epoch {completed}: CE={loss_sum / rows_seen:.6f}")
        if selection == "best_val":
            score = _validation_macro_f1(
                model, context.data.val, len(context.data.class_names), device,
                int(config["training"].get("validation_chunk_rows", 262_144)),
            )
            if score > best_score + 1e-12:
                best_score, best_epoch = score, completed
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = int(config["training"].get("early_stopping_patience", 5))
            else:
                patience -= 1
        append_training_history(config, {
            "stage": "C", "encoder_role": "full_model", "dataset": "ALL",
            "epoch": completed, "objective": "ce",
            "learning_rate": optimizer.param_groups[0]["lr"], "rows": rows_seen,
            "optimizer_steps": steps, "examples_seen": examples,
            "train_ce_loss": loss_sum / rows_seen, "train_total_loss": loss_sum / rows_seen,
            "val_macro_f1": score if selection == "best_val" else np.nan,
            "best_val_macro_f1": best_score if selection == "best_val" else np.nan,
            "improved": (best_epoch == completed) if selection == "best_val" else np.nan,
            "patience_left": patience if selection == "best_val" else np.nan,
        })
        save_progress(config["training"]["checkpoint_dir"], "DC", {
            "epoch": completed, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "optimizer_steps": steps, "examples_seen": examples, "best_score": best_score,
            "best_state": best_state, "best_epoch": best_epoch, "patience": patience,
        })
        if selection == "best_val" and patience <= 0:
            break
    if selection == "best_val" and best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "optimizer_steps": steps, "examples_seen": examples, "epochs_completed": completed,
        "wall_seconds": time.monotonic() - started,
        "selected_epoch": best_epoch if selection == "best_val" else completed,
    }
    prior = dict(getattr(model, "training_summary", {})); prior["C"] = summary
    model.training_summary = prior
    _atomic_torch_save({
        "model_state": model.state_dict(), "class_names": context.data.class_names,
        "dataset_names": context.data.active_datasets,
        "match_info": getattr(model, "match_info", None), "training_summary": model.training_summary,
    }, stage_file(config["training"]["checkpoint_dir"], "C", BASELINE_MODEL_FILE))
    write_validation_report_ooc(model, context, config)
    clear_progress(config["training"]["checkpoint_dir"], "DC")
    return model


def load_dense_ooc(config: dict, context: OutOfCoreContext):
    model = build_dense_ooc(config, context)
    checkpoint = torch.load(
        stage_file(config["training"]["checkpoint_dir"], "C", BASELINE_MODEL_FILE), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state"])
    model.training_summary = checkpoint.get("training_summary") or {}
    return model
