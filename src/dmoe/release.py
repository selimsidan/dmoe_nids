"""Release helpers: copy small result files into the repository, publish bundles.

``sync_results`` copies the small, shareable files from OUTPUT_ROOT into the
repository's ``results/`` and ``paper/`` folders: per-run CSV tables and
manifests, the cross-seed summary, the split description and the paper
exhibits. It never copies data, caches, checkpoints, bundles, per-row
predictions or logs, and it rewrites any absolute path under OUTPUT_ROOT to a
path relative to it.

``publish_hub`` uploads every run's bundle to the Hugging Face repository
named in ``configs/base.yaml`` (``hub.repo_id``) under
``bundles/<condition>/seed<k>/`` plus a ``recommended.json`` pointer to M's
recommended seed. Dry run by default; needs ``HF_TOKEN`` to execute.
"""
from __future__ import annotations

import json
import os
import shutil

import pandas as pd

from . import paths

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COPY_SUFFIXES = {".csv", ".json", ".tex"}
FIGURE_SUFFIXES = {".pdf", ".png"}
NEVER_COPY_SUFFIXES = {".npz", ".npy", ".pt", ".safetensors", ".log", ".txt", ".tmp"}


def _relativize(text: str, output_root: str) -> str:
    root = os.path.abspath(output_root).rstrip(os.sep)
    return text.replace(root + os.sep, "").replace(root, ".")


def _copy_text(source: str, target: str, output_root: str) -> None:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(_relativize(text, output_root))


def sync_results(output_root: str | None = None, repo_root: str = REPO_ROOT, *, figures: bool = True) -> list[str]:
    """Copy shareable results into ``results/`` and ``paper/``; return the written paths."""
    root = output_root or paths.output_root()
    written = []
    plan = [
        (os.path.join(root, "runs"), os.path.join(repo_root, "results", "runs"), False),
        (paths.summary_dir(root), os.path.join(repo_root, "results", "summary"), figures),
        (paths.paper_dir(root), os.path.join(repo_root, "paper"), True),
    ]
    for source_dir, target_dir, with_figures in plan:
        if not os.path.isdir(source_dir):
            continue
        for directory, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if d not in {"checkpoints", "bundle"}]
            for name in files:
                suffix = os.path.splitext(name)[1].lower()
                if suffix in NEVER_COPY_SUFFIXES:
                    continue
                if suffix not in COPY_SUFFIXES and not (with_figures and suffix in FIGURE_SUFFIXES):
                    continue
                source = os.path.join(directory, name)
                target = os.path.join(target_dir, os.path.relpath(source, source_dir))
                if suffix in COPY_SUFFIXES:
                    _copy_text(source, target, root)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copy2(source, target)
                written.append(target)
    split_file = os.path.join(root, "cache", "split.json")
    if os.path.isfile(split_file):
        target = os.path.join(repo_root, "results", "splits", "split.json")
        _copy_text(split_file, target, root)
        written.append(target)
    print(f"[sync] copied {len(written)} file(s) into results/ and paper/")
    return written


def recommended_pointer(output_root: str, condition: str = "M") -> dict:
    path = os.path.join(paths.summary_dir(output_root), "Recommended_Seeds.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing; run scripts/aggregate.py first")
    frame = pd.read_csv(path)
    row = frame[frame["condition"] == condition].iloc[0]
    return {"condition": condition, "seed": int(row["recommended_seed"]), "rule": str(row["rule"]),
            "val_macro_f1": float(row["val_macro_f1"])}


def publish_hub(output_root: str | None = None, *, repo_id: str | None = None, execute: bool = False) -> list[str]:
    from .hub import hub_repo_id

    root = output_root or paths.output_root()
    repo = repo_id or hub_repo_id()
    uploads = []
    runs = os.path.join(root, "runs")
    for condition in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        for seed_dir in sorted(os.listdir(os.path.join(runs, condition))):
            bundle = os.path.join(runs, condition, seed_dir, "bundle")
            if os.path.isfile(os.path.join(bundle, "metadata.json")):
                uploads.append((bundle, f"bundles/{condition}/{seed_dir}"))
    pointer = recommended_pointer(root)
    for local, remote in uploads:
        print(f"[hub] {'upload' if execute else 'would upload'} {paths.relative_to_root(local, root)} -> {repo}:{remote}")
    print(f"[hub] recommended.json -> {pointer}")
    if not execute:
        print("[hub] dry run; pass --execute (and set HF_TOKEN) to publish")
        return [remote for _, remote in uploads]
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="model", exist_ok=True)
    for local, remote in uploads:
        api.upload_folder(repo_id=repo, folder_path=local, path_in_repo=remote)
    api.upload_file(
        repo_id=repo, path_or_fileobj=json.dumps(pointer, indent=2).encode(), path_in_repo="recommended.json",
    )
    return [remote for _, remote in uploads]
