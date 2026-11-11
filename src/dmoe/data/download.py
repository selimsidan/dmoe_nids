"""Download helpers driven by ``configs/sources.yaml``.

Only datasets with a ``direct_url`` are fetched automatically (into
``DATA_ROOT/<dataset>/``, unzipped when the file is an archive, then
verified). For the others the manual steps are printed: the official page,
where to place the CSV, and the verification command.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile

from .. import paths
from ..training.config import CONFIG_ROOT, load_yaml
from .registry import DATASET_NAMES

SOURCES_FILE = os.path.join(CONFIG_ROOT, "sources.yaml")


def load_sources(path: str = SOURCES_FILE) -> dict:
    return load_yaml(path).get("datasets", {})


def manual_steps(dataset: str, source: dict, data_root: str) -> str:
    page = source.get("page_url") or "TODO (see docs/DATA.md)"
    target = paths.dataset_csv_path(dataset, data_root)
    return (
        f"{dataset}: download it manually from {page}\n"
        f"    then place the CSV at {target}\n"
        f"    and run: python scripts/verify_data.py --only {dataset}"
    )


def download_dataset(dataset: str, source: dict, data_root: str) -> str:
    url = source["direct_url"]
    target_dir = os.path.join(data_root, dataset)
    os.makedirs(os.path.join(target_dir, "data"), exist_ok=True)
    archive = os.path.join(target_dir, os.path.basename(url.split("?", 1)[0]) or f"{dataset}.download")
    print(f"[download] {dataset}: {url}", flush=True)
    with urllib.request.urlopen(url) as response, open(archive + ".part", "wb") as handle:
        shutil.copyfileobj(response, handle, length=16 * 2**20)
    os.replace(archive + ".part", archive)
    expected_csv = paths.dataset_csv_path(dataset, data_root)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            member = source.get("archive_member") or next(
                (name for name in bundle.namelist() if name.endswith(f"{dataset}.csv")), None
            )
            if member is None:
                raise FileNotFoundError(f"{archive} does not contain {dataset}.csv; set archive_member in sources.yaml")
            with bundle.open(member) as stream, open(expected_csv, "wb") as handle:
                shutil.copyfileobj(stream, handle, length=16 * 2**20)
        os.remove(archive)
    elif archive != expected_csv:
        os.replace(archive, expected_csv)
    return expected_csv


def download_all(data_root: str | None = None, datasets: list[str] | None = None) -> dict[str, str]:
    """Download what can be downloaded; return ``{dataset: "downloaded" | "manual"}``."""
    root = data_root or paths.data_root()
    sources = load_sources()
    status = {}
    for dataset in datasets or list(DATASET_NAMES):
        source = sources.get(dataset, {})
        if source.get("direct_url"):
            download_dataset(dataset, source, root)
            status[dataset] = "downloaded"
        else:
            print(manual_steps(dataset, source, root), flush=True)
            status[dataset] = "manual"
    return status
