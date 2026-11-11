"""Check that the four NF-v3 CSVs are where the code expects them and look right.

For each dataset: the file exists at ``DATA_ROOT/<dataset>/data/<dataset>.csv``,
its header contains the 47 features and the ``Attack`` label column, its row
count matches ``configs/checksums.json`` and, when a SHA-256 is recorded
there, its content hash matches (SHA-1 from the publisher's BagIt manifest, and
SHA-256 once recorded). Every failure prints one line naming the
expected path and pointing to ``docs/DATA.md``.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

from .. import paths
from ..training.config import CONFIG_ROOT
from .registry import DATASET_NAMES, FEATURE_COLUMNS, LABEL_COLUMN
from .source_hash import cached_digests

CHECKSUMS_FILE = os.path.join(CONFIG_ROOT, "checksums.json")
HELP = "see docs/DATA.md"


@dataclass
class DatasetCheck:
    dataset: str
    path: str
    ok: bool = True
    rows: int | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    sha1: str | None = None
    problems: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.problems.append(message)


def load_checksums(path: str = CHECKSUMS_FILE) -> dict:
    with open(path) as handle:
        return json.load(handle)


def read_header(path: str) -> list[str]:
    with open(path, newline="") as handle:
        return [column.strip() for column in next(csv.reader(handle))]


def count_rows(path: str, chunk_bytes: int = 64 * 2**20) -> int:
    """Data rows (lines minus the header), counted in binary chunks."""
    lines = 0
    last = b"\n"
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            lines += block.count(b"\n")
            last = block[-1:]
    if last != b"\n":
        lines += 1
    return max(0, lines - 1)


def check_dataset(
    dataset: str,
    expected: dict,
    data_root: str | None = None,
    *,
    count: bool = True,
    hash_files: bool = True,
    hash_cache_dir: str | None = None,
) -> DatasetCheck:
    path = paths.dataset_csv_path(dataset, data_root)
    result = DatasetCheck(dataset, path)
    if not os.path.isfile(path):
        result.fail(f"{dataset}: file not found at {path}; {HELP}")
        return result
    result.size_bytes = os.path.getsize(path)
    header = read_header(path)
    missing = [c for c in [*FEATURE_COLUMNS, LABEL_COLUMN] if c not in header]
    if missing:
        result.fail(f"{dataset}: header of {path} lacks {len(missing)} expected column(s) {missing[:6]}; {HELP}")
    if count:
        result.rows = count_rows(path)
        if expected.get("rows") is not None and result.rows != int(expected["rows"]):
            result.fail(f"{dataset}: {path} has {result.rows:,} rows, expected {int(expected['rows']):,}; {HELP}")
    if expected.get("size_bytes") is not None and result.size_bytes != int(expected["size_bytes"]):
        result.fail(f"{dataset}: {path} is {result.size_bytes:,} bytes, expected {int(expected['size_bytes']):,}; {HELP}")
    if hash_files and (expected.get("sha256") or expected.get("sha1")):
        digests = cached_digests(path, hash_cache_dir)
        result.sha256, result.sha1 = digests["sha256"], digests["sha1"]
        for name in ("sha256", "sha1"):
            if expected.get(name) and digests[name] != expected[name]:
                result.fail(f"{dataset}: {name.upper()} of {path} does not match configs/checksums.json; {HELP}")
    return result


def verify_data(
    data_root: str | None = None,
    datasets: list[str] | None = None,
    *,
    count: bool = True,
    hash_files: bool = True,
    hash_cache_dir: str | None = None,
    checksums_path: str = CHECKSUMS_FILE,
) -> list[DatasetCheck]:
    checksums = load_checksums(checksums_path)
    results = []
    for dataset in datasets or list(DATASET_NAMES):
        result = check_dataset(
            dataset, checksums.get(dataset, {}), data_root,
            count=count, hash_files=hash_files, hash_cache_dir=hash_cache_dir,
        )
        status = "ok" if result.ok else "FAILED"
        rows = f" rows={result.rows:,}" if result.rows is not None else ""
        print(f"[verify] {dataset}: {status}{rows}", flush=True)
        for problem in result.problems:
            print(f"[verify]   {problem}", flush=True)
        results.append(result)
    return results
