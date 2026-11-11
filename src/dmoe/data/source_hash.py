"""SHA-256 and SHA-1 of large source CSVs, cached so each file version is hashed once.

The split signature signs the source by content. Hashing a multi-GB CSV takes
minutes, so the digest is cached in a small JSON file keyed by the file's
name, size and modification time. A changed file (new size or mtime) is
re-hashed; an unchanged file returns the cached digest instantly. The cache
key never enters any signature, only the digest does.
"""
from __future__ import annotations

import hashlib
import json
import os

CACHE_FILE = "source_digests.json"
_CHUNK = 16 * 2**20


def file_digests(path: str, algorithms: tuple[str, ...] = ("sha256",)) -> dict[str, str]:
    """Several digests of one file in a single read pass."""
    digests = {name: hashlib.new(name) for name in algorithms}
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            for digest in digests.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def file_sha256(path: str) -> str:
    return file_digests(path)["sha256"]


def _cache_key(path: str) -> str:
    stat = os.stat(path)
    return f"{os.path.basename(path)}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}"


def _load_cache(cache_dir: str | None) -> dict[str, str]:
    if not cache_dir:
        return {}
    path = os.path.join(cache_dir, CACHE_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def cached_digests(path: str, cache_dir: str | None = None) -> dict[str, str]:
    """SHA-256 and SHA-1 of ``path`` (one read), reusing cached values when possible."""
    key = _cache_key(path)
    cache = _load_cache(cache_dir)
    value = cache.get(key)
    if isinstance(value, dict) and {"sha256", "sha1"} <= set(value):
        return value
    print(f"[data] hashing {os.path.basename(path)} (once per file version)", flush=True)
    value = file_digests(path, ("sha256", "sha1"))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache[key] = value
        target = os.path.join(cache_dir, CACHE_FILE)
        temporary = target + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
        os.replace(temporary, target)
    return value


def source_sha256(path: str, cache_dir: str | None = None) -> str:
    """SHA-256 of ``path`` (the digest that signs the split)."""
    return cached_digests(path, cache_dir)["sha256"]
