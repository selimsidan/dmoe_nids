"""Google Colab setup shared by every notebook.

``setup()`` resolves the three storage roots, mounts Google Drive when asked,
exports the roots as environment variables (so training subprocesses see
them) and prints what it resolved and from where.

Root resolution, for ``DATA_ROOT`` and ``OUTPUT_ROOT``:

1. the value typed in the notebook's SETTINGS cell (``"typed"``);
2. the Colab secret ``NIDS_DATA_DIR`` / ``NIDS_OUTPUT_DIR`` (``"secret"``);
   a missing secret, or one the notebook is not allowed to read, counts as unset;
3. the generic default for the chosen storage (``"default"``).

``SCRATCH_ROOT`` is always ``/content/scratch`` (fast local disk, wiped when the
runtime ends; the split cache is copied there each session).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from . import paths

REPO_URL = "https://github.com/selimsidan/dmoe_nids.git"
SCRATCH = "/content/scratch"
DEFAULTS = {
    "drive": {"data": "/content/drive/MyDrive/NIDS_datasets", "output": "/content/drive/MyDrive/NIDS_outputs"},
    "local": {"data": "/content/data", "output": "/content/outputs"},
}
SECRETS = {"data": "NIDS_DATA_DIR", "output": "NIDS_OUTPUT_DIR"}


@dataclass(frozen=True)
class ColabRoot:
    path: str
    source: str  # typed | secret | default


def read_secret(name: str, userdata=None) -> str | None:
    """Value of a Colab secret, or None when it is missing or access is denied."""
    if userdata is None:
        try:
            from google.colab import userdata as colab_userdata  # type: ignore
        except ImportError:
            return None
        userdata = colab_userdata
    try:
        value = userdata.get(name)
    except Exception:  # SecretNotFoundError, NotebookAccessError, TimeoutException
        return None
    return value or None


def resolve_colab_root(kind: str, typed: str | None, storage: str, userdata=None) -> ColabRoot:
    if storage not in DEFAULTS:
        raise ValueError(f"STORAGE must be 'drive' or 'local', got {storage!r}")
    if typed:
        return ColabRoot(typed, "typed")
    secret = read_secret(SECRETS[kind], userdata)
    if secret:
        return ColabRoot(secret, "secret")
    return ColabRoot(DEFAULTS[storage][kind], "default")


def _git(*args: str, cwd: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def setup(
    storage: str = "drive",
    data_root: str | None = None,
    output_root: str | None = None,
    *,
    require_data: bool = True,
    userdata=None,
    mount=None,
) -> dict[str, ColabRoot]:
    """Resolve and export the storage roots; mount Drive in ``drive`` mode."""
    roots = {
        "data": resolve_colab_root("data", data_root, storage, userdata),
        "output": resolve_colab_root("output", output_root, storage, userdata),
        "scratch": ColabRoot(SCRATCH, "default"),
    }
    if storage == "drive":
        if mount is None:
            from google.colab import drive  # type: ignore

            mount = drive.mount
        if not os.path.isdir("/content/drive/MyDrive"):
            mount("/content/drive")
    if require_data and not os.path.isdir(roots["data"].path):
        raise FileNotFoundError(
            f"DATA_ROOT {roots['data'].path} ({roots['data'].source}) does not exist. "
            "Create it and upload the datasets as described in docs/DATA.md, or set DATA_ROOT."
        )
    os.makedirs(roots["output"].path, exist_ok=True)
    os.makedirs(roots["scratch"].path, exist_ok=True)
    for kind, root in roots.items():
        os.environ[paths.ENV_VARS[kind]] = root.path
    try:
        commit = _git("rev-parse", "--short", "HEAD", cwd=os.path.dirname(os.path.abspath(__file__)))
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    print(f"storage      : {storage}")
    print(f"commit       : {commit}")
    print(f"gpu          : {gpu_name() or 'none (CPU runtime)'}")
    for kind in ("data", "output", "scratch"):
        root = roots[kind]
        print(f"{kind + '_root':<13}: {root.path}  [{root.source}]")
    return roots


def clone_or_update(ref: str = "main", target: str = "/content/dmoe_nids", url: str = REPO_URL, userdata=None) -> str:
    """Clone (or fetch and check out) the repository at ``ref`` and return the commit.

    The repository is public, so an anonymous clone is tried first. Only if
    that fails and a ``GITHUB_TOKEN`` Colab secret exists (for example for a
    private fork) is the clone retried with a temporary authorization header,
    which is removed afterwards. The token is never printed or written to disk.
    """
    def run(*args: str, cwd: str | None = None, extra_env: dict | None = None) -> None:
        env = {**os.environ, **(extra_env or {})}
        subprocess.run(["git", *args], cwd=cwd, check=True, env=env, capture_output=True, text=True)

    def attempt(extra_env: dict | None = None) -> None:
        if os.path.isdir(os.path.join(target, ".git")):
            run("fetch", "--quiet", "origin", cwd=target, extra_env=extra_env)
        else:
            run("clone", "--quiet", url, target, extra_env=extra_env)
        run("checkout", "--quiet", ref, cwd=target)
        if _git("rev-parse", "--abbrev-ref", "HEAD", cwd=target) != "HEAD":
            run("pull", "--quiet", "--ff-only", "origin", ref, cwd=target, extra_env=extra_env)

    try:
        attempt()
    except subprocess.CalledProcessError:
        token = read_secret("GITHUB_TOKEN", userdata)
        if not token:
            raise RuntimeError(f"could not clone {url}; check REPO_REF or your connection") from None
        import base64

        header = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        attempt({
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {header}",
        })
        del token, header
    commit = _git("rev-parse", "HEAD", cwd=target)
    print(f"repository   : {url} @ {ref} ({commit[:12]})")
    return commit
