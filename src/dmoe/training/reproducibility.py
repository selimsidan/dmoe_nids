"""Process-level seeding and the environment record stored in every manifest.

Guarantee: the same code, data, seeds, library versions and GPU model give
identical results. A different GPU model (or CUDA/cuDNN version) can change
the last decimals, because floating-point reductions may be ordered
differently; the GPU and versions are therefore recorded in every manifest.
"""
from __future__ import annotations

import os
import platform
import random
import subprocess

import numpy as np
import torch


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (all devices) and select deterministic kernels.

    The data pipeline additionally builds explicit per-stage NumPy generators
    from ``seed``; this covers weight initialization and dropout.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def _git_commit() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return value + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict:
    """Versions, hardware and commit; recorded in manifests and bundle metadata."""
    import pandas
    import sklearn

    info = {
        "commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return info
