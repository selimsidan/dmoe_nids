"""Process setup for B4 (XGBoost) when torch is also present.

torch and XGBoost ship separate OpenMP runtimes on macOS; XGBoost crashes in
a process where torch was loaded first. Processes that train or run a B4
model call ``prepare_for_trees()`` before anything imports torch: it loads
XGBoost first and keeps torch on one thread (B4 uses torch only to move
arrays). On Linux (Colab) this is not required and changes no result.
"""
from __future__ import annotations

import sys


def prepare_for_trees() -> bool:
    """Load XGBoost before torch; return False when it is too late or XGBoost is missing."""
    if "torch" in sys.modules and "xgboost" not in sys.modules:
        return False
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    import torch

    torch.set_num_threads(1)
    return True
