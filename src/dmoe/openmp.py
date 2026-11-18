"""Process setup for B4 (LightGBM) when torch is also present.

torch and LightGBM ship separate OpenMP runtimes on macOS; LightGBM crashes in
a process where torch was loaded first. Processes that train or run a B4
model call ``prepare_for_lightgbm()`` before anything imports torch: it loads
LightGBM first and keeps torch on one thread (B4 uses torch only to move
arrays). On Linux (Colab) this is not required and changes no result.
"""
from __future__ import annotations

import sys


def prepare_for_lightgbm() -> bool:
    """Load LightGBM before torch; return False when it is too late or LightGBM is missing."""
    if "torch" in sys.modules and "lightgbm" not in sys.modules:
        return False
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    import torch

    torch.set_num_threads(1)
    return True
