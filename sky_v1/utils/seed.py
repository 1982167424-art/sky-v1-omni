"""Deterministic seeding for stdlib / numpy / torch when available."""
from __future__ import annotations

import os
import random

import numpy as np

from .logging import get_logger

log = get_logger("utils.seed")


def set_global_seed(seed: int = 1337) -> int:
    """Set seeds for stdlib random, PYTHONHASHSEED, numpy, torch (if installed).

    Returns the normalized seed actually used (clamped to u32).
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    # clamp to 32-bit unsigned for cross-platform determinism
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch is optional in M1
        pass
    log.debug("Global seed set", seed=seed)
    return seed
