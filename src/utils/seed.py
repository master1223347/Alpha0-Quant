"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set process-wide seed for deterministic behavior where possible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass
