"""Small shared helpers used across scripts."""
from __future__ import annotations

import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(X: np.ndarray, y: np.ndarray):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
