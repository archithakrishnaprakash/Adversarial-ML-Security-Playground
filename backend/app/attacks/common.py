from __future__ import annotations

import numpy as np
import torch


def to_tensor(X: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(X, dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t


def clip01(X: np.ndarray) -> np.ndarray:
    return np.clip(X, 0.0, 1.0)
