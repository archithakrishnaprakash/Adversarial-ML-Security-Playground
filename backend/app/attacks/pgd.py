from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.attacks.common import clip01


def pgd_attack(
    gradient_source: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float | None = None,
    steps: int = 10,
) -> np.ndarray:
    """Projected Gradient Descent (Madry et al., 2018) — iterative FGSM with
    projection back onto the epsilon L-infinity ball around the original
    input after every step. Generally a much stronger attack than single-step
    FGSM at the same epsilon budget.
    """
    if alpha is None:
        alpha = max(epsilon / 4, 1e-3)

    gradient_source.eval()
    X_orig = X.copy()
    X_adv = X.copy()
    yt = torch.tensor(y, dtype=torch.long)

    for _ in range(steps):
        Xt = torch.tensor(X_adv, dtype=torch.float32, requires_grad=True)
        logits = gradient_source(Xt)
        loss = nn.functional.cross_entropy(logits, yt)
        loss.backward()

        grad_sign = Xt.grad.data.sign().numpy()
        X_adv = X_adv + alpha * grad_sign

        # project back into the epsilon ball around the original input
        perturbation = np.clip(X_adv - X_orig, -epsilon, epsilon)
        X_adv = clip01(X_orig + perturbation)

    return X_adv.astype(np.float32)
