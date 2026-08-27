from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.attacks.common import clip01


def fgsm_attack(gradient_source: nn.Module, X: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Perturbs each input by epsilon in the direction that most increases the
    loss, using only the sign of the gradient — a single step, computationally
    cheap, and the classic "hello world" of adversarial attacks.

    x_adv = x + epsilon * sign(grad_x Loss(model(x), y))
    """
    gradient_source.eval()
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.long)

    logits = gradient_source(Xt)
    loss = nn.functional.cross_entropy(logits, yt)
    loss.backward()

    grad_sign = Xt.grad.data.sign().numpy()
    X_adv = X + epsilon * grad_sign
    return clip01(X_adv).astype(np.float32)
