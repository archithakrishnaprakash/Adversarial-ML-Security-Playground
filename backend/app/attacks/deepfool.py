from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.attacks.common import clip01


def _deepfool_single(gradient_source: nn.Module, x: np.ndarray, num_classes: int, max_iter: int, overshoot: float):
    x_adv = x.copy()
    x_shape = x.shape

    for _ in range(max_iter):
        xt = torch.tensor(x_adv, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        logits = gradient_source(xt)[0]
        orig_label = int(logits.argmax().item())

        # gradient of the original-class logit
        grads = []
        logit_values = []
        for k in range(num_classes):
            gradient_source.zero_grad(set_to_none=True)
            if xt.grad is not None:
                xt.grad.zero_()
            logits_k = gradient_source(xt)[0]
            logits_k[k].backward(retain_graph=True)
            grads.append(xt.grad.detach().clone().numpy().reshape(-1))
            logit_values.append(logits_k[k].item())

        grads = np.stack(grads)  # (C, D)
        logit_values = np.array(logit_values)

        w = grads - grads[orig_label]
        f = logit_values - logit_values[orig_label]

        w[orig_label] = np.inf  # ignore the true class itself
        norms = np.linalg.norm(w, axis=1) + 1e-8
        ratios = np.abs(f) / norms
        ratios[orig_label] = np.inf

        target = int(np.argmin(ratios))
        r_i = (np.abs(f[target]) / (norms[target] ** 2)) * w[target]
        x_adv = x_adv + (1 + overshoot) * r_i.reshape(x_shape)
        x_adv = clip01(x_adv)

        new_pred = gradient_source(torch.tensor(x_adv, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item()
        if new_pred != orig_label:
            break

    return x_adv


def deepfool_attack(
    gradient_source: nn.Module,
    X: np.ndarray,
    num_classes: int,
    max_iter: int = 25,
    overshoot: float = 0.02,
) -> np.ndarray:
    """Simplified DeepFool (Moosavi-Dezfooli et al., 2016).

    Iteratively pushes each point toward the nearest linearized decision
    boundary — finds a (locally) minimal perturbation that flips the
    prediction, rather than taking a fixed epsilon step like FGSM/PGD. Run
    per-sample since each one takes a different path to the boundary.
    """
    gradient_source.eval()
    out = np.zeros_like(X)
    for i in range(len(X)):
        out[i] = _deepfool_single(gradient_source, X[i], num_classes, max_iter, overshoot)
    return out.astype(np.float32)
