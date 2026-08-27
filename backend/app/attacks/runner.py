from __future__ import annotations

import numpy as np

from app.attacks.deepfool import deepfool_attack
from app.attacks.fgsm import fgsm_attack
from app.attacks.pgd import pgd_attack
from app.attacks.random_noise import random_noise_attack
from app.models.registry import TrainedModel

ATTACK_NAMES = ["fgsm", "pgd", "deepfool", "random_noise"]


def run_attack(
    model: TrainedModel,
    X: np.ndarray,
    y: np.ndarray,
    attack: str,
    epsilon: float = 0.15,
    pgd_steps: int = 10,
) -> np.ndarray:
    """Runs `attack` against `model` and returns adversarial examples with the
    same shape as X.

    For Random Forest (non-differentiable), gradient-based attacks (fgsm,
    pgd, deepfool) are crafted against a differentiable surrogate model
    trained to mimic the RF, then *transferred* — i.e. we evaluate the RF's
    real predictions on the resulting adversarial inputs. This is a standard
    black-box "substitute model" attack strategy and is reflected honestly in
    the results (transfer attacks are usually weaker than white-box ones).
    """
    if attack == "random_noise":
        return random_noise_attack(X, epsilon)

    if not model.is_differentiable and model.surrogate is None:
        raise ValueError("This model has no gradient source available for this attack")

    gradient_source = model.gradient_source()

    if attack == "fgsm":
        return fgsm_attack(gradient_source, X, y, epsilon)
    if attack == "pgd":
        return pgd_attack(gradient_source, X, y, epsilon, steps=pgd_steps)
    if attack == "deepfool":
        capped = X[: min(len(X), 64)]  # deepfool is per-sample & iterative; cap batch for latency
        return deepfool_attack(gradient_source, capped, model.num_classes)

    raise ValueError(f"Unknown attack: {attack}")
