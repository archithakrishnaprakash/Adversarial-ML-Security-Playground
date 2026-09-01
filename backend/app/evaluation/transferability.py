from __future__ import annotations

import numpy as np

from app.attacks.runner import run_attack
from app.models.registry import TrainedModel


def compute_transferability_matrix(
    models: list[TrainedModel],
    attack: str = "fgsm",
    epsilon: float = 0.15,
    n_samples: int = 100,
    seed: int = 0,
) -> dict:
    """For every pair of models trained on the *same* dataset, generates
    adversarial examples against the source model and measures how often
    they also fool the target model — the classic transferability question:
    "do adversarial examples crafted for model A also break model B?"

    Only models sharing a dataset are compared (an adversarial digit image
    can't meaningfully be "transferred" to a network-intrusion classifier).
    Models from a different dataset than the majority are reported as
    skipped rather than silently dropped, so the caller can see why the
    matrix isn't square.
    """
    if len(models) < 2:
        raise ValueError("Need at least 2 trained models to compute transferability")

    dataset = models[0].dataset
    comparable = [m for m in models if m.dataset == dataset]
    skipped = [m.model_id for m in models if m.dataset != dataset]

    # use a shared sample set (from the first comparable model's test split)
    # so every source model attacks the *same* inputs — otherwise the matrix
    # would conflate "different model" with "different sample"
    X_test, y_test = comparable[0].X_test, comparable[0].y_test
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X, y = X_test[idx], y_test[idx]

    matrix = {}
    errors = {}
    for source in comparable:
        row = {}
        if not source.supports_gradient_attacks and attack != "random_noise":
            errors[source.model_id] = "No gradient source available for this model"
            continue
        try:
            X_adv = run_attack(source, X, y, attack, epsilon=epsilon)
        except Exception as e:
            errors[source.model_id] = str(e)
            continue

        y_eval = y[: len(X_adv)]
        for target in comparable:
            target_preds = target.predict(X_adv)
            success_rate = float((target_preds != y_eval).mean())
            row[target.model_id] = round(success_rate, 4)
        matrix[source.model_id] = row

    return {
        "dataset": dataset,
        "attack": attack,
        "epsilon": epsilon,
        "n_samples": n,
        "model_labels": {m.model_id: f"{m.model_type} #{m.model_id}" for m in comparable},
        "matrix": matrix,
        "errors": errors,
        "skipped_models_different_dataset": skipped,
    }
