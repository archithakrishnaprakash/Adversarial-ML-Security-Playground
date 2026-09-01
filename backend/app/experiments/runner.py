"""
Automated Experiment Engine.

Runs a full grid of (model type x attack x epsilon x defense) configurations
in one call instead of clicking through Attack Lab / Defense Lab one
combination at a time — the same idea as a hyperparameter sweep, applied to
robustness evaluation. Each configuration is logged as its own record (via
`app.storage.log_experiment`) so results are individually reproducible later,
not just summarized away.

The grid is intentionally capped (`MAX_CONFIGURATIONS`) — this runs
synchronously inside a single API request with no background job queue, so an
unbounded grid (e.g. 4 models x 5 attacks x 10 epsilons x 5 defenses = 1000
runs) would simply time out the request. Ask for a smaller grid instead, or
call the API repeatedly with different slices.
"""
from __future__ import annotations

import numpy as np

from app.attacks.runner import run_attack
from app.defenses.preprocessing import apply_preprocessing_defense
from app.models.registry import fit_and_register, get_data
from app.storage import log_experiment

MAX_CONFIGURATIONS = 60
DEFENSE_CHOICES = ["none", "gaussian_smoothing", "feature_clipping", "normalization"]


def run_experiment_grid(
    dataset: str,
    model_types: list[str],
    attacks: list[str],
    epsilons: list[float],
    defenses: list[str] | None = None,
    n_samples: int = 150,
    seed: int = 0,
) -> dict:
    defenses = defenses or ["none"]
    n_configs = len(model_types) * len(attacks) * len(epsilons) * len(defenses)
    if n_configs > MAX_CONFIGURATIONS:
        raise ValueError(
            f"Grid has {n_configs} configurations, which exceeds the cap of "
            f"{MAX_CONFIGURATIONS} for a single synchronous run. Reduce the number of "
            f"models/attacks/epsilons/defenses, or split the grid across multiple calls."
        )
    for d in defenses:
        if d not in DEFENSE_CHOICES:
            raise ValueError(f"Unknown defense '{d}'. Must be one of {DEFENSE_CHOICES}")

    X_train, X_test, y_train, y_test = get_data(dataset)
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X, y = X_test[idx], y_test[idx]

    records = []
    trained_ids = {}
    for model_type in model_types:
        try:
            model = fit_and_register(dataset, model_type, X_train, y_train, X_test, y_test)
        except ValueError as e:
            records.append({"model_type": model_type, "error": str(e)})
            continue
        trained_ids[model_type] = model.model_id
        clean_acc = float((model.predict(X) == y).mean())

        for attack in attacks:
            for epsilon in epsilons:
                try:
                    Xa, ya = (X[:64], y[:64]) if attack == "deepfool" else (X, y)
                    X_adv = run_attack(model, Xa, ya, attack, epsilon=epsilon)
                    adv_acc = float((model.predict(X_adv) == ya).mean())
                except Exception as e:
                    for defense in defenses:
                        records.append(
                            {
                                "model_type": model_type,
                                "model_id": model.model_id,
                                "attack": attack,
                                "epsilon": epsilon,
                                "defense": defense,
                                "error": str(e),
                            }
                        )
                    continue

                for defense in defenses:
                    if defense == "none":
                        defended_acc = adv_acc
                    else:
                        X_defended = apply_preprocessing_defense(defense, X_adv)
                        defended_acc = float((model.predict(X_defended) == ya).mean())

                    record = {
                        "model_type": model_type,
                        "model_id": model.model_id,
                        "attack": attack,
                        "epsilon": epsilon,
                        "defense": defense,
                        "clean_accuracy": round(clean_acc, 4),
                        "adversarial_accuracy": round(adv_acc, 4),
                        "defended_accuracy": round(defended_acc, 4),
                        "n_samples": int(len(ya)),
                    }
                    records.append(record)
                    log_experiment(
                        "grid_experiment",
                        dataset,
                        model_type,
                        model.model_id,
                        record,
                        attack=attack,
                        epsilon=epsilon,
                    )

    return {
        "dataset": dataset,
        "n_configurations_run": len(records),
        "trained_model_ids": trained_ids,
        "records": records,
    }
