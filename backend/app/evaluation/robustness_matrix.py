"""
Robustness Matrix: the "systematic benchmark" view, as opposed to the
Attack Lab's one-model-one-attack-at-a-time exploration. Trains every
requested model architecture on the same dataset/split and evaluates every
requested attack at a single epsilon, producing the classic

                    FGSM      PGD      DeepFool
Logistic Regression  71.2%    63.4%      58.2%
Random Forest        79.1%    70.3%      66.8%
...

table. This is deliberately a thin wrapper around `evaluate_robustness` per
model (same metrics, same attack dispatch) rather than a separate
implementation — one row is one model's full robustness benchmark, just
reshaped for side-by-side comparison instead of read one at a time.
"""
from __future__ import annotations

from app.evaluation.robustness import evaluate_robustness
from app.models.registry import fit_and_register, get_data

MAX_MODELS = 6
MAX_ATTACKS = 6


def compute_robustness_matrix(
    dataset: str,
    model_types: list[str],
    attacks: list[str],
    epsilon: float = 0.15,
    n_samples: int = 200,
    seed: int = 0,
) -> dict:
    if len(model_types) == 0 or len(attacks) == 0:
        raise ValueError("Need at least one model type and one attack")
    if len(model_types) > MAX_MODELS:
        raise ValueError(f"Too many model types ({len(model_types)}); max is {MAX_MODELS}")
    if len(attacks) > MAX_ATTACKS:
        raise ValueError(f"Too many attacks ({len(attacks)}); max is {MAX_ATTACKS}")

    X_train, X_test, y_train, y_test = get_data(dataset)

    matrix: dict[str, dict[str, float | None]] = {}
    clean_accuracy: dict[str, float] = {}
    model_ids: dict[str, str] = {}
    errors: dict[str, str] = {}

    for model_type in model_types:
        try:
            model = fit_and_register(dataset, model_type, X_train, y_train, X_test, y_test)
        except ValueError as e:
            errors[model_type] = str(e)
            continue

        model_ids[model_type] = model.model_id
        result = evaluate_robustness(model, attacks, epsilon=epsilon, n_samples=n_samples, seed=seed)
        clean_accuracy[model_type] = result["clean_accuracy"]

        row = {}
        for attack in attacks:
            attack_result = result["attacks"].get(attack, {})
            row[attack] = attack_result.get("accuracy")  # None if the attack errored
        matrix[model_type] = row

    return {
        "dataset": dataset,
        "epsilon": epsilon,
        "n_samples": n_samples,
        "model_types": [m for m in model_types if m in matrix],
        "attacks": attacks,
        "matrix": matrix,
        "clean_accuracy": clean_accuracy,
        "model_ids": model_ids,
        "errors": errors,
    }
