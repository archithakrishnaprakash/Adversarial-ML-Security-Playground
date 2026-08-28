from __future__ import annotations

import numpy as np

from app.attacks.runner import run_attack
from app.evaluation.norms import confidence_shift, perturbation_norms
from app.models.registry import TrainedModel


def evaluate_robustness(
    model: TrainedModel,
    attacks: list[str],
    epsilon: float = 0.15,
    n_samples: int | None = None,
    seed: int = 0,
) -> dict:
    """Runs each attack in `attacks` over (a subsample of) the model's held-out
    test set and reports clean vs. adversarial accuracy, attack success rate,
    and an overall 0-100 robustness score.
    """
    rng = np.random.default_rng(seed)
    X, y = model.X_test, model.y_test
    if n_samples is not None and n_samples < len(X):
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X, y = X[idx], y[idx]

    clean_preds = model.predict(X)
    clean_acc = float((clean_preds == y).mean())

    per_attack = {}
    accuracies = [clean_acc]
    for attack in attacks:
        try:
            if attack == "deepfool":
                cap = min(len(X), 64)
                Xa, ya = X[:cap], y[:cap]
            else:
                Xa, ya = X, y
            X_adv = run_attack(model, Xa, ya, attack, epsilon=epsilon)
            adv_preds = model.predict(X_adv)
            adv_acc = float((adv_preds == ya).mean())
            baseline_preds = clean_preds[: len(ya)]
            success_rate = float((adv_preds != baseline_preds).mean())
            avg_perturbation = float(np.mean(np.abs(X_adv - Xa)))

            # L0 / L2 / Linf perturbation norms and confidence shift, the
            # standard adversarial-ML literature metrics — additive fields
            # alongside avg_perturbation, not a replacement for it.
            norms = perturbation_norms(Xa, X_adv)
            clean_probs_a = model.predict_proba(Xa)
            adv_probs_a = model.predict_proba(X_adv)
            conf_shift = confidence_shift(clean_probs_a, adv_probs_a, ya)

            per_attack[attack] = {
                "accuracy": round(adv_acc, 4),
                "attack_success_rate": round(success_rate, 4),
                "avg_perturbation": round(avg_perturbation, 4),
                "n_samples": int(len(ya)),
                "l0_norm": norms["l0"],
                "l2_norm": norms["l2"],
                "linf_norm": norms["linf"],
                "confidence_shift": conf_shift,
            }
            accuracies.append(adv_acc)
        except Exception as e:  # keep evaluating remaining attacks even if one fails
            per_attack[attack] = {"error": str(e)}

    # robustness score: weighted average of clean accuracy (small weight,
    # a model must still work) and how well accuracy holds up under attack
    # relative to clean accuracy (larger weight). Scaled to 0-100.
    successful = [v["accuracy"] for v in per_attack.values() if "accuracy" in v]
    if successful:
        avg_adv_acc = sum(successful) / len(successful)
        retention = avg_adv_acc / clean_acc if clean_acc > 0 else 0.0
        score = 0.3 * clean_acc + 0.7 * retention
    else:
        score = clean_acc
    robustness_score = round(min(max(score, 0.0), 1.0) * 100, 1)

    return {
        "model_id": model.model_id,
        "model_type": model.model_type,
        "dataset": model.dataset,
        "clean_accuracy": round(clean_acc, 4),
        "n_samples_evaluated": int(len(X)),
        "epsilon": epsilon,
        "attacks": per_attack,
        "robustness_score": robustness_score,
    }
