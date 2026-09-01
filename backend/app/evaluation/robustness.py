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
    test set and reports the standard adversarial-ML metrics per attack
    (robust accuracy, attack success rate, L1/L2/Linf perturbation norms,
    confidence shift), plus a single composite score for at-a-glance
    comparison — see the `aegis_robustness_index` note in the returned dict
    for what that composite score is and, just as importantly, isn't.
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

            # L1 / L2 / Linf perturbation norms and confidence shift — the
            # standard metrics reported in the adversarial-ML literature,
            # additive alongside avg_perturbation, not a replacement for it.
            norms = perturbation_norms(Xa, X_adv)
            clean_probs_a = model.predict_proba(Xa)
            adv_probs_a = model.predict_proba(X_adv)
            conf_shift = confidence_shift(clean_probs_a, adv_probs_a, ya)

            per_attack[attack] = {
                # "accuracy" here is what the literature calls robust accuracy:
                # accuracy measured *on the adversarial inputs*.
                "accuracy": round(adv_acc, 4),
                "accuracy_degradation": round(clean_acc - adv_acc, 4),
                "attack_success_rate": round(success_rate, 4),
                "avg_perturbation": round(avg_perturbation, 4),
                "n_samples": int(len(ya)),
                "l0_norm": norms["l0"],
                "l1_norm": norms["l1"],
                "l2_norm": norms["l2"],
                "linf_norm": norms["linf"],
                "confidence_shift": conf_shift,
            }
            accuracies.append(adv_acc)
        except Exception as e:  # keep evaluating remaining attacks even if one fails
            per_attack[attack] = {"error": str(e)}

    # ---- composite score ----
    # This is a *project-defined* metric, not a standardized industry
    # benchmark — it exists for at-a-glance comparison (the Leaderboard, the
    # Security Assessment risk rating) and should not be treated as more
    # rigorous than it is. For actual evaluation, use the per-attack standard
    # metrics above (robust accuracy, ASR, L1/L2/Linf, confidence shift).
    # Formula: a weighted blend of clean accuracy (a model that doesn't work
    # on clean data isn't "robust", it's just broken) and *retention* —
    # how much of that clean accuracy survives the average attack.
    successful = [v["accuracy"] for v in per_attack.values() if "accuracy" in v]
    mean_robust_accuracy = sum(successful) / len(successful) if successful else None
    if successful:
        retention = mean_robust_accuracy / clean_acc if clean_acc > 0 else 0.0
        score = 0.3 * clean_acc + 0.7 * retention
    else:
        score = clean_acc
    aegis_robustness_index = round(min(max(score, 0.0), 1.0) * 100, 1)

    return {
        "model_id": model.model_id,
        "model_type": model.model_type,
        "dataset": model.dataset,
        "clean_accuracy": round(clean_acc, 4),
        "mean_robust_accuracy": round(mean_robust_accuracy, 4) if mean_robust_accuracy is not None else None,
        "mean_accuracy_degradation": (
            round(clean_acc - mean_robust_accuracy, 4) if mean_robust_accuracy is not None else None
        ),
        "n_samples_evaluated": int(len(X)),
        "epsilon": epsilon,
        "attacks": per_attack,
        # kept for backward compatibility — identical value to
        # aegis_robustness_index below. New code should prefer the latter,
        # which makes the "this is a composite project metric" framing explicit.
        "robustness_score": aegis_robustness_index,
        "aegis_robustness_index": aegis_robustness_index,
        "aegis_robustness_index_note": (
            "Composite project metric (0.3 x clean accuracy + 0.7 x accuracy retention "
            "under attack), not a standardized industry benchmark. Use the per-attack "
            "standard metrics (robust accuracy, attack success rate, L1/L2/Linf, "
            "confidence shift) for rigorous evaluation."
        ),
    }
