"""
Perturbation-size metrics.

`avg_perturbation` (mean absolute difference) has been reported since the
first version of the robustness evaluator. These three L_p norms are the
standard, more precise way the adversarial-ML literature actually reports
perturbation size, so they're added alongside it rather than replacing it —
existing callers that only read `avg_perturbation` are unaffected.
"""
from __future__ import annotations

import numpy as np


def perturbation_norms(X: np.ndarray, X_adv: np.ndarray, l0_threshold: float = 1e-4) -> dict:
    """Per-batch average of the L0 (fraction of entries changed beyond
    `l0_threshold`), L1 (mean absolute perturbation, summed per sample), L2
    (Euclidean norm per sample), and L-infinity (max absolute change per
    sample) perturbation norms — the standard set reported in the
    adversarial-ML literature, all averaged over the batch.
    """
    diff = X_adv.reshape(len(X_adv), -1) - X.reshape(len(X), -1)
    l0 = float((np.abs(diff) > l0_threshold).mean(axis=1).mean())
    l1 = float(np.linalg.norm(diff, ord=1, axis=1).mean())
    l2 = float(np.linalg.norm(diff, ord=2, axis=1).mean())
    linf = float(np.abs(diff).max(axis=1).mean())
    return {"l0": round(l0, 4), "l1": round(l1, 4), "l2": round(l2, 4), "linf": round(linf, 4)}


def confidence_shift(clean_probs: np.ndarray, adv_probs: np.ndarray, y: np.ndarray) -> float:
    """Average drop in the model's confidence assigned to the *true* class,
    clean vs. adversarial. Positive means the attack eroded confidence in the
    correct answer (even on samples where the top-1 prediction didn't flip)."""
    idx = np.arange(len(y))
    clean_true_conf = clean_probs[idx, y]
    adv_true_conf = adv_probs[idx, y]
    return round(float((clean_true_conf - adv_true_conf).mean()), 4)
