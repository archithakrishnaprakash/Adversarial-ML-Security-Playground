"""
Black-box attacks.

Unlike FGSM/PGD/DeepFool, these never touch a gradient — they only call
`model.predict_proba`, exactly the way a real attacker who only has access to
a prediction API would operate. Two strategies are implemented:

- `transfer_attack`: query the target a limited number of times to train a
  differentiable substitute ("surrogate") model, craft a white-box attack
  (FGSM/PGD) against the surrogate, then transfer the result to the real
  target. Query-efficient (one query per training point used) but only as
  good as the surrogate's fidelity.
- `query_attack`: a zeroth-order random-search / hill-climbing attack that
  queries the target directly for every candidate perturbation. No surrogate
  needed, but far more query-hungry — this cost is exactly the point, and is
  reported back as `queries_used` so it can be compared against `transfer_attack`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.attacks.common import clip01
from app.attacks.fgsm import fgsm_attack
from app.attacks.pgd import pgd_attack
from app.models.torch_models import SmallNN

DEVICE = torch.device("cpu")


def train_query_surrogate(
    target_predict_proba,
    X_pool: np.ndarray,
    num_classes: int,
    in_features: int,
    n_queries: int = 200,
    seed: int = 0,
) -> tuple[nn.Module, int]:
    """Builds a substitute model by querying the target's predict_proba on a
    sample of `n_queries` points drawn from X_pool (representing data an
    attacker could plausibly collect — e.g. public traffic samples — without
    ever seeing the target's true training labels). Returns (surrogate, queries_used).
    """
    rng = np.random.default_rng(seed)
    n_queries = min(n_queries, len(X_pool))
    idx = rng.choice(len(X_pool), size=n_queries, replace=False)
    X_query = X_pool[idx]

    soft_labels = target_predict_proba(X_query)  # this IS the "query" — one call per point
    surrogate = SmallNN(in_features=in_features, num_classes=num_classes, hidden=64).to(DEVICE)
    opt = optim.Adam(surrogate.parameters(), lr=1e-2)
    Xt = torch.tensor(X_query, dtype=torch.float32, device=DEVICE)
    target = torch.tensor(soft_labels, dtype=torch.float32, device=DEVICE)

    surrogate.train()
    for _ in range(60):
        opt.zero_grad()
        logits = surrogate(Xt)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(target * log_probs).sum(dim=1).mean()
        loss.backward()
        opt.step()

    return surrogate, n_queries


def transfer_attack(
    target_predict_proba,
    X_pool: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epsilon: float,
    base_attack: str = "fgsm",
    n_queries: int = 200,
    pgd_steps: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Trains a query-limited surrogate, crafts `base_attack` (fgsm or pgd)
    against it, and returns the resulting adversarial inputs (meant to be
    evaluated against the *real* target afterward) plus the number of queries
    spent building the surrogate.
    """
    in_features = int(np.prod(X_pool.shape[1:]))
    surrogate, queries_used = train_query_surrogate(
        target_predict_proba, X_pool, num_classes, in_features, n_queries, seed
    )
    if base_attack == "pgd":
        X_adv = pgd_attack(surrogate, X, y, epsilon, steps=pgd_steps)
    else:
        X_adv = fgsm_attack(surrogate, X, y, epsilon)
    return X_adv, queries_used


def query_attack(
    target_predict_proba,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    max_queries_per_sample: int = 150,
    step_size: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Zeroth-order random-search attack: repeatedly try a random perturbation
    direction, query the target, and keep the step only if it reduced the
    target's confidence in the true class. Stops early per-sample once the
    prediction flips. No gradients, no surrogate — just the prediction API,
    which is the whole point of a pure black-box baseline.
    """
    rng = np.random.default_rng(seed)
    X_adv = X.copy()
    total_queries = 0

    for i in range(len(X_adv)):
        x = X_adv[i].copy()
        true_label = int(y[i])
        best_conf = float(target_predict_proba(x[None, ...])[0, true_label])
        total_queries += 1

        for _ in range(max_queries_per_sample):
            direction = rng.uniform(-1, 1, size=x.shape)
            candidate = clip01(x + step_size * direction)
            probs = target_predict_proba(candidate[None, ...])[0]
            total_queries += 1

            conf = float(probs[true_label])
            if conf < best_conf:
                x = candidate
                best_conf = conf
                if int(probs.argmax()) != true_label:
                    break  # prediction already flipped, no need to keep querying

        # keep the total perturbation within the epsilon budget, same contract
        # as the gradient-based attacks
        perturbation = np.clip(x - X[i], -epsilon, epsilon)
        X_adv[i] = clip01(X[i] + perturbation)

    return X_adv.astype(np.float32), total_queries
