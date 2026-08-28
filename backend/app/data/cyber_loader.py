"""
Synthetic network-intrusion dataset.

Real intrusion-detection datasets (NSL-KDD, CIC-IDS) require downloading
large files from external hosts, which this sandboxed environment can't
reach. Instead we generate a synthetic-but-realistic tabular dataset with
the same *shape* of problem: a handful of continuous/behavioural features,
two classes (BENIGN / ATTACK), and class-conditional feature distributions
that a classifier can actually learn and that an attacker can plausibly
perturb.

Swap `load_cyber_dataset` for a real CSV loader (e.g. pandas.read_csv on a
downloaded NSL-KDD file) if you want real data later — everything downstream
only assumes a 2D float feature matrix and a binary label vector.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "duration_sec",
    "src_bytes",
    "dst_bytes",
    "packet_rate",
    "failed_logins",
    "transaction_frequency",
    "amount_transferred",
    "location_risk_score",
    "port_entropy",
    "protocol_anomaly_score",
]
NUM_CLASSES = 2
CLASS_NAMES = ["BENIGN", "ATTACK"]


def load_cyber_dataset(
    n_samples: int = 4000, test_size: float = 0.2, random_state: int = 42
):
    """Returns X_train, X_test, y_train, y_test.

    X: float32 (N, 10) features, roughly standardized to [0, 1] range.
    y: int64 labels, 0 = BENIGN, 1 = ATTACK.
    """
    rng = np.random.default_rng(random_state)
    n_benign = n_samples // 2
    n_attack = n_samples - n_benign

    # BENIGN traffic: low variance, "normal" ranges
    benign = np.stack(
        [
            rng.gamma(2.0, 1.5, n_benign),       # duration_sec
            rng.gamma(2.0, 200.0, n_benign),     # src_bytes
            rng.gamma(2.0, 220.0, n_benign),     # dst_bytes
            rng.normal(20, 5, n_benign).clip(0),  # packet_rate
            rng.poisson(0.1, n_benign),          # failed_logins
            rng.normal(5, 2, n_benign).clip(0),  # transaction_frequency
            rng.gamma(2.0, 50.0, n_benign),      # amount_transferred
            rng.beta(1.5, 6.0, n_benign),        # location_risk_score
            rng.normal(0.3, 0.1, n_benign).clip(0, 1),  # port_entropy
            rng.beta(1.5, 6.0, n_benign),        # protocol_anomaly_score
        ],
        axis=1,
    )

    # ATTACK traffic: heavier tails, more failed logins, higher risk scores
    attack = np.stack(
        [
            rng.gamma(1.2, 4.0, n_attack),
            rng.gamma(1.5, 900.0, n_attack),
            rng.gamma(1.2, 120.0, n_attack),
            rng.normal(80, 25, n_attack).clip(0),
            rng.poisson(3.5, n_attack),
            rng.normal(18, 6, n_attack).clip(0),
            rng.gamma(1.2, 300.0, n_attack),
            rng.beta(5.0, 2.0, n_attack),
            rng.normal(0.75, 0.15, n_attack).clip(0, 1),
            rng.beta(5.0, 2.0, n_attack),
        ],
        axis=1,
    )

    X = np.concatenate([benign, attack], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n_benign), np.ones(n_attack)]).astype(np.int64)

    # min-max scale each feature to [0, 1] so gradient-based attacks behave
    # consistently regardless of a feature's raw units
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
