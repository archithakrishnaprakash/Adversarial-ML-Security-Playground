from __future__ import annotations

import numpy as np

from app.attacks.common import clip01


def random_noise_attack(X: np.ndarray, epsilon: float, seed: int | None = None) -> np.ndarray:
    """Uniform random noise in [-epsilon, epsilon], applied with no knowledge
    of the model at all. This is the baseline every real attack should beat —
    if FGSM/PGD don't meaningfully outperform this, the "attack" isn't doing
    anything smarter than noise.
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-epsilon, epsilon, size=X.shape)
    return clip01(X + noise).astype(np.float32)
