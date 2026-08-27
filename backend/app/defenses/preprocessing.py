from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_smoothing(X: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """Blurs each image slightly to wash out high-frequency adversarial
    perturbations. Only meaningful for image inputs (expects the last two
    dims to be spatial)."""
    if X.ndim < 3:
        return X  # not an image batch, no-op
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = gaussian_filter(X[i], sigma=sigma)
    return out


def feature_clipping(X: np.ndarray, low: float = 0.05, high: float = 0.95) -> np.ndarray:
    """Clips extreme feature/pixel values toward the bulk of the distribution.
    Adversarial perturbations often push values toward the edges of the valid
    range, so clipping removes some of the attack's headroom."""
    return np.clip(X, low, high)


def normalization(X: np.ndarray) -> np.ndarray:
    """Re-normalizes each sample to zero mean / unit variance, then rescales
    back to roughly [0, 1]. Removes some of the fine-grained structure an
    attack relies on."""
    flat = X.reshape(len(X), -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-8
    normed = (flat - mean) / std
    # rescale to [0, 1] per-sample so the model still sees a familiar range
    normed = (normed - normed.min(axis=1, keepdims=True)) / (
        normed.max(axis=1, keepdims=True) - normed.min(axis=1, keepdims=True) + 1e-8
    )
    return normed.reshape(X.shape)


PREPROCESSING_DEFENSES = {
    "gaussian_smoothing": gaussian_smoothing,
    "feature_clipping": feature_clipping,
    "normalization": normalization,
}


def apply_preprocessing_defense(name: str, X: np.ndarray, **kwargs) -> np.ndarray:
    if name not in PREPROCESSING_DEFENSES:
        raise ValueError(f"Unknown preprocessing defense: {name}")
    return PREPROCESSING_DEFENSES[name](X, **kwargs)
