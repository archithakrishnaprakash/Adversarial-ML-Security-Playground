"""
Image dataset loader.

We use scikit-learn's bundled `load_digits` dataset instead of downloading
real MNIST. It ships with scikit-learn (no internet access required), and is
conceptually identical for this project's purposes: small greyscale
handwritten digit images, 10 classes (0-9). Each image is 8x8 pixels instead
of MNIST's 28x28, which also keeps training fast enough to run live from a
web UI.

If you want to swap in real 28x28 MNIST later, replace `load_image_dataset`
with a torchvision.datasets.MNIST loader — the rest of the pipeline
(attacks, defenses, evaluation) is written to be shape-agnostic.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

IMAGE_SHAPE = (8, 8)
NUM_CLASSES = 10


def load_image_dataset(test_size: float = 0.2, random_state: int = 42):
    """Returns X_train, X_test, y_train, y_test.

    X arrays are float32, shape (N, 8, 8), scaled to [0, 1].
    y arrays are int64 class labels 0-9.
    """
    digits = load_digits()
    X = digits.images.astype(np.float32) / 16.0  # raw pixel range is 0-16
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
