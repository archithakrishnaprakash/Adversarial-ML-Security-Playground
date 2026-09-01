"""
Training-time (poisoning) attacks.

Everything else in this project attacks a model *after* it's trained
(evasion). Poisoning attacks tamper with the training data itself, so the
model learns something wrong from the start. Two are implemented:

- `label_flip_poison`: relabels a fraction of training samples to an
  incorrect class. Simple, doesn't require touching features at all, and
  still measurably degrades accuracy — a good illustration of why training
  pipeline integrity matters.
- `inject_backdoor`: stamps a fixed, recognizable "trigger" pattern onto a
  fraction of training samples and relabels *only those* to a chosen target
  class. A model trained on this data learns to associate the trigger with
  the target label, while behaving normally on clean inputs — the trigger
  acts like a hidden switch the attacker can flip at inference time.
"""
from __future__ import annotations

import numpy as np


def label_flip_poison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    poison_fraction: float,
    num_classes: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly selects `poison_fraction` of the training set and relabels
    each to a different, randomly chosen class. Returns
    (X_train, y_poisoned, poisoned_indices) — X_train is returned unchanged
    since label flipping never touches features.
    """
    if not 0.0 <= poison_fraction <= 1.0:
        raise ValueError("poison_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    n_poison = int(round(len(y_train) * poison_fraction))
    poisoned_idx = rng.choice(len(y_train), size=n_poison, replace=False)

    y_poisoned = y_train.copy()
    for idx in poisoned_idx:
        original = y_poisoned[idx]
        # pick any other class uniformly at random
        choices = [c for c in range(num_classes) if c != original]
        y_poisoned[idx] = rng.choice(choices)

    return X_train, y_poisoned, poisoned_idx


def _apply_trigger(x: np.ndarray, is_image: bool) -> np.ndarray:
    """Stamps a small, fixed, easily-recognizable pattern onto a single
    sample. For images: a bright 2x2 square in the bottom-right corner. For
    tabular data: pins the last two features to a fixed extreme value —
    playing the role of an unusual-but-plausible combination of flags an
    attacker fully controls (e.g. a crafted packet field)."""
    x = x.copy()
    if is_image:
        x[-2:, -2:] = 1.0
    else:
        x[-2:] = 0.98
    return x


def inject_backdoor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    poison_fraction: float,
    target_label: int,
    is_image: bool,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stamps the trigger pattern onto `poison_fraction` of the training set
    and relabels those samples to `target_label`. Returns
    (X_poisoned, y_poisoned, poisoned_indices).
    """
    if not 0.0 <= poison_fraction <= 1.0:
        raise ValueError("poison_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    n_poison = int(round(len(y_train) * poison_fraction))
    poisoned_idx = rng.choice(len(y_train), size=n_poison, replace=False)

    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()
    for idx in poisoned_idx:
        X_poisoned[idx] = _apply_trigger(X_poisoned[idx], is_image)
        y_poisoned[idx] = target_label

    return X_poisoned, y_poisoned, poisoned_idx


def apply_trigger_batch(X: np.ndarray, is_image: bool) -> np.ndarray:
    """Stamps the trigger onto every sample in a batch — used at evaluation
    time to measure backdoor *success rate*: what fraction of clean test
    samples get pushed to the target label once the trigger is present."""
    out = X.copy()
    for i in range(len(out)):
        out[i] = _apply_trigger(out[i], is_image)
    return out
