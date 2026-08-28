import numpy as np
import pytest

from app.attacks.poisoning import apply_trigger_batch, inject_backdoor, label_flip_poison


def _toy_image_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 8, 8)).astype(np.float32)
    y = rng.integers(0, 10, size=n)
    return X, y


def _toy_tabular_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    return X, y


def test_label_flip_changes_expected_fraction():
    X, y = _toy_image_data(n=100)
    X_out, y_poisoned, poisoned_idx = label_flip_poison(X, y, poison_fraction=0.2, num_classes=10, seed=1)

    assert X_out is X  # features untouched
    assert len(poisoned_idx) == 20
    # every poisoned label must differ from the original
    assert np.all(y_poisoned[poisoned_idx] != y[poisoned_idx])
    # every non-poisoned label must be unchanged
    unpoisoned_mask = np.ones(len(y), dtype=bool)
    unpoisoned_mask[poisoned_idx] = False
    assert np.array_equal(y_poisoned[unpoisoned_mask], y[unpoisoned_mask])


def test_label_flip_rejects_invalid_fraction():
    X, y = _toy_tabular_data()
    with pytest.raises(ValueError):
        label_flip_poison(X, y, poison_fraction=1.5, num_classes=2, seed=0)
    with pytest.raises(ValueError):
        label_flip_poison(X, y, poison_fraction=-0.1, num_classes=2, seed=0)


def test_label_flip_zero_fraction_is_a_no_op():
    X, y = _toy_tabular_data()
    _, y_poisoned, poisoned_idx = label_flip_poison(X, y, poison_fraction=0.0, num_classes=2, seed=0)
    assert len(poisoned_idx) == 0
    assert np.array_equal(y_poisoned, y)


def test_backdoor_relabels_poisoned_samples_to_target():
    X, y = _toy_image_data(n=100)
    X_poisoned, y_poisoned, poisoned_idx = inject_backdoor(
        X, y, poison_fraction=0.1, target_label=3, is_image=True, seed=2
    )
    assert len(poisoned_idx) == 10
    assert np.all(y_poisoned[poisoned_idx] == 3)
    # trigger pattern (bottom-right 2x2 block) should be stamped at max value
    for idx in poisoned_idx:
        assert np.allclose(X_poisoned[idx][-2:, -2:], 1.0)


def test_backdoor_tabular_stamps_last_two_features():
    X, y = _toy_tabular_data(n=60)
    X_poisoned, y_poisoned, poisoned_idx = inject_backdoor(
        X, y, poison_fraction=0.15, target_label=1, is_image=False, seed=3
    )
    for idx in poisoned_idx:
        assert np.allclose(X_poisoned[idx][-2:], 0.98)
    assert np.all(y_poisoned[poisoned_idx] == 1)


def test_apply_trigger_batch_stamps_every_sample():
    X, _ = _toy_image_data(n=10)
    triggered = apply_trigger_batch(X, is_image=True)
    assert triggered.shape == X.shape
    for i in range(len(triggered)):
        assert np.allclose(triggered[i][-2:, -2:], 1.0)
    # original batch must be untouched (apply_trigger_batch copies)
    assert not np.allclose(X[0][-2:, -2:], 1.0)
