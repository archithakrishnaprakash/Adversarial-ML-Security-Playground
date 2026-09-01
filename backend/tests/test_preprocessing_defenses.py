import numpy as np

from app.defenses.preprocessing import (
    apply_preprocessing_defense,
    feature_clipping,
    gaussian_smoothing,
    normalization,
)


def test_gaussian_smoothing_preserves_shape():
    X = np.random.default_rng(0).random((4, 8, 8)).astype(np.float32)
    out = gaussian_smoothing(X)
    assert out.shape == X.shape


def test_gaussian_smoothing_is_no_op_on_non_image_input():
    X = np.random.default_rng(0).random((4, 10)).astype(np.float32)
    out = gaussian_smoothing(X)
    assert np.array_equal(out, X)


def test_feature_clipping_bounds_values():
    X = np.array([[-1.0, 0.5, 2.0]])
    out = feature_clipping(X, low=0.0, high=1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalization_output_in_unit_range():
    X = np.random.default_rng(1).normal(5, 2, size=(3, 10)).astype(np.float32)
    out = normalization(X)
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6


def test_apply_preprocessing_defense_dispatches_correctly():
    X = np.random.default_rng(2).random((3, 8, 8)).astype(np.float32)
    for name in ("gaussian_smoothing", "feature_clipping", "normalization"):
        out = apply_preprocessing_defense(name, X)
        assert out.shape == X.shape


def test_apply_preprocessing_defense_rejects_unknown_name():
    X = np.zeros((2, 4))
    try:
        apply_preprocessing_defense("not_a_real_defense", X)
        assert False, "expected ValueError"
    except ValueError:
        pass
