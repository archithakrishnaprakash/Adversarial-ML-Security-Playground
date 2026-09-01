import numpy as np
import pytest

from app.evaluation.norms import confidence_shift, perturbation_norms


def test_perturbation_norms_zero_when_unchanged():
    X = np.random.default_rng(0).random((5, 8, 8)).astype(np.float32)
    norms = perturbation_norms(X, X.copy())
    assert norms["l0"] == 0.0
    assert norms["l1"] == 0.0
    assert norms["l2"] == 0.0
    assert norms["linf"] == 0.0


def test_perturbation_norms_l1_is_sum_of_abs_changes():
    X = np.zeros((1, 4))
    X_adv = np.array([[0.1, -0.2, 0.3, 0.0]])
    norms = perturbation_norms(X, X_adv)
    assert norms["l1"] == pytest.approx(0.6, abs=1e-6)


def test_perturbation_norms_l0_counts_changed_fraction():
    X = np.zeros((1, 4))
    X_adv = X.copy()
    X_adv[0, :2] = 1.0  # 2 of 4 entries changed
    norms = perturbation_norms(X, X_adv)
    assert norms["l0"] == 0.5


def test_perturbation_norms_linf_is_max_abs_change():
    X = np.zeros((1, 4))
    X_adv = np.array([[0.1, 0.9, 0.0, 0.0]])
    norms = perturbation_norms(X, X_adv)
    assert norms["linf"] == 0.9


def test_perturbation_norms_l2_matches_manual_calculation():
    X = np.zeros((1, 2))
    X_adv = np.array([[3.0, 4.0]])  # 3-4-5 triangle -> L2 norm = 5
    norms = perturbation_norms(X, X_adv)
    assert norms["l2"] == 5.0


def test_confidence_shift_positive_when_true_class_confidence_drops():
    y = np.array([0, 1])
    clean_probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    adv_probs = np.array([[0.4, 0.6], [0.2, 0.8]])
    shift = confidence_shift(clean_probs, adv_probs, y)
    # sample 0: true class 0 confidence dropped 0.9 -> 0.4 (delta 0.5)
    # sample 1: true class 1 confidence unchanged (delta 0.0)
    assert shift == 0.25


def test_confidence_shift_zero_when_unchanged():
    y = np.array([0, 1])
    probs = np.array([[0.7, 0.3], [0.3, 0.7]])
    assert confidence_shift(probs, probs.copy(), y) == 0.0
