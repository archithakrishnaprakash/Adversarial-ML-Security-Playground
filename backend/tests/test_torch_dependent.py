"""
These tests exercise everything that needs PyTorch: model training, the
gradient-based and black-box attacks, poisoning-lab training, and the
transferability/experiment-grid orchestration.

`pytest.importorskip("torch")` means this file is silently skipped (not
failed) in an environment without PyTorch installed — e.g. the sandbox this
project was originally built in, which couldn't fit PyTorch's ~800MB wheel on
disk. In a normal `pip install -r requirements.txt` environment, these run
like any other test.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.attacks import blackbox  # noqa: E402
from app.attacks.fgsm import fgsm_attack  # noqa: E402
from app.attacks.pgd import pgd_attack  # noqa: E402
from app.attacks.runner import run_attack  # noqa: E402
from app.evaluation.poisoning_eval import run_backdoor_experiment, run_label_flip_experiment  # noqa: E402
from app.evaluation.robustness import evaluate_robustness  # noqa: E402
from app.evaluation.transferability import compute_transferability_matrix  # noqa: E402
from app.experiments.runner import run_experiment_grid  # noqa: E402
from app.models.registry import MODEL_STORE, get_data, train_model  # noqa: E402
from app.threat_model import is_attack_applicable  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_model_store():
    MODEL_STORE.clear()
    yield
    MODEL_STORE.clear()


def test_train_differentiable_model_has_gradient_source():
    model = train_model("cyber", "logistic_regression")
    assert model.is_differentiable
    assert model.supports_gradient_attacks
    assert model.gradient_source() is model.torch_model
    assert 0.0 <= model.metrics["clean_accuracy"] <= 1.0


def test_train_random_forest_gets_a_surrogate():
    model = train_model("cyber", "random_forest")
    assert not model.is_differentiable
    assert model.supports_gradient_attacks  # via surrogate
    assert model.surrogate is not None
    assert model.gradient_source() is model.surrogate


def test_fgsm_output_bounded_and_shaped_correctly():
    model = train_model("image", "small_nn")
    X, y = model.X_test[:8], model.y_test[:8]
    X_adv = fgsm_attack(model.gradient_source(), X, y, epsilon=0.1)
    assert X_adv.shape == X.shape
    assert X_adv.min() >= 0.0 and X_adv.max() <= 1.0


def test_pgd_stays_within_epsilon_ball():
    model = train_model("image", "small_nn")
    X, y = model.X_test[:8], model.y_test[:8]
    epsilon = 0.1
    X_adv = pgd_attack(model.gradient_source(), X, y, epsilon=epsilon, steps=5)
    assert np.all(np.abs(X_adv - X) <= epsilon + 1e-5)


def test_run_attack_dispatches_random_noise_without_gradient_source():
    model = train_model("cyber", "random_forest")
    X, y = model.X_test[:5], model.y_test[:5]
    X_adv = run_attack(model, X, y, "random_noise", epsilon=0.1)
    assert X_adv.shape == X.shape


def test_threat_model_blocks_black_box_gradient_attack_end_to_end():
    applicable, _ = is_attack_applicable("fgsm", "black_box")
    assert applicable is False


def test_blackbox_transfer_attack_uses_no_gradients_on_target():
    model = train_model("cyber", "small_nn")
    X_pool, _, _, _ = get_data("cyber")
    X, y = model.X_test[:10], model.y_test[:10]
    X_adv, queries_used = blackbox.transfer_attack(
        model.predict_proba, X_pool, X, y, model.num_classes, epsilon=0.15, n_queries=50
    )
    assert X_adv.shape == X.shape
    assert queries_used == 50


def test_blackbox_query_attack_reports_query_count():
    model = train_model("cyber", "logistic_regression")
    X, y = model.X_test[:3], model.y_test[:3]
    X_adv, queries_used = blackbox.query_attack(
        model.predict_proba, X, y, epsilon=0.1, max_queries_per_sample=20
    )
    assert X_adv.shape == X.shape
    assert queries_used >= len(X)  # at least 1 query per sample


def test_label_flip_experiment_registers_two_models():
    result = run_label_flip_experiment("cyber", "logistic_regression", poison_fraction=0.1)
    assert result["clean_model_id"] in MODEL_STORE
    assert result["poisoned_model_id"] in MODEL_STORE
    assert MODEL_STORE[result["poisoned_model_id"]].metrics["poisoned"] is True


def test_backdoor_experiment_reports_success_rate_in_range():
    result = run_backdoor_experiment("image", "small_nn", poison_fraction=0.15, target_label=0)
    assert 0.0 <= result["backdoor_success_rate"] <= 1.0
    assert result["target_label"] == 0


def test_backdoor_rejects_invalid_target_label():
    with pytest.raises(ValueError):
        run_backdoor_experiment("cyber", "logistic_regression", poison_fraction=0.1, target_label=99)


def test_transferability_matrix_is_square_for_same_dataset_models():
    m1 = train_model("cyber", "logistic_regression")
    m2 = train_model("cyber", "small_nn")
    result = compute_transferability_matrix([m1, m2], attack="fgsm", epsilon=0.15, n_samples=20)
    assert set(result["matrix"].keys()) == {m1.model_id, m2.model_id}
    for row in result["matrix"].values():
        assert set(row.keys()) == {m1.model_id, m2.model_id}
        for rate in row.values():
            assert 0.0 <= rate <= 1.0


def test_transferability_requires_at_least_two_models():
    m1 = train_model("cyber", "logistic_regression")
    with pytest.raises(ValueError):
        compute_transferability_matrix([m1])


def test_experiment_grid_respects_configuration_cap():
    with pytest.raises(ValueError):
        run_experiment_grid(
            "cyber",
            model_types=["logistic_regression", "random_forest", "small_nn"],
            attacks=["fgsm", "pgd", "random_noise"],
            epsilons=[0.05, 0.1, 0.15, 0.2, 0.25],
            defenses=["none", "feature_clipping"],
        )  # 3*3*5*2 = 90 > MAX_CONFIGURATIONS (60)


def test_experiment_grid_runs_small_grid_and_logs_records():
    result = run_experiment_grid(
        "cyber",
        model_types=["logistic_regression"],
        attacks=["fgsm"],
        epsilons=[0.1],
        defenses=["none", "feature_clipping"],
        n_samples=30,
    )
    assert result["n_configurations_run"] == 2
    for record in result["records"]:
        assert "defended_accuracy" in record


def test_robustness_evaluation_includes_lp_norms():
    model = train_model("cyber", "logistic_regression")
    result = evaluate_robustness(model, ["fgsm"], epsilon=0.1, n_samples=30)
    fgsm_result = result["attacks"]["fgsm"]
    for key in ("l0_norm", "l2_norm", "linf_norm", "confidence_shift"):
        assert key in fgsm_result
