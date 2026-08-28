import pytest

from app.evaluation.risk_engine import generate_security_assessment, risk_rating_for_score


def _benchmark(score, clean_acc=0.98, attack_acc=0.6, asr=0.4):
    return {
        "model_id": "abc123",
        "model_type": "cnn",
        "dataset": "image",
        "clean_accuracy": clean_acc,
        "robustness_score": score,
        "attacks": {
            "fgsm": {"accuracy": attack_acc, "attack_success_rate": asr},
        },
    }


def test_risk_rating_thresholds():
    assert risk_rating_for_score(85) == "LOW"
    assert risk_rating_for_score(70) == "LOW"
    assert risk_rating_for_score(69.9) == "MEDIUM"
    assert risk_rating_for_score(40) == "MEDIUM"
    assert risk_rating_for_score(39.9) == "HIGH"
    assert risk_rating_for_score(0) == "HIGH"


def test_assessment_requires_a_real_benchmark():
    with pytest.raises(ValueError):
        generate_security_assessment({"not_a_benchmark": True})


def test_high_risk_recommends_do_not_deploy():
    result = generate_security_assessment(_benchmark(score=25))
    assert result["risk_rating"] == "HIGH"
    assert result["recommendation"] == "DO NOT DEPLOY"
    assert len(result["recommended_actions"]) > 0


def test_low_risk_recommends_safe_to_deploy():
    result = generate_security_assessment(_benchmark(score=90, attack_acc=0.95, asr=0.05))
    assert result["risk_rating"] == "LOW"
    assert result["recommendation"] == "SAFE TO DEPLOY"


def test_findings_generated_per_attack():
    result = generate_security_assessment(_benchmark(score=50))
    assert len(result["findings"]) >= 1
    assert any("FGSM" in f["text"] for f in result["findings"])


def test_findings_sorted_worst_first():
    benchmark = {
        "model_id": "x",
        "model_type": "cnn",
        "dataset": "image",
        "clean_accuracy": 0.98,
        "robustness_score": 30,
        "attacks": {
            "random_noise": {"accuracy": 0.97, "attack_success_rate": 0.01},  # low severity
            "pgd": {"accuracy": 0.4, "attack_success_rate": 0.6},  # high severity
        },
    }
    result = generate_security_assessment(benchmark)
    severities = [f["severity"] for f in result["findings"]]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    assert severities == sorted(severities, key=lambda s: order[s])


def test_transferability_findings_included():
    benchmark = _benchmark(score=80, attack_acc=0.9, asr=0.1)
    transferability = {
        "model_labels": {"m1": "cnn #m1", "m2": "mlp #m2"},
        "matrix": {"m1": {"m1": 0.9, "m2": 0.5}},
    }
    result = generate_security_assessment(benchmark, transferability=transferability)
    assert any("transfer to" in f["text"] for f in result["findings"])


def test_poisoning_findings_included():
    benchmark = _benchmark(score=80, attack_acc=0.9, asr=0.1)
    poisoning = {"poison_fraction": 0.1, "accuracy_degradation": 0.2}
    result = generate_security_assessment(benchmark, poisoning=poisoning)
    assert any("poisoning" in f["text"] for f in result["findings"])


def test_empty_attacks_defaults_to_low_severity_message():
    benchmark = {
        "model_id": "x",
        "model_type": "cnn",
        "dataset": "image",
        "clean_accuracy": 0.98,
        "robustness_score": 95,
        "attacks": {},
    }
    result = generate_security_assessment(benchmark)
    assert len(result["findings"]) == 1
    assert result["findings"][0]["severity"] == "LOW"
    assert "No significant vulnerabilities" in result["findings"][0]["text"]
