import pytest

from app.threat_model import applicable_attacks, full_matrix, is_attack_applicable


def test_white_box_allows_all_gradient_attacks():
    result = applicable_attacks("white_box")
    for attack in ("fgsm", "pgd", "deepfool"):
        assert result[attack]["applicable"] is True


def test_black_box_blocks_gradient_attacks():
    result = applicable_attacks("black_box")
    for attack in ("fgsm", "pgd", "deepfool"):
        assert result[attack]["applicable"] is False


def test_black_box_allows_query_and_transfer_and_noise():
    result = applicable_attacks("black_box")
    for attack in ("random_noise", "transfer", "query"):
        assert result[attack]["applicable"] is True


def test_unknown_capability_raises():
    with pytest.raises(ValueError):
        applicable_attacks("purple_box")


def test_is_attack_applicable_matches_applicable_attacks():
    for capability in ("white_box", "gray_box", "black_box"):
        table = applicable_attacks(capability)
        for attack in table:
            applicable, rationale = is_attack_applicable(attack, capability)
            assert applicable == table[attack]["applicable"]
            assert rationale == table[attack]["rationale"]


def test_unknown_attack_defaults_to_applicable():
    applicable, rationale = is_attack_applicable("some_future_attack", "white_box")
    assert applicable is True
    assert "No threat-model rule" in rationale


def test_full_matrix_covers_every_capability():
    matrix = full_matrix()
    assert set(matrix.keys()) == {"white_box", "gray_box", "black_box"}
    for capability_table in matrix.values():
        assert "fgsm" in capability_table
        assert "random_noise" in capability_table
