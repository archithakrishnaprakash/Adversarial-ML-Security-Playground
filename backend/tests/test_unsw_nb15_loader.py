import numpy as np
import pandas as pd
import pytest

from app.data import unsw_nb15_loader as loader

COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "attack_cat", "label",
]


def _make_fixture_csv(path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    protos = ["tcp", "udp", "icmp"]
    services = ["-", "http", "dns", "ftp"]
    states = ["FIN", "CON", "INT"]
    rows = []
    for i in range(n):
        is_attack = i % 3 == 0
        rows.append(
            {
                "id": i,
                "dur": rng.random() * 10,
                "proto": rng.choice(protos),
                "service": rng.choice(services),
                "state": rng.choice(states),
                "spkts": rng.integers(1, 100),
                "dpkts": rng.integers(1, 100),
                "sbytes": rng.integers(1, 10000),
                "dbytes": rng.integers(1, 10000),
                "rate": rng.random() * 1000,
                "sttl": rng.integers(1, 255),
                "dttl": rng.integers(1, 255),
                "attack_cat": "Exploits" if is_attack else "Normal",
                "label": 1 if is_attack else 0,
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False)


def test_is_available_false_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSW_NB15_DATA_DIR", str(tmp_path / "does_not_exist"))
    assert loader.is_available() is False


def test_is_available_true_with_training_set_file(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSW_NB15_DATA_DIR", str(tmp_path))
    _make_fixture_csv(tmp_path / "UNSW_NB15_training-set.csv")
    assert loader.is_available() is True


def test_is_available_true_with_any_single_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSW_NB15_DATA_DIR", str(tmp_path))
    _make_fixture_csv(tmp_path / "some_export.csv")
    assert loader.is_available() is True


def test_load_raises_clear_error_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSW_NB15_DATA_DIR", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="No UNSW-NB15 CSV files found"):
        loader.load_unsw_nb15()


def test_load_single_file_splits_and_scales(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSW_NB15_DATA_DIR", str(tmp_path))
    _make_fixture_csv(tmp_path / "combined.csv", n=100)

    X_train, X_test, y_train, y_test, feature_names = loader.load_unsw_nb15(
        test_size=0.2, max_samples=None
    )

    assert len(X_train) + len(X_test) == 100
    assert X_train.shape[1] == len(feature_names)
    # id, attack_cat, label must never appear as features
    assert "id" not in feature_names
    assert "attack_cat" not in feature_names
    assert "label" not in feature_names
    # categorical columns should be present but numerically encoded
    assert "proto" in feature_names
    # scaled to [0, 1] using train statistics (test can slightly exceed due
    # to clipping, so just check train is well-formed and test is clipped)
    assert X_train.min() >= 0.0 - 1e-6 and X_train.max() <= 1.0 + 1e-6
    assert X_test.min() >= 0.0 - 1e-6 and X_test.max() <= 1.0 + 1e-6
    assert set(np.unique(y_train)).issubset({0, 1})


def test_load_respects_max_samples_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSW_NB15_DATA_DIR", str(tmp_path))
    _make_fixture_csv(tmp_path / "combined.csv", n=200)

    X_train, X_test, y_train, y_test, _ = loader.load_unsw_nb15(max_samples=50)
    assert len(X_train) <= 50
    assert len(X_test) <= 13  # max(1, 50 // 4)


def test_load_train_and_test_files_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSW_NB15_DATA_DIR", str(tmp_path))
    _make_fixture_csv(tmp_path / "UNSW_NB15_training-set.csv", n=80, seed=1)
    _make_fixture_csv(tmp_path / "UNSW_NB15_testing-set.csv", n=20, seed=2)

    X_train, X_test, y_train, y_test, feature_names = loader.load_unsw_nb15(max_samples=None)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert X_train.shape[1] == X_test.shape[1] == len(feature_names)
