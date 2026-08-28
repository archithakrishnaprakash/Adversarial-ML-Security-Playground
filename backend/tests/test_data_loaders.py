import numpy as np

from app.data.cyber_loader import FEATURE_NAMES, load_cyber_dataset
from app.data.mnist_loader import load_image_dataset


def test_image_dataset_shapes_and_range():
    X_train, X_test, y_train, y_test = load_image_dataset()
    assert X_train.shape[1:] == (8, 8)
    assert X_test.shape[1:] == (8, 8)
    assert X_train.dtype == np.float32
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert set(np.unique(y_train)).issubset(set(range(10)))
    assert len(X_train) > len(X_test)  # default 80/20 split


def test_image_dataset_is_reproducible():
    a = load_image_dataset(random_state=42)
    b = load_image_dataset(random_state=42)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[2], b[2])


def test_cyber_dataset_shapes_and_labels():
    X_train, X_test, y_train, y_test = load_cyber_dataset(n_samples=500)
    assert X_train.shape[1] == len(FEATURE_NAMES)
    assert set(np.unique(y_train)).issubset({0, 1})
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0


def test_cyber_dataset_roughly_balanced():
    _, _, y_train, y_test = load_cyber_dataset(n_samples=1000)
    y_all = np.concatenate([y_train, y_test])
    benign_fraction = (y_all == 0).mean()
    assert 0.4 < benign_fraction < 0.6
