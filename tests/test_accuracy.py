import numpy as np
from ml_metrics.metrics import accuracy
from sklearn.metrics import accuracy_score
import pytest


def test_perfect_accuracy():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    assert accuracy(y_true, y_pred) == 1.0


def test_zero_accuracy():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    assert accuracy(y_true, y_pred) == 0.0


def test_compare_with_sklearn_random_data():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.integers(0, 2, size=100)
    assert np.isclose(
        accuracy(y_true, y_pred),
        accuracy_score(y_true, y_pred),
    )


def test_accuracy_raises_on_different_lengths():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0, 1, 1])
