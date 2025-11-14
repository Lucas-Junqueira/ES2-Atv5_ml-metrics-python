import numpy as np
from ml_metrics.metrics import confusion_matrix_binary
from sklearn.metrics import confusion_matrix


def test_confusion_matrix_basic():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    cm = confusion_matrix_binary(y_true, y_pred)
    # Esperado: tn=1 (0->0), fp=1 (0->1), fn=1 (1->0), tp=1 (1->1)
    expected = np.array([[1, 1],
                         [1, 1]])
    assert np.array_equal(cm, expected)


def test_confusion_matrix_matches_sklearn():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=50)
    y_pred = rng.integers(0, 2, size=50)

    my_cm = confusion_matrix_binary(y_true, y_pred)
    sk_cm = confusion_matrix(y_true, y_pred)
    assert np.array_equal(my_cm, sk_cm)
