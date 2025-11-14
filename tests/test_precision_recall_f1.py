import numpy as np
from ml_metrics.metrics import precision, recall, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1


def test_precision_recall_simple_case():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    # TP=1, FP=0, FN=1
    assert precision(y_true, y_pred) == 1.0
    assert recall(y_true, y_pred) == 0.5


def test_f1_simple_case():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    # p=1.0, r=0.5 -> F1=2/3
    assert np.isclose(f1_score(y_true, y_pred), 2/3)


def test_compare_with_sklearn_random_data():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)

    assert np.isclose(
        precision(y_true, y_pred),
        precision_score(y_true, y_pred),
    )

    assert np.isclose(
        recall(y_true, y_pred),
        recall_score(y_true, y_pred),
    )

    assert np.isclose(
        f1_score(y_true, y_pred),
        sk_f1(y_true, y_pred),
    )
