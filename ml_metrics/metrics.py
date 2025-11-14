import numpy as np

def _to_numpy(arr):
    """Converte entrada para numpy array 1D."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Os vetores y_true e y_pred devem ser 1D.")
    return arr


def accuracy(y_true, y_pred):
    """
    Acurácia: proporção de previsões corretas.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho.")
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, positive_label=1):
    """
    Precisão (binary): TP / (TP + FP)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho.")

    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true, y_pred, positive_label=1):
    """
    Revocação (sensibilidade): TP / (TP + FN)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho.")

    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true, y_pred, positive_label=1):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    p = precision(y_true, y_pred, positive_label=positive_label)
    r = recall(y_true, y_pred, positive_label=positive_label)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def confusion_matrix_binary(y_true, y_pred, positive_label=1):
    """
    Matriz de confusão 2x2 para problema binário.
    Retorna np.array([[tn, fp],
                      [fn, tp]])
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho.")

    pos = positive_label
    neg = 1 - positive_label if positive_label in (0, 1) else None
    if neg is None:
        raise ValueError("Para esta função, positive_label deve ser 0 ou 1.")

    tp = np.sum((y_true == pos) & (y_pred == pos))
    tn = np.sum((y_true == neg) & (y_pred == neg))
    fp = np.sum((y_true == neg) & (y_pred == pos))
    fn = np.sum((y_true == pos) & (y_pred == neg))

    return np.array([[tn, fp],
                     [fn, tp]])
