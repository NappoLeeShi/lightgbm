import numpy as np
from typing import Optional


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -500, 500)))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true > 0.5).astype(int) == y_pred.astype(int)))


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]
    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_cumsum = np.cumsum(y_true_sorted == 1) / n_pos
    fpr_cumsum = np.cumsum(y_true_sorted == 0) / n_neg
    tpr_cumsum = np.concatenate(([0.0], tpr_cumsum))
    fpr_cumsum = np.concatenate(([0.0], fpr_cumsum))
    return float(np.trapezoid(tpr_cumsum, fpr_cumsum))


def precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    pred_pos = (y_pred >= threshold).astype(int)
    tp = np.sum((pred_pos == 1) & (y_true.astype(int) == 1))
    fp = np.sum((pred_pos == 1) & (y_true.astype(int) == 0))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    pred_pos = (y_pred >= threshold).astype(int)
    tp = np.sum((pred_pos == 1) & (y_true.astype(int) == 1))
    fn = np.sum((pred_pos == 0) & (y_true.astype(int) == 1))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    p = precision(y_true, y_pred, threshold)
    r = recall(y_true, y_pred, threshold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def multiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.argmax(y_pred, axis=1) == y_true.astype(int)))
