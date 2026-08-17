import numpy as np
from .base import Objective


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class BinaryLogloss(Objective):
    name = "binary"

    @property
    def is_classification(self):
        return True

    def loss(self, y_true, y_pred):
        p = _sigmoid(y_pred)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def gradients(self, y_true, y_pred):
        p = _sigmoid(y_pred)
        return -(y_true - p)

    def hessians(self, y_true, y_pred):
        p = _sigmoid(y_pred)
        return p * (1 - p)

    def init_score(self, y):
        p = np.mean(y)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return float(np.log(p / (1 - p)))


class MulticlassLogloss(Objective):
    name = "multiclass"

    def __init__(self, num_classes: int = None):
        self.num_classes = num_classes

    @property
    def is_classification(self):
        return True

    def loss(self, y_true, y_pred):
        probs = self._softmax(y_pred)
        num_classes = probs.shape[1]
        y_int = y_true.astype(int)
        log_probs = np.log(np.clip(probs, 1e-10, 1.0))
        return float(-np.mean(log_probs[np.arange(len(y_int)), y_int]))

    def gradients(self, y_true, y_pred):
        probs = self._softmax(y_pred)
        num_classes = probs.shape[1]
        y_onehot = np.zeros_like(probs)
        y_int = y_true.astype(int)
        y_onehot[np.arange(len(y_int)), y_int] = 1.0
        return -(y_onehot - probs)

    def hessians(self, y_true, y_pred):
        probs = self._softmax(y_pred)
        return probs * (1 - probs)

    def init_score(self, y):
        if self.num_classes is None:
            self.num_classes = int(y.max()) + 1
        counts = np.bincount(y.astype(int), minlength=self.num_classes)
        probs = counts / len(y)
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= probs.sum()
        log_probs = np.log(probs)
        return log_probs - log_probs.mean()

    def _softmax(self, x):
        x_shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)
