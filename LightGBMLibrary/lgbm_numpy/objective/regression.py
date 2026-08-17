import numpy as np
from .base import Objective


class MeanSquaredError(Objective):
    name = "mse"

    def loss(self, y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def gradients(self, y_true, y_pred):
        return -(y_true - y_pred)

    def hessians(self, y_true, y_pred):
        return np.ones_like(y_true)

    def init_score(self, y):
        return float(np.mean(y))


class MeanAbsoluteError(Objective):
    name = "mae"

    def loss(self, y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    def gradients(self, y_true, y_pred):
        diff = y_pred - y_true
        return -np.sign(diff)

    def hessians(self, y_true, y_pred):
        return np.ones_like(y_true)

    def init_score(self, y):
        return float(np.median(y))


class HuberLoss(Objective):
    name = "huber"

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha

    def loss(self, y_true, y_pred):
        residual = y_true - y_pred
        delta = self._delta(y_true)
        abs_r = np.abs(residual)
        quadratic = np.minimum(abs_r, delta)
        linear = abs_r - quadratic
        return float(np.mean(0.5 * quadratic ** 2 + delta * linear))

    def gradients(self, y_true, y_pred):
        residual = y_true - y_pred
        delta = self._delta(y_true)
        abs_r = np.abs(residual)
        grad = np.where(abs_r <= delta, -residual, -delta * np.sign(residual))
        return grad

    def hessians(self, y_true, y_pred):
        residual = y_true - y_pred
        delta = self._delta(y_true)
        return np.where(np.abs(residual) <= delta, 1.0, 0.0)

    def init_score(self, y):
        return float(np.median(y))

    def _delta(self, y_true):
        residual = y_true - np.median(y_true)
        return np.percentile(np.abs(residual), self.alpha * 100)


class PoissonLoss(Objective):
    name = "poisson"

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, None)
        return float(np.mean(y_pred - y_true * np.log(y_pred)))

    def gradients(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, None)
        return -(y_true / y_pred - 1.0)

    def hessians(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, None)
        return y_true / (y_pred ** 2)

    def init_score(self, y):
        return float(np.log(np.mean(y)))


class QuantileLoss(Objective):
    name = "quantile"

    def __init__(self, quantile: float = 0.5):
        self.quantile = quantile

    def loss(self, y_true, y_pred):
        residual = y_true - y_pred
        return float(np.mean(np.where(residual >= 0, self.quantile * residual,
                                       (self.quantile - 1) * residual)))

    def gradients(self, y_true, y_pred):
        residual = y_true - y_pred
        return -np.where(residual >= 0, self.quantile, self.quantile - 1)

    def hessians(self, y_true, y_pred):
        return np.ones_like(y_true)

    def init_score(self, y):
        return float(np.percentile(y, self.quantile * 100))
