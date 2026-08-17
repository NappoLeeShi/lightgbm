import numpy as np
from typing import Optional, List, Tuple, Dict
from ..boosting import GradientBoostingMachine
from ..engine import EarlyStopping, LogCallback, LearningRateScheduler
from ..model import ModelSerializer
from ..metrics import rmse, mae, mse, logloss, accuracy, auc, f1_score


class LGBMRegressor:
    """High-level regressor interface mirroring sklearn's API style."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        lambda_l1: float = 0.0,
        lambda_l2: float = 1.0,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        goss: bool = False,
        max_bin: int = 255,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        callbacks = []
        if early_stopping_rounds is not None:
            callbacks.append(EarlyStopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
        if verbose:
            callbacks.append(LogCallback(period=1))

        self._model = GradientBoostingMachine(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_data_in_leaf=min_data_in_leaf,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            goss=goss,
            max_bin=max_bin,
            callbacks=callbacks,
            verbose=verbose,
            seed=seed,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self._model.fit(X, y, eval_set=eval_set)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return -rmse(y, y_pred)

    def save_model(self, filepath: str):
        ModelSerializer.save(self._model, filepath)

    def load_model(self, filepath: str):
        self._model = ModelSerializer.load(filepath)

    def get_params(self) -> Dict:
        return {
            "n_estimators": self._model.n_estimators,
            "learning_rate": self._model.learning_rate,
            "num_leaves": self._model.num_leaves,
            "max_depth": self._model.max_depth,
            "min_data_in_leaf": self._model.min_data_in_leaf,
            "lambda_l1": self._model.lambda_l1,
            "lambda_l2": self._model.lambda_l2,
        }


class LGBMClassifier:
    """High-level classifier interface mirroring sklearn's API style."""

    def __init__(
        self,
        objective: str = "binary",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        lambda_l1: float = 0.0,
        lambda_l2: float = 1.0,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        goss: bool = False,
        max_bin: int = 255,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        callbacks = []
        if early_stopping_rounds is not None:
            callbacks.append(EarlyStopping(
                stopping_rounds=early_stopping_rounds,
                metric_name="valid_0",
                verbose=verbose,
            ))
        if verbose:
            callbacks.append(LogCallback(period=1))

        self._model = GradientBoostingMachine(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_data_in_leaf=min_data_in_leaf,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            goss=goss,
            max_bin=max_bin,
            callbacks=callbacks,
            verbose=verbose,
            seed=seed,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self._model.fit(X, y, eval_set=eval_set)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            return np.argmax(proba, axis=1).astype(int)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y.astype(int)))

    def save_model(self, filepath: str):
        ModelSerializer.save(self._model, filepath)

    def load_model(self, filepath: str):
        self._model = ModelSerializer.load(filepath)

    def get_params(self) -> Dict:
        return {
            "objective": self._model.objective_name,
            "n_estimators": self._model.n_estimators,
            "learning_rate": self._model.learning_rate,
            "num_leaves": self._model.num_leaves,
            "max_depth": self._model.max_depth,
            "min_data_in_leaf": self._model.min_data_in_leaf,
        }
