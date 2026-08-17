from abc import ABC, abstractmethod
import numpy as np


class Objective(ABC):
    """Base class for all objective functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def is_classification(self) -> bool:
        return False

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute scalar loss value."""
        ...

    @abstractmethod
    def gradients(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute first-order gradients (negative direction)."""
        ...

    @abstractmethod
    def hessians(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute second-order gradients (hessians)."""
        ...

    def init_score(self, y: np.ndarray) -> float:
        """Initial prediction score (e.g. log-odds for binary, mean for regression)."""
        return 0.0
