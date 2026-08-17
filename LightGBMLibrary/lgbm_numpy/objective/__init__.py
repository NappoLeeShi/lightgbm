from .base import Objective
from .regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    HuberLoss,
    PoissonLoss,
    QuantileLoss,
)
from .classification import BinaryLogloss, MulticlassLogloss

OBJECTIVES = {
    "regression": MeanSquaredError,
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "huber": HuberLoss,
    "poisson": PoissonLoss,
    "quantile": QuantileLoss,
    "binary": BinaryLogloss,
    "multiclass": MulticlassLogloss,
}


def get_objective(name: str) -> Objective:
    name_lower = name.lower()
    if name_lower not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {name}. Available: {list(OBJECTIVES.keys())}")
    return OBJECTIVES[name_lower]()
