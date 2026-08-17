from .metrics import (
    rmse, mae, mse, logloss, accuracy, auc,
    precision, recall, f1_score, multiclass_accuracy,
)

REGRESSION_METRICS = {"rmse": rmse, "mae": mae, "mse": mse}
BINARY_METRICS = {"logloss": logloss, "accuracy": accuracy, "auc": auc,
                  "precision": precision, "recall": recall, "f1": f1_score}
MULTICLASS_METRICS = {"accuracy": multiclass_accuracy}

__all__ = [
    "rmse", "mae", "mse", "logloss", "accuracy", "auc",
    "precision", "recall", "f1_score", "multiclass_accuracy",
    "REGRESSION_METRICS", "BINARY_METRICS", "MULTICLASS_METRICS",
]
