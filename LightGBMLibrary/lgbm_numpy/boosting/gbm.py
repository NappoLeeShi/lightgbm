import numpy as np
from typing import Optional, List, Dict, Tuple
from ..dataset import Dataset
from ..tree import DecisionTree
from ..objective import get_objective, Objective
from ..metrics import REGRESSION_METRICS, BINARY_METRICS, MULTICLASS_METRICS
from ..engine import Callback, CallbackEnv


class GradientBoostingMachine:
    """
    Gradient Boosting Decision Tree (GBDT) engine implementing LightGBM's core ideas:
    - Histogram-based split finding
    - Leaf-wise (best-first) tree growth
    - Gradient-based One-Side Sampling (GOSS)
    - L1/L2 regularization
    - Feature subsampling
    - Early stopping
    """

    def __init__(
        self,
        objective: str = "regression",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        min_sum_hessian_in_leaf: float = 1e-3,
        lambda_l1: float = 0.0,
        lambda_l2: float = 1.0,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        goss: bool = False,
        goss_large_data_pct: float = 0.2,
        max_bin: int = 255,
        extra_trees: bool = False,
        callbacks: Optional[List[Callback]] = None,
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        self.objective_name = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.goss = goss
        self.goss_large_data_pct = goss_large_data_pct
        self.max_bin = max_bin
        self.extra_trees = extra_trees
        self.callbacks = callbacks or []
        self.verbose = verbose
        self.seed = seed

        self.obj: Objective = get_objective(objective)
        self.trees: List[DecisionTree] = []
        self.init_score: float = 0.0
        self.feature_names: Optional[List[str]] = None
        self._num_features: int = 0
        self._n_classes: int = 0
        self._is_fitted = False
        self._best_iteration: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
    ) -> "GradientBoostingMachine":
        self.feature_names = feature_names
        self._num_features = X.shape[1]

        train_data = Dataset(X, y, max_bins=self.max_bin, feature_names=feature_names,
                             categorical_features=categorical_features)
        train_data.bin()

        self._n_classes = int(y.max()) + 1 if self.obj.is_classification and self.obj.name == "multiclass" else 0
        if hasattr(self.obj, 'num_classes') and self.obj.num_classes is None:
            self.obj.num_classes = self._n_classes

        self.init_score = self.obj.init_score(y)
        y_pred = np.full(len(y), self.init_score, dtype=np.float64)

        eval_data = []
        if eval_set:
            for X_val, y_val in eval_set:
                val_data = Dataset(X_val, y_val, max_bins=self.max_bin)
                val_data.bin()
                eval_data.append((val_data, y_val))

        self.trees = []
        env = CallbackEnv()
        env.model = self

        for cb in self.callbacks:
            cb.init(env)

        for i in range(self.n_estimators):
            env.iteration = i
            gradients = self.obj.gradients(y, y_pred)
            hessians = self.obj.hessians(y, y_pred)

            sample_indices = self._apply_bagging(len(y), i)
            if self.goss:
                sample_indices = self._apply_goss(gradients, hessians, sample_indices)

            tree = DecisionTree(
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_data_in_leaf=self.min_data_in_leaf,
                min_sum_hessian_in_leaf=self.min_sum_hessian_in_leaf,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
                feature_fraction=self.feature_fraction,
                learning_rate=self.learning_rate,
                max_bin=self.max_bin,
                extra_trees=self.extra_trees,
            )
            tree.fit(
                train_data.binned_data,
                np.array(train_data.num_bins_per_feature),
                gradients,
                hessians,
                train_data.total_bins,
            )

            y_pred += tree.predict_by_indices(train_data.binned_data)
            self.trees.append(tree)

            train_loss = self.obj.loss(y, y_pred)
            env.train_loss = train_loss
            env.valid_losses = {}

            for idx, (val_data, y_val) in enumerate(eval_data):
                val_pred = self._predict_dataset(val_data)
                val_loss = self.obj.loss(y_val, val_pred)
                env.valid_losses[f"valid_{idx}"] = val_loss

            env.model = self
            stop = False
            for cb in self.callbacks:
                result = cb(env)
                if result is True:
                    stop = True
                    break

            if stop:
                break

        self._is_fitted = True
        if hasattr(env, 'best_iteration'):
            self._best_iteration = env.best_iteration
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        data = Dataset(X, np.zeros(len(X)), max_bins=self.max_bin)
        data.bin()
        return self._predict_dataset(data)

    def _predict_dataset(self, data: Dataset) -> np.ndarray:
        pred = np.full(data.n_samples, self.init_score, dtype=np.float64)
        for tree in self.trees:
            pred += tree.predict_by_indices(data.binned_data)
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict(X)
        if self.obj.name == "binary":
            prob = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
            return np.column_stack([1 - prob, prob])
        elif self.obj.name == "multiclass":
            exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
            return exp_raw / exp_raw.sum(axis=1, keepdims=True)
        return raw

    def _apply_bagging(self, n_samples: int, iteration: int) -> np.ndarray:
        if self.bagging_fraction >= 1.0 or self.bagging_freq == 0:
            return np.arange(n_samples)
        if iteration % self.bagging_freq != 0:
            return np.arange(n_samples)
        rng = np.random.RandomState(self.seed + iteration if self.seed else iteration)
        n_select = max(1, int(n_samples * self.bagging_fraction))
        return rng.choice(n_samples, size=n_select, replace=False)

    def _apply_goss(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        abs_grads = np.abs(gradients[indices])
        n = len(indices)
        n_large = max(1, int(n * self.goss_large_data_pct))

        top_indices = np.argpartition(abs_grads, -n_large)[-n_large:]
        large_mask = np.zeros(n, dtype=bool)
        large_mask[top_indices] = True

        large_idx = indices[large_mask]
        small_idx = indices[~large_mask]

        a = self.goss_large_data_pct
        b = 1.0 - a
        if len(small_idx) > 0 and a > 0:
            weight = b / a
            # Just subsample small gradient instances
            rng = np.random.RandomState()
            n_keep = max(1, int(len(small_idx) * weight))
            n_keep = min(n_keep, len(small_idx))
            small_keep = rng.choice(small_idx, size=n_keep, replace=False)
            return np.concatenate([large_idx, small_keep])
        return large_idx

    @property
    def name(self) -> str:
        return self.obj.name

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return self.obj.loss(y, y_pred)
