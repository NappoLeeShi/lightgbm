import numpy as np
from typing import Dict, List, Optional, Tuple
from ..histogram import BinMapper


class Dataset:
    """Container for training data with feature binning support."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_bins: int = 255,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
    ):
        self.X = X.astype(np.float64)
        self.y = y.astype(np.float64)
        self.n_samples, self.n_features = X.shape
        self.max_bins = max_bins
        self.feature_names = (
            feature_names
            or [f"f{i}" for i in range(self.n_features)]
        )
        self.categorical_features = set(categorical_features or [])
        self.bin_mappers: List[BinMapper] = []
        self.binned_data: Optional[np.ndarray] = None
        self.num_bins_per_feature: List[int] = []
        self.total_bins: int = 0
        self._is_binned = False

    def bin(self) -> "Dataset":
        self.bin_mappers = []
        binned_cols = []
        self.num_bins_per_feature = []

        for j in range(self.n_features):
            col = self.X[:, j]
            mapper = BinMapper(max_bins=self.max_bins)
            bins = mapper.fit_transform(col)
            self.bin_mappers.append(mapper)
            binned_cols.append(bins)
            self.num_bins_per_feature.append(mapper.num_bins)

        self.binned_data = np.column_stack(binned_cols).astype(np.int32)
        self.total_bins = sum(self.num_bins_per_feature)
        self._is_binned = True
        return self

    def get_binned_col(self, feature_idx: int) -> np.ndarray:
        if not self._is_binned:
            raise RuntimeError("Dataset has not been binned. Call .bin() first.")
        return self.binned_data[:, feature_idx]

    def get_feature_bins_range(self, feature_idx: int) -> Tuple[int, int]:
        start = sum(self.num_bins_per_feature[:feature_idx])
        end = start + self.num_bins_per_feature[feature_idx]
        return start, end

    def __repr__(self):
        return (
            f"Dataset(samples={self.n_samples}, features={self.n_features}, "
            f"binned={self._is_binned}, max_bins={self.max_bins})"
        )
