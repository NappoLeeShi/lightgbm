import numpy as np
from typing import Tuple, Optional


class BinMapper:
    """Maps continuous feature values to discrete bin indices using quantile binning."""

    def __init__(self, max_bins: int = 255):
        self.max_bins = max_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.missing_bin: int = 0

    def fit(self, feature: np.ndarray) -> "BinMapper":
        valid = feature[~np.isnan(feature)]
        if len(valid) == 0:
            self.bin_edges = np.array([0.0])
            return self
        n_unique = len(np.unique(valid))
        actual_bins = min(self.max_bins, n_unique)
        if actual_bins <= 1:
            self.bin_edges = np.array([valid.min()])
            return self
        quantiles = np.linspace(0, 100, actual_bins + 1)
        edges = np.percentile(valid, quantiles)
        edges = np.unique(edges)
        self.bin_edges = edges
        return self

    def transform(self, feature: np.ndarray) -> np.ndarray:
        if self.bin_edges is None:
            raise RuntimeError("BinMapper has not been fitted yet.")
        bins = np.searchsorted(self.bin_edges, feature, side="right") - 1
        bins = np.clip(bins, 0, len(self.bin_edges) - 2)
        bins[np.isnan(feature)] = self.missing_bin
        return bins.astype(np.int32)

    def fit_transform(self, feature: np.ndarray) -> np.ndarray:
        return self.fit(feature).transform(feature)

    @property
    def num_bins(self) -> int:
        if self.bin_edges is None:
            return 0
        return len(self.bin_edges) - 1


class Histogram:
    """Histogram of gradient statistics for a single feature."""

    __slots__ = ("num_bins", "sum_gradients", "sum_hessians", "count")

    def __init__(self, num_bins: int):
        self.num_bins = num_bins
        self.sum_gradients = np.zeros(num_bins, dtype=np.float64)
        self.sum_hessians = np.zeros(num_bins, dtype=np.float64)
        self.count = np.zeros(num_bins, dtype=np.int64)

    def build(
        self,
        bin_indices: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        if indices is not None:
            b = bin_indices[indices]
            g = gradients[indices]
            h = hessians[indices]
        else:
            b = bin_indices
            g = gradients
            h = hessians
        np.add.at(self.sum_gradients, b, g)
        np.add.at(self.sum_hessians, b, h)
        np.add.at(self.count, b, 1)

    def subtract(self, other: "Histogram"):
        self.sum_gradients -= other.sum_gradients
        self.sum_hessians -= other.sum_hessians
        self.count -= other.count

    def copy(self) -> "Histogram":
        h = Histogram(self.num_bins)
        h.sum_gradients = self.sum_gradients.copy()
        h.sum_hessians = self.sum_hessians.copy()
        h.count = self.count.copy()
        return h
