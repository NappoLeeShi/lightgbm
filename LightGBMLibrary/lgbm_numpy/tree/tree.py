import numpy as np
from typing import Optional, List, Tuple, Dict
from ..histogram import Histogram


class TreeNode:
    """A single node in a decision tree."""

    __slots__ = (
        "feature_idx", "bin_threshold", "threshold_value",
        "left_child", "right_child",
        "left_indices", "right_indices", "indices",
        "leaf_value", "is_leaf",
        "sum_gradient", "sum_hessian", "num_data",
        "depth", "parent",
    )

    def __init__(self):
        self.feature_idx: int = -1
        self.bin_threshold: int = -1
        self.threshold_value: float = np.nan
        self.left_child: Optional["TreeNode"] = None
        self.right_child: Optional["TreeNode"] = None
        self.left_indices: Optional[np.ndarray] = None
        self.right_indices: Optional[np.ndarray] = None
        self.leaf_value: float = 0.0
        self.is_leaf: bool = False
        self.sum_gradient: float = 0.0
        self.sum_hessian: float = 0.0
        self.num_data: int = 0
        self.indices: Optional[np.ndarray] = None
        self.depth: int = 0
        self.parent: Optional["TreeNode"] = None


class DecisionTree:
    """
    LightGBM-style decision tree with:
    - Histogram-based split finding
    - Leaf-wise (best-first) growth
    - L1/L2 regularization
    - Min data in leaf / depth constraints
    - Per-leaf histogram management with subtraction
    """

    def __init__(
        self,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        min_sum_hessian_in_leaf: float = 1e-3,
        lambda_l1: float = 0.0,
        lambda_l2: float = 1.0,
        feature_fraction: float = 1.0,
        learning_rate: float = 0.1,
        max_bin: int = 255,
        extra_trees: bool = False,
    ):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.feature_fraction = feature_fraction
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.extra_trees = extra_trees
        self.root: Optional[TreeNode] = None
        self.leaves: List[TreeNode] = []
        self._num_leaves_found = 0

    def fit(
        self,
        binned_data: np.ndarray,
        num_bins_per_feature: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        total_bins: int,
    ) -> "DecisionTree":
        n_features = binned_data.shape[1]
        indices = np.arange(len(gradients))

        self.root = TreeNode()
        self.root.indices = indices
        self.root.num_data = len(indices)
        self.root.sum_gradient = float(np.sum(gradients[indices]))
        self.root.sum_hessian = float(np.sum(hessians[indices]))
        self.root.depth = 0

        self.leaves = [self.root]
        self._num_leaves_found = 1

        # Feature subsampling
        if self.feature_fraction < 1.0:
            n_select = max(1, int(n_features * self.feature_fraction))
            rng = np.random.RandomState()
            selected_features = rng.choice(n_features, size=n_select, replace=False)
        else:
            selected_features = np.arange(n_features)

        # Build histograms for all features at root
        leaf_histograms: Dict[int, List[Optional[Histogram]]] = {}
        root_hists = [None] * n_features
        for j in selected_features:
            num_bins = num_bins_per_feature[j]
            h = Histogram(num_bins)
            h.build(binned_data[:, j], gradients, hessians, indices)
            root_hists[j] = h
        leaf_histograms[id(self.root)] = root_hists

        while len(self.leaves) < self.num_leaves:
            best_gain = -np.inf
            best_leaf = None
            best_feature = -1
            best_bin_split = -1

            for leaf in self.leaves:
                if leaf.num_data < 2 * self.min_data_in_leaf:
                    continue
                if self.max_depth > 0 and leaf.depth >= self.max_depth:
                    continue

                leaf_id = id(leaf)
                leaf_hists = leaf_histograms.get(leaf_id)
                if leaf_hists is None:
                    continue

                for j in selected_features:
                    hist = leaf_hists[j]
                    if hist is None:
                        continue

                    gain, best_bin = self._find_best_split_for_feature(
                        hist, leaf, num_bins_per_feature[j]
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_leaf = leaf
                        best_feature = j
                        best_bin_split = best_bin

            if best_leaf is None or best_gain <= 1e-10:
                break

            # Split the node
            leaf_id = id(best_leaf)
            parent_hists = leaf_histograms.pop(leaf_id)

            left_idx, right_idx = self._split_indices(
                best_leaf, best_feature, best_bin_split, binned_data
            )

            left = TreeNode()
            left.indices = left_idx
            left.num_data = len(left_idx)
            left.sum_gradient = float(np.sum(gradients[left_idx]))
            left.sum_hessian = float(np.sum(hessians[left_idx]))
            left.depth = best_leaf.depth + 1
            left.parent = best_leaf

            right = TreeNode()
            right.indices = right_idx
            right.num_data = len(right_idx)
            right.sum_gradient = float(np.sum(gradients[right_idx]))
            right.sum_hessian = float(np.sum(hessians[right_idx]))
            right.depth = best_leaf.depth + 1
            right.parent = best_leaf

            best_leaf.feature_idx = best_feature
            best_leaf.bin_threshold = best_bin_split
            best_leaf.left_child = left
            best_leaf.right_child = right
            best_leaf.left_indices = left_idx
            best_leaf.right_indices = right_idx
            best_leaf.is_leaf = False

            self.leaves.remove(best_leaf)
            self.leaves.append(left)
            self.leaves.append(right)
            self._num_leaves_found += 1

            # Build histograms for children: left = parent - right (or rebuild)
            # Right child gets histograms by subtraction: parent_hist - left_child_hist
            # But easier and more correct: build left from scratch, subtract to get right
            left_hists = [None] * n_features
            right_hists = [None] * n_features

            for j in selected_features:
                num_bins = num_bins_per_feature[j]
                h_left = Histogram(num_bins)
                h_left.build(binned_data[:, j], gradients, hessians, left_idx)
                left_hists[j] = h_left

                h_right = parent_hists[j]
                if h_right is not None:
                    h_right = h_right.copy()
                    h_right.subtract(h_left)
                    right_hists[j] = h_right

            leaf_histograms[id(left)] = left_hists
            leaf_histograms[id(right)] = right_hists

        # Free indices memory, set leaf values
        self._set_leaf_values(self.root, gradients, hessians)
        self._free_indices(self.root)
        return self

    def _split_indices(
        self,
        node: TreeNode,
        feature_idx: int,
        bin_threshold: int,
        binned_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask_left = binned_data[node.indices, feature_idx] <= bin_threshold
        left_idx = node.indices[mask_left]
        right_idx = node.indices[~mask_left]
        return left_idx, right_idx

    def _find_best_split_for_feature(
        self,
        hist: Histogram,
        leaf: TreeNode,
        num_bins: int,
    ) -> Tuple[float, int]:
        best_gain = -np.inf
        best_bin = -1

        left_sum_g = 0.0
        left_sum_h = 0.0
        left_count = 0

        parent_l2 = leaf.sum_gradient ** 2 / (leaf.sum_hessian + self.lambda_l2)

        for b in range(num_bins):
            left_sum_g += hist.sum_gradients[b]
            left_sum_h += hist.sum_hessians[b]
            left_count += int(hist.count[b])

            right_count = leaf.num_data - left_count
            if right_count < self.min_data_in_leaf:
                break
            if left_count < self.min_data_in_leaf:
                continue

            right_sum_g = leaf.sum_gradient - left_sum_g
            right_sum_h = leaf.sum_hessian - left_sum_h

            if left_sum_h < self.min_sum_hessian_in_leaf:
                continue
            if right_sum_h < self.min_sum_hessian_in_leaf:
                continue

            gain = self._compute_split_gain(
                left_sum_g, left_sum_h,
                right_sum_g, right_sum_h,
                parent_l2,
            )

            if self.extra_trees:
                gain *= (0.5 + np.random.rand() * 0.5)

            if gain > best_gain:
                best_gain = gain
                best_bin = b

        return best_gain, best_bin

    def _compute_split_gain(
        self,
        left_g: float, left_h: float,
        right_g: float, right_h: float,
        parent_l2: float,
    ) -> float:
        left_l2 = left_g ** 2 / (left_h + self.lambda_l2)
        right_l2 = right_g ** 2 / (right_h + self.lambda_l2)

        gain = 0.5 * (left_l2 + right_l2 - parent_l2)
        gain -= self.lambda_l1 * (np.abs(left_g) + np.abs(right_g) - np.abs(left_g + right_g))
        return gain

    def _set_leaf_values(
        self,
        node: TreeNode,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ):
        if node is None:
            return
        if node.left_child is None and node.right_child is None:
            node.is_leaf = True
            w = -node.sum_gradient / (node.sum_hessian + self.lambda_l2)
            node.leaf_value = w * self.learning_rate
            return
        self._set_leaf_values(node.left_child, gradients, hessians)
        self._set_leaf_values(node.right_child, gradients, hessians)

    def _free_indices(self, node: TreeNode):
        if node is None:
            return
        node.indices = None
        node.left_indices = None
        node.right_indices = None
        self._free_indices(node.left_child)
        self._free_indices(node.right_child)

    def predict(self, X_binned: np.ndarray) -> np.ndarray:
        n = X_binned.shape[0]
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            result[i] = self._predict_one(X_binned[i], self.root)
        return result

    def _predict_one(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf:
            return node.leaf_value
        if x[node.feature_idx] <= node.bin_threshold:
            return self._predict_one(x, node.left_child)
        else:
            return self._predict_one(x, node.right_child)

    def predict_by_indices(self, X_binned: np.ndarray) -> np.ndarray:
        n = X_binned.shape[0]
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            node = self.root
            while not node.is_leaf:
                if X_binned[i, node.feature_idx] <= node.bin_threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            result[i] = node.leaf_value
        return result

    def num_leaves_found(self) -> int:
        return self._count_leaves(self.root)

    def _count_leaves(self, node: TreeNode) -> int:
        if node is None or node.is_leaf:
            return 1 if node is not None else 0
        return self._count_leaves(node.left_child) + self._count_leaves(node.right_child)

    def max_depth_found(self) -> int:
        return self._tree_depth(self.root)

    def _tree_depth(self, node: TreeNode) -> int:
        if node is None or node.is_leaf:
            return 0 if node is None else node.depth
        return max(self._tree_depth(node.left_child), self._tree_depth(node.right_child))
