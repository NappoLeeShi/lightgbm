import numpy as np
from histogram import (
    build_histogram
)
from node import Node

from objective import (
    calculate_gain,
    leaf_weight
)


class DecisionTree:

    def __init__(
        self,
        max_depth=3,
        min_samples_split=20,
        lamda=1,
        gamma=0
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.lamda = lamda
        self.gamma = gamma

        self.root = None

    def fit(
        self,
        X,
        gradients,
        hessians
    ):

        self.root = self._build_tree(
            X,
            gradients,
            hessians,
            depth=0
        )

    def _best_split(
        self,
        X,
        gradients,
        hessians
    ):

        best_gain = -float("inf")

        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):

            values = X[:, feature]

            histogram, bins = (
                build_histogram(
                    values,
                    gradients,
                    hessians,
                    max_bins=16
                )
            )

            GL = 0
            HL = 0

            G_total = np.sum(
                gradients
            )   

            H_total = np.sum(
                hessians
            )

            for i in range(
                len(histogram)-1
            ):

                GL += histogram[i]["G"]
                HL += histogram[i]["H"]

                GR = G_total - GL
                HR = H_total - HL

                gain = calculate_gain(
                    GL,
                    HL,
                    GR,
                    HR,
                    self.lamda,
                    self.gamma
                )

                if gain > best_gain:

                    best_gain = gain

                    best_feature = feature

                    best_threshold = bins[i+1]
                    


        return (
            best_feature,
            best_threshold
        )

    def _build_tree(
        self,
        X,
        gradients,
        hessians,
        depth
    ):

        G = np.sum(
            gradients
        )

        H = np.sum(
            hessians
        )

        # stop condition
        if (
            depth >= self.max_depth
            or
            len(X) < self.min_samples_split
        ):

            return Node(
                value=leaf_weight(
                    G,
                    H,
                    self.lamda
                )
            )

        feature, threshold = (
            self._best_split(
                X,
                gradients,
                hessians
            )
        )

        if feature is None:

            return Node(
                value=leaf_weight(
                    G,
                    H,
                    self.lamda
                )
            )

        left_mask = (
            X[:, feature]
            <= threshold
        )

        right_mask = ~left_mask

        left_child = self._build_tree(
            X[left_mask],
            gradients[left_mask],
            hessians[left_mask],
            depth + 1
        )

        right_child = self._build_tree(
            X[right_mask],
            gradients[right_mask],
            hessians[right_mask],
            depth + 1
        )

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def _predict_sample(
        self,
        x,
        node
    ):

        if node.is_leaf():

            return node.value

        if (
            x[node.feature]
            <= node.threshold
        ):

            return self._predict_sample(
                x,
                node.left
            )

        return self._predict_sample(
            x,
            node.right
        )

    def predict(
        self,
        X
    ):

        return np.array([
            self._predict_sample(
                row,
                self.root
            )
            for row in X
        ])