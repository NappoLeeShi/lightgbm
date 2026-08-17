import numpy as np
from goss import (
    goss_sampling
)
from tree import DecisionTree

from objective import (
    sigmoid,
    compute_gradient,
    compute_hessian
)


class GradientBoostingLightGBM:

    def __init__(
        self,
        n_estimators=20,
        learning_rate=0.1,
        max_depth=4
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.trees = []

    def fit(
        self,
        X,
        y
    ):

        eps = 1e-15

        p = np.clip(
            np.mean(y),
            eps,
            1-eps
        )

        self.base_score = np.log(
            p/(1-p)
        )

        pred = np.full(
            len(y),
            self.base_score
        )

        for _ in range(
            self.n_estimators
        ):

            prob = sigmoid(pred)

            gradients = compute_gradient(
                y,
                prob
            )

            hessians = compute_hessian(
                prob
            )

            tree = DecisionTree(
                max_depth=self.max_depth
            )
            X_goss,\
            g_goss,\
            h_goss = goss_sampling(
                X,
                gradients,
                hessians,
                top_rate=0.2,
                other_rate=0.1
            )
            
            tree.fit(
                X_goss,
                g_goss,
                h_goss
            )

            update = tree.predict(X)

            pred += (
                self.learning_rate
                *
                update
            )

            self.trees.append(tree)

    def predict_proba(
        self,
        X
    ):

        pred = np.full(
            len(X),
            self.base_score
        )

        for tree in self.trees:

            pred += (
                self.learning_rate
                *
                tree.predict(X)
            )

        return sigmoid(pred)

    def predict(
        self,
        X
    ):

        prob = self.predict_proba(X)

        return (
            prob >= 0.5
        ).astype(int)