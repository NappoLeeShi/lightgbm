import json
import numpy as np
from typing import Dict, Any, Optional, List
from .boosting import GradientBoostingMachine


class ModelSerializer:
    """Save and load trained GBM models to/from JSON."""

    @staticmethod
    def save(model: GradientBoostingMachine, filepath: str):
        model_data = {
            "objective": model.objective_name,
            "n_estimators": len(model.trees),
            "learning_rate": model.learning_rate,
            "init_score": model.init_score,
            "num_features": model._num_features,
            "n_classes": model._n_classes,
            "trees": [ModelSerializer._serialize_tree(t.root) for t in model.trees],
        }
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @staticmethod
    def load(filepath: str) -> GradientBoostingMachine:
        with open(filepath, "r") as f:
            model_data = json.load(f)

        gbm = GradientBoostingMachine(
            objective=model_data["objective"],
            n_estimators=model_data["n_estimators"],
            learning_rate=model_data["learning_rate"],
        )
        gbm.init_score = model_data["init_score"]
        gbm._num_features = model_data["num_features"]
        gbm._n_classes = model_data["n_classes"]

        gbm.trees = []
        for tree_data in model_data["trees"]:
            from .tree import DecisionTree
            tree = DecisionTree(learning_rate=model_data["learning_rate"])
            tree.root = ModelSerializer._deserialize_tree(tree_data)
            gbm.trees.append(tree)

        gbm._is_fitted = True
        return gbm

    @staticmethod
    def _serialize_tree(node) -> Dict[str, Any]:
        if node is None:
            return None
        data = {
            "is_leaf": node.is_leaf,
            "feature_idx": int(node.feature_idx),
            "bin_threshold": int(node.bin_threshold),
            "leaf_value": float(node.leaf_value),
            "sum_gradient": float(node.sum_gradient),
            "sum_hessian": float(node.sum_hessian),
            "num_data": int(node.num_data),
            "depth": int(node.depth),
        }
        if not node.is_leaf:
            data["left"] = ModelSerializer._serialize_tree(node.left_child)
            data["right"] = ModelSerializer._serialize_tree(node.right_child)
        return data

    @staticmethod
    def _deserialize_tree(data: Dict[str, Any]):
        from .tree import TreeNode
        if data is None:
            return None
        node = TreeNode()
        node.is_leaf = data["is_leaf"]
        node.feature_idx = data["feature_idx"]
        node.bin_threshold = data["bin_threshold"]
        node.leaf_value = data["leaf_value"]
        node.sum_gradient = data["sum_gradient"]
        node.sum_hessian = data["sum_hessian"]
        node.num_data = data["num_data"]
        node.depth = data["depth"]
        if not node.is_leaf:
            node.left_child = ModelSerializer._deserialize_tree(data.get("left"))
            node.right_child = ModelSerializer._deserialize_tree(data.get("right"))
        return node
