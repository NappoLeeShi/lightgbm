"""Smoke tests for lgbm_numpy framework."""
import sys
import os
import numpy as np
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgbm_numpy.api import LGBMRegressor, LGBMClassifier
from lgbm_numpy.objective import get_objective
from lgbm_numpy.histogram import BinMapper, Histogram
from lgbm_numpy.tree import DecisionTree
from lgbm_numpy.boosting import GradientBoostingMachine
from lgbm_numpy.model import ModelSerializer
from lgbm_numpy.dataset import Dataset
from lgbm_numpy.metrics import rmse, mae, mse, logloss, accuracy, auc, f1_score


def test_objectives():
    print("=== Testing Objective Functions ===")
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_pred = rng.randn(100)

    for name in ["mse", "mae", "huber", "poisson", "quantile"]:
        obj = get_objective(name)
        loss = obj.loss(y_true, y_pred)
        grad = obj.gradients(y_true, y_pred)
        hess = obj.hessians(y_true, y_pred)
        init = obj.init_score(y_true)
        print(f"  {name:10s}: loss={loss:.4f}, grad_shape={grad.shape}, hess_shape={hess.shape}, init={init:.4f}")

    y_true_cls = (rng.rand(100) > 0.5).astype(float)
    for name in ["binary"]:
        obj = get_objective(name)
        loss = obj.loss(y_true_cls, y_pred)
        grad = obj.gradients(y_true_cls, y_pred)
        hess = obj.hessians(y_true_cls, y_pred)
        init = obj.init_score(y_true_cls)
        print(f"  {name:10s}: loss={loss:.4f}, grad_shape={grad.shape}, hess_shape={hess.shape}, init={init:.4f}")

    print("  [OK] All objectives passed\n")


def test_binning():
    print("=== Testing Histogram & Binning ===")
    rng = np.random.RandomState(42)
    X = rng.randn(500, 5)

    mapper = BinMapper(max_bins=32)
    bins = mapper.fit_transform(X[:, 0])
    print(f"  Feature 0: num_bins={mapper.num_bins}, unique_bins={len(np.unique(bins))}")

    hist = Histogram(mapper.num_bins)
    g = rng.randn(500)
    h = np.abs(rng.randn(500)) + 0.1
    hist.build(bins, g, h)
    print(f"  Histogram sum_g={hist.sum_gradients.sum():.4f}, sum_h={hist.sum_hessians.sum():.4f}, count={hist.count.sum()}")

    print("  [OK] Binning passed\n")


def test_dataset():
    print("=== Testing Dataset ===")
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = rng.randn(200)

    ds = Dataset(X, y, max_bins=32)
    ds.bin()
    print(f"  {ds}")
    print(f"  total_bins={ds.total_bins}, bins_per_feature={ds.num_bins_per_feature}")
    print("  [OK] Dataset passed\n")


def test_tree():
    print("=== Testing Decision Tree ===")
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5)
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(300)

    ds = Dataset(X, y, max_bins=32)
    ds.bin()

    obj = get_objective("mse")
    init = obj.init_score(y)
    y_pred = np.full(len(y), init)

    grad = obj.gradients(y, y_pred)
    hess = obj.hessians(y, y_pred)

    tree = DecisionTree(
        num_leaves=15,
        min_data_in_leaf=10,
        lambda_l2=1.0,
        learning_rate=0.1,
        max_bin=32,
    )
    t0 = time.time()
    tree.fit(ds.binned_data, np.array(ds.num_bins_per_feature), grad, hess, ds.total_bins)
    t1 = time.time()
    pred = tree.predict_by_indices(ds.binned_data)
    print(f"  Tree built in {t1-t0:.3f}s, leaves={tree.num_leaves_found()}, max_depth={tree.max_depth_found()}")
    print(f"  Train RMSE after 1 tree: {np.sqrt(np.mean((y - (init + pred))**2)):.4f}")
    print("  [OK] Tree passed\n")


def test_regressor():
    print("=== Testing LGBMRegressor ===")
    rng = np.random.RandomState(42)
    X_train = rng.randn(500, 10)
    y_train = np.sum(X_train[:, :3] ** 2, axis=1) + 0.5 * rng.randn(500)
    X_test = rng.randn(200, 10)
    y_test = np.sum(X_test[:, :3] ** 2, axis=1) + 0.5 * rng.randn(200)

    model = LGBMRegressor(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10,
        lambda_l2=1.0,
        verbose=False,
        seed=42,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    t1 = time.time()

    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    print(f"  Trained in {t1-t0:.3f}s")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Score (neg RMSE): {model.score(X_test, y_test):.4f}")
    print(f"  Params: {model.get_params()}")
    print("  [OK] Regressor passed\n")


def test_regressor_goss():
    print("=== Testing LGBMRegressor with GOSS ===")
    rng = np.random.RandomState(42)
    X_train = rng.randn(500, 10)
    y_train = np.sum(X_train[:, :3] ** 2, axis=1) + 0.5 * rng.randn(500)

    model = LGBMRegressor(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10,
        goss=True,
        goss_large_data_pct=0.2,
        verbose=False,
        seed=42,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()

    y_pred = model.predict(X_train)
    train_rmse = rmse(y_train, y_pred)
    print(f"  Trained in {t1-t0:.3f}s, Train RMSE: {train_rmse:.4f}")
    print("  [OK] GOSS Regressor passed\n")


def test_classifier():
    print("=== Testing LGBMClassifier ===")
    rng = np.random.RandomState(42)
    X_train = rng.randn(500, 10)
    w = rng.randn(5)
    logits = X_train[:, :5] @ w
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_train = (rng.rand(500) < probs).astype(float)

    X_test = rng.randn(200, 10)
    logits_test = X_test[:, :5] @ w
    probs_test = 1.0 / (1.0 + np.exp(-logits_test))
    y_test = (rng.rand(200) < probs_test).astype(float)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=15,
        min_data_in_leaf=10,
        verbose=False,
        seed=42,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    t1 = time.time()

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    y_proba = model.predict_proba(X_test)[:, 1]
    ll = logloss(y_test, y_proba)
    auc_val = auc(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"  Trained in {t1-t0:.3f}s")
    print(f"  Accuracy: {acc:.4f}, Logloss: {ll:.4f}, AUC: {auc_val:.4f}, F1: {f1:.4f}")
    print(f"  Params: {model.get_params()}")
    print("  [OK] Classifier passed\n")


def test_early_stopping():
    print("=== Testing Early Stopping ===")
    rng = np.random.RandomState(42)
    X_train = rng.randn(300, 5)
    y_train = np.sum(X_train[:, :2], axis=1) + 0.5 * rng.randn(300)
    X_val = rng.randn(100, 5)
    y_val = np.sum(X_val[:, :2], axis=1) + 0.5 * rng.randn(100)

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=10,
        early_stopping_rounds=20,
        verbose=False,
        seed=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    actual_iters = len(model._model.trees)
    print(f"  Requested 500 iterations, stopped at {actual_iters}")
    print("  [OK] Early stopping passed\n")


def test_serialization():
    print("=== Testing Model Serialization ===")
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = np.sum(X[:, :2], axis=1) + 0.1 * rng.randn(200)

    model = LGBMRegressor(n_estimators=20, learning_rate=0.1, num_leaves=10, verbose=False, seed=42)
    model.fit(X, y)
    y_pred_before = model.predict(X)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    model.save_model(path)

    model2 = LGBMRegressor(verbose=False)
    model2.load_model(path)
    y_pred_after = model2.predict(X)

    max_diff = np.max(np.abs(y_pred_before - y_pred_after))
    print(f"  Max prediction diff after save/load: {max_diff:.10f}")
    assert max_diff < 1e-10, "Save/load mismatch!"
    os.unlink(path)
    print("  [OK] Serialization passed\n")


def test_metrics():
    print("=== Testing Metrics ===")
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_pred = y_true + 0.1 * rng.randn(100)
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  MAE:  {mae(y_true, y_pred):.4f}")
    print(f"  MSE:  {mse(y_true, y_pred):.4f}")

    y_true_cls = (rng.rand(100) > 0.5).astype(float)
    y_score = rng.rand(100)
    y_pred_cls = (y_score >= 0.5).astype(float)
    print(f"  Logloss: {logloss(y_true_cls, y_score):.4f}")
    print(f"  Accuracy: {accuracy(y_true_cls, y_pred_cls):.4f}")
    print(f"  AUC: {auc(y_true_cls, y_score):.4f}")
    print(f"  F1: {f1_score(y_true_cls, y_pred_cls):.4f}")
    print("  [OK] Metrics passed\n")


if __name__ == "__main__":
    print("lgbm_numpy — Smoke Tests\n")
    t_start = time.time()

    test_objectives()
    test_binning()
    test_dataset()
    test_tree()
    test_regressor()
    test_regressor_goss()
    test_classifier()
    test_early_stopping()
    test_serialization()
    test_metrics()

    t_total = time.time() - t_start
    print(f"All tests passed in {t_total:.2f}s")
