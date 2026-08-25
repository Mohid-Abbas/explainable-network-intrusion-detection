import numpy as np
import pandas as pd
import pytest
from src.models.boosted_trees import train_xgboost, train_lightgbm
from src.models.baseline import train_logistic_regression
from src.models.evaluate import compute_metrics


@pytest.fixture
def toy_data():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = pd.Series(np.random.randint(0, 3, size=200))
    return X, y


def test_train_xgboost(toy_data):
    X, y = toy_data
    model = train_xgboost(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_lightgbm(toy_data):
    X, y = toy_data
    model = train_lightgbm(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_logistic_regression(toy_data):
    X, y = toy_data
    model = train_logistic_regression(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_compute_metrics(toy_data):
    X, y = toy_data
    model = train_xgboost(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    metrics = compute_metrics(y, preds, y_proba=proba, labels=[0, 1, 2])
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["macro_f1"] > 0
