import optuna
import numpy as np
import pandas as pd
from typing import Optional
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.config import MODEL_PARAMS, RANDOM_SEED
from src.models.baseline import train_logistic_regression
from src.models.boosted_trees import train_xgboost, train_lightgbm


def objective_xgb(trial, X: np.ndarray, y: pd.Series, cv_splits: int = 3) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


def hyperparameter_tune(
    X: np.ndarray,
    y: pd.Series,
    model_type: str = "xgboost",
    n_trials: int = 30,
) -> dict:
    if model_type == "xgboost":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective_xgb(t, X, y), n_trials=n_trials)
        return study.best_params
    else:
        raise NotImplementedError(f"Tuning not implemented for {model_type}")


def run_training_pipeline(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_type: str = "xgboost",
    tune: bool = False,
) -> dict:
    if model_type == "logistic_regression":
        model = train_logistic_regression(X_train, y_train)
    elif model_type == "xgboost":
        if tune:
            best_params = hyperparameter_tune(X_train, y_train, model_type="xgboost", n_trials=20)
            model = train_xgboost(X_train, y_train, params=best_params)
        else:
            model = train_xgboost(X_train, y_train)
    elif model_type == "lightgbm":
        model = train_lightgbm(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return {"model": model, "model_type": model_type}
