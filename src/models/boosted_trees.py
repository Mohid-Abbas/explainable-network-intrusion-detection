import joblib
import numpy as np
import pandas as pd
from typing import Optional
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import MODEL_PARAMS


def train_xgboost(X_train: np.ndarray, y_train: pd.Series, params: Optional[dict] = None) -> XGBClassifier:
    if params is None:
        params = MODEL_PARAMS["xgboost"]
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: np.ndarray, y_train: pd.Series, params: Optional[dict] = None) -> LGBMClassifier:
    if params is None:
        params = MODEL_PARAMS["lightgbm"]
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model
