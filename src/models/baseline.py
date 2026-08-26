import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
