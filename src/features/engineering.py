import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")


def remove_low_variance_features(X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    keep_cols = X.columns[sel.get_support()]
    return X[keep_cols]


def remove_highly_correlated_features(
    X: pd.DataFrame, threshold: float = 0.95
) -> pd.DataFrame:
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=drop_cols, errors="ignore")
