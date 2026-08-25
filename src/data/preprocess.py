import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from src.config import TEST_SIZE, VAL_SIZE, RANDOM_SEED, TARGET_COLUMN, PROCESSED_DATA_DIR
from src.data.clean import handle_infinite_and_nan, impute_remaining_nulls, remove_duplicates
from src.data.ingest import drop_identifier_columns


def full_clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_infinite_and_nan(df)
    df = drop_high_null_columns(df)
    df = impute_remaining_nulls(df)
    df = remove_duplicates(df)
    df = drop_identifier_columns(df)
    return df


def encode_labels(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    return df, le


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "robust",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    null_frac = df.isnull().mean()
    drop_cols = null_frac[null_frac > threshold].index.tolist()
    return df.drop(columns=drop_cols, errors="ignore")
