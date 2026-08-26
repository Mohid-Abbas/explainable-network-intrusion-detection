
import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN


def handle_infinite_and_nan(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    null_frac = df.isnull().mean()
    drop_cols = null_frac[null_frac > threshold].index.tolist()
    return df.drop(columns=drop_cols, errors="ignore")


def impute_remaining_nulls(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if strategy == "median":
        fill_values = df[numeric_cols].median()
    elif strategy == "mean":
        fill_values = df[numeric_cols].mean()
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        if col == TARGET_COLUMN:
            continue
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()
