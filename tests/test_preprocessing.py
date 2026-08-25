import pandas as pd
import numpy as np
import pytest
from src.data.clean import handle_infinite_and_nan, impute_remaining_nulls, remove_duplicates, drop_high_null_columns


def test_handle_infinite_and_nan():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, np.nan], "b": [1, 2, 3, 4]})
    cleaned = handle_infinite_and_nan(df)
    assert np.isnan(cleaned.loc[1, "a"])
    assert np.isnan(cleaned.loc[2, "a"])
    assert np.isnan(cleaned.loc[3, "a"])


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 2


def test_drop_high_null_columns():
    df = pd.DataFrame({"a": [1, 2, None, None], "b": [1, 2, 3, 4]})
    cleaned = drop_high_null_columns(df, threshold=0.4)
    assert "a" not in cleaned.columns
    assert "b" in cleaned.columns


def test_impute_remaining_nulls():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "x"]})
    cleaned = impute_remaining_nulls(df)
    assert cleaned["a"].isnull().sum() == 0
