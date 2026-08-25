import os
import glob
import pandas as pd
from typing import List, Optional
from src.config import RAW_DATA_DIR, DROP_COLUMNS, TARGET_COLUMN


def load_csv_files(file_pattern: str = "*.csv", directory: Optional[str] = None) -> pd.DataFrame:
    if directory is None:
        directory = str(RAW_DATA_DIR)
    files = glob.glob(os.path.join(directory, file_pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def consolidate_cic_ids2017(raw_dir: Optional[str] = None) -> pd.DataFrame:
    if raw_dir is None:
        raw_dir = str(RAW_DATA_DIR)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    files = []
    for day in days:
        pattern = os.path.join(raw_dir, f"*{day}*.csv")
        files.extend(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CIC-IDS2017 files found in {raw_dir}")
    frames = []
    for f in sorted(files):
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def basic_schema_check(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")


def drop_identifier_columns(df: pd.DataFrame, extra_drop: Optional[List[str]] = None) -> pd.DataFrame:
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    if extra_drop:
        drop_cols.extend([c for c in extra_drop if c in df.columns])
    return df.drop(columns=drop_cols, errors="ignore")
