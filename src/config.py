import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
TARGET_COLUMN = "Label"
DROP_COLUMNS = ["Timestamp", "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port"]

MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    },
}

SHAP_CONFIG = {
    "max_display": 10,
    "sample_size": 1000,
}

API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

DASHBOARD_PORT = 8501

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
