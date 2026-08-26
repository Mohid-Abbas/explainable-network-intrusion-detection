import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.config import (
    DROP_COLUMNS,
    FIGURES_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)
from src.data.clean import (
    drop_high_null_columns,
    handle_infinite_and_nan,
    impute_remaining_nulls,
    remove_duplicates,
)
from src.data.ingest import basic_schema_check, consolidate_cic_ids2017
from src.data.preprocess import encode_labels, scale_features, stratified_split
from src.explainability.shap_explainer import Explainer
from src.features.engineering import (
    remove_highly_correlated_features,
    remove_low_variance_features,
)
from src.models.baseline import train_logistic_regression
from src.models.boosted_trees import train_lightgbm, train_xgboost
from src.models.evaluate import compute_metrics, plot_confusion_matrix


def run_pipeline(model_type: str = "xgboost", use_smote: bool = False):
    print("[1/8] Loading raw data...")
    df = consolidate_cic_ids2017(str(RAW_DATA_DIR))
    print(f"  Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    basic_schema_check(df)

    print("[2/8] Cleaning...")
    df = handle_infinite_and_nan(df)
    df = drop_high_null_columns(df, threshold=0.3)
    df = impute_remaining_nulls(df, strategy="median")
    df = remove_duplicates(df)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    print(f"  After cleaning: {df.shape[0]:,} rows, {df.shape[1]} columns")

    print("[3/8] Encoding labels...")
    df, label_encoder = encode_labels(df, target_col=TARGET_COLUMN)
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    print(f"  Classes: {dict(class_counts)}")

    print("[4/8] Feature engineering...")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X = remove_low_variance_features(X, threshold=0.01)
    X = remove_highly_correlated_features(X, threshold=0.95)
    feature_names = X.columns.tolist()
    print(f"  Features after selection: {len(feature_names)}")

    print("[5/8] Splitting and scaling...")
    train_df = pd.concat([X, y], axis=1)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(train_df, target_col=TARGET_COLUMN)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test, scaler_type="robust")
    print(f"  Train: {X_train_s.shape}, Val: {X_val_s.shape}, Test: {X_test_s.shape}")

    print(f"[6/8] Training {model_type}...")
    if model_type == "xgboost":
        model = train_xgboost(X_train_s, y_train)
    elif model_type == "lightgbm":
        model = train_lightgbm(X_train_s, y_train)
    elif model_type == "logistic_regression":
        model = train_logistic_regression(X_train_s, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print("[7/8] Evaluating...")
    test_preds = model.predict(X_test_s)
    test_proba = model.predict_proba(X_test_s)
    labels = label_encoder.classes_.tolist()
    metrics = compute_metrics(y_test, test_preds, y_proba=test_proba, labels=labels)
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    if metrics.get("roc_auc_macro"):
        print(f"  ROC AUC (macro): {metrics['roc_auc_macro']:.4f}")

    cm_path = str(FIGURES_DIR / "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], labels, save_path=cm_path)
    print(f"  Saved confusion matrix to {cm_path}")

    print("[8/8] Explainability and serialization...")
    explainer = Explainer(model, background_data=X_train_s[:1000], feature_names=feature_names)
    explainer.fit(X_train_s)

    summary_path = str(FIGURES_DIR / "shap_summary.png")
    explainer.global_summary(X_test_s, save_path=summary_path)
    print(f"  Saved SHAP summary to {summary_path}")

    model_path = str(MODELS_DIR / "model_v1.pkl")
    explainer_path = str(MODELS_DIR / "shap_explainer_v1.pkl")
    le_path = str(MODELS_DIR / "label_encoder_v1.pkl")
    scaler_path = str(MODELS_DIR / "scaler_v1.pkl")

    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    joblib.dump(label_encoder, le_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved model to {model_path}")
    print(f"  Saved explainer to {explainer_path}")
    print(f"  Saved label encoder to {le_path}")
    print(f"  Saved scaler to {scaler_path}")

    metrics_path = str(MODELS_DIR / "metrics_v1.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_type": model_type,
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "roc_auc_macro": metrics.get("roc_auc_macro"),
            "num_features": len(feature_names),
            "train_size": int(X_train_s.shape[0]),
            "test_size": int(X_test_s.shape[0]),
        }, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    print("\nPipeline complete.")
    return model, explainer, label_encoder, scaler, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm", "logistic_regression"])
    args = parser.parse_args()
    run_pipeline(model_type=args.model)
