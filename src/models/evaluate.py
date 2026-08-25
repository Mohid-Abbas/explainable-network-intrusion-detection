import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.config import FIGURES_DIR, RANDOM_SEED


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    results = {
        "classification_report": report,
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }
    if y_proba is not None and labels is not None:
        try:
            roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            results["roc_auc_macro"] = roc
        except Exception:
            results["roc_auc_macro"] = None
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
