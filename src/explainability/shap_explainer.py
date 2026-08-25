import joblib
import numpy as np
import pandas as pd
import shap
from typing import Optional, List, Dict, Any
from src.config import MODEL_PARAMS, SHAP_CONFIG, MODELS_DIR
from src.models.boosted_trees import train_xgboost, train_lightgbm


class Explainer:
    def __init__(self, model, background_data: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or []
        self.explainer = None
        self._fitted = False

    def fit(self, X: np.ndarray, max_samples: int = SHAP_CONFIG["sample_size"]) -> None:
        sample = X[:max_samples] if len(X) > max_samples else X
        try:
            self.explainer = shap.TreeExplainer(self.model)
            shap_values = self.explainer.shap_values(sample)
            self._fitted = True
        except Exception:
            self.explainer = shap.KernelExplainer(self.model.predict_proba, sample[:50])
            shap_values = self.explainer.shap_values(sample[:50])
            self._fitted = True

    def explain_instance(self, x: np.ndarray) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Explainer not fitted")
        x = x.reshape(1, -1)
        shap_values = self.explainer.shap_values(x)
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        if vals.ndim > 1:
            vals = vals[:, 0]
        indices = np.argsort(np.abs(vals))[::-1][:5]
        features = []
        for idx in indices:
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            features.append({"feature": name, "shap_value": float(vals[idx])})
        return {"top_features": features}

    def global_summary(self, X: np.ndarray, save_path: Optional[str] = None) -> None:
        if not self._fitted:
            raise RuntimeError("Explainer not fitted")
        shap_values = self.explainer.shap_values(X[:1000])
        shap.summary_plot(shap_values, X[:1000], show=False, max_display=SHAP_CONFIG["max_display"])
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
