import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.config import MODELS_DIR, MODEL_PARAMS
from src.explainability.shap_explainer import Explainer


class InferenceEngine:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(MODELS_DIR)
        self.model_dir = model_dir
        self.model = None
        self.explainer = None
        self.label_encoder = None
        self.feature_names: List[str] = []
        self.model_version = "v1"

    def load(self) -> None:
        model_path = os.path.join(self.model_dir, "model_v1.pkl")
        explainer_path = os.path.join(self.model_dir, "shap_explainer_v1.pkl")
        le_path = os.path.join(self.model_dir, "label_encoder_v1.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)
        if os.path.exists(explainer_path):
            self.explainer = joblib.load(explainer_path)
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)

    def _dict_to_vector(self, data: Dict[str, Any]) -> np.ndarray:
        if not self.feature_names:
            raise RuntimeError("Feature names not set")
        row = []
        for f in self.feature_names:
            val = data.get(f, 0.0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            row.append(val)
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        x = self._dict_to_vector(features)
        proba = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        if self.label_encoder:
            pred_label = str(self.label_encoder.inverse_transform([pred_idx])[0])
        else:
            pred_label = str(pred_idx)
        is_malicious = pred_label.lower() != "benign"
        explanation = {"top_features": []}
        if self.explainer:
            try:
                explanation = self.explainer.explain_instance(x[0])
            except Exception:
                explanation = {"top_features": []}
        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "is_malicious": is_malicious,
            "top_features": explanation.get("top_features", []),
            "model_version": self.model_version,
        }
