from typing import Any

import numpy as np
import pandas as pd


def feature_perturbation_test(
    model,
    X: np.ndarray,
    y: pd.Series,
    indices: np.ndarray,
    noise_scale: float = 0.1,
    feature_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    X_adv = X[indices].copy()
    original_preds = model.predict(X_adv)
    original_proba = model.predict_proba(X_adv)
    noise = np.random.normal(0, noise_scale, X_adv.shape)
    X_adv = X_adv + noise
    if feature_range is not None:
        X_adv = np.clip(X_adv, feature_range[0], feature_range[1])
    adv_preds = model.predict(X_adv)
    adv_proba = model.predict_proba(X_adv)
    flip_count = int(np.sum(original_preds != adv_preds))
    conf_drop = float(np.mean(np.max(original_proba, axis=1) - np.max(adv_proba, axis=1)))
    return {
        "flip_count": flip_count,
        "total_tested": len(indices),
        "flip_rate": flip_count / len(indices),
        "avg_confidence_drop": conf_drop,
    }
