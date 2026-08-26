# Technical Report: Explainable Network Intrusion Detection

## 1. Introduction

This report documents the architectural decisions, trade-offs, and results for the Explainable Network Intrusion Detection System.

## 2. Dataset & Preprocessing

### Dataset Choice
CIC-IDS2017 was selected for its realistic flow-based traffic, modern attack diversity, and widespread recognition in security ML literature.

### Preprocessing Pipeline
1. **Ingestion:** Consolidate multiple daily CSVs into a single DataFrame.
2. **Cleaning:** Replace infinite/NaN values (common when flow duration = 0), drop columns with >30% nulls, impute remaining nulls with median, remove duplicates.
3. **Feature Engineering:** Drop identifier columns to prevent label leakage, remove highly correlated features (threshold 0.95), remove low-variance features (threshold 0.01).
4. **Scaling:** RobustScaler to handle outliers in byte counts and duration.
5. **Splitting:** Stratified train/validation/test split to preserve class distribution.

## 3. Model Selection

- **Baseline:** Logistic Regression (sanity check)
- **Primary:** XGBoost / LightGBM (gradient boosted trees — strong tabular performance, fast inference, native SHAP compatibility)
- **Optional comparison:** LSTM (treating flows as sequences) — stretch goal

**Rationale:** Tree-based models are known to excel on tabular data and integrate seamlessly with SHAP TreeExplainer, making them ideal for an explainability-first project.

## 4. Explainability Strategy

- **Global explainability:** SHAP summary plot identifying the most influential features across the entire test set.
- **Local explainability:** SHAP force plot / waterfall plot for individual predictions.
- **API integration:** Top-5 SHAP features returned with every prediction in the FastAPI response, enabling the dashboard to display real-time explanations.

## 5. Adversarial Robustness

- **Method:** Select correctly classified attack samples, apply Gaussian noise perturbation within realistic bounds, measure flip rate and average confidence drop.
- **Purpose:** Demonstrate model behavior under perturbation and produce a discussion point for interviews.

## 6. Deployment Architecture

- **API:** FastAPI + Uvicorn with Pydantic validation and CORS support.
- **Dashboard:** Streamlit for rapid prototyping of the demo UI.
- **Containerization:** Separate Dockerfiles for API and dashboard, orchestrated with `docker-compose.yml`.
- **Deployment:** Target HuggingFace Spaces or Render for public demo hosting.

## 7. Results & Discussion

*(To be populated after training and evaluation runs.)*

## 8. Trade-offs

| Decision | Trade-off |
|----------|-----------|
| Tree-based model over deep learning | Faster training and SHAP compatibility; may lose subtle pattern capture that LSTMs can exploit |
| SHAP over LIME | TreeExplainer is exact and fast for trees; LIME is model-agnostic but slower and less stable |
| Streamlit over React | Faster to build and deploy; less customization than a full React app |
| Synthetic perturbation over real adversarial examples | Simpler to implement; less representative of real evasion techniques |

## 9. Future Work

- Graph Neural Network approach
- Live packet capture pipeline
- Drift detection and continuous retraining
- Cross-dataset generalization testing
