# Product Requirements Document (PRD)
## Project: Explainable Network Intrusion Detection System
**Repository:** `explainable-network-intrusion-detection`
**Version:** 1.0
**Status:** Draft for team kickoff

---

## 1. Overview

### 1.1 Problem Statement
Traditional signature-based Intrusion Detection Systems (IDS) fail against novel/zero-day attacks and generate high false-positive rates that overwhelm security teams. ML-based IDS improve detection accuracy but are typically "black boxes" — a Security Operations Center (SOC) analyst cannot easily trust or act on a flagged alert without knowing *why* it was flagged.

This project builds a network intrusion detection system that:
1. Classifies network traffic as benign or malicious (with attack-type granularity)
2. Explains every prediction using explainability techniques (SHAP/LIME)
3. Demonstrates resilience awareness against adversarial evasion attempts
4. Is deployable as a real-time scoring API, not just a notebook

### 1.2 Why This Project (Context)
Built as a portfolio-grade "advanced" ML project targeting international AI/ML engineering roles (SOC tooling, threat detection, applied ML/security). The differentiator versus typical student IDS projects is the **explainability layer + adversarial robustness check + production-style deployment**, not just a classifier with an accuracy score.

### 1.3 Target Audience for the Project Itself
- Primary: Hiring managers/recruiters reviewing the GitHub repo
- Secondary: Interviewers who may ask you to walk through design decisions
- Tertiary: A hypothetical SOC analyst persona who is the "end user" of the explainability output — this framing should show up in the README/UI

---

## 2. Goals & Success Metrics

### 2.1 Functional Goals
- [ ] Multi-class classification of network flows (benign vs. specific attack categories: DoS, Probe, R2L, U2R, or dataset-equivalent categories)
- [ ] Explainability report generated per prediction (top contributing features)
- [ ] Real-time-style inference API (single flow or batch scoring)
- [ ] Basic adversarial evasion test demonstrating model behavior under perturbation
- [ ] Dashboard/UI to visualize alerts + explanations (even minimal)

### 2.2 Non-Functional Goals
- Reproducible pipeline (anyone can clone → run → get same results)
- Clean, documented, modular codebase (not one giant notebook)
- Deployed and publicly demoable (not just "runs on my machine")
- Professional documentation (README, architecture diagram, model card)

### 2.3 Success Metrics (Model-Level)
| Metric | Target | Why |
|---|---|---|
| Macro F1-score | > 0.90 | Class imbalance makes accuracy misleading |
| Recall on minority attack classes (e.g., U2R, R2L) | > 0.70 | These are rare but high-severity |
| False Positive Rate | < 5% | Alert fatigue is a real SOC problem |
| Inference latency (single flow) | < 200ms | Needed for "real-time" framing |
| SHAP explanation generation time | < 1s per prediction | UX for the dashboard |

### 2.4 Success Metrics (Project-Level)
- Public GitHub repo with clean commit history and README
- Live demo link (HuggingFace Spaces / Render / Railway)
- Model card published (e.g., on Hugging Face Hub)
- Written technical report explaining architectural decisions and trade-offs

---

## 3. Scope

### 3.1 In Scope
- Binary + multi-class classification on flow-based network traffic
- Classical ML (XGBoost/LightGBM) as primary model
- Optional deep learning comparison (LSTM or simple feedforward NN) as a benchmark
- SHAP-based global + local explainability
- One adversarial robustness experiment (evasion via feature perturbation, e.g., using a simple FGSM-style or manual perturbation test)
- REST API for inference (FastAPI)
- Minimal frontend dashboard (Streamlit or simple React page) to visualize alerts
- Dockerized deployment

### 3.2 Out of Scope (v1)
- Real packet capture / live network sniffing (use existing labeled datasets, not live traffic)
- Full production-grade SIEM integration
- Graph Neural Network approach (noted as a stretch/future goal, not v1 requirement)
- Multi-tenant/auth system for the dashboard
- Continuous retraining/drift monitoring pipeline (mention as "future work" in README)

---

## 4. Dataset

### 4.1 Recommended Dataset
**CIC-IDS2017** (Canadian Institute for Cybersecurity) — recommended primary choice because:
- Realistic, labeled, flow-based traffic (via CICFlowMeter, 80+ features)
- Covers modern attack types: DoS, DDoS, Brute Force, Infiltration, Botnet, Port Scan, Web Attacks
- Widely recognized in security ML literature (interviewers will know it)

**Alternative/secondary datasets** (team can choose one as primary, or use for cross-dataset generalization testing):
- `NSL-KDD` — smaller, classic benchmark, good for quick iteration
- `UNSW-NB15` — modern attack diversity, good secondary validation set

### 4.2 Data Handling Requirements
- Document class distribution (expect severe imbalance — some attack classes <0.1% of data)
- Train/validation/test split must be stratified
- No data leakage: ensure flows from same session don't span train/test
- Store raw data outside git (use `.gitignore` + a `data/README.md` with download instructions), since these datasets are large (multi-GB)

---

## 5. System Architecture & Execution Flow

### 5.1 High-Level Pipeline

```
[Raw PCAP/CSV Flow Data]
        ↓
[1. Data Ingestion & Cleaning]
        ↓
[2. Feature Engineering & Preprocessing]
        ↓
[3. Handle Class Imbalance (SMOTE/class weights)]
        ↓
[4. Model Training (XGBoost/LightGBM + optional LSTM)]
        ↓
[5. Model Evaluation (per-class metrics, confusion matrix)]
        ↓
[6. Explainability Layer (SHAP — global + local)]
        ↓
[7. Adversarial Robustness Test]
        ↓
[8. Model Export (serialized .pkl / ONNX)]
        ↓
[9. Inference API (FastAPI) ← model + SHAP explainer loaded here]
        ↓
[10. Dashboard (Streamlit/React) — sends traffic samples to API, displays verdict + explanation]
        ↓
[11. Dockerize both API + Dashboard]
        ↓
[12. Deploy (HF Spaces / Render / Railway)]
```

### 5.2 Step-by-Step Execution Detail

**Step 1 — Data Ingestion**
- Download CIC-IDS2017 CSVs
- Load into pandas, handle mixed types, drop/flag infinite or NaN values (common in this dataset due to flow duration=0 cases)
- Consolidate multiple attack-day CSVs into one labeled dataset

**Step 2 — Feature Engineering**
- Drop identifier columns that would leak the label (e.g., raw IPs/ports if they correlate trivially with source files)
- Correlation analysis to drop redundant features (CIC-IDS2017 has many near-duplicate flow features)
- Normalize/scale features (StandardScaler or RobustScaler given outliers in flow duration/byte counts)
- Encode labels (benign=0, attack categories=1..n)

**Step 3 — Class Imbalance Handling**
- Compute class weights for tree-based models
- Optionally apply SMOTE/ADASYN on minority classes in the training set only (never on validation/test)
- Document before/after class distribution

**Step 4 — Model Training**
- Baseline: Logistic Regression (sanity check)
- Primary: XGBoost or LightGBM (gradient boosted trees — strong tabular performance, fast, SHAP-compatible natively)
- Comparison model: simple LSTM or 1D-CNN if treating flows as sequences (stretch goal, not blocking)
- Hyperparameter tuning via Optuna or GridSearchCV (document search space and final params)

**Step 5 — Evaluation**
- Per-class precision/recall/F1 (not just overall accuracy)
- Confusion matrix (normalized)
- ROC-AUC per class (one-vs-rest)
- Document which attack types are hardest to detect and hypothesize why

**Step 6 — Explainability**
- Global: SHAP summary plot — which features matter most across the whole model
- Local: SHAP force/waterfall plot — for a single flagged flow, show which features pushed it toward "malicious"
- This output must be exposed through the API (not just a notebook plot) — return top-5 contributing features with their SHAP values in the API response

**Step 7 — Adversarial Robustness Test**
- Pick a successfully-detected attack sample
- Manually perturb a small number of features within realistic bounds (e.g., slightly alter packet timing/size features) to see if the model's confidence drops or flips
- Document findings: "the model's decision changed with X% feature perturbation" — this is a strong interview talking point even if the result is simple

**Step 8 — Model Export**
- Serialize final model (`joblib`/`pickle`, or export to ONNX for portability)
- Save the fitted SHAP explainer alongside it
- Version the model artifact (e.g., `model_v1.pkl`) — do not commit large binaries to git; use Git LFS or store in HF Hub/S3 and download at build time

**Step 9 — Inference API**
- FastAPI service with endpoints:
  - `POST /predict` — single flow → verdict + confidence + top SHAP features
  - `POST /predict/batch` — CSV upload → batch verdicts
  - `GET /health` — health check
  - `GET /model/info` — model version, training date, metrics summary
- Pydantic schema validation for input features

**Step 10 — Dashboard**
- Simple Streamlit app (fastest to build) OR a lightweight React page if the team wants a more "product" feel
- Features: upload/select a traffic sample → show prediction + SHAP explanation chart + confidence score
- This is what you'll screen-record for the README demo GIF

**Step 11 — Containerization**
- Separate Dockerfiles for API and dashboard, or a `docker-compose.yml` running both
- Keep image lean (multi-stage build, avoid shipping training dependencies in the inference image)

**Step 12 — Deployment**
- Recommended: Hugging Face Spaces (free, good for ML demos, easy Docker support) or Render free tier
- Add the live link prominently in the README

---

## 6. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.10+ | |
| Data processing | pandas, numpy | |
| Class imbalance | imbalanced-learn (SMOTE/ADASYN) | |
| ML modeling | scikit-learn, XGBoost or LightGBM | Tree-based = fast + SHAP-native support |
| Deep learning (optional) | PyTorch | Only if team builds the LSTM comparison |
| Explainability | SHAP | LIME as a secondary/optional comparison |
| Hyperparameter tuning | Optuna | |
| Experiment tracking | MLflow or Weights & Biases | Tracks runs, params, metrics — strong "advanced" signal |
| API | FastAPI + Uvicorn | |
| Data validation | Pydantic | |
| Dashboard | Streamlit (fastest) or React + Vite | |
| Containerization | Docker, docker-compose | |
| Deployment | Hugging Face Spaces / Render / Railway | Free-tier friendly |
| Model hosting | Hugging Face Hub (model card) | |
| Version control | Git + GitHub, Git LFS for artifacts | |
| CI (stretch) | GitHub Actions (lint + test on push) | Strong MLOps signal if time allows |

---

## 7. File / Repository Structure

```
explainable-network-intrusion-detection/
├── README.md                        # Project overview, architecture diagram, demo link, results
├── LICENSE
├── .gitignore
├── requirements.txt
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml                   # (stretch) lint/test on push
│
├── data/
│   ├── README.md                    # Dataset download instructions (raw data NOT committed)
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_explainability_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Paths, constants, hyperparameter defaults
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py                # Load & consolidate raw CSVs
│   │   ├── clean.py                 # Handle NaN/inf, drop leakage columns
│   │   └── preprocess.py            # Scaling, encoding, train/val/test split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py           # Correlation filtering, feature selection
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                 # Training loop, hyperparameter tuning
│   │   ├── baseline.py              # Logistic regression baseline
│   │   ├── boosted_trees.py         # XGBoost/LightGBM model
│   │   ├── deep_model.py            # (optional) LSTM/CNN comparison
│   │   └── evaluate.py              # Metrics, confusion matrix, ROC-AUC
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py        # Fit/load SHAP explainer, generate plots
│   │   └── adversarial_test.py      # Perturbation/evasion experiment
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app entrypoint
│       ├── schemas.py               # Pydantic request/response models
│       └── inference.py             # Load model + explainer, run predictions
│
├── dashboard/
│   ├── app.py                       # Streamlit app (or /src, /public if React)
│   └── requirements.txt
│
├── models/
│   ├── model_v1.pkl                 # (or tracked via Git LFS / external storage)
│   └── shap_explainer_v1.pkl
│
├── reports/
│   ├── figures/                     # Confusion matrices, SHAP plots, ROC curves
│   ├── model_card.md                # Hugging Face-style model card
│   └── technical_report.md          # Full write-up: decisions, trade-offs, results
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model_inference.py
│   └── test_api.py
│
├── Dockerfile.api
├── Dockerfile.dashboard
└── docs/
    └── architecture_diagram.png     # Visual version of the pipeline above
```

---

## 8. API Contract (Draft)

**POST `/predict`**
Request:
```json
{
  "flow_duration": 128473,
  "total_fwd_packets": 12,
  "total_bwd_packets": 8,
  "...": "...(remaining CICFlowMeter features)"
}
```
Response:
```json
{
  "prediction": "DoS Hulk",
  "confidence": 0.94,
  "is_malicious": true,
  "top_features": [
    {"feature": "flow_duration", "shap_value": 0.31},
    {"feature": "fwd_packet_length_max", "shap_value": 0.22},
    {"feature": "bwd_packets_per_sec", "shap_value": -0.11}
  ],
  "model_version": "v1"
}
```

---

## 9. Team Roles (adjust to your team size)

| Role | Responsibilities |
|---|---|
| Data/ML Lead | Data pipeline, preprocessing, class imbalance handling |
| Model Lead | Training, tuning, evaluation, model comparison |
| Explainability/Research Lead | SHAP integration, adversarial test, technical report |
| Backend/API Lead | FastAPI service, Docker, deployment |
| Frontend/Dashboard Lead | Streamlit/React dashboard, demo polish |
| Docs/PM | README, architecture diagram, model card, video demo |

(On a small team, one person can own 2 roles — e.g., Backend + Docs.)

---

## 10. Milestones & Timeline (suggested — adjust to your pace)

| Week | Milestone |
|---|---|
| Week 1 | Repo setup, dataset acquired, EDA complete, cleaning + preprocessing pipeline done |
| Week 2 | Baseline + primary model trained, class imbalance strategy validated, initial metrics documented |
| Week 3 | SHAP explainability integrated (global + local), adversarial test run and documented |
| Week 4 | FastAPI built, model served, dashboard built, Docker setup, deployed live, README + technical report + model card finalized |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dataset is large (multi-GB), hard to handle in git | Keep raw data out of git; document download steps; work with a sampled subset during dev |
| Class imbalance tanks minority-class recall | Use SMOTE/class weights + report per-class metrics honestly, don't hide behind accuracy |
| SHAP is slow on large models/datasets | Use `TreeExplainer` (fast, exact for tree models) instead of `KernelExplainer`; explain on a sample, not the full test set, for the summary plot |
| Free-tier deployment sleeps/has cold starts | Mention this explicitly in README so it doesn't look broken during review |
| Scope creep (GNN, live packet capture, etc.) | Keep those explicitly in a "Future Work" section — don't block v1 on them |

---

## 12. Deliverables Checklist

- [ ] Clean, modular codebase in `src/`
- [ ] Trained model + evaluation report (per-class metrics, confusion matrix)
- [ ] SHAP explainability integrated into both training analysis and live API responses
- [ ] Adversarial robustness experiment documented with results
- [ ] FastAPI inference service (dockerized)
- [ ] Dashboard for demo purposes (dockerized)
- [ ] Live deployed demo link
- [ ] `README.md` with architecture diagram, results table, and demo GIF/video
- [ ] `model_card.md` (published to Hugging Face Hub if possible)
- [ ] `technical_report.md` explaining every major design decision and trade-off

---

## 13. Future Work (explicitly out of v1, mention in README)
- Graph Neural Network approach modeling network topology
- Live packet capture (real-time sniffing via Scapy/PyShark) instead of static datasets
- Continuous retraining pipeline with drift detection
- Cross-dataset generalization testing (train on CIC-IDS2017, test on UNSW-NB15)
