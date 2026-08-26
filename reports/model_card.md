# Model Card: Explainable Network Intrusion Detection v1

## Model Details

- **Model type:** XGBoost Gradient Boosted Trees
- **Version:** v1
- **Training date:** (update after training)
- **Dataset:** CIC-IDS2017 (flow-based network traffic)
- **Task:** Multi-class classification (benign + attack categories)

## Intended Use

This model is intended for research and educational purposes to demonstrate explainable ML for network intrusion detection. It is not intended for production use without further validation and hardening.

## Metrics

| Metric | Value |
|--------|-------|
| Macro F1-score | (to be filled after training) |
| Macro Precision | (to be filled after training) |
| Macro Recall | (to be filled after training) |
| ROC-AUC (macro, OvR) | (to be filled after training) |

## Training Data

- **Source:** CIC-IDS2017
- **Size:** Multi-GB (see `data/README.md`)
- **Splits:** Stratified train / validation / test
- **Class imbalance handling:** Class weights + optional SMOTE

## Explainability

- **Method:** SHAP TreeExplainer
- **Global:** Summary plot across 1000 samples
- **Local:** Top-5 contributing features per prediction returned by API

## Adversarial Robustness

- **Test:** Feature perturbation (Gaussian noise injection) on correctly classified attack samples
- **Findings:** (to be documented after experiment)

## Limitations

- Trained on CIC-IDS2017 only; generalization to other networks is untested
- Free-tier deployment may have cold-start latency
- Not intended for real-time live packet capture without adaptation
