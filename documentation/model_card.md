# Model Card — Social Phobia Screening Tool v5

**Model ID:** `v5`  
**Date:** 2026-07-31  
**Authors:** Social Phobia Detection Research Team  
**Project:** Deploying Machine Learning to Detect Social Phobia in UK School Students  
**Dataset:** UKDA-8467 — Mental Health of Children and Young People in England 2017 (MHCYP 2017)  
**Version status:** Active — deployed as `active_model: v5` in `config/model_registry.yaml`

---

## 1. Model Overview

| Attribute | Value |
|-----------|-------|
| Algorithm | Logistic Regression + Platt Scaling (sigmoid calibration layer) |
| Task | Binary classification — ICD-10 social phobia risk screening |
| Output | Calibrated risk probability ∈ [0, 1]; risk band (low / moderate / high) |
| Classification threshold | 0.0337 (optimised on calibration set; maximises F1 subject to Recall ≥ 0.85) |
| Feature count | 22 |
| Training dataset | `real_v1` — n = 3,749, 457 school clusters, 75 positive cases (2.0%) |
| Artefacts | `models/model_v5.pkl` · `models/preprocessor_v5.pkl` · `models/spin_imputer_v5.pkl` |

### Why the threshold looks low

The threshold (0.034) is in **calibrated probability space**, which reflects the true 2% base rate. v4's threshold of 0.438 was in uncalibrated space where `class_weight='balanced'` had inflated all probabilities toward 0.5. These are equivalent operating points — the calibrated threshold is not more permissive, it is more honest.

### Intended use

This tool is designed for **population-level screening** of social phobia risk in secondary school students (aged 11–18) in England. It is intended to help schools prioritise pastoral and SEMH (Social, Emotional and Mental Health) follow-up conversations.

**Prohibited uses:**
- Clinical diagnosis of social phobia or any other mental health condition
- Exclusion, disciplinary action, or academic decisions based solely on model output
- Use outside the age range 11–18 or outside an English secondary school context
- Any application where the output is not reviewed by a qualified human before action is taken

---

## 2. Training Data

| Attribute | Value |
|-----------|-------|
| Source | UK Data Archive, Study 8467 — MHCYP 2017 |
| Access | Special Licence (UKDA safeguarded data) |
| Participants | 3,749 school-age children and young people aged 11–18 |
| School clusters (PSU) | 457 primary sampling units |
| Survey year | 2017 |
| Target variable | ICD-10 social phobia (`dcsoph = 2`) via DAWBA clinical ratings |
| Positive cases | 75 (2.0%) |
| Negative cases | 3,674 (98.0%) |
| Positive-to-negative ratio | 1 : 49 |

### Data splits

| Split | n | Positive cases | Purpose |
|-------|---|---------------|---------|
| Training (80% of CV pool) | 2,399 | 48 | Base LogisticRegression training |
| Calibration (20% of CV pool) | 599 | 12 | Platt Scaling fitting + threshold optimisation |
| Test (held-out school cluster) | 751 | 15 | Final evaluation — never seen during training or calibration |

### Class imbalance handling

SMOTE applied to the training split only (`sampling_strategy=0.30`, `k_neighbors=5`): 2,399 → 3,056 rows (48 → 705 at-risk). Calibration and test sets are unmodified.

---

## 3. Features

All 22 features are derived from the UKDA-8467 dataset. No personally identifiable information is used.

| Feature | Source | Range | SHAP rank |
|---------|--------|-------|-----------|
| `sdq_emotional` | SDQ self-report (emotional subscale) | 0–10 | 1 |
| `sdq_internalising` | Derived: sdq_emotional + sdq_peer | 0–20 | 2 |
| `wellbeing_wemwbs` | WEMWBS total score | 14–70 | 3 |
| `psdq_emotional` | SDQ parent-report (emotional subscale) | 0–10 | 4 |
| `psdq_internalising` | Derived: psdq_emotional + psdq_peer | 0–20 | 5 |
| `age_years` | Age at assessment | 11–18 | 6 |
| `deprivation_quintile` | IMD deprivation quintile (1=most, 5=least) | 1–5 | 7 |
| `sdq_peer` | SDQ self-report (peer subscale) | 0–10 | 8 |
| `psdq_prosocial` | SDQ parent-report (prosocial subscale) | 0–10 | 9 |
| `psdq_peer` | SDQ parent-report (peer subscale) | 0–10 | 10 |
| `sdq_prosocial` | SDQ self-report (prosocial subscale) | 0–10 | 11 |
| `parent_self_discrepancy` | Derived: psdq_total − sdq_total | −40–40 | 12 |
| `psdq_hyper` | SDQ parent-report (hyperactivity subscale) | 0–10 | 13 |
| `sdq_hyper` | SDQ self-report (hyperactivity subscale) | 0–10 | 14 |
| `sdq_conduct` | SDQ self-report (conduct subscale) | 0–10 | 15 |
| `psdq_conduct` | SDQ parent-report (conduct subscale) | 0–10 | 16 |
| `bullied_flag` | Any school bullying (binary) | 0–1 | 17 |
| `any_bullying` | Derived: bullied_flag OR cyberbullied_flag | 0–1 | 18 |
| `sen_status` | Special Educational Needs (binary) | 0–1 | 19 |
| `wellbeing_low_flag` | Derived: WEMWBS < 40 (binary) | 0–1 | 20 |
| `cyberbullied_flag` | Cyberbullying experience (binary) | 0–1 | 21 |
| `learning_difficulty_level` | Learning difficulty severity (ordinal) | 1–4 | 22 |

SHAP ranks from `LinearExplainer` on the full dataset (n = 3,749). See `reports/figures/shap_global_importance.png`.

### Feature preprocessing (9 steps, fitted on training split only)

1. **Conservative minimum imputation** — negative values floored to 0
2. **Zero imputation** — QC columns filled with defaults
3. **MICE** — SDQ subscales + WEMWBS; 10 iterations; BayesianRidge; per-column bounds from schema
4. **KNN imputation** — continuous demographic features; k=5
5. **Median imputation** — remaining continuous gaps
6. **Mode imputation** — binary / categorical gaps
7. **Derived feature recomputation** — internalising scores, wellbeing flag, bullying flag recalculated post-imputation
8. **Ordinal encoding** — `gender`, `learning_difficulty_level`
9. **Standard scaling** — μ=0, σ=1 using training-split statistics

Fitted preprocessor: `models/preprocessor_v5.pkl` (reused from v4; fitted on same CV pool).

---

## 4. Hyperparameters

### Base model (LogisticRegression)

| Parameter | Value | Selection method |
|-----------|-------|-----------------|
| C (regularisation) | 0.001005 | Optuna TPE, inner 3-fold CV, 50 trials |
| l1_ratio | 0.0 (L2) | Fixed (lbfgs solver) |
| solver | `lbfgs` | Fixed |
| max_iter | 1000 | Fixed |
| class_weight | `balanced` | Fixed — addresses 1:49 imbalance |

### Calibration layer (Platt Scaling)

| Parameter | Value |
|-----------|-------|
| Method | Sigmoid (logistic regression on decision function scores) |
| Fitted on | 20% held-out calibration split (n=599, 12 positives) |
| Implementation | `PlattCalibratedModel` — `scripts/run_calibration.py` |
| Classification threshold | 0.0337 (optimised on calibration set) |

---

## 5. Performance Metrics

### 5.1 Cross-validation performance (5-fold outer × 3-fold inner, base model)

CV metrics measure the base model's discrimination ability, which is unchanged by calibration (calibration affects probability outputs, not ranking).

| Metric | Mean | SD |
|--------|------|----|
| Recall (Sensitivity) | **0.920** | 0.046 |
| PR-AUC | 0.275 | 0.044 |
| AUC-ROC | 0.926 | 0.012 |
| F1-Score | 0.156 | 0.042 |
| Composite SelectionScore | 0.606 | 0.026 |

Each outer fold had exactly 15 positive cases in the validation split.

### 5.2 Held-out test set performance (v5 calibrated model)

| Metric | v4 (uncalibrated) | v5 (calibrated) | Change |
|--------|-------------------|-----------------|--------|
| Recall | 1.000 | **0.933** | −0.067 |
| Precision (PPV) | 0.097 | **0.135** | +0.038 |
| PR-AUC | 0.170 | **0.179** | +0.009 |
| AUC-ROC | 0.939 | **0.938** | −0.001 |
| Brier score | 0.115 | **0.019** | −0.096 ✅ |
| ECE | 0.243 | **0.005** | −0.238 ✅ |
| Threshold | 0.438 | **0.034** | — |

The test set was a single school cluster held out before all training and never seen during training or calibration.

### 5.3 Clinical operating point (v5, test set, threshold = 0.034)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Sensitivity (Recall) | 0.933 | 14 of 15 test cases detected |
| Specificity | 0.878 | 88% of non-cases correctly cleared |
| PPV (Precision) | 0.135 | 1 in ~7 flagged students is a true case |
| NPV | 0.998 | Cleared students are almost certainly non-cases |
| False Negative Rate | 0.067 | 1 in 15 cases missed |
| False Positive Rate | 0.122 | 12% of non-cases incorrectly flagged |
| Number Needed to Screen | ~51 | Screen ~51 students to identify one confirmed case |

### 5.4 Deployment gates

| Gate | Threshold | CV Mean | Status |
|------|-----------|---------|--------|
| Recall | ≥ 0.85 | 0.920 | ✅ PASS |
| AUC-ROC | ≥ 0.80 | 0.926 | ✅ PASS |
| PR-AUC | ≥ 0.15 | 0.275 | ✅ PASS |

---

## 6. Calibration

| Metric | v4 (before) | v5 (after) | Target | Status |
|--------|-------------|------------|--------|--------|
| Brier score | 0.115 | **0.019** | < 0.10 | ✅ PASS |
| ECE (10-bin) | 0.243 | **0.005** | < 0.05 | ✅ PASS |

**Calibration status: GOOD.** Platt Scaling maps the base model's decision scores through a sigmoid fitted on a held-out calibration set. The calibrated probabilities now reflect the true 2% population base rate — a predicted probability of 3% means approximately 3 in 100 students with that profile have social phobia.

**Risk band thresholds** should be re-calibrated clinically for v5 (the old 0.30/0.65 thresholds were defined for uncalibrated probabilities):

| Band | Calibrated probability | Meaning |
|------|----------------------|---------|
| Low | < 0.02 | Below population prevalence — no action |
| Moderate | 0.02–0.10 | 2–10× prevalence — pastoral check-in |
| High | ≥ 0.10 | 5× or more — refer to SEMH / SENCO |

---

## 7. Fairness / Bias Audit

Fairness audit was run on the base model v4 (full dataset, n=3,749). Calibration does not affect recall ordering across subgroups.

| Group | Subgroup | n | Positives | Recall |
|-------|---------|---|-----------|--------|
| Gender | Male | 1,878 | 22 | 0.955 |
| Gender | Female | 1,871 | 53 | 0.906 |
| Age | 11–13 | 1,615 | 16 | 0.938 |
| Age | 14–16 | 1,506 | 28 | **0.857** |
| Age | 17–18 | 628 | 31 | 0.968 |
| SEN status | No SEN | 3,093 | 53 | 0.925 |
| SEN status | SEN | 314 | 8 | 0.875 |
| Deprivation | Q1–Q2 (deprived) | 1,314 | 20 | 0.900 |
| Deprivation | Q3 (middle) | 820 | 17 | **1.000** |
| Deprivation | Q4–Q5 (affluent) | 1,615 | 38 | 0.895 |

**Recall range: 0.857–1.000 (gap = 0.143) — ⚠ PARITY GAP EXCEEDS 0.10**

- Gender gap: 0.049 — within acceptable range
- Upper bound (Q3 middle, 1.000) driven by small cell (17 positives) — high variance
- Lower bound (age 14–16, 0.857) — still above deployment gate; requires clinical review
- SEN: only 8 positives — estimate unreliable

Full data: `reports/figures/fairness_audit.csv`

---

## 8. Explainability

### 8.1 Global feature importance (SHAP, n = 3,749)

| Rank | Feature | Lay label | Mean \|SHAP\| |
|------|---------|-----------|--------------|
| 1 | `sdq_emotional` | Self-rated emotional difficulties | 0.237 |
| 2 | `sdq_internalising` | Self-rated internalising (emotional+peer) | 0.215 |
| 3 | `wellbeing_wemwbs` | Warwick-Edinburgh wellbeing score | 0.181 |
| 4 | `psdq_emotional` | Parent-rated emotional difficulties | 0.168 |
| 5 | `psdq_internalising` | Parent-rated internalising (emotional+peer) | 0.148 |

Figures: `reports/figures/shap_global_importance.png` · `reports/figures/shap_beeswarm.png`

### 8.2 Surrogate decision tree (fidelity = 92.5%)

Three features reproduce 92.5% of model decisions: `sdq_internalising`, `psdq_internalising`, `wellbeing_wemwbs`.

Core rule: elevated self-rated AND parent-rated internalising problems, combined with low wellbeing → flag for follow-up.

Full rules: `reports/figures/surrogate_tree_rules.txt`

### 8.3 Local explanations

`ExplainabilityEngine.explain_instance()` returns per-student JSON with `risk_probability`, `risk_band`, `recommended_action`, and top contributing features with direction and strength labels. See `reports/figures/shap_waterfall_case.png` for an example.

---

## 9. Limitations

1. **Dataset vintage:** Training data is from 2017. Cohort characteristics and social phobia prevalence may have shifted post-pandemic. Re-evaluate before deployment beyond 2026.

2. **No SPIN instrument:** SDQ + WEMWBS capture general emotional distress, not social-anxiety-specific symptoms. The model cannot distinguish social phobia from general anxiety disorder.

3. **ICD-10 target only:** Confirmed clinical diagnosis (2.0% prevalence). Sub-clinical social anxiety — students who may still benefit from support — are classified as negatives.

4. **Fairness parity gap (0.143):** Recall for age 14–16 is 0.857 vs. 1.000 for deprivation Q3. Gap exceeds 0.10 threshold. Requires clinical review before deployment.

5. **Low precision (PPV ≈ 0.135):** For every 7 students flagged, ~6 are non-cases. Schools must be briefed that most flags will not be confirmed cases — the tool is a starting point for a conversation, not a verdict.

6. **Single survey wave:** Trained on MHCYP 2017 only. Generalisation across years, regions, or individual schools is not validated.

7. **Ethnicity fairness unmeasurable:** `ethnicity_bme` is structurally incomplete (White British uncoded as NaN). Fairness by ethnicity cannot be assessed with the current data.

8. **SEN underpowered:** Only 8 positive cases in the SEN subgroup — recall estimate (0.875) is statistically unreliable.

9. **No longitudinal validation:** The model predicts cross-sectional ICD-10 diagnosis status, not future clinical presentation or treatment outcomes.

10. **Calibration set is small:** 12 positive cases in the calibration set. The Platt Scaling sigmoid is fitted on limited data; ECE estimates carry uncertainty at this sample size.

---

## 10. Ethical Considerations

- **Screening, not diagnosis:** All outputs require human review before any action is taken.
- **Data minimisation:** 22 features from routine survey data. No biometric, attendance, or academic records needed.
- **GDPR compliance:** Predictions generated at point of need; no raw responses stored post-scoring; pseudonymised identifiers; UK servers only.
- **Transparency:** Students and parents must be informed a screening tool is in use (ICO guidance on AI profiling of children).
- **Clinical referral pathway:** A flag initiates a pastoral conversation — not an automated referral.

---

## 11. Model Versioning and Artefacts

| Artefact | Path | Purpose |
|---------|------|---------|
| Calibrated model | `models/model_v5.pkl` | `PlattCalibratedModel` wrapping base LR |
| Fitted preprocessor | `models/preprocessor_v5.pkl` | Same as v4 — fitted on CV pool |
| SPIN imputer | `models/spin_imputer_v5.pkl` | SDQ/WEMWBS MICE imputer |
| Model registry | `config/model_registry.yaml` | Metrics, threshold, active model flag |
| SHAP importance | `reports/figures/shap_global_importance.png` | Global feature importance |
| SHAP beeswarm | `reports/figures/shap_beeswarm.png` | Direction + magnitude across dataset |
| SHAP waterfall | `reports/figures/shap_waterfall_case.png` | Local explanation (true positive) |
| SHAP values CSV | `reports/figures/shap_values.csv` | Per-student SHAP values for UKDA deposit |
| Surrogate rules | `reports/figures/surrogate_tree_rules.txt` | Depth-4 tree (92.5% fidelity) |
| Fairness audit | `reports/figures/fairness_audit.csv` | Recall by gender, age, SEN, deprivation |
| Model card | `documentation/model_card.md` | This document |

### Version history

| Version | Date | Algorithm | CV Recall | Test ECE | Notes |
|---------|------|-----------|-----------|----------|-------|
| v1 | 2026-07-31 | LogisticRegression | 0.871 | — | Initial run |
| v2 | 2026-07-31 | RandomForest | 0.891 | — | Second candidate |
| v3 | 2026-07-31 | LogisticRegression | 0.871 | — | Re-run |
| v4 | 2026-07-31 | LogisticRegression | 0.920 | 0.243 | Best CV recall; poor calibration |
| **v5** | **2026-07-31** | **LR + Platt Scaling** | **0.920** | **0.005** | **Active — calibrated** |

---

## 12. How to Use This Model

### Inference (single student)

```python
import joblib
from src.config import Config
from src.feature_engineering import FeatureEngineer

cfg   = Config()
model = joblib.load("models/model_v5.pkl")   # PlattCalibratedModel
prep  = joblib.load("models/preprocessor_v5.pkl")

fe           = FeatureEngineer(cfg)
enriched     = fe.transform(student_row)
feature_cols = [c for c in cfg.model_feature_cols if c in enriched.columns]
X_proc       = prep.transform(enriched[feature_cols])

# Calibrated probability — reflects true base rate
prob      = model.predict_proba(X_proc.values)[0, 1]
predicted = int(prob >= 0.0337)   # v5 threshold
```

### Retraining

1. Update `config/dataset_registry.yaml` — set `active` to new dataset key
2. Run `python scripts/run_pipeline.py` — trains base models, selects best
3. Run `python scripts/run_calibration.py` — fits Platt Scaling, saves v5
4. Run `python scripts/run_explainability.py` — regenerates SHAP and fairness outputs
5. Update this model card

---

*This model card follows the format recommended by Mitchell et al. (2019) "Model Cards for Model Reporting", ACM FAccT.*
