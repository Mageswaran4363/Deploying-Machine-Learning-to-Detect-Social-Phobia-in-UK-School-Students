# Deploying Machine Learning to Detect Social Phobia in UK School Students

**Project Title:** Deploying Machine Learning to Detect Social Phobia in UK School Students: A Pathway to Early Intervention and Educational Impact  
**Report type:** Final Project Report  
**Date:** 2026-07-31  
**Dataset:** UKDA Study 8467 — Mental Health of Children and Young People in England 2017 (MHCYP 2017)  
**Active model:** v5 (LogisticRegression + Platt Scaling)  
**Model artefacts:** `models/model_v5.pkl` · `models/preprocessor_v5.pkl`

---

## Executive Summary

This project developed and validated a machine learning screening tool for the early identification of social phobia in UK secondary school students (aged 11–18), using the Mental Health of Children and Young People in England 2017 survey (MHCYP 2017; UKDA-8467). Working from a nationally representative sample of 3,749 school-aged participants nested within 457 school clusters, and using ICD-10 clinical diagnoses as the target label, a Logistic Regression classifier was trained using a nested cross-validation framework with school-cluster grouping to prevent data leakage.

The final model (v5) achieves **cross-validated recall of 0.920** and **test-set recall of 0.933** against the pre-specified gate of ≥ 0.85 — meaning the tool correctly identifies 14 of 15 confirmed social phobia cases in a held-out school. A Platt Scaling calibration layer reduced the Expected Calibration Error from 0.243 to **0.005**, ensuring that predicted probabilities reflect the true 2% population base rate. SHAP-based explainability confirmed that self-reported and parent-reported emotional distress (SDQ subscales) and low wellbeing (WEMWBS) are the dominant predictive features, consistent with the clinical literature on social phobia.

The model meets all three deployment gates (Recall ≥ 0.85 ✅, AUC-ROC ≥ 0.80 ✅, PR-AUC ≥ 0.15 ✅) and is accompanied by a detailed model card, fairness audit, surrogate decision tree, and SHAP explainability outputs for UKDA deposit.

---

## 1. Introduction

### 1.1 Background and Motivation

Social phobia (social anxiety disorder) is among the most prevalent anxiety disorders in adolescence, with a typical onset between ages 11 and 15 and population prevalence estimates of 1–2% in England (NICE CG159, 2013; Morales-Muñoz et al., 2022). Despite its substantial burden — including impaired academic attainment, school avoidance, and downstream risk of depression — the condition remains systematically under-identified in school settings, partly because its presentation (silence, withdrawal, avoidance of participation) is often misread as introversion or poor engagement.

Current identification pathways are reactive: a teacher or parent raises a concern, the child is referred to CAMHS or a school counsellor, and the waiting list begins. A proactive, data-driven screening tool — one that analyses routinely collected wellbeing survey data — could shift this pathway toward earlier identification, enabling pastoral staff to initiate supportive conversations before distress escalates.

### 1.2 Research Questions

**RQ1 — Classification Accuracy**
> How accurately can a machine learning model classify social phobia risk in school-aged children using SDQ and WEMWBS data, and does it meet the clinical deployment gate of sensitivity ≥ 85%?

**RQ2 — Policy and Intervention Impact**
> To what extent can calibrated risk predictions and explainable model outputs inform school-level interventions and contribute to safeguarding and mental health policy?

### 1.3 Dataset Pivot

The original project proposal planned to use the Social Phobia Inventory (SPIN, Connor et al., 2000) as both the feature source and the label. Upon accessing UKDA-8467, it was found that MHCYP 2017 uses the Development and Well-Being Assessment (DAWBA) to produce ICD-10 clinical ratings — not the SPIN instrument. This required a fundamental redesign:

- **Target label**: ICD-10 social phobia (`dcsoph = 2`) via DAWBA clinical ratings
- **Features**: SDQ subscales (parent + self-report) and WEMWBS wellbeing score, supplemented by demographics
- **Age range**: Extended to 11–18 to maximise case count (75 cases vs ~59 at the originally planned 11–17)

This pivot is documented in the Architecture Decision Record (`reports/ukda-dataset-profiling-report.md`, Section 6). The ICD-10 target is appropriate for the UK CAMHS referral context and the use of SDQ + WEMWBS is justified by large between-group effect sizes (reported in Section 3).

---

## 2. Dataset

### 2.1 Source

The Mental Health of Children and Young People in England 2017 (MHCYP 2017) survey is a nationally representative cross-sectional study of the mental health of 5–19 year olds in England, conducted by the NHS Digital and archived at the UK Data Archive (Study 8467). Access was obtained under Special Licence conditions. The dataset spans 9,117 participants (age 2–19) across a complex multi-stage probability sample of 457+ primary sampling units (PSUs, representing school clusters).

### 2.2 Working Subset

The analysis subset (`real_v1.csv`) comprised **3,749 participants aged 11–18**, drawn from **457 school PSU clusters**. This age range was selected to match the secondary school population relevant to the research question.

| Characteristic | Value |
|----------------|-------|
| N (total) | 3,749 |
| Age, M (SD) | 14.1 (2.2) |
| Age range | 11–18 years |
| Male | 50.1% |
| Female | 49.9% |
| SEN status (any) | 9.2% |
| Deprivation Q1–Q2 (most deprived) | 35.0% |
| Deprivation Q4–Q5 (least deprived) | 43.3% |
| School PSU clusters | 457 |

### 2.3 Case Identification

ICD-10 social phobia was identified via DAWBA clinical ratings (`dcsoph = 2`), yielding **75 confirmed cases (2.0%; 95% CI: 1.6–2.5%)** — a positive-to-negative ratio of 1:49. This prevalence is consistent with published UK estimates for clinical-threshold social phobia in school-age adolescents (NICE CG159, 2013).

The ICD-10 label was preferred over the DSM-5 equivalent (`dcspph = 2`, ~32 cases) because it yields more than twice as many positives, improving cross-validation stability, while remaining the clinically relevant framework for CAMHS referral pathways in England.

---

## 3. Descriptive Statistics

### 3.1 SDQ and WEMWBS by Case Status

Large between-group differences were observed across all primary features:

**Table 1. SDQ and WEMWBS Scores by Case Status**

| Measure | Full sample M (SD) | Cases M (n=75) | Controls M (n=3,674) | Cohen *d* | *p* |
|---------|-------------------|-----------|--------------|----|---|
| SDQ Emotional (self) | 2.66 (2.13) | 6.54 | 2.59 | **1.92** | < .001 |
| SDQ Peer (self) | 1.74 (1.56) | 3.74 | 1.70 | 1.32 | < .001 |
| SDQ Emotional (parent) | 2.36 (2.30) | 5.85 | 2.30 | 1.58 | < .001 |
| SDQ Peer (parent) | 1.65 (1.86) | 3.69 | 1.61 | 1.13 | < .001 |
| SDQ Total (self, 0–40) | 9.58 (5.52) | 18.07 | 9.41 | — | < .001 |
| SDQ Total (parent, 0–40) | 8.26 (6.47) | 16.11 | 8.11 | — | < .001 |
| WEMWBS (14–70) | 51.8 (9.6) | 36.8 | 52.1 | −1.63 | < .001 |

All differences represent large effects by conventional benchmarks (Cohen, 1988). Self-reported emotional problems showed the largest case-control separation (*d* = 1.92), followed by parent-rated emotional problems (*d* = 1.58) and low wellbeing (*d* = −1.63). These effect sizes confirm that the SDQ–WEMWBS feature set provides strong discriminative signal despite the absence of social-phobia-specific instruments.

### 3.2 Bullying Exposure

Over half of social phobia cases (52.0%) reported any bullying exposure, compared with 37.5% of controls — a 14.5 percentage point difference consistent with the known co-occurrence of peer victimisation and social anxiety in adolescent populations (Schwartz et al., 2019).

### 3.3 Missing Data

**Table 2. Missing Data by Feature Group**

| Feature group | Missing rate | Imputation method |
|--------------|-------------|-------------------|
| SDQ self-report subscales | 15.1% | MICE (BayesianRidge, 10 iterations) |
| SDQ parent-report subscales | 8.4% | MICE (BayesianRidge, 10 iterations) |
| WEMWBS wellbeing | 16.6% | MICE (BayesianRidge, 10 iterations) |
| SEN status | 9.1% | Mode imputation |
| Bullying indicators | 15.3–16.1% | Mode imputation |
| Ethnicity (BME) | 19.9% | Structural — not used in modelling |

MICE imputation used per-column bounds from the validated scale ranges (SDQ subscales: 0–10; WEMWBS: 14–70) to prevent imputed values from violating instrument constraints. Derived features (internalising composites, wellbeing flag, combined bullying flag) were recomputed after imputation to maintain internal consistency.

---

## 4. Methods

### 4.1 Feature Engineering

Twenty-two features were extracted and engineered from the UKDA-8467 dataset. No personally identifiable information was retained. Key engineered features include:

- **Internalising composites**: `sdq_internalising` = sdq_emotional + sdq_peer; `psdq_internalising` = psdq_emotional + psdq_peer
- **Parent-self discrepancy**: `parent_self_discrepancy` = psdq_total − sdq_total (captures agreement between informants)
- **Wellbeing flag**: binary indicator for WEMWBS < 40 (below the threshold associated with substantially elevated mental health risk)
- **Combined bullying flag**: any bullying (school or cyber)

### 4.2 Preprocessing Pipeline (9 steps)

The following steps were applied sequentially, fitted only on training data and applied to validation and test sets without refitting:

1. Conservative minimum imputation (floor negative values at 0)
2. Zero imputation for quality-control placeholder columns
3. MICE imputation for SDQ subscales and WEMWBS (10 iterations, BayesianRidge, bounds-constrained)
4. KNN imputation for continuous demographic features (k = 5)
5. Median imputation for remaining continuous gaps
6. Mode imputation for binary and categorical gaps
7. Derived feature recomputation (internalising composites, wellbeing flag, bullying flag)
8. Ordinal encoding for gender and learning difficulty level
9. Standard scaling (μ = 0, σ = 1) using training-split statistics

### 4.3 Class Imbalance

Social phobia is rare in the general population (2.0%; 1:49 ratio). Two complementary strategies were applied:

- **SMOTE** (Synthetic Minority Oversampling Technique): applied to training splits only, never to validation or test sets. `sampling_strategy = 0.30`, `k_neighbors = 5`. Training data expanded from 2,399 to ~3,056 rows (48 → 705 positives) in the calibration training split.
- **`class_weight='balanced'`**: LogisticRegression loss-weighted by inverse class frequency, ensuring the base model prioritises recall over precision.

### 4.4 Model Selection Framework

A nested cross-validation framework was used to select and evaluate models without information leakage between hyperparameter tuning and performance estimation:

- **Outer loop**: 5-fold StratifiedGroupKFold, grouped by school PSU cluster. Each outer fold holds out an entire set of schools for validation, preventing students from the same school appearing in both training and test.
- **Inner loop**: 3-fold StratifiedGroupKFold within each outer training set, used for Bayesian hyperparameter optimisation (Optuna, 50 trials, TPE sampler).
- **Primary metric**: a composite selection score (0.5 × Recall + 0.3 × PR-AUC + 0.2 × AUC-ROC), with Recall given the highest weight to minimise missed cases.

With 75 positives distributed across 5 folds, each outer validation fold contained exactly **15 confirmed cases** — the minimum required for stable recall estimation.

Four candidate models were evaluated: Logistic Regression, Random Forest, Support Vector Machine (with CalibratedClassifierCV), and Gradient Boosting.

### 4.5 Calibration

Model v4 (the best-performing Logistic Regression) had an Expected Calibration Error (ECE) of 0.243 — substantially above the deployment gate of < 0.05. This is a known consequence of `class_weight='balanced'`, which distorts predicted probabilities away from the true base rate.

**Platt Scaling** was applied post-training to produce model v5:

1. The CV pool (n = 2,998) was split 80/20 (train: n = 2,399, calibration: n = 599).
2. A fresh Logistic Regression (same hyperparameters as v4) was trained on the 80% with SMOTE.
3. A `PlattCalibratedModel` — a LogisticRegression fitted on the base model's `decision_function` scores vs. true labels — was fitted on the 20% calibration set.
4. The classification threshold (0.0337) was optimised on the calibration set to maximise F1 subject to Recall ≥ 0.85.

The calibrated probabilities now reflect the true 2% base rate. A predicted probability of 0.034 corresponds approximately to one chance in 29.

---

## 5. Results

### 5.1 Cross-Validation Performance

**Table 3. Cross-Validation Results by Model Version (5-fold outer)**

| Version | Algorithm | CV Recall | CV PR-AUC | CV AUC-ROC |
|---------|-----------|-----------|-----------|------------|
| v1 | LogisticRegression | 0.871 ± 0.018 | 0.300 ± 0.121 | 0.926 ± 0.020 |
| v2 | RandomForest | 0.891 ± 0.061 | 0.286 ± 0.154 | 0.913 ± 0.019 |
| v3 | LogisticRegression | 0.871 ± 0.018 | 0.300 ± 0.121 | 0.926 ± 0.020 |
| v4 (selected) | LogisticRegression | **0.920 ± 0.046** | **0.275 ± 0.044** | **0.926 ± 0.012** |

Model v4 was selected on the basis of highest CV recall (0.920) and lowest PR-AUC standard deviation (±0.044), indicating greater consistency across folds. The random-baseline PR-AUC for 2% prevalence is 0.02; all models exceed the deployment gate of ≥ 0.15 (7.5× above random).

### 5.2 Test Set Performance (model v5, calibrated)

The test set comprised a single held-out school cluster (n = 751, 15 confirmed social phobia cases) that was never seen during training, hyperparameter optimisation, or calibration.

**Table 4. Test Set Performance — Before and After Calibration**

| Metric | v4 (uncalibrated) | v5 (calibrated) |
|--------|-------------------|-----------------|
| Recall (Sensitivity) | 1.000 | **0.933** ✅ |
| Precision (PPV) | 0.097 | **0.135** |
| PR-AUC | 0.170 | **0.179** |
| AUC-ROC | 0.939 | **0.938** |
| Brier score | 0.115 | **0.019** ✅ |
| ECE (10-bin) | 0.243 | **0.005** ✅ |
| Classification threshold | 0.438 | **0.034** |

Platt Scaling produced a 98% reduction in ECE (0.243 → 0.005), exceeding the < 0.05 gate. Recall decreased marginally (1.000 → 0.933) because the calibrated model's threshold operates in true probability space: at 0.034, one previously borderline case fell below threshold. The model still detects 14 of 15 confirmed cases in the held-out school — a clinically acceptable operating point.

**Table 5. Confusion Matrix (v5, test set, threshold = 0.034)**

| | Predicted Negative | Predicted Positive |
|--|--|--|
| **True Negative** (n = 736) | 647 (TN) | 89 (FP) |
| **True Positive** (n = 15) | 1 (FN) | 14 (TP) |

- **Sensitivity (Recall)**: 0.933 — 14 of 15 cases detected
- **Specificity**: 0.879 — 88% of non-cases correctly cleared
- **NPV**: 0.998 — cleared students are almost certainly non-cases
- **PPV (Precision)**: 0.135 — approximately 1 in 7 flagged students is a true case

For every confirmed social phobia case identified, approximately 6 students are incorrectly flagged. Schools must understand that the tool is a starting point for a pastoral conversation, not a verdict. The high NPV (0.998) means that students not flagged by the tool are very unlikely to have social phobia.

### 5.3 Deployment Gate Summary

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| Recall (sensitivity) | ≥ 0.85 | 0.920 (CV) / 0.933 (test) | ✅ PASS |
| AUC-ROC | ≥ 0.80 | 0.926 (CV) / 0.938 (test) | ✅ PASS |
| PR-AUC | ≥ 0.15 | 0.275 (CV) / 0.179 (test) | ✅ PASS |
| ECE (calibration) | < 0.05 | 0.005 (test) | ✅ PASS |

### 5.4 Hyperparameters (final model)

| Parameter | Value |
|-----------|-------|
| Algorithm | LogisticRegression (sklearn) + Platt Scaling |
| Regularisation C | 0.001005 (Optuna TPE, 50 trials) |
| Penalty | L2 (lbfgs solver) |
| class_weight | balanced |
| SMOTE sampling_strategy | 0.30 |
| Classification threshold | 0.034 (calibration-set optimised) |
| Calibration method | Platt Scaling (sigmoid, 20% held-out calibration set) |

---

## 6. Explainability

### 6.1 Global Feature Importance (SHAP)

SHAP (SHapley Additive exPlanations) values were computed using `shap.LinearExplainer` on all 3,749 participants. The global importance ranking (mean absolute SHAP value) is:

**Table 6. Top 10 Features by Global SHAP Importance**

| Rank | Feature | Lay label | Mean \|SHAP\| |
|------|---------|-----------|--------------|
| 1 | `sdq_emotional` | Self-rated emotional difficulties | 0.237 |
| 2 | `sdq_internalising` | Self-rated internalising (emotional + peer) | 0.215 |
| 3 | `wellbeing_wemwbs` | Warwick-Edinburgh wellbeing score | 0.181 |
| 4 | `psdq_emotional` | Parent-rated emotional difficulties | 0.168 |
| 5 | `psdq_internalising` | Parent-rated internalising (emotional + peer) | 0.148 |
| 6 | `age_years` | Age at assessment | 0.112 |
| 7 | `deprivation_quintile` | IMD deprivation quintile | 0.094 |
| 8 | `sdq_peer` | Self-rated peer relationship problems | 0.087 |
| 9 | `psdq_prosocial` | Parent-rated prosocial behaviour | 0.079 |
| 10 | `psdq_peer` | Parent-rated peer relationship problems | 0.071 |

The dominance of SDQ emotional subscales and WEMWBS is consistent with the large between-group effect sizes observed in the descriptive statistics (Section 3) and with the clinical literature on social phobia, where emotional avoidance and low wellbeing are core features.

### 6.2 Surrogate Decision Tree

A depth-4 decision tree was fitted to the model's predictions (n = 3,749). The surrogate tree achieved **92.5% fidelity** — meaning it reproduces the model's binary predictions for 92.5% of cases using only three features: `sdq_internalising`, `psdq_internalising`, and `wellbeing_wemwbs`.

**Core decision rule (practitioner summary):**
> A student is flagged at elevated risk if they score high on **both** self-reported and parent-reported internalising problems, **and** their wellbeing score is low.

This three-feature summary is accessible to educators and school mental health leads without requiring any knowledge of machine learning.

### 6.3 Local Explanations

For any individual student, the model returns:
- A calibrated risk probability (0–1, reflecting the true 2% base rate)
- A risk band (Low / Moderate / High)
- The top contributing features with direction and strength labels
- A recommended action (no action / pastoral check-in / SEMH referral)

All local explanations use lay labels drawn from `config/schema.json`, ensuring that outputs are written for educators, not data scientists.

---

## 7. Fairness Audit

A recall parity audit was conducted across four protected / demographic dimensions. All recall estimates are from model v4 on the full dataset (n = 3,749); calibration does not affect recall ranking across subgroups.

**Table 7. Recall Parity by Subgroup**

| Group | Subgroup | n | Positives | Recall |
|-------|---------|---|-----------|--------|
| Gender | Male | 1,878 | 22 | 0.955 |
| Gender | Female | 1,871 | 53 | 0.906 |
| Age | 11–13 | 1,615 | 16 | 0.938 |
| Age | 14–16 | 1,506 | 28 | 0.857 |
| Age | 17–18 | 628 | 31 | 0.968 |
| SEN status | No SEN | 3,093 | 53 | 0.925 |
| SEN status | SEN | 314 | 8 | 0.875 |
| Deprivation | Q1–Q2 (deprived) | 1,314 | 20 | 0.900 |
| Deprivation | Q3 (middle) | 820 | 17 | 1.000 |
| Deprivation | Q4–Q5 (affluent) | 1,615 | 38 | 0.895 |

**Recall range: 0.857 – 1.000 (gap = 0.143) — ⚠ Parity gap exceeds 0.10**

Key observations:
- **Gender**: gap of 0.049 (Male 0.955, Female 0.906) — within acceptable range; both above 0.85 gate.
- **Age**: 14–16 group shows lowest recall (0.857) — still above the gate, but warrants attention. This age group may present with more subtle symptom profiles.
- **Deprivation Q3 upper bound (1.000)**: based on 17 positives — high variance, not substantively meaningful.
- **SEN**: 8 positives — recall estimate (0.875) is statistically unreliable; confidence interval spans the full deployment gate range.
- **Ethnicity**: `ethnicity_bme` was structurally incomplete in UKDA-8467 (White British not coded as 0, resulting in 19.9% NaN). Fairness by ethnicity could not be reliably assessed and was excluded from the audit.

The parity gap of 0.143 exceeds the ≤ 0.10 target and should be reviewed by a clinical psychologist or SEMH specialist before deployment, particularly regarding the 14–16 age group.

---

## 8. Limitations

1. **Dataset vintage (2017):** The training data is nine years old. Adolescent mental health profiles changed substantially after the COVID-19 pandemic. Social media use, prevalence patterns, and school demographics may differ from the current population.

2. **No SPIN instrument:** SDQ and WEMWBS measure general emotional distress, not social-anxiety-specific symptoms. The model cannot distinguish social phobia from general anxiety disorder. A student with high SDQ emotional scores due to depression would receive a similar risk flag to a student with social phobia.

3. **ICD-10 clinical threshold:** The model targets confirmed clinical-threshold diagnosis (2.0% prevalence). Sub-clinical social anxiety — affecting a much larger proportion of students — is classified as negative. The tool is not suited to detecting students who may benefit from light-touch universal support.

4. **Low precision (PPV ≈ 0.135):** For every 7 students flagged, approximately 6 are non-cases. This is unavoidable at 2% prevalence even with strong recall. Schools must be briefed that most flags will not be confirmed — the tool surfaces students for a conversation, not a label.

5. **Fairness parity gap (0.143):** Recall for age 14–16 is 0.857 vs. 1.000 for deprivation Q3. Gap exceeds 0.10. The age 14–16 finding requires clinical review.

6. **SEN subgroup underpowered:** Only 8 positive cases in the SEN subgroup. Recall estimate (0.875) carries wide confidence intervals and should not be interpreted as definitive.

7. **Single survey wave:** Trained on MHCYP 2017 only. External validation on a different year or a school-recruited sample would be required before deployment.

8. **Ethnicity fairness unmeasurable:** `ethnicity_bme` is structurally incomplete; fairness by ethnicity cannot be audited with this dataset.

9. **Calibration set is small:** 12 positive cases were available for Platt Scaling. The ECE of 0.005 should be interpreted cautiously; confidence intervals at this sample size are wide.

10. **No longitudinal validation:** The model predicts cross-sectional ICD-10 diagnosis status at a point in time. It does not predict future clinical presentation, symptom trajectory, or response to intervention.

---

## 9. Policy Implications and Recommendations

### 9.1 For Schools

**Adopt a two-stage approach:**
The model output should be used to produce a **prioritised watchlist** of students for pastoral follow-up — not as a referral trigger. Stage 1: the model flags students above threshold. Stage 2: a trained member of staff (SENCO, mental health lead, or form tutor) conducts a brief wellbeing check-in with flagged students before any action is taken.

**Risk band interpretation (calibrated probabilities):**

| Band | Calibrated probability | Suggested action |
|------|----------------------|-----------------|
| Low | < 0.02 | No action — below population prevalence |
| Moderate | 0.02–0.10 | Brief check-in conversation |
| High | ≥ 0.10 | Discuss with SEMH team; consider CAMHS referral if symptoms confirmed |

**Staff training:** School staff using the tool should receive at minimum a 30-minute briefing on: (a) what social phobia is and how it presents, (b) what the risk bands mean, (c) the difference between a screening flag and a diagnosis, and (d) the referral pathway.

### 9.2 For Local Authorities and Commissioners

- A 2.0% social phobia prevalence in secondary schools implies approximately **20 students per 1,000** have a confirmed clinical-threshold condition. In a 1,200-student school, this corresponds to ~24 unidentified cases at any one time.
- The tool screens 51 students for every confirmed case identified. At school scale, this is operationally feasible — a small watchlist, not a mass referral.
- Local authorities should consider piloting the tool in schools that already administer SDQ or WEMWBS as part of routine wellbeing monitoring, as no additional data collection is required.

### 9.3 For Researchers

- A prospective validation study using school-recruited SPIN data would resolve the key limitation of this project: the mismatch between the SPIN-based research question and the ICD-10 label used in MHCYP 2017.
- The SDQ + WEMWBS feature backbone should be treated as a **proxy instrument** until a study can directly compare model performance between SDQ-based and SPIN-based features on a matched sample.
- Future work should investigate whether the 14–16 age group fairness gap is explained by symptom presentation differences (e.g., social phobia in early adolescence may present differently to late adolescence) or by sampling artefacts.

---

## 10. Ethical Considerations

- **Screening, not diagnosis:** The model output is a risk indicator, not a clinical finding. All outputs require human review by a trained member of staff before any action is taken.
- **GDPR compliance:** No personally identifiable information is used as model input. Predictions are generated at the point of need; raw survey responses are not stored post-scoring. Data is held on UK servers only.
- **Transparency:** In any deployment context, students and parents must be informed that a screening tool is in use, in accordance with ICO guidance on AI profiling of children (UK GDPR Article 22).
- **Harms of false positives:** Being incorrectly flagged as at-risk could cause unnecessary anxiety for the student and family. Pastoral check-in conversations must be framed as supportive and non-stigmatising.
- **Harms of false negatives:** A missed case (1 in 15 at the test threshold) receives no additional support via this pathway. The tool does not replace existing universal mental health provision.
- **Algorithmic bias:** The ethnicity fairness audit could not be completed due to structural data limitations. Any deployment must address this gap before use with ethnically diverse student populations.

---

## 11. Conclusion

This project demonstrates that a Logistic Regression classifier, trained on SDQ subscales and WEMWBS wellbeing data from the MHCYP 2017 national survey, can detect social phobia in school-aged children with cross-validated recall of 0.920 and test recall of 0.933 — exceeding the pre-specified clinical deployment gate of 0.85. Platt Scaling calibration reduced the Expected Calibration Error from 0.243 to 0.005, ensuring that predicted probabilities reflect the true 2% population base rate and can be meaningfully communicated to school staff.

The model's decisions are transparent: three features (self-rated internalising, parent-rated internalising, and low wellbeing) reproduce 92.5% of predictions in a surrogate decision tree, and SHAP explanations confirm that the dominant predictors align with established clinical understanding of social phobia.

The tool does not require new data collection — it can be applied to routinely administered SDQ surveys already in use in English schools. With appropriate staff training, clear communication to students and parents, and a structured referral pathway, it has the potential to shift social phobia identification from reactive to proactive, enabling earlier access to support for a condition that causes significant distress and impairs educational outcomes.

Further work is required to: (a) validate the tool on a prospective school-recruited sample with SPIN data, (b) resolve the ethnicity fairness gap, (c) investigate the 14–16 age group recall discrepancy, and (d) update the model as more recent national wellbeing survey data becomes available.

---

## 12. Project Artefacts

| Artefact | Path |
|---------|------|
| Active model | `models/model_v5.pkl` |
| Fitted preprocessor | `models/preprocessor_v5.pkl` |
| SPIN imputer | `models/spin_imputer_v5.pkl` |
| Model registry | `config/model_registry.yaml` |
| Model card | `documentation/model_card.md` |
| Dataset profiling report | `reports/ukda-dataset-profiling-report.md` |
| Population statistics | `reports/population-statistics.md` |
| SHAP global importance | `reports/figures/shap_global_importance.png` |
| SHAP beeswarm | `reports/figures/shap_beeswarm.png` |
| SHAP waterfall (case) | `reports/figures/shap_waterfall_case.png` |
| SHAP values CSV | `reports/figures/shap_values.csv` |
| Surrogate tree rules | `reports/figures/surrogate_tree_rules.txt` |
| Fairness audit | `reports/figures/fairness_audit.csv` |

---

## References

Cohen, J. (1988) *Statistical Power Analysis for the Behavioral Sciences*. 2nd edn. Hillsdale, NJ: Lawrence Erlbaum Associates.

Connor, K.M. et al. (2000) 'Psychometric properties of the Social Phobia Inventory (SPIN)', *The British Journal of Psychiatry*, 176(4), pp. 379–386.

Chawla, N.V. et al. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357.

Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30.

Mitchell, M. et al. (2019) 'Model cards for model reporting', *Proceedings of the Conference on Fairness, Accountability, and Transparency*, pp. 220–229.

Morales-Muñoz, I. et al. (2022) 'Anxiety disorders across middle childhood and early adolescence in a UK population-based cohort', *JCPP Advances*, 2(2), e12089.

NICE (2013) *Social anxiety disorder: recognition, assessment and treatment*. Clinical guideline CG159. London: National Institute for Health and Care Excellence.

Platt, J. (1999) 'Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods', in *Advances in Large Margin Classifiers*, pp. 61–74.

Schwartz, C. et al. (2019) 'Six decades of preventing and treating childhood anxiety disorders', *BMJ Mental Health*, 22(3), pp. 103–110.

Thornton, C. et al. (2013) 'Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms', *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 847–855.
