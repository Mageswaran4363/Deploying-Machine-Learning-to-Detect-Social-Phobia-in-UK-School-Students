# Research Roadmap: Deploying Machine Learning to Detect Social Phobia in UK School Students

**Project Title:** Deploying Machine Learning to Detect Social Phobia in UK School Students: A Pathway to Early Intervention and Educational Impact
**Date:** 2026-06-18
**Source Document:** QR-Project-Proposal.pdf

---

## 1. Research Objectives

| # | Objective |
|---|-----------|
| O1 | Develop a machine learning screening tool capable of detecting early indicators of social phobia in anonymised student data |
| O2 | Validate the tool against the Social Phobia Inventory (SPIN) questionnaire administered in partner schools, ensuring accuracy and reliability |
| O3 | Deploy the validated tool as a secure, user-friendly web application accessible to UK schools and local authorities for systematic early identification |
| O4 | Disseminate findings to academic, educational, and policy stakeholders to inform consistent screening protocols and safeguarding practice |

---

## 2. Research Questions

**RQ1 — Classification Accuracy**
> How accurately can machine learning models classify social phobia risk in school-aged children using SPIN data compared to traditional observation methods?

**RQ2 — Policy and Intervention Impact**
> To what extent can early detection through a validated screening tool inform school-level interventions and contribute to safeguarding and educational policy?

---

## 3. Required Dataset Variables

### 3.1 Primary Instrument: SPIN Questionnaire
The Social Phobia Inventory (Connor et al., 2000) is a 17-item self-report scale. Each item is rated 0–4, yielding a total score of 0–68. Subscales cover three domains:

| Subscale | Items | Domain |
|----------|-------|--------|
| Fear | ~6 items | Fear of embarrassment / social situations |
| Avoidance | ~7 items | Avoidance of social interactions |
| Physiological Arousal | ~4 items | Physical symptoms (blushing, sweating, trembling) |

**Derived variables:**
- Total SPIN score (continuous, 0–68)
- Binary risk label: at-risk (≥19) vs. not at-risk (<19), per validated cut-off
- Subscale scores for fear, avoidance, and physiological arousal

### 3.2 Demographic Variables

| Variable | Type | Rationale |
|----------|------|-----------|
| Age | Continuous / ordinal | Prevalence peaks in early adolescence |
| Gender | Categorical | Gender differences in social anxiety presentation |
| Academic performance (e.g., grades, teacher ratings) | Ordinal / continuous | Correlate of social phobia severity |
| School year / Key Stage | Categorical | Developmental stage context |
| Attendance rate | Continuous | Avoidance behaviour proxy |
| SEN / EHCP status | Binary | Co-occurring conditions flag |

### 3.3 Data Quality Requirements
- Minimum recommended sample: ≥200 students per school for meaningful cross-validation
- Missing data threshold: imputation required if >10% missing per variable
- Anonymisation: all personally identifiable information removed before analysis, compliant with UK GDPR

---

## 4. Candidate Machine Learning Models

### 4.1 Supervised Classification (Primary Task: Risk Level Prediction)

| Model | Rationale | Key Strength | Key Weakness |
|-------|-----------|--------------|--------------|
| **Logistic Regression** | Interpretable baseline; well-suited to binary/ordinal outcomes | Transparent coefficients; easy to audit | Assumes linearity; limited on complex interactions |
| **Decision Tree** | Produces rule-based explanations accessible to educators | Highly interpretable; no scaling needed | Prone to overfitting on small samples |
| **Random Forest** | Ensemble method proven in mental health risk tasks | Handles feature interactions; built-in feature importance | Black-box ensemble; larger memory footprint |
| **Gradient Boosting / XGBoost** | State-of-the-art on tabular data; achieves ~95% accuracy in comparable studies (Jabbar et al., 2018) | High predictive accuracy; handles class imbalance | Requires careful hyperparameter tuning |
| **Support Vector Machine (SVM)** | Effective in high-dimensional, small-sample clinical settings | Good generalisation with small n | Kernel selection adds complexity; less interpretable |
| **Multilayer Perceptron (MLP)** | Captures non-linear patterns across SPIN subscales | Flexible architecture | Needs larger dataset; difficult to explain to stakeholders |

### 4.2 Unsupervised / Clustering (Secondary Task: Subgroup Profiling)

| Model | Rationale |
|-------|-----------|
| **K-Means Clustering** | Identify distinct anxiety profiles (e.g., fear-dominant vs. avoidance-dominant), consistent with Hwang & Lee (2019) |
| **Hierarchical Clustering** | Explore natural groupings in SPIN subscale space without pre-specifying k |

### 4.3 Interpretability Layer (Required for Deployment)

Given the safeguarding context, all deployed models must include an explainability component:

- **SHAP (SHapley Additive exPlanations):** Item-level contribution to individual predictions
- **LIME:** Local surrogate explanations for individual student outputs
- **Feature importance plots:** For educator-facing reporting

### 4.4 Recommended Model Selection Strategy
1. Establish Logistic Regression as the interpretable baseline
2. Compare Random Forest and XGBoost for accuracy
3. Select the final model based on F1-score and sensitivity (minimising false negatives is prioritised — missing an at-risk student carries greater harm than a false positive)
4. Wrap the selected model with SHAP explanations before deployment

---

## 5. Risks and Limitations

### 5.1 Data Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Low school recruitment / insufficient sample size | High | High | Begin outreach in Week 1; target multiple schools; partner with local authorities early |
| Self-report bias in SPIN responses | Medium | High | Supplement with teacher observation ratings where possible; acknowledge in validity discussion |
| Class imbalance (fewer at-risk students in general population) | High | High | Apply SMOTE or class-weighted training; report precision/recall not just accuracy |
| Missing data (incomplete survey responses) | Medium | Medium | Use multiple imputation (MICE); exclude participants with >30% missing |

### 5.2 Methodological Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Overfitting on small dataset | High | High | Stratified k-fold cross-validation (k=5 or 10); held-out test set from separate school |
| Model generalises poorly across school demographics | Medium | High | Include schools with diverse socioeconomic and ethnic profiles; report performance by subgroup |
| Accuracy–interpretability trade-off | Medium | Medium | Prioritise interpretable models (RF + SHAP) over maximum accuracy; co-design outputs with educators |
| SPIN cut-off not validated for all age groups in UK | Low | Medium | Validate against NICE CG159 guidance; consult clinical psychologist |

### 5.3 Ethical and Legal Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GDPR non-compliance | Low | Critical | Full UK GDPR protocol; no PII in model inputs; data held on UK servers |
| Parental consent dropout reducing sample | Medium | High | Provide clear, plain-English information sheets; opt-in design |
| False positives causing unnecessary distress to students | Medium | High | Tool outputs risk bands not diagnoses; mandate follow-up by trained staff before any action |
| False negatives leaving at-risk students undetected | Medium | Critical | Optimise for high sensitivity (recall); set threshold conservatively |
| Algorithmic bias against protected groups | Low | High | Audit model performance across gender, age, SEN status |

### 5.4 Deployment Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Low adoption by schools after deployment | Medium | High | Co-design UI with educators; offer training workshops |
| Data security breach via web application | Low | Critical | Penetration testing before launch; no raw student data stored post-scoring |
| Tool misused as a diagnostic instrument | Medium | High | Clearly label outputs as screening (not diagnostic); embed clinical referral pathway |

---

## 6. Eight-Week Research Roadmap

The proposal's 5-month timeline is condensed into an intensive 8-week academic schedule appropriate for a quantitative research project.

```
Week  1   2   3   4   5   6   7   8
      |---|---|---|---|---|---|---|---|
W1-2  Ethics & School Recruitment
W2-3      Survey Design & Pilot
W3-4          Data Collection
W5                EDA & Preprocessing
W6-7                  Model Dev & Evaluation
W8                            Deployment & Reporting
```

---

### Week 1 — Ethics, Governance, and Stakeholder Engagement
**Goal:** Secure ethical approval and confirm school partnerships

- Submit ethics application (institutional review board + school safeguarding leads)
- Draft parental consent forms and student information sheets (plain English, age-appropriate)
- Identify and contact 3–5 UK partner schools via local authority contacts
- Confirm GDPR data processing agreements with each school
- Set up anonymised, encrypted data storage environment

**Deliverable:** Signed ethics approval and at least 2 confirmed school partners

---

### Week 2 — Survey Instrument Preparation and Pilot
**Goal:** Finalise the SPIN administration protocol and pilot for clarity

- Finalise digital SPIN questionnaire (17 items, 0–4 Likert scale) with demographic questions
- Design the secure online data collection form (e.g., Qualtrics or Microsoft Forms via school SSO)
- Pilot with a small group (10–15 students) to check comprehension and timing
- Refine question wording based on pilot feedback
- Train school staff on administration procedures

**Deliverable:** Validated digital survey instrument ready for deployment

---

### Week 3 — Data Collection
**Goal:** Administer SPIN questionnaire to full student cohort across partner schools

- Roll out SPIN survey across partner schools (target: 200+ students minimum)
- Collect demographic variables: age, gender, year group, academic performance band, attendance rate, SEN status
- Monitor response rates daily; follow up with non-responding classes
- Maintain audit log of data collection activities for GDPR compliance

**Deliverable:** Raw anonymised dataset with SPIN responses + demographics

---

### Week 4 — Exploratory Data Analysis (EDA) and Preprocessing
**Goal:** Understand the dataset and prepare it for modelling

- Assess data completeness; apply MICE imputation for missing values
- Compute descriptive statistics: SPIN score distributions, subscale breakdowns, demographic profiles
- Apply validated SPIN cut-off (score ≥19 = at-risk) to generate binary risk labels
- Visualise class distribution; quantify imbalance ratio
- Correlation analysis: SPIN subscales vs. demographic variables
- Feature engineering: total score, subscale scores, interaction terms (e.g., age × avoidance score)
- Split data: 70% training / 15% validation / 15% test (stratified by risk class and school)

**Deliverable:** Clean, labelled dataset; EDA report with key findings

---

### Week 5 — Baseline Model Development
**Goal:** Train and evaluate initial models; establish baseline performance

- Train Logistic Regression and Decision Tree as interpretable baselines
- Apply SMOTE to training set to address class imbalance
- Evaluate with stratified 5-fold cross-validation
- Metrics: Accuracy, Precision, Recall (Sensitivity), F1-score, AUC-ROC
- Prioritise Recall — minimising false negatives is clinically critical
- Log all experiments in a reproducible notebook (Jupyter)

**Deliverable:** Baseline model results table; initial feature importance rankings

---

### Week 6 — Advanced Model Development and Tuning
**Goal:** Train ensemble and advanced models; optimise the best performer

- Train Random Forest, XGBoost, SVM, and MLP classifiers
- Hyperparameter tuning via Grid Search / Bayesian Optimisation on validation set
- Compare all models on held-out test set; select final model
- Attach SHAP explainer to the selected model
- Run K-Means clustering (k=3: low/moderate/high) for student profile analysis
- Validate cluster profiles with reference to SPIN subscale literature

**Deliverable:** Final model selected with full evaluation metrics; SHAP feature importance plots

---

### Week 7 — Expert Validation and Risk Categorisation
**Goal:** Validate model outputs with clinical and educational experts; define risk bands

- Share anonymised model outputs and SHAP explanations with consulting clinical psychologist
- Define risk bands: Low (<19), Moderate (19–29), High (≥30) aligned with SPIN scoring norms
- Conduct structured feedback session with school SENCO or mental health lead
- Refine model threshold based on expert feedback (trading precision for recall if needed)
- Conduct bias audit: check performance parity across gender, age group, SEN status
- Document model card (intended use, limitations, performance metrics)

**Deliverable:** Clinically validated risk categorisation framework; bias audit report

---

### Week 8 — Tool Deployment, Reporting, and Dissemination
**Goal:** Deploy web application and complete project outputs

- Deploy screening model as a secure web application (Flask/FastAPI backend + simple HTML/CSS frontend)
- Implement role-based access: school staff see risk band + recommended action, not raw scores
- User acceptance testing with 2–3 school staff members
- Write final project report covering methodology, results, limitations, and policy implications
- Prepare 1-page policy brief for local authority / DfE stakeholders
- Draft academic paper outline for peer-reviewed submission
- Archive anonymised dataset and model artefacts per GDPR retention policy

**Deliverable:** Live web application; final report; policy brief; academic paper draft

---

## Summary Table

| Week | Phase | Key Output |
|------|-------|------------|
| 1 | Ethics & Recruitment | Ethics approval, school partnerships |
| 2 | Survey Preparation | Piloted digital SPIN instrument |
| 3 | Data Collection | Raw anonymised dataset |
| 4 | EDA & Preprocessing | Clean dataset, EDA report |
| 5 | Baseline Modelling | Baseline performance benchmarks |
| 6 | Advanced Modelling | Final model + SHAP explanations |
| 7 | Expert Validation | Validated risk bands, bias audit |
| 8 | Deployment & Dissemination | Web app, final report, policy brief |

---

## Key Success Metrics

| Metric | Target |
|--------|--------|
| Minimum student sample | ≥200 participants |
| Model Sensitivity (Recall) | ≥85% (at-risk students correctly identified) |
| Model AUC-ROC | ≥0.80 |
| False Negative Rate | <15% |
| Expert validation agreement | ≥80% concordance with clinical psychologist review |
| School adoption (pilot) | ≥2 schools actively using tool by Week 8 |

---

## References

Connor, K.M. et al. (2000) 'Psychometric properties of the Social Phobia Inventory (SPIN)', *The British Journal of Psychiatry*, 176(4), pp. 379–386.

Hwang, W. and Lee, T. (2019) 'Clustering analysis of social anxiety patterns in school-aged children using SPIN and SASC', *Journal of Child Psychology and Psychiatry*, 60(7), pp. 765–774.

Jabbar, M.A., Deekshatulu, B.L. and Chandra, P. (2018) 'Prediction of social phobia in adolescents using machine learning techniques', *International Journal of Engineering & Technology*, 7(2.7), pp. 321–325.

Morales-Muñoz, I. et al. (2022) 'Anxiety disorders across middle childhood and early adolescence in a UK population-based cohort', *JCPP Advances*, 2(2), e12089.

NICE (2013) *Social anxiety disorder: recognition, assessment and treatment*. Clinical guideline CG159. London: NICE.

Schwartz, C. et al. (2019) 'Six decades of preventing and treating childhood anxiety disorders', *BMJ Mental Health*, 22(3), pp. 103–110.
