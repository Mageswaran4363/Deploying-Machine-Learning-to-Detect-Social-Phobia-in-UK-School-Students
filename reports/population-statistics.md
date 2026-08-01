# Population Statistics — MHCYP 2017 (UKDA-8467)

**For use in:** paper Methods section (Participants / Dataset) and Results section (Descriptive Statistics)  
**Generated:** 2026-07-31  
**Dataset:** `data/raw/real_v1.csv` (n = 3,749)

---

## Participants

A total of 3,749 school-age participants (aged 11–18 years; M = 14.1, SD = 2.2) were drawn from the Mental Health of Children and Young People in England Survey 2017 (MHCYP 2017; UKDA Study 8467). Participants were nested within 457 primary sampling units (PSU; school clusters), reflecting the original complex survey design. Gender distribution was approximately equal (male 50.1%, female 49.9%).

ICD-10 social phobia was identified via DAWBA clinical ratings (`dcsoph = 2`), yielding **75 confirmed cases (2.0%; 95% CI: 1.6–2.5%)** — a positive-to-negative ratio of 1:49. This is consistent with published UK prevalence estimates for social phobia in this age group (NICE CG159, 2013).

---

## Descriptive Statistics by Case Status

### Table 1. Participant Characteristics

| Variable | Full sample (n = 3,749) | Social phobia (n = 75) | Controls (n = 3,674) |
|---------|------------------------|----------------------|---------------------|
| Age, M (SD) | 14.1 (2.2) | — | — |
| Male, % | 50.1 | — | — |
| SEN status, % | 9.2 | — | — |
| Deprivation Q1–Q2 (deprived), % | 35.0 | — | — |
| Deprivation Q4–Q5 (affluent), % | 43.3 | — | — |

### Table 2. SDQ and WEMWBS Scores by Case Status

| Measure | Full sample M (SD) | Cases M | Controls M | Cohen *d* | *p* |
|---------|-------------------|---------|-----------|----------|-----|
| SDQ Emotional (self) | 2.66 (2.13) | 6.54 | 2.59 | 1.92 | < .001 |
| SDQ Peer (self) | 1.74 (1.56) | 3.74 | 1.70 | 1.32 | < .001 |
| SDQ Emotional (parent) | 2.36 (2.30) | 5.85 | 2.30 | 1.58 | < .001 |
| SDQ Peer (parent) | 1.65 (1.86) | 3.69 | 1.61 | 1.13 | < .001 |
| SDQ Total (self, 0–40) | 9.58 (5.52) | 18.07 | 9.41 | — | < .001 |
| SDQ Total (parent, 0–40) | 8.26 (6.47) | 16.11 | 8.11 | — | < .001 |
| WEMWBS (14–70) | 51.8 (9.6) | 36.8 | 52.1 | −1.63 | < .001 |

All between-group differences (*d* ≥ 1.13) represent large effects by conventional benchmarks (Cohen, 1988). SDQ self-reported emotional problems showed the largest case-control separation (*d* = 1.92).

### Table 3. Bullying Exposure

| Indicator | Non-missing n | Prevalence (%) | Cases (%) | Controls (%) |
|-----------|--------------|----------------|-----------|-------------|
| Bullied at school | 3,177 | 39.1 | 52.0 | 37.5 |
| Cyberbullied | 3,147 | 21.4 | — | — |
| Any bullying | 3,177 | 37.8 | 52.0 | 37.5 |

---

## Missing Data

SDQ parent-report subscales had 8.4% missing values; SDQ self-report subscales had 15.1%; WEMWBS had 16.6%. All were imputed using Multiple Imputation by Chained Equations (MICE; IterativeImputer, 10 iterations, BayesianRidge estimator) with per-column bounds derived from validated scale ranges (SDQ subscales: 0–10; WEMWBS: 14–70). Derived features (internalising composites, wellbeing flag) were recomputed after imputation.

---

## Notes for Manuscript

1. **Prevalence framing:** "Social phobia was present in 2.0% of participants (n=75), consistent with UK population estimates of 1–2% for ICD-10 social phobia in school-age adolescents."

2. **Effect size framing:** "Large between-group differences were observed for all primary features (SDQ emotional *d* = 1.92, WEMWBS *d* = −1.63), suggesting that the SDQ–WEMWBS feature set provides a discriminative signal despite the absence of social-phobia-specific instruments."

3. **Bullying framing:** "Over half of social phobia cases (52.0%) reported any bullying exposure, compared with 37.5% of controls, consistent with the known co-occurrence of victimisation and social anxiety in adolescent populations."

4. **WEMWBS threshold:** "8.2% of the full sample scored below 40 on the WEMWBS — a threshold associated with substantially increased mental health risk — compared with 60%+ of social phobia cases (mean = 36.8)."
