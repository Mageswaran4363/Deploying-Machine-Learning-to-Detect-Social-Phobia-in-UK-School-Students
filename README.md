# Deploying Machine Learning to Detect Social Phobia in UK School Students

**Dataset**: UKDA Study 8467 — Mental Health of Children and Young People in England 2017  
**Model**: Logistic Regression + Platt Scaling calibration (v5)  
**Key results**: Recall 0.933 · AUC-ROC 0.938 · ECE 0.005 · n = 3,749

---

## Viewing Results Without Running

The notebook `notebooks/social_phobia_pipeline.ipynb` has **pre-executed outputs** — all charts, metrics, and tables are already saved inside it. You can open it in VS Code or Jupyter and read through the results without running a single cell.

---

## Running the Notebook (requires the dataset)

### 1. Prerequisites

- Python 3.11 or 3.12 recommended (developed on 3.14)
- Git

### 2. Clone the repository

```bash
git clone https://github.com/Mageswaran4363/Deploying-Machine-Learning-to-Detect-Social-Phobia-in-UK-School-Students.git
cd Deploying-Machine-Learning-to-Detect-Social-Phobia-in-UK-School-Students
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Place the dataset

The raw data is from **UKDA Study 8467** under a Special Licence and cannot be shared publicly. Once you have the file, place it here:

```
data/raw/real_v1.csv
```

Apply for access at: https://ukdataservice.ac.uk/

### 5. Register the kernel and launch the notebook

```bash
python -m ipykernel install --user --name social-phobia-venv --display-name "Social Phobia (venv)"
jupyter notebook notebooks/social_phobia_pipeline.ipynb
```

Then select the kernel **Social Phobia (venv)** from the kernel menu and run all cells.

> **Section 6 (Nested CV Training)** has `RETRAIN = False` by default — it skips the 3–4 minute training run and loads the pre-saved model v5 artefacts. Flip to `RETRAIN = True` only if you want to retrain from scratch.

---

## Project Structure

```
├── config/
│   ├── model_registry.yaml      # Version history & metrics for all models
│   ├── schema.json              # Data schema (v2.0)
│   └── dataset_registry.yaml
├── data/
│   └── raw/                     # Place real_v1.csv here (not in repo)
├── documentation/
│   └── model_card.md            # Full model card for v5
├── models/                      # Saved pkl artefacts (v4 base + v5 calibrated)
├── notebooks/
│   └── social_phobia_pipeline.ipynb   # Main notebook (pre-executed)
├── reports/
│   ├── final-project-report.md
│   ├── Academic Paper - Social Phobia Detection in UK Schools.docx
│   └── figures/                 # SHAP plots, fairness audit
├── scripts/
│   ├── run_calibration.py       # Produces model v5 from v4
│   ├── run_explainability.py    # SHAP / surrogate tree
│   └── run_pipeline.py          # Standalone training run
├── src/                         # Core library
│   ├── calibration.py           # PlattCalibratedModel
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── tests/                       # Pytest suite
└── requirements.txt
```

---

## Model Summary

| Version | Model | Recall | AUC-ROC | ECE |
|---------|-------|--------|---------|-----|
| v4 | Logistic Regression | 0.846 | 0.915 | 0.243 |
| **v5** | **LR + Platt Scaling** | **0.933** | **0.938** | **0.005** |

Threshold: **0.0337** (optimised on calibration holdout for high sensitivity)  
Number needed to screen: **7** (flag 7 students to identify 1 true case)

---

## Data Access

The raw dataset (`real_v1.csv`) is under a **UKDA Special Licence** and cannot be shared or committed to any public repository. To reproduce results:

1. Apply for access to UKDA Study 8467 at https://ukdataservice.ac.uk/
2. Place the approved file at `data/raw/real_v1.csv`
3. Run the notebook
